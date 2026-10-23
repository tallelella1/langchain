"""API 核心编排：问答、流式、就绪检查与入库任务（P2.4）。

终端与 API 复用同一套 ``rag`` 核心逻辑；API 层只做参数校验、任务状态和
JSON 序列化，不复制检索或生成规则。
"""

from __future__ import annotations

import json
import threading
import uuid
from collections import OrderedDict
from datetime import datetime
from typing import Any, Callable, Iterator

import config
from rag import conversation, documents, indexer, sources, tools, vector_store

STORE = conversation.ConversationStore(max_turns=config.SESSION_MAX_TURNS)


class TaskBusyError(RuntimeError):
    """已有入库任务在运行；本轮不引入队列（属于 P3）。"""


class TaskNotFoundError(KeyError):
    pass


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, dict)):
        try:
            json.dumps(value, ensure_ascii=False)
            return value
        except TypeError:
            pass
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump()
    return str(value)


def _public_source(source: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "citation",
        "source",
        "doc_id",
        "chunk_key",
        "parent_id",
        "chapter",
        "section",
        "location",
        "location_type",
        "matched_chunk_id",
        "matched_chunk_key",
        "score",
        "metric",
        "text",
    )
    return {name: _jsonable(source[name]) for name in fields if name in source}


def public_result(result: dict[str, Any], mode: str) -> dict[str, Any]:
    return {
        "session_id": result.get("session_id") or "",
        "question": result.get("question") or "",
        "resolved_query": result.get("resolved_query") or result.get("question") or "",
        "mode": mode,
        "answer": result.get("answer") or "",
        "sources": [_public_source(source) for source in result.get("sources", [])],
        "citations": list(result.get("citations", [])),
        "invalid_citations": list(result.get("invalid_citations", [])),
        "insufficient": bool(result.get("insufficient", False)),
        "model_calls": int(result.get("model_calls", 0)),
        "tool_calls": int(result.get("tool_calls", 0)),
        "tool_errors": list(result.get("tool_errors", [])),
        "usage": _jsonable(result.get("usage")),
    }


def ask(request: Any) -> dict[str, Any]:
    """执行一次非流式问答；request 需有 AskRequest 的字段。"""
    session_id = request.session_id or uuid.uuid4().hex
    if request.mode == "tool":
        result = tools.answer_tool_turn(
            request.question,
            store=STORE,
            session_id=session_id,
            k=request.k,
            strategy=request.strategy,
            rerank_strategy=request.rerank,
        )
    else:
        result = conversation.answer_turn(
            request.question,
            store=STORE,
            session_id=session_id,
            k=request.k,
            strategy=request.strategy,
            rerank_strategy=request.rerank,
        )
    return public_result(result, request.mode)


def stream_events(request: Any) -> Iterator[dict[str, Any]]:
    """执行一次流式问答，产出 start/delta/sources/done/error 事件。"""
    session_id = request.session_id or uuid.uuid4().hex
    if request.mode == "tool":
        events = tools.stream_tool_turn(
            request.question,
            store=STORE,
            session_id=session_id,
            k=request.k,
            strategy=request.strategy,
            rerank_strategy=request.rerank,
        )
    else:
        events = conversation.stream_turn(
            request.question,
            store=STORE,
            session_id=session_id,
            k=request.k,
            strategy=request.strategy,
            rerank_strategy=request.rerank,
        )
    for event in events:
        yield _jsonable(event)


class IngestTaskManager:
    """单任务后台入库；任务状态保存在进程内（P3 再引入持久队列）。"""

    def __init__(
        self,
        *,
        max_tasks: int = config.API_MAX_INGEST_TASKS,
        build: Callable[..., Any] = indexer.build_index,
    ) -> None:
        self.max_tasks = max_tasks
        self._build = build
        self._tasks: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._running = False

    def start(self, *, force: bool = False) -> dict[str, Any]:
        with self._lock:
            if self._running:
                raise TaskBusyError("已有入库任务在运行，请稍后查询状态")
            task_id = uuid.uuid4().hex
            task = {
                "task_id": task_id,
                "status": "pending",
                "force": force,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "started_at": None,
                "finished_at": None,
                "collection": None,
                "index_version": None,
                "record_count": None,
                "reused": None,
                "error": None,
            }
            self._tasks[task_id] = task
            self._running = True
            self._trim_locked()
            thread = threading.Thread(
                target=self._run,
                args=(task_id, force),
                name=f"ingest-{task_id[:8]}",
                daemon=True,
            )
            thread.start()
            return dict(task)

    def _trim_locked(self) -> None:
        while len(self._tasks) > self.max_tasks:
            for key, value in list(self._tasks.items()):
                if value["status"] not in {"pending", "running"}:
                    del self._tasks[key]
                    break
            else:
                break

    def _run(self, task_id: str, force: bool) -> None:
        with self._lock:
            task = self._tasks[task_id]
            task["status"] = "running"
            task["started_at"] = datetime.now().isoformat(timespec="seconds")
        try:
            config.validate_config(require_embedding=True)
            result = self._build(force=force)
            with self._lock:
                task.update(
                    {
                        "status": "succeeded",
                        "collection": result.collection,
                        "index_version": result.manifest.get("index_version"),
                        "record_count": result.record_count,
                        "reused": result.reused,
                        "finished_at": datetime.now().isoformat(timespec="seconds"),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                task.update(
                    {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "finished_at": datetime.now().isoformat(timespec="seconds"),
                    }
                )
        finally:
            with self._lock:
                self._running = False

    def get(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise TaskNotFoundError(task_id)
            return dict(task)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(task) for task in self._tasks.values()]


TASKS = IngestTaskManager()


def readiness() -> dict[str, Any]:
    """区分进程存活与依赖就绪；不调用收费模型。"""
    checks: dict[str, Any] = {}
    errors: list[str] = []
    missing = config.missing_env_vars(
        config.EMBED_API_KEY_ENV,
        config.EMBED_BASE_URL_ENV,
        config.LLM_API_KEY_ENV,
        config.LLM_BASE_URL_ENV,
    )
    checks["config"] = {"ok": not missing, "missing": missing}
    if missing:
        errors.append("缺少环境变量: " + ", ".join(missing))

    try:
        entries = sources.load_entries()
        checks["sources"] = {
            "ok": True,
            "count": len(entries),
            "formats": sorted({entry["format"] for entry in entries}),
        }
    except Exception as exc:  # noqa: BLE001
        checks["sources"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        errors.append(f"来源清单不可用: {exc}")
        entries = []

    manifest: dict[str, Any] | None = None
    try:
        manifest = documents.load_index_manifest()
        checks["manifest"] = {
            "ok": True,
            "collection": manifest.get("collection"),
            "index_version": manifest.get("index_version"),
            "chunk_count": manifest.get("chunk_count"),
            "document_count": len(manifest.get("documents") or {}),
            "stale_sources": bool(manifest.get("stale_sources", False)),
        }
    except Exception as exc:  # noqa: BLE001
        checks["manifest"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        errors.append(f"索引清单不可用: {exc}")

    if manifest is not None:
        registry_sources = sorted(entry["source"] for entry in entries)
        manifest_sources = sorted(manifest.get("sources") or [])
        sync_ok = registry_sources == manifest_sources
        checks["sources_sync"] = {
            "ok": sync_ok,
            "registry": registry_sources,
            "manifest": manifest_sources,
        }
        if not sync_ok:
            errors.append("来源清单与活动索引不一致，请重新运行入库任务")

    if manifest is not None:
        try:
            summary = vector_store.inspect_target(manifest["collection"])
            loaded = str(summary.get("load_state", "")).lower() == "loaded"
            checks["milvus"] = {
                "ok": loaded,
                "collection": summary["collection"],
                "load_state": summary["load_state"],
                "schema_version": summary.get("schema_version", 1),
                "row_count": summary.get("row_count"),
            }
            if not loaded:
                errors.append(f"集合未加载: {summary['collection']}")
        except Exception as exc:  # noqa: BLE001
            checks["milvus"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            errors.append(f"Milvus 不可用: {exc}")
    else:
        checks["milvus"] = {"ok": False, "error": "缺少索引清单"}

    return {
        "ready": all(check.get("ok") for check in checks.values()),
        "checks": checks,
        "errors": errors,
    }


def knowledge_status() -> dict[str, Any]:
    manifest = documents.load_index_manifest()
    return {
        "collection": manifest.get("collection"),
        "index_version": manifest.get("index_version"),
        "schema_version": manifest.get("schema_version"),
        "strategy": manifest.get("strategy"),
        "embedding_model": manifest.get("embedding_model"),
        "embedding_dim": manifest.get("embedding_dim"),
        "chunk_count": manifest.get("chunk_count"),
        "parent_count": manifest.get("parent_count"),
        "created_at": manifest.get("created_at"),
        "previous_collection": manifest.get("previous_collection"),
        "stale_sources": bool(manifest.get("stale_sources", False)),
        "sources": manifest.get("sources") or [],
        "documents": [
            {
                "doc_id": doc_id,
                "source": doc.get("source"),
                "format": doc.get("format"),
                "document_version": doc.get("document_version"),
                "content_hash": doc.get("content_hash"),
                "chunk_count": doc.get("chunk_count"),
                "parent_count": doc.get("parent_count"),
            }
            for doc_id, doc in (manifest.get("documents") or {}).items()
        ],
    }
