"""P2.4：FastAPI 问答、流式 SSE、管理鉴权、任务状态与健康检查。"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

import config
from api import service
from api.main import create_app
from api.service import IngestTaskManager, TaskBusyError
from rag import answering, retrieval

RESULT = {
    "session_id": "s1",
    "question": "基础版多少人？",
    "resolved_query": "基础版多少人？",
    "mode": "fixed",
    "answer": "10 人 [片段 1]",
    "sources": [
        {
            "citation": 1,
            "source": "knowledge.txt",
            "chapter": "四、成员与权限规则",
            "section": "1. 成员数计算口径",
            "location": "四、成员与权限规则 / 1. 成员数计算口径",
            "text": "基础版正式成员上限为 10 人。",
        }
    ],
    "citations": [1],
    "invalid_citations": [],
    "insufficient": False,
    "model_calls": 1,
    "tool_calls": 0,
    "tool_errors": [],
    "usage": None,
}


def fake_ask(request):
    return RESULT


def fake_stream(request):
    yield {"event": "start", "question": request.question}
    yield {"event": "delta", "text": "10 人"}
    yield {"event": "sources", "sources": RESULT["sources"]}
    yield {
        "event": "done",
        "answer": "10 人 [片段 1]",
        "sources": RESULT["sources"],
        "citations": [1],
        "invalid_citations": [],
        "insufficient": False,
        "model_calls": 1,
        "tool_calls": 0,
    }


class FakeTasks:
    def __init__(self, busy=False):
        self.busy = busy
        self.started = []

    def start(self, *, force=False):
        if self.busy:
            raise service.TaskBusyError("已有任务")
        task = {
            "task_id": "t1",
            "status": "pending",
            "force": force,
            "created_at": "2026-10-23T00:00:00",
            "started_at": None,
            "finished_at": None,
            "collection": None,
            "index_version": None,
            "record_count": None,
            "reused": None,
            "error": None,
        }
        self.started.append(task)
        return task

    def get(self, task_id):
        if task_id != "t1":
            raise service.TaskNotFoundError(task_id)
        return self.started[0] if self.started else {"task_id": "t1", "status": "pending", "force": False, "created_at": "2026-10-23T00:00:00"}

    def list(self):
        return self.started


def make_client(*, busy=False, ready=True, ask=fake_ask, stream=fake_stream):
    tasks = FakeTasks(busy=busy)
    app = create_app(
        ask_fn=ask,
        stream_fn=stream,
        task_manager=tasks,
        readiness_fn=lambda: {
            "ready": ready,
            "checks": {"milvus": {"ok": ready}},
            "errors": [] if ready else ["Milvus 不可用"],
        },
        knowledge_fn=lambda: {"collection": "docs_v3", "documents": []},
    )
    return TestClient(app), tasks


def test_health_live_and_ready(monkeypatch):
    client, _ = make_client(ready=True)
    assert client.get("/health/live").json() == {"status": "alive"}
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["ready"] is True

    client, _ = make_client(ready=False)
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["ready"] is False


def test_ask_validation_and_response():
    client, _ = make_client()
    response = client.post("/ask", json={"question": "基础版多少人？"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "10 人 [片段 1]"
    assert body["sources"][0]["citation"] == 1
    assert body["mode"] == "fixed"

    assert client.post("/ask", json={"question": "   "}).status_code == 422
    assert client.post("/ask", json={"question": "问题", "k": 0}).status_code == 422


def test_ask_error_mapping():
    def failing(request):
        raise retrieval.RetrievalError("数据库不可达")

    client, _ = make_client(ask=failing)
    assert client.post("/ask", json={"question": "问题"}).status_code == 503

    def model_failing(request):
        raise answering.AnswerGenerationError("模型失败")

    client, _ = make_client(ask=model_failing)
    assert client.post("/ask", json={"question": "问题"}).status_code == 502


def test_stream_sse_events_and_error_without_done():
    client, _ = make_client()
    response = client.post("/ask/stream", json={"question": "基础版多少人？"})
    assert response.status_code == 200
    text = response.text
    assert "event: start" in text
    assert "event: delta" in text
    assert "event: sources" in text
    assert "event: done" in text

    def broken(request):
        yield {"event": "start", "question": request.question}
        raise ConnectionError("model stream broke")

    client, _ = make_client(stream=broken)
    text = client.post("/ask/stream", json={"question": "问题"}).text
    assert "event: error" in text
    assert "event: done" not in text


def test_admin_auth_and_ingest_task(monkeypatch):
    monkeypatch.setattr(config, "ADMIN_TOKEN", None)
    client, _ = make_client()
    assert client.post("/admin/ingest", json={}).status_code == 503

    monkeypatch.setattr(config, "ADMIN_TOKEN", "secret")
    assert client.post("/admin/ingest", json={}).status_code == 401
    assert (
        client.post(
            "/admin/ingest", json={}, headers={"X-Admin-Token": "wrong"}
        ).status_code
        == 401
    )
    response = client.post(
        "/admin/ingest", json={"force": True}, headers={"X-Admin-Token": "secret"}
    )
    assert response.status_code == 202
    assert response.json()["task_id"] == "t1"
    assert response.json()["status"] == "pending"

    response = client.get(
        "/admin/ingest/t1", headers={"X-Admin-Token": "secret"}
    )
    assert response.status_code == 200
    assert response.json()["task_id"] == "t1"
    assert (
        client.get(
            "/admin/ingest/missing", headers={"X-Admin-Token": "secret"}
        ).status_code
        == 404
    )
    assert (
        client.get(
            "/admin/knowledge", headers={"X-Admin-Token": "secret"}
        ).json()["collection"]
        == "docs_v3"
    )


def test_admin_ingest_busy_returns_409(monkeypatch):
    monkeypatch.setattr(config, "ADMIN_TOKEN", "secret")
    client, _ = make_client(busy=True)
    response = client.post(
        "/admin/ingest", json={}, headers={"X-Admin-Token": "secret"}
    )
    assert response.status_code == 409


def test_root_serves_page():
    client, _ = make_client()
    response = client.get("/")
    assert response.status_code == 200
    assert "Atguigu 助手" in response.text
    assert "固定检索" in response.text
    admin = client.get("/admin")
    assert admin.status_code == 200
    assert "知识管理" in admin.text
    assert "创建入库任务" in admin.text


def test_ingest_task_manager_success_and_busy(monkeypatch):
    monkeypatch.setattr(config, "validate_config", lambda **kwargs: None)
    release = threading.Event()
    started = threading.Event()

    class Built:
        collection = "docs_v9"
        record_count = 38
        reused = False
        manifest = {"index_version": 9}

    def build(*, force=False):
        started.set()
        release.wait(timeout=5)
        return Built()

    manager = IngestTaskManager(max_tasks=5, build=build)
    first = manager.start(force=True)
    assert started.wait(timeout=2)
    with pytest.raises(TaskBusyError):
        manager.start()
    release.set()
    for _ in range(100):
        task = manager.get(first["task_id"])
        if task["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.02)
    assert task["status"] == "succeeded"
    assert task["collection"] == "docs_v9"
    assert task["record_count"] == 38


def test_ingest_task_manager_failure(monkeypatch):
    monkeypatch.setattr(config, "validate_config", lambda **kwargs: None)

    def build(*, force=False):
        raise RuntimeError("embedding down")

    manager = IngestTaskManager(build=build)
    task = manager.start()
    for _ in range(100):
        current = manager.get(task["task_id"])
        if current["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.02)
    assert current["status"] == "failed"
    assert "embedding down" in current["error"]


def test_readiness_reports_missing_dependencies(monkeypatch):
    monkeypatch.setattr(config, "missing_env_vars", lambda *names: ["BAILIAN_API_KEY"])
    monkeypatch.setattr(
        service.sources, "load_entries", lambda: [{"source": "knowledge.txt", "format": "txt"}]
    )
    monkeypatch.setattr(
        service.documents,
        "load_index_manifest",
        lambda: {"collection": "docs_v3", "index_version": 3, "documents": {}},
    )
    monkeypatch.setattr(
        service.vector_store,
        "inspect_target",
        lambda collection: {
            "collection": collection,
            "load_state": "Loaded",
            "schema_version": 2,
            "row_count": 38,
        },
    )
    result = service.readiness()
    assert result["ready"] is False
    assert "BAILIAN_API_KEY" in str(result["errors"])
