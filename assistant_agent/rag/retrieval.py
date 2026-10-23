"""步骤 6：向量检索与结果归一化。

只对当前问题做一次查询 Embedding；把 PyMilvus 的嵌套结果统一成固定契约：
id / chunk_id / text / source / score / metric。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config  # noqa: E402
from rag import models, vector_store  # noqa: E402
from rag.documents import load_index_manifest  # noqa: E402


class RetrievalError(RuntimeError):
    """检索链路（查询向量、数据库搜索、结果字段、父块补全）失败。"""


def normalize_hits(raw_hits: Sequence[Any]) -> list[dict[str, Any]]:
    """把 PyMilvus 结果转成统一命中结构。

    兼容三种形状：单查询嵌套 ``[[hit, ...]]``、扁平 ``[hit, ...]``、单条 ``hit``。
    其他形状明确报错，不静默丢弃命中。
    """
    if not raw_hits:
        return []
    if isinstance(raw_hits, dict):
        rows = [raw_hits]
    elif not isinstance(raw_hits, (list, tuple)):
        raise RetrievalError(
            f"检索结果结构不符合预期: {type(raw_hits).__name__}"
        )
    elif all(isinstance(item, dict) for item in raw_hits):
        rows = list(raw_hits)
    else:
        first = raw_hits[0]
        if first is None:
            rows = []
        elif isinstance(first, (list, tuple)):
            rows = list(first)
        else:
            raise RetrievalError(
                "检索结果结构不符合预期: 既不是命中列表，也不是单查询嵌套结果"
            )

    hits: list[dict[str, Any]] = []
    for index, hit in enumerate(rows):
        if not isinstance(hit, dict):
            raise RetrievalError(
                f"检索结果第 {index + 1} 条结构不符合预期: {type(hit).__name__}"
            )
        entity = hit.get("entity") or hit
        if not isinstance(entity, dict):
            raise RetrievalError(
                f"检索结果第 {index + 1} 条 entity 结构不符合预期"
            )
        record_id = hit.get("id", entity.get("id"))
        missing = [
            name
            for name, value in (
                ("id", record_id),
                ("text", entity.get("text")),
                ("source", entity.get("source")),
                ("chunk_id", entity.get("chunk_id")),
                ("distance", hit.get("distance")),
            )
            if value is None
        ]
        if missing:
            raise RetrievalError(
                f"检索结果第 {index + 1} 条缺少字段: {', '.join(missing)}"
            )
        hits.append(
            {
                "id": record_id,
                "chunk_id": entity["chunk_id"],
                "text": entity["text"],
                "source": entity["source"],
                "score": hit["distance"],
                "metric": vector_store.METRIC_TYPE,
            }
        )
    return hits


def expand_hits_with_parents(
    hits: Sequence[dict[str, Any]], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    """按子块→父块映射补全完整规则段；同一父块只保留一条（最高分）。"""
    if not hits:
        return []
    parents = manifest.get("parents") or {}
    children = manifest.get("children") or {}
    if not isinstance(parents, dict) or not isinstance(children, dict):
        raise RetrievalError("索引清单 parents/children 结构无效")
    expanded: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for hit in hits:
        key = f"{hit['source']}#{hit['chunk_id']}"
        child = children.get(key)
        if not isinstance(child, dict):
            raise RetrievalError(f"检索命中缺少 parent_id 映射: {key}")
        parent_id = child.get("parent_id")
        parent = parents.get(parent_id)
        if not isinstance(parent, dict):
            raise RetrievalError(f"检索命中的父块不存在: {parent_id}")
        if parent.get("source") != hit["source"]:
            raise RetrievalError(
                f"父块来源与命中来源不一致: {parent.get('source')} != {hit['source']}"
            )
        parent_text = parent.get("text")
        if not isinstance(parent_text, str) or not parent_text.strip():
            raise RetrievalError(f"父块缺少 text: {parent_id}")

        if parent_id not in expanded:
            expanded[parent_id] = {
                **hit,
                "text": parent_text,
                "parent_id": parent_id,
                "parent_title": parent.get("title") or parent.get("section", ""),
                "chapter": parent.get("chapter", ""),
                "section": parent.get("section", ""),
                "plan": parent.get("plan", ""),
                "matched_chunk_id": hit["chunk_id"],
                "matched_chunk_ids": [hit["chunk_id"]],
                "matched_text": hit["text"],
                "context_length": len(parent_text),
            }
            order.append(parent_id)
            continue

        entry = expanded[parent_id]
        entry["matched_chunk_ids"].append(hit["chunk_id"])
        if hit["score"] > entry["score"]:
            entry["score"] = hit["score"]
            entry["matched_chunk_id"] = hit["chunk_id"]
            entry["matched_text"] = hit["text"]
            entry["id"] = hit["id"]
            entry["chunk_id"] = hit["chunk_id"]
    return [expanded[parent_id] for parent_id in order]


def retrieve(
    question: str,
    k: int = config.TOP_K,
    *,
    embedder: Any = None,
    client: Any = None,
    expand_parents: bool = True,
    manifest: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """检索当前问题的 Top-K 子块；默认按 parent_id 补全为完整规则段。"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("查询问题不能为空")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k 必须是正整数，实际 {k!r}")

    if expand_parents and manifest is None:
        try:
            manifest = load_index_manifest()
        except Exception as exc:
            raise RetrievalError(f"加载索引清单失败: {exc}") from exc

    if embedder is None:
        embedder = models.init_embedding_model()
    try:
        query_vector = models.embed_question(embedder, question)
    except ValueError:
        raise
    except Exception as exc:
        raise RetrievalError(f"查询向量化失败: {type(exc).__name__}: {exc}") from exc

    owns_client = client is None
    if owns_client:
        client = vector_store.open_query_client()
    try:
        if owns_client:
            vector_store.validate_query_target(client)
        raw_hits = vector_store.search_raw(client, query_vector, k)
    except vector_store.DimensionMismatchError:
        raise
    except vector_store.VectorStoreError as exc:
        raise RetrievalError(f"检索失败: {exc}") from exc
    except Exception as exc:
        raise RetrievalError(f"检索失败: {type(exc).__name__}: {exc}") from exc
    finally:
        if owns_client:
            client.close()

    hits = normalize_hits(raw_hits)
    if expand_parents:
        return expand_hits_with_parents(hits, manifest or {})
    return hits
