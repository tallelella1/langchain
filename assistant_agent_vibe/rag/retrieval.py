"""检索与结果归一化（步骤 6、P1.2 混合检索）。

- ``vector``：只做 COSINE 向量召回（首版基线）；
- ``hybrid``：向量召回 + 中文 BM25 关键词召回，用 RRF 按名次融合，
  可选 LLM 重排；两路分数分别保留，不直接相加不可比较的分数；
- 默认按 parent_id 补全为完整规则段，并保留子块到父块的引用映射。
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Callable, Sequence

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config  # noqa: E402
from rag import models, reranking, vector_store  # noqa: E402
from rag.documents import load_index_manifest  # noqa: E402
from rag.lexical import BM25Index, build_index  # noqa: E402


class RetrievalError(RuntimeError):
    """检索链路（查询向量、数据库搜索、结果字段、父块补全）失败。"""


def hit_key(hit: dict[str, Any]) -> str:
    """命中的稳定 key：新版集合用 chunk_key，旧集合用 source#chunk_id。"""
    key = hit.get("chunk_key")
    if isinstance(key, str) and key:
        return key
    return f"{hit['source']}#{hit['chunk_id']}"


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
        distance = hit.get("distance", entity.get("distance"))
        missing = [
            name
            for name, value in (
                ("id", record_id),
                ("text", entity.get("text")),
                ("source", entity.get("source")),
                ("chunk_id", entity.get("chunk_id")),
                ("distance", distance),
            )
            if value is None
        ]
        if missing:
            raise RetrievalError(
                f"检索结果第 {index + 1} 条缺少字段: {', '.join(missing)}"
            )
        normalized = {
            "id": record_id,
            "chunk_id": entity["chunk_id"],
            "text": entity["text"],
            "source": entity["source"],
            "score": distance,
            "metric": vector_store.METRIC_TYPE,
        }
        for name in ("chunk_key", "doc_id", "parent_id", "location_type", "location"):
            if entity.get(name) not in (None, ""):
                normalized[name] = entity[name]
        if entity.get("page") is not None:
            normalized["page"] = entity["page"]
        hits.append(normalized)
    return hits


def _attach_chunk_keys(
    hits: Sequence[dict[str, Any]], manifest: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """为旧集合命中补上稳定 key；新版集合直接使用 Milvus 返回的 chunk_key。"""
    children = (manifest or {}).get("children") or {}
    attached: list[dict[str, Any]] = []
    for hit in hits:
        entry = dict(hit)
        if not entry.get("chunk_key"):
            child = _child_for_key(children, entry) if isinstance(children, dict) else None
            entry["chunk_key"] = (
                child.get("chunk_key") if isinstance(child, dict) else None
            ) or hit_key(entry)
        attached.append(entry)
    return attached


def _child_for_key(
    children: dict[str, Any], hit: dict[str, Any]
) -> dict[str, Any] | None:
    key = hit_key(hit)
    child = children.get(key)
    if isinstance(child, dict):
        return child
    derived = f"{hit['source']}#{hit['chunk_id']}"
    child = children.get(derived)
    return child if isinstance(child, dict) else None


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
        key = hit_key(hit)
        child = _child_for_key(children, hit)
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
            merged = {
                **hit,
                "text": parent_text,
                "parent_id": parent_id,
                "parent_title": parent.get("title") or parent.get("section", ""),
                "chapter": parent.get("chapter", ""),
                "section": parent.get("section", ""),
                "plan": parent.get("plan", ""),
                "matched_chunk_id": hit["chunk_id"],
                "matched_chunk_ids": [hit["chunk_id"]],
                "matched_chunk_key": key,
                "matched_chunk_keys": [key],
                "matched_text": hit["text"],
                "context_length": len(parent_text),
            }
            if parent.get("location"):
                merged["location"] = parent["location"]
            if parent.get("location_type"):
                merged["location_type"] = parent["location_type"]
            expanded[parent_id] = merged
            order.append(parent_id)
            continue

        entry = expanded[parent_id]
        if key not in entry["matched_chunk_keys"]:
            entry["matched_chunk_keys"].append(key)
            entry["matched_chunk_ids"].append(hit["chunk_id"])
        if hit["score"] > entry["score"]:
            entry["score"] = hit["score"]
            entry["matched_chunk_id"] = hit["chunk_id"]
            entry["matched_chunk_key"] = key
            entry["matched_text"] = hit["text"]
            entry["id"] = hit["id"]
            entry["chunk_id"] = hit["chunk_id"]
            for name in ("chunk_key", "doc_id", "semantic_score", "bm25_score"):
                if name in hit:
                    entry[name] = hit[name]
    return [expanded[parent_id] for parent_id in order]


# =========================
# 混合检索
# =========================


_LEXICAL_LOCK = threading.Lock()
_LEXICAL_CACHE: dict[str, Any] = {"signature": None, "index": None, "rows": None}


def clear_lexical_cache() -> None:
    with _LEXICAL_LOCK:
        _LEXICAL_CACHE.update({"signature": None, "index": None, "rows": None})


def _lexical_signature(manifest: dict[str, Any] | None, collection: str) -> tuple:
    manifest = manifest or {}
    return (
        collection,
        manifest.get("index_version"),
        manifest.get("collection"),
        manifest.get("chunk_count"),
    )


def _build_lexical_index(
    client: Any,
    collection: str,
    manifest: dict[str, Any] | None,
) -> tuple[BM25Index, dict[str, dict[str, Any]]]:
    signature = _lexical_signature(manifest, collection)
    with _LEXICAL_LOCK:
        if _LEXICAL_CACHE["signature"] == signature and _LEXICAL_CACHE["index"] is not None:
            return _LEXICAL_CACHE["index"], _LEXICAL_CACHE["rows"]
    try:
        rows = vector_store.fetch_all_records(client, collection)
    except vector_store.VectorStoreError as exc:
        raise RetrievalError(f"建立 BM25 索引失败: {exc}") from exc
    documents: list[tuple[str, str]] = []
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = hit_key(row)
        if key in by_key:
            raise RetrievalError(f"集合中存在重复 chunk_key: {key}")
        documents.append((key, row["text"]))
        by_key[key] = row
    index = build_index(
        documents,
        k1=config.BM25_K1,
        b=config.BM25_B,
    )
    with _LEXICAL_LOCK:
        _LEXICAL_CACHE.update(
            {"signature": signature, "index": index, "rows": by_key}
        )
    return index, by_key


def _manifest_child(
    manifest: dict[str, Any] | None, key: str
) -> dict[str, Any]:
    children = (manifest or {}).get("children") or {}
    child = children.get(key) if isinstance(children, dict) else None
    return child if isinstance(child, dict) else {}


def _bm25_hits(
    question: str,
    client: Any,
    collection: str,
    manifest: dict[str, Any] | None,
    k: int,
) -> list[dict[str, Any]]:
    index, rows = _build_lexical_index(client, collection, manifest)
    hits: list[dict[str, Any]] = []
    for lexical_hit in index.search(question, k):
        row = rows[lexical_hit.key]
        child = _manifest_child(manifest, lexical_hit.key)
        hit = {
            **row,
            "chunk_key": lexical_hit.key,
            "score": lexical_hit.score,
            "metric": "BM25",
            "bm25_score": lexical_hit.score,
        }
        for name in ("chapter", "section", "plan"):
            if child.get(name):
                hit[name] = child[name]
        hits.append(hit)
    return hits


def _fuse(
    question: str,
    semantic_hits: Sequence[dict[str, Any]],
    bm25_hits: Sequence[dict[str, Any]],
    *,
    k: int,
    rerank_strategy: str,
    chat_model: Any,
) -> list[dict[str, Any]]:
    semantic_scored = []
    for hit in semantic_hits:
        entry = dict(hit)
        entry["semantic_score"] = hit["score"]
        semantic_scored.append(entry)
    fused = reranking.rrf_fuse(
        [("semantic", semantic_scored), ("bm25", list(bm25_hits))],
        key=hit_key,
        rrf_k=config.RRF_K,
    )
    for hit in fused:
        semantic = hit.get("semantic_score")
        if semantic is None:
            for candidate in semantic_scored:
                if hit_key(candidate) == hit_key(hit):
                    hit["semantic_score"] = candidate["score"]
                    break
        hit["score"] = hit["rrf_score"]
        hit["metric"] = "RRF"
    candidates = fused[: max(config.RERANK_CANDIDATES, k)]
    return reranking.rerank_hits(
        question,
        candidates,
        top_n=k,
        strategy=rerank_strategy,
        chat_model=chat_model,
    )


def retrieve(
    question: str,
    k: int = config.TOP_K,
    *,
    strategy: str | None = None,
    embedder: Any = None,
    client: Any = None,
    expand_parents: bool = True,
    manifest: dict[str, Any] | None = None,
    rerank_strategy: str | None = None,
    chat_model: Any = None,
) -> list[dict[str, Any]]:
    """检索当前问题的 Top-K 子块；默认按 parent_id 补全为完整规则段。"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("查询问题不能为空")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k 必须是正整数，实际 {k!r}")
    strategy = (strategy or config.RETRIEVAL_STRATEGY).lower()
    if strategy not in {"vector", "hybrid"}:
        raise ValueError(f"未知检索策略: {strategy}")
    rerank_strategy = (rerank_strategy or config.RERANK_STRATEGY).lower()

    needs_manifest = (
        expand_parents or strategy == "hybrid" or rerank_strategy != "none"
    )
    if needs_manifest and manifest is None:
        try:
            manifest = load_index_manifest()
        except Exception as exc:
            raise RetrievalError(f"加载索引清单失败: {exc}") from exc
    collection = (
        (manifest or {}).get("collection") or config.COLLECTION_NAME
    )

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
        client = vector_store.open_query_client(collection)
    try:
        if owns_client:
            vector_store.validate_query_target(client, collection)
        semantic_k = (
            max(k, config.HYBRID_SEMANTIC_K) if strategy == "hybrid" else k
        )
        if rerank_strategy != "none":
            semantic_k = max(semantic_k, config.RERANK_CANDIDATES)
        raw_hits = vector_store.search_raw(client, query_vector, semantic_k, collection)
        hits = normalize_hits(raw_hits)
        if manifest is not None or strategy == "hybrid" or rerank_strategy != "none":
            hits = _attach_chunk_keys(hits, manifest)
        if strategy == "hybrid":
            bm25_hits = _bm25_hits(
                question,
                client,
                collection,
                manifest,
                max(k, config.HYBRID_BM25_K),
            )
            hits = _fuse(
                question,
                hits,
                bm25_hits,
                k=k,
                rerank_strategy=rerank_strategy,
                chat_model=chat_model,
            )
        else:
            hits = reranking.rerank_hits(
                question,
                hits[: max(k, config.RERANK_CANDIDATES)],
                top_n=k,
                strategy=rerank_strategy,
                chat_model=chat_model,
            )
    except vector_store.DimensionMismatchError:
        raise
    except vector_store.VectorStoreError as exc:
        raise RetrievalError(f"检索失败: {exc}") from exc
    except RetrievalError:
        raise
    except Exception as exc:
        raise RetrievalError(f"检索失败: {type(exc).__name__}: {exc}") from exc
    finally:
        if owns_client:
            client.close()

    if expand_parents:
        return expand_hits_with_parents(hits[:k], manifest or {})
    return hits[:k]
