"""版本化索引构建、回滚与旧集合清理（P1.3）。

路线：
1. 计算来源内容哈希与切分/模型配置摘要，未变化时直接复用当前索引；
2. 在**新集合**完成解析、切分、向量化、写入与读回核对；
3. 用固定问题集抽查新集合，通过后才原子切换 ``data/index_manifest.json``；
4. 旧清单归档到 ``data/manifests/<collection>.json``，切换失败时旧版本继续可用；
5. 旧集合按显式 ``--prune`` 回收，不在构建成功时自动删除。

中断或切换失败会留下未被引用的新集合；它不会成为当前索引，需人工检查或
使用显式清理命令处理。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config  # noqa: E402
from rag import documents, models, retrieval, sources, vector_store  # noqa: E402


class BuildError(RuntimeError):
    """新索引构建、校验或切换失败。"""


@dataclass
class BuildResult:
    collection: str
    manifest: dict[str, Any]
    reused: bool
    previous_collection: str | None
    record_count: int


def _read_manifest_lenient(path: Path) -> dict[str, Any] | None:
    """读取当前清单用于版本计算；内容损坏时返回 None，不伪装成已校验。"""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _active_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return documents.load_index_manifest(path)
    except Exception:
        return _read_manifest_lenient(path)


def _collection_index_version(
    collection: str | None, prefix: str
) -> int:
    if not collection:
        return 0
    if collection.startswith(prefix):
        suffix = collection[len(prefix) :]
        if suffix.isdigit():
            return int(suffix)
    return 0


def next_index_version(
    client: Any,
    previous: dict[str, Any] | None,
    *,
    prefix: str = config.COLLECTION_PREFIX,
) -> tuple[int, str]:
    """下一个未被占用的版本号与集合名；失败构建留下的集合会跳过。"""
    start = int((previous or {}).get("index_version") or 0) + 1
    start = max(
        start,
        _collection_index_version((previous or {}).get("collection"), prefix) + 1,
        1,
    )
    version = start
    while client.has_collection(f"{prefix}{version}"):
        version += 1
    return version, f"{prefix}{version}"


def is_unchanged(
    loaded: Sequence[Any],
    previous: dict[str, Any] | None,
) -> bool:
    """来源集合、内容哈希、切分配置与 Embedding 身份全部一致才复用。"""
    if not previous or previous.get("manifest_version") != documents.MANIFEST_VERSION:
        return False
    if previous.get("schema_version") != config.SCHEMA_VERSION:
        return False
    if previous.get("strategy") != config.CHUNKING_STRATEGY:
        return False
    if previous.get("embedding_model") != config.EMBED_MODEL_NAME:
        return False
    if previous.get("embedding_dim") != config.EMBED_DIM:
        return False
    if previous.get("chunk_config_hash") != documents.chunk_config_hash():
        return False
    previous_documents = previous.get("documents") or {}
    if set(previous_documents) != {item.doc_id for item in loaded}:
        return False
    for item in loaded:
        entry = previous_documents.get(item.doc_id) or {}
        if entry.get("source") != item.source:
            return False
        if entry.get("content_hash") != item.content_hash:
            return False
        if entry.get("format") != item.format:
            return False
    return True


def document_versions(
    loaded: Sequence[Any], previous: dict[str, Any] | None
) -> dict[str, int]:
    """按来源内容哈希递增 document_version；新来源从 1 开始。"""
    previous_documents = (previous or {}).get("documents") or {}
    versions: dict[str, int] = {}
    for item in loaded:
        previous_entry = previous_documents.get(item.doc_id) or {}
        old_version = previous_entry.get("document_version")
        if (
            previous_entry.get("content_hash") == item.content_hash
            and isinstance(old_version, int)
            and old_version > 0
        ):
            versions[item.source] = old_version
        elif isinstance(old_version, int) and old_version > 0:
            versions[item.source] = old_version + 1
        else:
            versions[item.source] = 1
    return versions


def build_records_v2(
    chunks: Sequence[Any],
    vectors: Sequence[Sequence[float]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    if len(chunks) != len(vectors):
        raise BuildError(
            f"切片与向量数量不符: {len(chunks)} 个切片，{len(vectors)} 个向量"
        )
    records: list[dict[str, Any]] = []
    for chunk, vector in zip(chunks, vectors):
        metadata = chunk.metadata
        document = manifest["documents"].get(metadata["doc_id"])
        if not isinstance(document, dict):
            raise BuildError(f"切片缺少文档登记: {metadata['doc_id']}")
        record = {
            "id": metadata["chunk_id"],
            "vector": list(vector),
            "text": chunk.page_content,
            "source": metadata["source"],
            "chunk_id": metadata["chunk_id"],
            "chunk_key": metadata["chunk_key"],
            "doc_id": metadata["doc_id"],
            "parent_id": metadata["parent_id"],
            "content_hash": document["content_hash"],
            "document_version": document["document_version"],
            "index_version": manifest["index_version"],
            "embedding_model": manifest["embedding_model"],
            "chunk_config_hash": manifest["chunk_config_hash"],
            "location_type": metadata["location_type"],
            "location": metadata["location"],
            "page": metadata.get("page", -1),
        }
        vector_store.validate_record(record, schema_version=2)
        records.append(record)
    if len({row["id"] for row in records}) != len(records):
        raise BuildError("切片 id 重复，拒绝写入")
    return records


V2_READBACK_FIELDS = [
    "id",
    "text",
    "source",
    "chunk_id",
    "chunk_key",
    "doc_id",
    "parent_id",
    "content_hash",
    "document_version",
    "index_version",
]


def verify_readback_v2(
    chunks: Sequence[Any],
    rows: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    expected = {chunk.metadata["chunk_id"]: chunk for chunk in chunks}
    if len(rows) != len(expected):
        raise BuildError(f"读回数量不符: 期望 {len(expected)}，实际 {len(rows)}")
    actual_ids = {row.get("id") for row in rows}
    if actual_ids != set(expected):
        raise BuildError(
            f"读回 ID 不符: 缺少 {sorted(set(expected) - actual_ids)}，"
            f"多出 {sorted(actual_ids - set(expected))}"
        )
    for row in rows:
        chunk = expected[row["id"]]
        metadata = chunk.metadata
        if row.get("text") != chunk.page_content:
            raise BuildError(f"读回文本不符: id={row['id']}")
        for field in ("source", "chunk_key", "doc_id", "parent_id"):
            if row.get(field) != metadata[field]:
                raise BuildError(f"读回 {field} 不符: id={row['id']}")
        if row.get("chunk_id") != metadata["chunk_id"]:
            raise BuildError(f"读回 chunk_id 不符: id={row['id']}")
        document = manifest["documents"][metadata["doc_id"]]
        if row.get("content_hash") != document["content_hash"]:
            raise BuildError(f"读回 content_hash 不符: id={row['id']}")
        if row.get("document_version") != document["document_version"]:
            raise BuildError(f"读回 document_version 不符: id={row['id']}")
        if row.get("index_version") != manifest["index_version"]:
            raise BuildError(f"读回 index_version 不符: id={row['id']}")


def _load_verification_cases(ids: Iterable[str]) -> list[dict[str, Any]]:
    from evals.run_eval import load_cases

    wanted = set(ids)
    cases = [case for case in load_cases() if case["id"] in wanted]
    missing = wanted - {case["id"] for case in cases}
    if missing:
        raise BuildError(f"固定问题集缺少: {', '.join(sorted(missing))}")
    return cases


def verify_fixed_questions(
    manifest: dict[str, Any],
    client: Any,
    embedder: Any,
    *,
    ids: Iterable[str] = config.BUILD_VERIFY_IDS,
) -> None:
    """切换前用固定问题抽查新集合；缺少任一预期事实就拒绝切换。"""
    cases = _load_verification_cases(ids)
    for case in cases:
        hits = retrieval.retrieve(
            case["question"],
            config.TOP_K,
            manifest=manifest,
            client=client,
            embedder=embedder,
        )
        text = "\n".join(hit.get("text", "") for hit in hits)
        missing = [fact for fact in case["expected_facts"] if fact not in text]
        if missing:
            raise BuildError(
                f"固定问题 {case['id']} 未召回预期事实: {'、'.join(missing)}"
            )


def _create_collection(
    client: Any,
    collection: str,
    manifest: dict[str, Any],
) -> None:
    try:
        client.create_collection(
            collection_name=collection,
            schema=vector_store.build_schema(2),
            index_params=vector_store.build_index_params(client),
            consistency_level=vector_store.CONSISTENCY_LEVEL,
            properties=vector_store.embedding_identity(
                schema_version=2,
                chunk_config_hash=manifest["chunk_config_hash"],
                index_version=manifest["index_version"],
            ),
        )
        vector_store.ensure_loaded(client, collection)
    except Exception as exc:  # noqa: BLE001
        raise BuildError(
            f"创建或加载新集合失败: {collection}；{type(exc).__name__}: {exc}"
        ) from exc


def build_index(
    entries: Sequence[dict[str, str]] | None = None,
    *,
    force: bool = False,
    client: Any = None,
    embedder: Any = None,
    manifest_path: str | Path | None = None,
    archive_dir: str | Path | None = None,
    collection_prefix: str = config.COLLECTION_PREFIX,
    verify: bool = True,
) -> BuildResult:
    """构建新版本集合并原子切换；返回新清单与集合名。"""
    manifest_path = Path(manifest_path or config.INDEX_MANIFEST_FILE)
    archive_dir = Path(archive_dir or config.MANIFEST_ARCHIVE_DIR)
    owns_client = client is None
    if client is None:
        client = vector_store.open_target(create_database=True)
    previous = _active_manifest(manifest_path)
    new_collection: str | None = None
    try:
        entries = list(entries or sources.load_entries())
        loaded = documents.load_sources_documents(entries)
        if not force and is_unchanged(loaded, previous):
            collection = previous.get("collection")
            if not collection:
                raise BuildError("当前索引清单缺少 collection，无法复用")
            vector_store.validate_query_target(client, collection)
            return BuildResult(
                collection=collection,
                manifest=previous,
                reused=True,
                previous_collection=previous.get("previous_collection"),
                record_count=int(previous.get("chunk_count") or 0),
            )

        version, collection = next_index_version(
            client, previous, prefix=collection_prefix
        )
        new_collection = collection
        versions = document_versions(loaded, previous)
        _, chunks, manifest = documents.load_and_split(
            [entry["source"] for entry in entries],
            document_versions=versions,
            collection=collection,
            collection_prefix=collection_prefix,
            index_version=version,
            previous_collection=(previous or {}).get("collection"),
        )
        manifest["collection"] = collection
        manifest["index_version"] = version
        manifest["created_at"] = datetime.now().isoformat(timespec="seconds")

        _create_collection(client, collection, manifest)

        if embedder is None:
            embedder = models.init_embedding_model()
        texts = [chunk.page_content for chunk in chunks]
        try:
            vectors = models.embed_documents(embedder, texts)
        except Exception as exc:  # noqa: BLE001
            raise BuildError(f"文档向量化失败: {type(exc).__name__}: {exc}") from exc

        records = build_records_v2(chunks, vectors, manifest)
        vector_store.insert_records(
            client, records, collection_name=collection, schema_version=2
        )
        vector_store.flush(client, collection)

        rows = vector_store.fetch_records_by_ids(
            client,
            [record["id"] for record in records],
            collection_name=collection,
            output_fields=V2_READBACK_FIELDS,
        )
        verify_readback_v2(chunks, rows, manifest)
        total = vector_store.count_records(client, collection)
        if total != len(records):
            raise BuildError(
                f"集合实际记录数 {total} 与切片数 {len(records)} 不符"
            )

        hits = vector_store.search_raw(client, vectors[0], 1, collection)
        top = hits[0][0] if hits and hits[0] else None
        if not top or top.get("id") != records[0]["id"]:
            raise BuildError("新集合自向量搜索未召回首条记录")

        if verify:
            verify_fixed_questions(manifest, client, embedder)

        if previous and previous.get("collection"):
            documents.archive_manifest(
                previous, archive_dir / f"{previous['collection']}.json"
            )
        try:
            documents.save_index_manifest(manifest, manifest_path)
        except Exception as exc:  # noqa: BLE001
            raise BuildError(
                f"新集合 {collection} 已构建，但清单切换失败；"
                f"当前索引保持不变。{type(exc).__name__}: {exc}"
            ) from exc
        return BuildResult(
            collection=collection,
            manifest=manifest,
            reused=False,
            previous_collection=(previous or {}).get("collection"),
            record_count=len(records),
        )
    except BuildError:
        raise
    except Exception as exc:  # noqa: BLE001
        suffix = (
            f"；新集合 {new_collection} 未切换，可人工检查"
            if new_collection
            else ""
        )
        raise BuildError(
            f"索引构建失败: {type(exc).__name__}: {exc}{suffix}"
        ) from exc
    finally:
        if owns_client:
            client.close()


def rollback(
    *,
    client: Any = None,
    manifest_path: str | Path | None = None,
    archive_dir: str | Path | None = None,
) -> dict[str, Any]:
    """切换回清单记录的上一版本；旧版本按 stale_sources 明确标记后继续可用。"""
    manifest_path = Path(manifest_path or config.INDEX_MANIFEST_FILE)
    archive_dir = Path(archive_dir or config.MANIFEST_ARCHIVE_DIR)
    active = documents.load_index_manifest(manifest_path)
    previous_collection = active.get("previous_collection")
    if not previous_collection:
        raise BuildError("当前清单没有可回滚的 previous_collection")
    archive_path = archive_dir / f"{previous_collection}.json"
    if not archive_path.is_file():
        raise BuildError(f"回滚清单不存在: {archive_path}")
    try:
        target = json.loads(archive_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildError(f"回滚清单无法读取: {archive_path}；{exc}") from exc
    if not isinstance(target, dict):
        raise BuildError(f"回滚清单格式无效: {archive_path}")
    target["stale_sources"] = True
    target["rollback_from"] = active.get("collection")
    target["rolled_back_at"] = datetime.now().isoformat(timespec="seconds")
    target["previous_collection"] = active.get("collection")
    documents.validate_index_manifest(target)

    owns_client = client is None
    if client is None:
        client = vector_store.open_query_client(previous_collection)
    try:
        vector_store.ensure_loaded(client, previous_collection)
        vector_store.validate_query_target(client, previous_collection)
        documents.archive_manifest(
            active, archive_dir / f"{active.get('collection')}.json"
        )
        documents.save_index_manifest(target, manifest_path)
        return target
    finally:
        if owns_client:
            client.close()


def prune_plan(
    *,
    keep: int = 1,
    manifest_path: str | Path | None = None,
    archive_dir: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    """返回（保留集合，可删除的归档集合）；只处理有归档清单的旧集合。"""
    if keep < 0:
        raise BuildError("keep 不能为负数")
    manifest_path = Path(manifest_path or config.INDEX_MANIFEST_FILE)
    archive_dir = Path(archive_dir or config.MANIFEST_ARCHIVE_DIR)
    active = documents.load_index_manifest(manifest_path)
    active_collection = active.get("collection")
    entries: list[tuple[int, str]] = []
    for path in archive_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        collection = payload.get("collection")
        if not isinstance(collection, str):
            continue
        version = payload.get("index_version")
        if not isinstance(version, int):
            version = _collection_index_version(collection, config.COLLECTION_PREFIX)
        entries.append((version, collection))
    entries.sort(reverse=True)
    kept: list[str] = [active_collection] if active_collection else []
    candidates: list[str] = []
    for _, collection in entries:
        if collection == active_collection:
            continue
        if len(kept) < keep + 1:
            kept.append(collection)
        else:
            candidates.append(collection)
    return kept, candidates


def prune_old_collections(
    *,
    keep: int = 1,
    yes: bool = False,
    client: Any = None,
    manifest_path: str | Path | None = None,
    archive_dir: str | Path | None = None,
) -> list[str]:
    kept, candidates = prune_plan(
        keep=keep, manifest_path=manifest_path, archive_dir=archive_dir
    )
    if not yes:
        raise BuildError(
            f"将保留 {kept}；可删除 {candidates}；"
            "确认后加 --yes 执行（不会删除当前集合）"
        )
    owns_client = client is None
    if client is None:
        client = vector_store.open_target(create_database=False)
    deleted: list[str] = []
    try:
        for collection in candidates:
            try:
                client.drop_collection(collection_name=collection)
            except Exception as exc:  # noqa: BLE001
                raise BuildError(
                    f"删除旧集合失败: {collection}；{type(exc).__name__}: {exc}"
                ) from exc
            deleted.append(collection)
        return deleted
    finally:
        if owns_client:
            client.close()
