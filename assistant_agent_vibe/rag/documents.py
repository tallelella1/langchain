"""章节切分、父块映射、多文档清单与版本校验（P1.3 / P2.1）。

策略 ``section-parent-v2``：
- 来源由 ``data/sources.json`` 显式登记，默认只有 ``knowledge.txt``；
- 每个来源先由 ``rag.loaders`` 解析为章节/页面/标题段落，再按 220/80 切分子块；
- 父块保存完整规则段、来源定位与子块映射；子块保存稳定 ``chunk_key``；
- 清单记录 ``doc_id/content_hash/document_version/index_version`` 与切分/模型摘要；
- 清单原子切换；旧清单归档后可用于回滚。

用法：
    python -m rag.documents           # 输出块数/父块数、长度分布并预览
    python -m rag.documents --limit 5
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config  # noqa: E402
from rag import loaders, sources  # noqa: E402
from rag.paths import normalize_source  # noqa: E402

SEPARATORS = ["\n==============================\n", "\n\n", "\n", "。", "，", " ", ""]
PLAN_NAMES = ("试用版", "基础版", "专业版", "企业版")
MANIFEST_VERSION = 2
LEGACY_STRATEGY = "section-parent-v1"

CHILD_META_KEYS = (
    "source",
    "doc_id",
    "chunk_id",
    "chunk_key",
    "parent_id",
    "chapter",
    "section",
    "title",
    "plan",
    "start_index",
    "child_index",
    "location_type",
    "location",
)


class ManifestError(RuntimeError):
    """索引清单缺失、损坏或与当前配置/源文件不匹配。"""


def load_knowledge_documents() -> list[Document]:
    """只加载 knowledge.txt（不扫描 asset/load）；保留首版接口。"""
    if not config.KNOWLEDGE_FILE.is_file():
        raise FileNotFoundError(f"知识库文件不存在: {config.KNOWLEDGE_FILE}")
    loader = TextLoader(file_path=str(config.KNOWLEDGE_FILE), encoding="utf-8")
    return loader.load()


def build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        separators=SEPARATORS,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        keep_separator="end",
        add_start_index=True,
    )


def source_sha256(path: str | Path | None = None) -> str:
    path = Path(path or config.KNOWLEDGE_FILE)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def meaningful_length(text: str, chapter: str, section: str) -> int:
    """去掉章节标题、小节标题和分隔线后剩余的有效字符数。"""
    headings = {chapter.strip(), section.strip()}
    parts: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or set(stripped) <= {"="} or stripped in headings:
            continue
        parts.append(stripped)
    return len("".join(parts))


def _detect_plan(section: str) -> str:
    return "、".join(name for name in PLAN_NAMES if name in section)


def load_sources_documents(
    entries: Sequence[dict[str, str]] | None = None,
) -> list[loaders.LoadedSource]:
    """按来源清单解析全部文件；任一文件失败都停止，不跳过。"""
    entries = list(entries or sources.load_entries())
    if not entries:
        raise ValueError("至少需要一个知识来源")
    loaded: list[loaders.LoadedSource] = []
    seen_doc_ids: set[str] = set()
    for entry in entries:
        path = sources.resolve_source(entry["source"])
        document = loaders.load_source(path, entry["source"])
        if document.doc_id in seen_doc_ids:
            raise ValueError(
                f"doc_id 冲突: {document.doc_id}；来源 {entry['source']}"
            )
        seen_doc_ids.add(document.doc_id)
        loaded.append(document)
    return loaded


def _build_chunks_and_parents(
    loaded_sources: Sequence[loaders.LoadedSource],
) -> tuple[list[Document], list[dict[str, Any]]]:
    splitter = build_splitter()
    chunks: list[Document] = []
    parents: list[dict[str, Any]] = []
    next_chunk_id = 0

    for loaded in loaded_sources:
        source = loaded.source
        doc_id = loaded.doc_id
        doc_chunk_index = 0
        for section_index, section in enumerate(loaded.sections):
            parent_text = f"{section.chapter}\n\n{section.text}".strip()
            pieces = splitter.create_documents([parent_text])

            kept: list[tuple[str, int]] = []
            for piece in pieces:
                text = piece.page_content
                if (
                    meaningful_length(text, section.chapter, section.section)
                    < config.MIN_CHILD_CHARS
                ):
                    continue
                kept.append((text, int(piece.metadata.get("start_index", 0))))
            if not kept:
                continue

            parent_id = f"{doc_id}#p{section_index:03d}"
            plan = _detect_plan(section.section)
            child_ids: list[int] = []
            child_keys: list[str] = []
            for child_index, (text, local_start) in enumerate(kept):
                chunk_key = f"{doc_id}#c{doc_chunk_index:04d}"
                metadata = {
                    "source": source,
                    "doc_id": doc_id,
                    "format": loaded.format,
                    "chunk_id": next_chunk_id,
                    "chunk_key": chunk_key,
                    "parent_id": parent_id,
                    "chapter": section.chapter,
                    "section": section.section,
                    "title": section.section,
                    "plan": plan,
                    "start_index": section.start_index + local_start,
                    "child_index": child_index,
                    "location_type": section.location_type,
                    "location": section.location,
                }
                if section.page is not None:
                    metadata["page"] = section.page
                chunks.append(Document(page_content=text, metadata=metadata))
                child_ids.append(next_chunk_id)
                child_keys.append(chunk_key)
                next_chunk_id += 1
                doc_chunk_index += 1

            parent = {
                "parent_id": parent_id,
                "doc_id": doc_id,
                "source": source,
                "chapter": section.chapter,
                "section": section.section,
                "title": section.section,
                "plan": plan,
                "text": parent_text,
                "start_index": section.start_index,
                "length": len(parent_text),
                "child_ids": child_ids,
                "child_keys": child_keys,
                "location_type": section.location_type,
                "location": section.location,
            }
            if section.page is not None:
                parent["page"] = section.page
            parents.append(parent)

    return chunks, parents


def chunk_config(embedding_model: str | None = None) -> dict[str, Any]:
    return {
        "strategy": config.CHUNKING_STRATEGY,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "min_child_chars": config.MIN_CHILD_CHARS,
        "schema_version": config.SCHEMA_VERSION,
        "embedding_model": embedding_model or config.EMBED_MODEL_NAME,
        "embedding_dim": config.EMBED_DIM,
    }


def chunk_config_hash(embedding_model: str | None = None) -> str:
    payload = json.dumps(
        chunk_config(embedding_model), ensure_ascii=False, sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _combined_source_sha256(documents: dict[str, dict[str, Any]]) -> str:
    payload = "|".join(
        f"{doc['source']}:{doc['content_hash']}" for doc in documents.values()
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_index_manifest(
    chunks: list[Document],
    parents: list[dict[str, Any]],
    loaded_sources: Sequence[loaders.LoadedSource],
    *,
    collection: str | None = None,
    collection_prefix: str | None = None,
    index_version: int = 1,
    document_versions: dict[str, int] | None = None,
    previous_collection: str | None = None,
) -> dict[str, Any]:
    """父块全文 + 子块→父块映射 + 多文档版本信息。"""
    document_versions = document_versions or {}
    documents: dict[str, dict[str, Any]] = {}
    for loaded in loaded_sources:
        doc_chunks = [chunk for chunk in chunks if chunk.metadata["doc_id"] == loaded.doc_id]
        doc_parents = [
            parent for parent in parents if parent.get("doc_id") == loaded.doc_id
        ]
        documents[loaded.doc_id] = {
            "doc_id": loaded.doc_id,
            "source": loaded.source,
            "format": loaded.format,
            "content_hash": loaded.content_hash,
            "document_version": int(
                document_versions.get(loaded.source, 1)
            ),
            "chunk_count": len(doc_chunks),
            "parent_count": len(doc_parents),
            "location_type": loaded.sections[0].location_type,
        }

    children: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        metadata = chunk.metadata
        children[metadata["chunk_key"]] = {
            "chunk_key": metadata["chunk_key"],
            "chunk_id": metadata["chunk_id"],
            "doc_id": metadata["doc_id"],
            "parent_id": metadata["parent_id"],
            "source": metadata["source"],
            "format": metadata["format"],
            "chapter": metadata["chapter"],
            "section": metadata["section"],
            "title": metadata["title"],
            "plan": metadata["plan"],
            "start_index": metadata["start_index"],
            "child_index": metadata["child_index"],
            "location_type": metadata["location_type"],
            "location": metadata["location"],
            "length": len(chunk.page_content),
            "text": chunk.page_content,
        }
        if metadata.get("page") is not None:
            children[metadata["chunk_key"]]["page"] = metadata["page"]

    source_list = [loaded.source for loaded in loaded_sources]
    combined_hash = _combined_source_sha256(documents)
    annotation = chunk_config()
    return {
        "manifest_version": MANIFEST_VERSION,
        "schema_version": config.SCHEMA_VERSION,
        "version": 2,
        "strategy": config.CHUNKING_STRATEGY,
        "collection": collection or config.COLLECTION_NAME,
        "collection_prefix": collection_prefix or config.COLLECTION_PREFIX,
        "index_version": int(index_version),
        "previous_collection": previous_collection,
        "source": source_list[0] if len(source_list) == 1 else ",".join(source_list),
        "sources": source_list,
        "source_sha256": combined_hash,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "min_child_chars": config.MIN_CHILD_CHARS,
        "chunking": annotation,
        "chunk_config_hash": chunk_config_hash(),
        "embedding_model": config.EMBED_MODEL_NAME,
        "embedding_dim": config.EMBED_DIM,
        "documents": documents,
        "chunk_count": len(chunks),
        "parent_count": len(parents),
        "parents": {parent["parent_id"]: parent for parent in parents},
        "children": children,
    }


def load_and_split(
    paths: Iterable[str | Path] | None = None,
    *,
    document_versions: dict[str, int] | None = None,
    collection: str | None = None,
    collection_prefix: str | None = None,
    index_version: int = 1,
    previous_collection: str | None = None,
) -> tuple[list[Document], list[Document], dict[str, Any]]:
    """返回（原始 Document、子块、索引清单）；纯离线。"""
    entries = (
        sources.entries_from_paths(paths) if paths is not None else sources.load_entries()
    )
    loaded_sources = load_sources_documents(entries)
    chunks, parents = _build_chunks_and_parents(loaded_sources)
    if not chunks:
        raise ValueError("章节切分结果为空")
    if any(not chunk.page_content.strip() for chunk in chunks):
        raise ValueError("切分结果包含空块")
    chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    if chunk_ids != list(range(len(chunks))):
        raise ValueError("chunk_id 不是连续序号")
    chunk_keys = [chunk.metadata["chunk_key"] for chunk in chunks]
    if len(set(chunk_keys)) != len(chunk_keys):
        raise ValueError("chunk_key 存在重复")

    documents = [
        Document(
            page_content="\n".join(section.text for section in loaded.sections),
            metadata={
                "source": loaded.source,
                "doc_id": loaded.doc_id,
                "format": loaded.format,
                "content_hash": loaded.content_hash,
            },
        )
        for loaded in loaded_sources
    ]
    manifest = build_index_manifest(
        chunks,
        parents,
        loaded_sources,
        collection=collection,
        collection_prefix=collection_prefix,
        index_version=index_version,
        document_versions=document_versions,
        previous_collection=previous_collection,
    )
    return documents, chunks, manifest


# =========================
# 清单校验
# =========================


def _manifest_int(mapping: dict[str, Any], key: str, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{context} 的 {key} 必须是整数，实际 {value!r}")
    return value


def _manifest_str(
    mapping: dict[str, Any], key: str, context: str, *, allow_empty: bool = False
) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ManifestError(f"{context} 缺少有效 {key}")
    return value


def _validate_children_and_parents(
    manifest: dict[str, Any],
    *,
    source_of_doc: dict[str, str],
    check_text: bool,
) -> None:
    parents = manifest.get("parents")
    children = manifest.get("children")
    if not isinstance(parents, dict) or not parents:
        raise ManifestError("索引清单缺少父块")
    if not isinstance(children, dict) or not children:
        raise ManifestError("索引清单缺少子块")
    if _manifest_int(manifest, "parent_count", "索引清单") != len(parents):
        raise ManifestError("索引清单 parent_count 与实际父块数量不符")
    if _manifest_int(manifest, "chunk_count", "索引清单") != len(children):
        raise ManifestError("索引清单 chunk_count 与实际子块数量不符")

    for key, parent in parents.items():
        context = f"父块 {key}"
        if not isinstance(parent, dict):
            raise ManifestError(f"{context} 结构无效")
        if parent.get("parent_id") != key:
            raise ManifestError(f"{context} 的 parent_id 与键不一致")
        doc_id = _manifest_str(parent, "doc_id", context)
        if doc_id not in source_of_doc:
            raise ManifestError(f"{context} 的 doc_id 不存在: {doc_id}")
        if _manifest_str(parent, "source", context) != source_of_doc[doc_id]:
            raise ManifestError(f"{context} 的 source 与文档登记不一致")
        _manifest_str(parent, "chapter", context)
        _manifest_str(parent, "section", context)
        _manifest_str(parent, "title", context)
        _manifest_str(parent, "plan", context, allow_empty=True)
        _manifest_str(parent, "location", context)
        _manifest_str(parent, "location_type", context)
        text = _manifest_str(parent, "text", context)
        if _manifest_int(parent, "length", context) != len(text):
            raise ManifestError(f"{context} 的 length 与 text 不一致")
        _manifest_int(parent, "start_index", context)
        child_ids = parent.get("child_ids")
        if (
            not isinstance(child_ids, list)
            or not child_ids
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in child_ids
            )
        ):
            raise ManifestError(f"{context} 的 child_ids 必须是非空整数列表")
        if len(set(child_ids)) != len(child_ids):
            raise ManifestError(f"{context} 的 child_ids 存在重复")
        child_keys = parent.get("child_keys")
        if (
            not isinstance(child_keys, list)
            or not child_keys
            or any(not isinstance(value, str) or not value for value in child_keys)
        ):
            raise ManifestError(f"{context} 的 child_keys 必须是非空字符串列表")
        if len(set(child_keys)) != len(child_keys):
            raise ManifestError(f"{context} 的 child_keys 存在重复")

    children_by_parent: dict[str, list[str]] = {key: [] for key in parents}
    for key, child in children.items():
        context = f"子块 {key}"
        if not isinstance(child, dict):
            raise ManifestError(f"{context} 结构无效")
        if child.get("chunk_key") != key:
            raise ManifestError(f"{context} 的 chunk_key 与键不一致")
        chunk_id = _manifest_int(child, "chunk_id", context)
        if chunk_id < 0:
            raise ManifestError(f"{context} 的 chunk_id 不能为负数")
        doc_id = _manifest_str(child, "doc_id", context)
        if doc_id not in source_of_doc:
            raise ManifestError(f"{context} 的 doc_id 不存在: {doc_id}")
        if _manifest_str(child, "source", context) != source_of_doc[doc_id]:
            raise ManifestError(f"{context} 的 source 与文档登记不一致")
        parent_id = _manifest_str(child, "parent_id", context)
        parent = parents.get(parent_id)
        if not isinstance(parent, dict):
            raise ManifestError(f"{context} 的 parent_id 不存在: {parent_id}")
        if parent.get("doc_id") != doc_id:
            raise ManifestError(f"{context} 与父块 doc_id 不一致")
        _manifest_str(child, "chapter", context)
        _manifest_str(child, "section", context)
        _manifest_str(child, "title", context)
        _manifest_str(child, "plan", context, allow_empty=True)
        _manifest_str(child, "location", context)
        _manifest_str(child, "location_type", context)
        if _manifest_int(child, "child_index", context) < 0:
            raise ManifestError(f"{context} 的 child_index 不能为负数")
        if _manifest_int(child, "length", context) <= 0:
            raise ManifestError(f"{context} 的 length 必须大于 0")
        _manifest_int(child, "start_index", context)
        if key not in parent["child_keys"]:
            raise ManifestError(f"{context} 不在父块 child_keys 中")
        if check_text:
            text = _manifest_str(child, "text", context)
            if _manifest_int(child, "length", context) != len(text):
                raise ManifestError(f"{context} 的 length 与 text 不一致")
        children_by_parent[parent_id].append(key)

    for key, parent in parents.items():
        if sorted(children_by_parent[key]) != sorted(parent["child_keys"]):
            raise ManifestError(f"父块 {key} 的 child_keys 与 children 映射不一致")
        if len(children_by_parent[key]) != len(parent["child_ids"]):
            raise ManifestError(f"父块 {key} 的 child_ids 与 child_keys 数量不一致")


def _validate_legacy_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("strategy") != LEGACY_STRATEGY:
        raise ManifestError(
            f"索引清单 strategy 为 {manifest.get('strategy')}，"
            f"期望旧版 {LEGACY_STRATEGY}"
        )
    collection = _manifest_str(manifest, "collection", "索引清单")
    if not collection.startswith(config.COLLECTION_PREFIX):
        raise ManifestError(f"索引清单 collection 名称无效: {collection}")
    if manifest.get("embedding_model") != config.EMBED_MODEL_NAME:
        raise ManifestError(
            f"索引清单 embedding_model 为 {manifest.get('embedding_model')}，"
            f"当前配置为 {config.EMBED_MODEL_NAME}；换模型必须重新入库"
        )
    if _manifest_int(manifest, "embedding_dim", "索引清单") != int(config.EMBED_DIM):
        raise ManifestError("索引清单 embedding_dim 与当前配置不一致")
    for key in ("chunk_size", "chunk_overlap", "min_child_chars"):
        if _manifest_int(manifest, key, "索引清单") != getattr(config, key.upper()):
            raise ManifestError(f"索引清单 {key} 与当前配置不一致")
    source = _manifest_str(manifest, "source", "索引清单")
    if not manifest.get("stale_sources"):
        path = sources.resolve_source(source)
        if manifest.get("source_sha256") != source_sha256(path):
            raise ManifestError("源文件已变化，索引清单过期；请重新运行 python ingest.py")

    parents = manifest.get("parents")
    children = manifest.get("children")
    if not isinstance(parents, dict) or not parents:
        raise ManifestError("索引清单缺少父块")
    if not isinstance(children, dict) or not children:
        raise ManifestError("索引清单缺少子块")
    if _manifest_int(manifest, "parent_count", "索引清单") != len(parents):
        raise ManifestError("索引清单 parent_count 与实际父块数量不符")
    if _manifest_int(manifest, "chunk_count", "索引清单") != len(children):
        raise ManifestError("索引清单 chunk_count 与实际子块数量不符")
    for key, parent in parents.items():
        context = f"父块 {key}"
        if not isinstance(parent, dict) or parent.get("parent_id") != key:
            raise ManifestError(f"{context} 结构无效")
        if _manifest_str(parent, "source", context) != source:
            raise ManifestError(f"{context} 的 source 与索引清单不一致")
        text = _manifest_str(parent, "text", context)
        if _manifest_int(parent, "length", context) != len(text):
            raise ManifestError(f"{context} 的 length 与 text 不一致")
    for key, child in children.items():
        context = f"子块 {key}"
        if not isinstance(child, dict):
            raise ManifestError(f"{context} 结构无效")
        chunk_id = _manifest_int(child, "chunk_id", context)
        parent_id = _manifest_str(child, "parent_id", context)
        if key != f"{source}#{chunk_id}":
            raise ManifestError(f"{context} 的键与 source/chunk_id 不一致")
        parent = parents.get(parent_id)
        if not isinstance(parent, dict):
            raise ManifestError(f"{context} 的 parent_id 不存在: {parent_id}")
        if chunk_id not in parent.get("child_ids", []):
            raise ManifestError(f"{context} 不在父块 child_ids 中")


def validate_index_manifest(manifest: dict[str, Any]) -> None:
    """深度核对清单结构、版本与源文件；不匹配必须重新入库。"""
    if not isinstance(manifest, dict):
        raise ManifestError("索引清单格式无效")
    manifest_version = manifest.get("manifest_version", 1)
    if manifest_version == 1:
        _validate_legacy_manifest(manifest)
        return
    if manifest_version != MANIFEST_VERSION:
        raise ManifestError(f"不支持的索引清单版本: {manifest_version}")

    if manifest.get("schema_version") != config.SCHEMA_VERSION:
        raise ManifestError(
            f"索引清单 schema_version 为 {manifest.get('schema_version')}，"
            f"当前配置为 {config.SCHEMA_VERSION}"
        )
    if manifest.get("strategy") != config.CHUNKING_STRATEGY:
        raise ManifestError(
            f"索引清单 strategy 为 {manifest.get('strategy')}，"
            f"当前配置为 {config.CHUNKING_STRATEGY}"
        )
    collection = _manifest_str(manifest, "collection", "索引清单")
    prefix = _manifest_str(manifest, "collection_prefix", "索引清单")
    if not collection.startswith(prefix):
        raise ManifestError(
            f"索引清单 collection 名称无效: {collection}，期望前缀 {prefix}"
        )
    if _manifest_int(manifest, "index_version", "索引清单") <= 0:
        raise ManifestError("索引清单 index_version 必须大于 0")
    if manifest.get("embedding_model") != config.EMBED_MODEL_NAME:
        raise ManifestError(
            f"索引清单 embedding_model 为 {manifest.get('embedding_model')}，"
            f"当前配置为 {config.EMBED_MODEL_NAME}；查询与索引必须使用同一模型"
        )
    if _manifest_int(manifest, "embedding_dim", "索引清单") != int(config.EMBED_DIM):
        raise ManifestError("索引清单 embedding_dim 与当前配置不一致")

    chunking = manifest.get("chunking")
    if not isinstance(chunking, dict):
        raise ManifestError("索引清单缺少 chunking 配置")
    expected_hash = chunk_config_hash(manifest.get("embedding_model"))
    if manifest.get("chunk_config_hash") != expected_hash:
        raise ManifestError("索引清单 chunk_config_hash 与配置摘要不一致")
    if chunking.get("strategy") != manifest.get("strategy"):
        raise ManifestError("索引清单 chunking.strategy 不一致")

    documents = manifest.get("documents")
    if not isinstance(documents, dict) or not documents:
        raise ManifestError("索引清单缺少 documents")
    source_of_doc: dict[str, str] = {}
    for doc_id, document in documents.items():
        context = f"文档 {doc_id}"
        if not isinstance(document, dict):
            raise ManifestError(f"{context} 结构无效")
        if document.get("doc_id") != doc_id:
            raise ManifestError(f"{context} 的 doc_id 与键不一致")
        source = _manifest_str(document, "source", context)
        source_of_doc[doc_id] = source
        _manifest_str(document, "format", context)
        content_hash = _manifest_str(document, "content_hash", context)
        if len(content_hash) != 64:
            raise ManifestError(f"{context} 的 content_hash 长度无效")
        if _manifest_int(document, "document_version", context) <= 0:
            raise ManifestError(f"{context} 的 document_version 必须大于 0")
        if not manifest.get("stale_sources"):
            path = sources.resolve_source(source)
            if source_sha256(path) != content_hash:
                raise ManifestError(
                    f"源文件已变化: {source}；请重新运行 python ingest.py 构建新版本"
                )

    if sorted(manifest.get("sources") or []) != sorted(source_of_doc.values()):
        raise ManifestError("索引清单 sources 与 documents 不一致")
    if manifest.get("source_sha256") != _combined_source_sha256(documents):
        raise ManifestError("索引清单 source_sha256 与源文件内容哈希不一致")

    _validate_children_and_parents(
        manifest, source_of_doc=source_of_doc, check_text=True
    )
    counts: Counter[tuple[str, str]] = Counter()
    for child in manifest["children"].values():
        counts[("chunk", child["doc_id"])] += 1
    for parent in manifest["parents"].values():
        counts[("parent", parent["doc_id"])] += 1
    for doc_id, document in documents.items():
        if _manifest_int(document, "chunk_count", f"文档 {doc_id}") != counts[
            ("chunk", doc_id)
        ]:
            raise ManifestError(f"文档 {doc_id} 的 chunk_count 与实际不符")
        if _manifest_int(document, "parent_count", f"文档 {doc_id}") != counts[
            ("parent", doc_id)
        ]:
            raise ManifestError(f"文档 {doc_id} 的 parent_count 与实际不符")


def active_collection(manifest: dict[str, Any] | None = None) -> str:
    manifest = manifest if manifest is not None else load_index_manifest()
    return manifest.get("collection") or config.COLLECTION_NAME


def save_index_manifest(
    manifest: dict[str, Any], path: str | Path | None = None
) -> Path:
    path = Path(path or config.INDEX_MANIFEST_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temp_path, path)
    return path


def load_index_manifest(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path or config.INDEX_MANIFEST_FILE)
    if not path.is_file():
        raise FileNotFoundError(
            f"索引清单不存在: {path}；请先运行 python ingest.py 生成父块映射"
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"索引清单无法读取: {path}；{exc}") from exc
    validate_index_manifest(manifest)
    return manifest


def archive_manifest(
    manifest: dict[str, Any], path: str | Path | None = None
) -> Path:
    """把旧清单归档到 data/manifests/<collection>.json，供回滚。"""
    collection = manifest.get("collection") or config.COLLECTION_NAME
    path = Path(path or (config.MANIFEST_ARCHIVE_DIR / f"{collection}.json"))
    return save_index_manifest(manifest, path)


def summarize_chunks(chunks: list[Document]) -> dict:
    lengths = [len(chunk.page_content) for chunk in chunks]
    return {
        "count": len(chunks),
        "parent_count": len({chunk.metadata.get("parent_id") for chunk in chunks}),
        "document_count": len({chunk.metadata.get("doc_id") for chunk in chunks}),
        "min_length": min(lengths) if lengths else 0,
        "avg_length": sum(lengths) / len(lengths) if lengths else 0.0,
        "max_length": max(lengths) if lengths else 0,
        "sources": dict(Counter(chunk.metadata["source"] for chunk in chunks)),
        "formats": dict(
            Counter(chunk.metadata.get("format", "") for chunk in chunks)
        ),
        "plans": dict(Counter(chunk.metadata.get("plan", "") for chunk in chunks)),
    }


def preview_chunks(chunks: list[Document]) -> None:
    """输出序号、父块、章节、长度、来源与原文。"""
    for chunk in chunks:
        metadata = chunk.metadata
        print(
            f"\n--- chunk_id={metadata['chunk_id']} "
            f"| chunk_key={metadata['chunk_key']} "
            f"| doc_id={metadata['doc_id']} "
            f"| parent_id={metadata['parent_id']} "
            f"| chapter={metadata['chapter']} "
            f"| section={metadata['section']} "
            f"| plan={metadata.get('plan') or '-'} "
            f"| location={metadata.get('location', '')} "
            f"| len={len(chunk.page_content)} "
            f"| source={metadata['source']} ---"
        )
        print(chunk.page_content)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="知识来源章节切分与父块预览")
    parser.add_argument("--limit", type=int, default=None, help="只预览前 N 个子块")
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="临时指定来源路径（可重复），不修改来源清单",
    )
    args = parser.parse_args(argv)

    config.validate_config()
    _, chunks, manifest = load_and_split(args.source)
    summary = summarize_chunks(chunks)
    print(
        f"策略: {manifest['strategy']}；子块 {summary['count']}，父块 {summary['parent_count']}；"
        f"文档 {summary['document_count']}；长度 min/avg/max: "
        f"{summary['min_length']}/{summary['avg_length']:.1f}/{summary['max_length']}"
    )
    print(f"来源块数: {summary['sources']}")
    print(f"套餐分布: {summary['plans']}")
    selected = chunks if args.limit is None else chunks[: args.limit]
    preview_chunks(selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
