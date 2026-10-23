"""2.1 章节切分、父块映射和索引清单（离线，不调用模型/数据库）。

策略 ``section-parent-v1``：
- 先按一级章节（分隔线 + 标题）识别章节，再按 ``N. 小节标题`` 识别套餐名/规则小节；
- 小节全文作为父块（完整规则与适用范围），小节再按 220/80 切分为子块；
- 子块元数据保留 chapter/section/plan/parent_id；只有标题或分隔线的子块不独立入库；
- 生成索引清单（父块全文 + 子块→父块映射），供检索按 parent_id 补全上下文。

用法：
    python -m rag.documents           # 输出块数/父块数、长度分布并预览
    python -m rag.documents --limit 5
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config  # noqa: E402  （sys.path 调整后导入）

SEPARATORS = ["\n==============================\n", "\n\n", "\n", "。", "，", " ", ""]
CHAPTER_PATTERN = re.compile(r"\n?={10,}\n([^\n]+)\n={10,}\n")
SECTION_PATTERN = re.compile(r"(?m)^(\d+)\.\s+(.+?)\s*$")
PLAN_NAMES = ("试用版", "基础版", "专业版", "企业版")

CHILD_META_KEYS = (
    "source",
    "chunk_id",
    "parent_id",
    "chapter",
    "section",
    "title",
    "plan",
    "start_index",
    "child_index",
)


class ManifestError(RuntimeError):
    """索引清单缺失、损坏或与当前配置/源文件不匹配。"""


def normalize_source(source: str | Path) -> str:
    """把加载器给出的路径规范为项目内相对路径（统一使用 /）。"""
    path = Path(source).resolve()
    try:
        return path.relative_to(config.PROJECT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def load_knowledge_documents() -> list[Document]:
    """只加载 knowledge.txt（不扫描 asset/load）。"""
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


def source_sha256() -> str:
    return hashlib.sha256(config.KNOWLEDGE_FILE.read_bytes()).hexdigest()


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


def _parse_sections(chapter: str, body: str, body_start: int) -> list[dict[str, Any]]:
    """把一级章节正文拆成小节目录；无编号时整章为一个小节。"""
    sections: list[dict[str, Any]] = []
    matches = list(SECTION_PATTERN.finditer(body))
    if not matches:
        if body.strip():
            sections.append(
                {
                    "chapter": chapter,
                    "section": chapter,
                    "text": body.strip(),
                    "start_index": body_start,
                }
            )
        return sections

    prefix = body[: matches[0].start()].strip()
    if prefix:
        sections.append(
            {
                "chapter": chapter,
                "section": f"{chapter}（章首说明）",
                "text": prefix,
                "start_index": body_start,
            }
        )

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        section_text = body[match.start() : end].strip()
        if not section_text:
            continue
        sections.append(
            {
                "chapter": chapter,
                "section": f"{match.group(1)}. {match.group(2).strip()}",
                "text": section_text,
                "start_index": body_start + match.start(),
            }
        )
    return sections


def parse_sections(text: str) -> list[dict[str, Any]]:
    """识别文档说明、一级章节与规则小节（不切分正文）。"""
    matches = list(CHAPTER_PATTERN.finditer(text))
    sections: list[dict[str, Any]] = []
    if not matches:
        if text.strip():
            sections.append(
                {
                    "chapter": "文档说明",
                    "section": "文档说明",
                    "text": text.strip(),
                    "start_index": 0,
                }
            )
        return sections

    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(
            {
                "chapter": "文档说明",
                "section": "文档说明",
                "text": preamble,
                "start_index": 0,
            }
        )

    for index, match in enumerate(matches):
        chapter = match.group(1).strip()
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.extend(_parse_sections(chapter, text[body_start:body_end], body_start))
    return sections


def _build_chunks_and_parents(
    sections: list[dict[str, Any]],
) -> tuple[list[Document], list[dict[str, Any]]]:
    splitter = build_splitter()
    source = normalize_source(config.KNOWLEDGE_FILE)
    chunks: list[Document] = []
    parents: list[dict[str, Any]] = []
    next_chunk_id = 0

    for section in sections:
        chapter = section["chapter"]
        section_title = section["section"]
        parent_text = f"{chapter}\n\n{section['text']}".strip()
        pieces = splitter.create_documents([parent_text])

        kept: list[tuple[str, int]] = []
        for piece in pieces:
            text = piece.page_content
            if meaningful_length(text, chapter, section_title) < config.MIN_CHILD_CHARS:
                continue
            kept.append((text, int(piece.metadata.get("start_index", 0))))
        if not kept:
            # 只有标题/分隔线的“小节”不成为知识，也不生成父块。
            continue

        parent_id = f"{source}#p{len(parents):03d}"
        plan = _detect_plan(section_title)
        child_ids: list[int] = []
        for child_index, (text, local_start) in enumerate(kept):
            metadata = {
                "source": source,
                "chunk_id": next_chunk_id,
                "parent_id": parent_id,
                "chapter": chapter,
                "section": section_title,
                "title": section_title,
                "plan": plan,
                "start_index": section["start_index"] + local_start,
                "child_index": child_index,
            }
            chunks.append(Document(page_content=text, metadata=metadata))
            child_ids.append(next_chunk_id)
            next_chunk_id += 1

        parents.append(
            {
                "parent_id": parent_id,
                "source": source,
                "chapter": chapter,
                "section": section_title,
                "title": section_title,
                "plan": plan,
                "text": parent_text,
                "start_index": section["start_index"],
                "length": len(parent_text),
                "child_ids": child_ids,
            }
        )

    return chunks, parents


def build_index_manifest(
    chunks: list[Document], parents: list[dict[str, Any]]
) -> dict[str, Any]:
    """父块全文 + 子块→父块映射；用于查询时上下文补全与失效检查。"""
    children = {}
    for chunk in chunks:
        metadata = chunk.metadata
        children[f"{metadata['source']}#{metadata['chunk_id']}"] = {
            "source": metadata["source"],
            "chunk_id": metadata["chunk_id"],
            "parent_id": metadata["parent_id"],
            "chapter": metadata["chapter"],
            "section": metadata["section"],
            "title": metadata["title"],
            "plan": metadata["plan"],
            "start_index": metadata["start_index"],
            "child_index": metadata["child_index"],
            "length": len(chunk.page_content),
        }
    return {
        "version": 1,
        "strategy": config.CHUNKING_STRATEGY,
        "source": normalize_source(config.KNOWLEDGE_FILE),
        "source_sha256": source_sha256(),
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "min_child_chars": config.MIN_CHILD_CHARS,
        "collection": config.COLLECTION_NAME,
        "embedding_model": config.EMBED_MODEL_NAME,
        "embedding_dim": config.EMBED_DIM,
        "chunk_count": len(chunks),
        "parent_count": len(parents),
        "parents": {parent["parent_id"]: parent for parent in parents},
        "children": children,
    }


def load_and_split() -> tuple[list[Document], list[Document], dict[str, Any]]:
    """返回（原始 Document、子块、索引清单）；纯离线。"""
    documents = load_knowledge_documents()
    text = documents[0].page_content
    sections = parse_sections(text)
    chunks, parents = _build_chunks_and_parents(sections)
    if not chunks:
        raise ValueError("章节切分结果为空")
    if any(not chunk.page_content.strip() for chunk in chunks):
        raise ValueError("切分结果包含空块")
    chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    if chunk_ids != list(range(len(chunks))):
        raise ValueError("chunk_id 不是连续序号")
    return documents, chunks, build_index_manifest(chunks, parents)


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


def validate_index_manifest(manifest: dict[str, Any]) -> None:
    """深度核对清单与当前源文件、配置及父子映射；不匹配必须重新入库。"""
    if not isinstance(manifest, dict):
        raise ManifestError("索引清单格式无效")
    if manifest.get("strategy") != config.CHUNKING_STRATEGY:
        raise ManifestError(
            f"索引清单 strategy 为 {manifest.get('strategy')}，"
            f"当前配置为 {config.CHUNKING_STRATEGY}"
        )
    if manifest.get("collection") != config.COLLECTION_NAME:
        raise ManifestError(
            f"索引清单 collection 为 {manifest.get('collection')}，"
            f"当前配置为 {config.COLLECTION_NAME}；请重新入库"
        )
    if manifest.get("embedding_model") != config.EMBED_MODEL_NAME:
        raise ManifestError(
            f"索引清单 embedding_model 为 {manifest.get('embedding_model')}，"
            f"当前配置为 {config.EMBED_MODEL_NAME}；换模型必须重新入库"
        )
    if _manifest_int(manifest, "embedding_dim", "索引清单") != int(config.EMBED_DIM):
        raise ManifestError(
            f"索引清单 embedding_dim 为 {manifest.get('embedding_dim')}，"
            f"当前配置为 {config.EMBED_DIM}"
        )
    if _manifest_int(manifest, "chunk_size", "索引清单") != config.CHUNK_SIZE:
        raise ManifestError("索引清单 chunk_size 与当前配置不一致")
    if _manifest_int(manifest, "chunk_overlap", "索引清单") != config.CHUNK_OVERLAP:
        raise ManifestError("索引清单 chunk_overlap 与当前配置不一致")
    if _manifest_int(manifest, "min_child_chars", "索引清单") != config.MIN_CHILD_CHARS:
        raise ManifestError("索引清单 min_child_chars 与当前配置不一致")
    if manifest.get("source_sha256") != source_sha256():
        raise ManifestError("源文件已变化，索引清单过期；请重新运行 python ingest.py")
    source = normalize_source(config.KNOWLEDGE_FILE)
    if manifest.get("source") != source:
        raise ManifestError(
            f"索引清单 source 为 {manifest.get('source')}，期望 {source}"
        )

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
        if _manifest_str(parent, "source", context) != source:
            raise ManifestError(f"{context} 的 source 与索引清单不一致")
        _manifest_str(parent, "chapter", context)
        _manifest_str(parent, "section", context)
        _manifest_str(parent, "title", context)
        _manifest_str(parent, "plan", context, allow_empty=True)
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

    for key, child in children.items():
        context = f"子块 {key}"
        if not isinstance(child, dict):
            raise ManifestError(f"{context} 结构无效")
        chunk_id = _manifest_int(child, "chunk_id", context)
        if key != f"{source}#{chunk_id}":
            raise ManifestError(f"{context} 的键与 source/chunk_id 不一致")
        parent_id = _manifest_str(child, "parent_id", context)
        parent = parents.get(parent_id)
        if not isinstance(parent, dict):
            raise ManifestError(f"{context} 的 parent_id 不存在: {parent_id}")
        if _manifest_str(child, "source", context) != source:
            raise ManifestError(f"{context} 的 source 与索引清单不一致")
        if parent.get("source") != source:
            raise ManifestError(f"{context} 的父块来源不一致")
        _manifest_str(child, "chapter", context)
        _manifest_str(child, "section", context)
        _manifest_str(child, "title", context)
        _manifest_str(child, "plan", context, allow_empty=True)
        if _manifest_int(child, "child_index", context) < 0:
            raise ManifestError(f"{context} 的 child_index 不能为负数")
        if _manifest_int(child, "length", context) <= 0:
            raise ManifestError(f"{context} 的 length 必须大于 0")
        _manifest_int(child, "start_index", context)
        if chunk_id not in parent["child_ids"]:
            raise ManifestError(f"{context} 不在父块 child_ids 中")

    children_by_parent: dict[str, list[int]] = {key: [] for key in parents}
    for child in children.values():
        children_by_parent[child["parent_id"]].append(child["chunk_id"])
    for key, parent in parents.items():
        if sorted(children_by_parent[key]) != sorted(parent["child_ids"]):
            raise ManifestError(f"父块 {key} 的 child_ids 与 children 映射不一致")


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


def summarize_chunks(chunks: list[Document]) -> dict:
    lengths = [len(chunk.page_content) for chunk in chunks]
    return {
        "count": len(chunks),
        "parent_count": len({chunk.metadata.get("parent_id") for chunk in chunks}),
        "min_length": min(lengths) if lengths else 0,
        "avg_length": sum(lengths) / len(lengths) if lengths else 0.0,
        "max_length": max(lengths) if lengths else 0,
        "sources": dict(Counter(chunk.metadata["source"] for chunk in chunks)),
        "plans": dict(Counter(chunk.metadata.get("plan", "") for chunk in chunks)),
    }


def preview_chunks(chunks: list[Document]) -> None:
    """输出序号、父块、章节、长度、来源与原文。"""
    for chunk in chunks:
        metadata = chunk.metadata
        print(
            f"\n--- chunk_id={metadata['chunk_id']} "
            f"| parent_id={metadata['parent_id']} "
            f"| chapter={metadata['chapter']} "
            f"| section={metadata['section']} "
            f"| plan={metadata.get('plan') or '-'} "
            f"| len={len(chunk.page_content)} "
            f"| source={metadata['source']} ---"
        )
        print(chunk.page_content)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="knowledge.txt 章节切分与父块预览")
    parser.add_argument("--limit", type=int, default=None, help="只预览前 N 个子块")
    args = parser.parse_args(argv)

    config.validate_config()
    _, chunks, manifest = load_and_split()
    summary = summarize_chunks(chunks)
    print(
        f"策略: {manifest['strategy']}；子块 {summary['count']}，父块 {summary['parent_count']}；"
        f"长度 min/avg/max: "
        f"{summary['min_length']}/{summary['avg_length']:.1f}/{summary['max_length']}"
    )
    print(f"来源块数: {summary['sources']}")
    print(f"套餐分布: {summary['plans']}")
    selected = chunks if args.limit is None else chunks[: args.limit]
    preview_chunks(selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
