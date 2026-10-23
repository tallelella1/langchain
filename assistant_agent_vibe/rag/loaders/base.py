"""多格式加载器的公共数据结构与校验（P2.1）。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class LoaderError(RuntimeError):
    """文件无法解析、文本为空或格式不支持。"""


@dataclass
class Section:
    """一个可独立成为父块的章节/页面/标题段。"""

    chapter: str
    section: str
    text: str
    location_type: str
    location: str
    start_index: int = 0
    page: int | None = None
    heading_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedSource:
    source: str
    doc_id: str
    format: str
    content_hash: str
    sections: list[Section]
    path: Path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_doc_id(source: str) -> str:
    """由规范化来源路径生成稳定 doc_id，不依赖切片顺序。"""
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]
    stem = re.sub(r"[^0-9A-Za-z一-鿿]+", "-", Path(source).stem)
    stem = stem.strip("-").lower() or "doc"
    return f"{stem}-{digest}"


def ensure_sections(
    sections: list[Section], *, path: Path, fmt: str
) -> list[Section]:
    """去掉空白 section；全部为空时明确失败，不能作为空文本成功入库。"""
    kept = [section for section in sections if section.text.strip()]
    if not kept:
        raise LoaderError(f"{fmt} 未解析出任何可用文本: {path}")
    for section in kept:
        if not section.chapter.strip() or not section.section.strip():
            raise LoaderError(f"{fmt} section 缺少章节/小节标题: {path}")
        if not section.location.strip():
            raise LoaderError(f"{fmt} section 缺少来源定位: {path}")
    return kept


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise LoaderError(f"文件不是有效的 UTF-8: {path}") from exc
    except OSError as exc:
        raise LoaderError(f"文件不可读: {path}") from exc
