"""DOCX 加载器：按标题层级分节，表格按“字段 | 值”保留列对应关系。"""

from __future__ import annotations

import re
from pathlib import Path

from rag.loaders.base import (
    LoadedSource,
    LoaderError,
    Section,
    derive_doc_id,
    ensure_sections,
    file_sha256,
)

try:
    from docx import Document as DocxDocument
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph
except ImportError:  # pragma: no cover - 环境已安装 python-docx
    DocxDocument = None
    Table = Paragraph = None
    qn = None

HEADING_PATTERN = re.compile(r"(?:Heading|标题)\s*(\d+)", re.IGNORECASE)
STYLE_ID_PATTERN = re.compile(r"Heading(\d+)", re.IGNORECASE)


def heading_level(paragraph) -> int | None:
    style = paragraph.style
    name = (getattr(style, "name", "") or "").strip()
    match = HEADING_PATTERN.search(name)
    if match:
        return int(match.group(1))
    style_id = (getattr(style, "style_id", "") or "").strip()
    match = STYLE_ID_PATTERN.search(style_id)
    if match:
        return int(match.group(1))
    return None


def table_text(table) -> str:
    rows: list[str] = []
    for row in table.rows:
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        if any(cells):
            rows.append(" | ".join(cells))
    if not rows:
        return ""
    return "[表格]\n" + "\n".join(rows)


def _iter_blocks(document):
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def parse_sections(document) -> list[Section]:
    sections: list[Section] = []
    stack: list[str] = []
    buffer: list[str] = []
    buffer_start = 0
    offset = 0
    paragraph_index = 0
    table_index = 0
    buffer_paragraphs: list[int] = []

    def flush() -> None:
        nonlocal buffer, buffer_start, buffer_paragraphs
        if not buffer:
            return
        text = "\n".join(buffer).strip()
        if text:
            if stack:
                chapter = stack[0]
                title = " > ".join(stack)
                location_type = "heading"
                location = title
                start_paragraph = buffer_paragraphs[0] if buffer_paragraphs else 0
                end_paragraph = buffer_paragraphs[-1] if buffer_paragraphs else 0
            else:
                chapter = "正文"
                title = "正文"
                location_type = "paragraph"
                start_paragraph = buffer_paragraphs[0] if buffer_paragraphs else 0
                end_paragraph = buffer_paragraphs[-1] if buffer_paragraphs else 0
                location = f"段落 {start_paragraph}-{end_paragraph}"
            sections.append(
                Section(
                    chapter=chapter,
                    section=title,
                    text=text,
                    location_type=location_type,
                    location=location,
                    start_index=buffer_start,
                    heading_path=list(stack),
                    metadata={
                        "paragraph_start": start_paragraph,
                        "paragraph_end": end_paragraph,
                    },
                )
            )
        buffer = []
        buffer_paragraphs = []

    for block in _iter_blocks(document):
        if isinstance(block, Paragraph):
            paragraph_index += 1
            text = block.text.strip()
            if not text:
                continue
            level = heading_level(block)
            if level is not None:
                flush()
                stack = stack[: level - 1] + [text]
                buffer = [text]
                buffer_start = offset
                buffer_paragraphs = [paragraph_index]
                offset += len(text) + 1
                continue
            if not buffer:
                buffer_start = offset
            buffer.append(text)
            buffer_paragraphs.append(paragraph_index)
            offset += len(text) + 1
        else:
            table_index += 1
            text = table_text(block)
            if not text:
                continue
            if not buffer:
                buffer_start = offset
            buffer.append(text)
            if paragraph_index:
                buffer_paragraphs.append(paragraph_index)
            offset += len(text) + 1
    flush()
    return sections


def load(path: Path, source: str) -> LoadedSource:
    if DocxDocument is None:
        raise LoaderError("解析 DOCX 需要安装 python-docx")
    try:
        document = DocxDocument(str(path))
        sections = parse_sections(document)
    except LoaderError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise LoaderError(
            f"DOCX 解析失败: {path}；{type(exc).__name__}: {exc}"
        ) from exc
    return LoadedSource(
        source=source,
        doc_id=derive_doc_id(source),
        format="docx",
        content_hash=file_sha256(path),
        sections=ensure_sections(sections, path=path, fmt="DOCX"),
        path=path,
    )
