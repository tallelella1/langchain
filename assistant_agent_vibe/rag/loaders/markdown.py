"""Markdown 加载器：保留标题层级与代码块内容。"""

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
    read_text_file,
)

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE_PATTERN = re.compile(r"^\s*(```|~~~)")


def _section(
    chapter: str,
    path: list[str],
    text: str,
    start_index: int,
) -> Section | None:
    body = text.strip()
    if not body:
        return None
    section_title = " > ".join(path) if path else chapter
    return Section(
        chapter=chapter,
        section=section_title,
        text=body,
        location_type="heading",
        location=section_title,
        start_index=start_index,
        heading_path=list(path),
    )


def parse_sections(text: str) -> list[Section]:
    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    total = 0
    for line in lines:
        offsets.append(total)
        total += len(line)

    sections: list[Section] = []
    stack: list[str] = []
    buffer: list[str] = []
    buffer_start = 0
    in_fence = False

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        raw = "".join(buffer)
        if stack:
            chapter = stack[0]
            section = _section(chapter, stack, raw, buffer_start)
        else:
            section = _section("文档说明", ["文档说明"], raw, buffer_start)
        if section is not None:
            sections.append(section)
        buffer = []

    for line_no, line in enumerate(lines):
        if FENCE_PATTERN.match(line):
            in_fence = not in_fence
            if not buffer:
                buffer_start = offsets[line_no]
            buffer.append(line)
            continue
        match = HEADING_PATTERN.match(line.rstrip("\r\n")) if not in_fence else None
        if not match:
            if not buffer and not line.strip():
                continue
            if not buffer:
                buffer_start = offsets[line_no]
            buffer.append(line)
            continue

        flush()
        level = len(match.group(1))
        title = match.group(2).strip()
        if not title:
            continue
        stack = stack[: level - 1] + [title]
        buffer = [line]
        buffer_start = offsets[line_no]
    flush()

    if not sections and text.strip():
        return [
            Section(
                chapter="文档说明",
                section="文档说明",
                text=text.strip(),
                location_type="heading",
                location="文档说明",
                start_index=0,
            )
        ]
    return sections


def load(path: Path, source: str) -> LoadedSource:
    text = read_text_file(path)
    if not text.strip():
        raise LoaderError(f"Markdown 文件为空: {path}")
    sections = parse_sections(text)
    return LoadedSource(
        source=source,
        doc_id=derive_doc_id(source),
        format="markdown",
        content_hash=file_sha256(path),
        sections=ensure_sections(sections, path=path, fmt="Markdown"),
        path=path,
    )
