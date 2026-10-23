"""TXT 章节切分加载器（沿用首版分隔线与编号小节规则）。"""

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

CHAPTER_PATTERN = re.compile(r"\n?={10,}\n([^\n]+)\n={10,}\n")
SECTION_PATTERN = re.compile(r"(?m)^(\d+)\.\s+(.+?)\s*$")


def _parse_sections(chapter: str, body: str, body_start: int) -> list[Section]:
    sections: list[Section] = []
    matches = list(SECTION_PATTERN.finditer(body))
    if not matches:
        if body.strip():
            sections.append(
                Section(
                    chapter=chapter,
                    section=chapter,
                    text=body.strip(),
                    location_type="section",
                    location=f"{chapter} / {chapter}",
                    start_index=body_start,
                )
            )
        return sections

    prefix = body[: matches[0].start()].strip()
    if prefix:
        title = f"{chapter}（章首说明）"
        sections.append(
            Section(
                chapter=chapter,
                section=title,
                text=prefix,
                location_type="section",
                location=f"{chapter} / {title}",
                start_index=body_start,
            )
        )

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        section_text = body[match.start() : end].strip()
        if not section_text:
            continue
        title = f"{match.group(1)}. {match.group(2).strip()}"
        sections.append(
            Section(
                chapter=chapter,
                section=title,
                text=section_text,
                location_type="section",
                location=f"{chapter} / {title}",
                start_index=body_start + match.start(),
            )
        )
    return sections


def parse_sections(text: str) -> list[Section]:
    """识别文档说明、一级章节与规则小节（不切分正文）。"""
    matches = list(CHAPTER_PATTERN.finditer(text))
    sections: list[Section] = []
    if not matches:
        if text.strip():
            sections.append(
                Section(
                    chapter="文档说明",
                    section="文档说明",
                    text=text.strip(),
                    location_type="section",
                    location="文档说明 / 文档说明",
                    start_index=0,
                )
            )
        return sections

    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(
            Section(
                chapter="文档说明",
                section="文档说明",
                text=preamble,
                location_type="section",
                location="文档说明 / 文档说明",
                start_index=0,
            )
        )

    for index, match in enumerate(matches):
        chapter = match.group(1).strip()
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.extend(_parse_sections(chapter, text[body_start:body_end], body_start))
    return sections


def load(path: Path, source: str) -> LoadedSource:
    text = read_text_file(path)
    if not text.strip():
        raise LoaderError(f"TXT 文件为空: {path}")
    sections = parse_sections(text)
    return LoadedSource(
        source=source,
        doc_id=derive_doc_id(source),
        format="txt",
        content_hash=file_sha256(path),
        sections=ensure_sections(sections, path=path, fmt="TXT"),
        path=path,
    )
