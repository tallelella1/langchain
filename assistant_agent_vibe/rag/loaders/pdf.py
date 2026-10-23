"""PDF 加载器：按页保留页码；扫描件不伪装成空文本成功。"""

from __future__ import annotations

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
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError
except ImportError:  # pragma: no cover - 环境已安装 pypdf
    PdfReader = None
    PdfReadError = Exception


def load(path: Path, source: str) -> LoadedSource:
    if PdfReader is None:
        raise LoaderError("解析 PDF 需要安装 pypdf")
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            if not reader.decrypt(""):
                raise LoaderError(f"PDF 已加密且无法用空密码解密: {path}")
        sections: list[Section] = []
        offset = 0
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            title = f"第 {page_number} 页"
            sections.append(
                Section(
                    chapter=path.stem,
                    section=title,
                    text=text,
                    location_type="page",
                    location=title,
                    start_index=offset,
                    page=page_number,
                )
            )
            offset += len(text)
    except LoaderError:
        raise
    except PdfReadError as exc:
        raise LoaderError(f"PDF 解析失败: {path}；{exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise LoaderError(f"PDF 解析失败: {path}；{type(exc).__name__}: {exc}") from exc

    if not sections:
        raise LoaderError(
            f"PDF 没有可提取文本（可能是扫描件，需要 OCR；本轮不做 OCR）: {path}"
        )
    return LoadedSource(
        source=source,
        doc_id=derive_doc_id(source),
        format="pdf",
        content_hash=file_sha256(path),
        sections=ensure_sections(sections, path=path, fmt="PDF"),
        path=path,
    )
