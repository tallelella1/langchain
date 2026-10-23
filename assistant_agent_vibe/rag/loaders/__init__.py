"""多格式加载器注册表（P2.1）。

支持 TXT、Markdown、PDF、DOCX；扫描 PDF 的 OCR、复杂表格重建与版面识别
不在本轮范围。``asset/load`` 中的学习素材只有被显式加入来源清单才入库。
"""

from __future__ import annotations

from pathlib import Path

from rag.loaders import docx, markdown, pdf, text
from rag.loaders.base import LoadedSource, LoaderError, Section

LOADERS = {
    ".txt": text.load,
    ".md": markdown.load,
    ".markdown": markdown.load,
    ".pdf": pdf.load,
    ".docx": docx.load,
}

SUPPORTED_SUFFIXES = tuple(sorted(LOADERS))


def loader_for(path: str | Path):
    suffix = Path(path).suffix.lower()
    loader = LOADERS.get(suffix)
    if loader is None:
        raise LoaderError(
            f"不支持的文档格式: {suffix or '无扩展名'}；"
            f"当前支持 {', '.join(SUPPORTED_SUFFIXES)}"
        )
    return loader


def load_source(path: str | Path, source: str) -> LoadedSource:
    """按扩展名解析文件；解析失败或空文本都抛出 LoaderError。"""
    path = Path(path)
    if not path.is_file():
        raise LoaderError(f"知识文件不存在: {path}")
    loader = loader_for(path)
    try:
        return loader(path, source)
    except LoaderError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise LoaderError(
            f"加载文件失败: {path}；{type(exc).__name__}: {exc}"
        ) from exc


__all__ = [
    "LOADERS",
    "SUPPORTED_SUFFIXES",
    "LoadedSource",
    "LoaderError",
    "Section",
    "loader_for",
    "load_source",
]
