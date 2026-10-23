"""P2.1：Markdown/PDF/DOCX 解析与来源定位（离线）。"""

from pathlib import Path

import pytest

from rag import loaders

ASSET = Path(__file__).resolve().parents[1] / "asset" / "load"


def test_markdown_preserves_heading_hierarchy_and_code_fence():
    source = loaders.load_source(ASSET / "06-load.md", "asset/load/06-load.md")
    assert source.format == "markdown"
    assert source.sections
    first = source.sections[0]
    assert first.location_type == "heading"
    assert first.heading_path == ["自然语言处理技术文档"]
    nested = [section for section in source.sections if len(section.heading_path) > 1]
    assert nested
    assert any("2.1" in section.section for section in nested)
    code = "\n".join(section.text for section in source.sections)
    assert "pipeline" in code
    assert "# 创建文本分类管道" in code


def test_pdf_keeps_page_numbers_and_skips_empty_pages():
    source = loaders.load_source(ASSET / "04-sample.pdf", "asset/load/04-sample.pdf")
    assert source.format == "pdf"
    assert source.sections
    assert all(section.location_type == "page" for section in source.sections)
    assert all(section.page and section.page >= 1 for section in source.sections)
    assert all(section.location == f"第 {section.page} 页" for section in source.sections)
    assert len({section.page for section in source.sections}) == len(source.sections)


def test_docx_keeps_paragraph_location_and_table_fields():
    source = loaders.load_source(
        ASSET / "05-sgg_chat.docx", "asset/load/05-sgg_chat.docx"
    )
    assert source.format == "docx"
    text = "\n".join(section.text for section in source.sections)
    assert source.sections[0].location_type in {"heading", "paragraph"}
    assert "|" in text  # 表格列没有被拍平成无分隔文本
    assert "培训" in text or "公司" in text


def test_empty_unsupported_and_missing_files_are_reported(tmp_path):
    empty = tmp_path / "empty.md"
    empty.write_text("   \n", encoding="utf-8")
    with pytest.raises(loaders.LoaderError, match="为空"):
        loaders.load_source(empty, "empty.md")

    unsupported = tmp_path / "data.csv"
    unsupported.write_text("a,b\n", encoding="utf-8")
    with pytest.raises(loaders.LoaderError, match="不支持"):
        loaders.load_source(unsupported, "data.csv")

    with pytest.raises(loaders.LoaderError, match="不存在"):
        loaders.load_source(tmp_path / "missing.txt", "missing.txt")


def test_doc_id_is_stable_for_same_source():
    first = loaders.load_source(ASSET / "06-load.md", "a/b.md")
    second = loaders.load_source(ASSET / "06-load.md", "a/b.md")
    other = loaders.load_source(ASSET / "06-load.md", "a/c.md")
    assert first.doc_id == second.doc_id
    assert first.doc_id != other.doc_id
    assert len(first.content_hash) == 64
