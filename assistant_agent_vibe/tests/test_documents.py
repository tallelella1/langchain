"""2.1：章节切分、父块映射与索引清单（离线）。"""

import json
from pathlib import Path

import pytest

import config
from rag import documents
from rag.documents import (
    ManifestError,
    load_and_split,
    load_index_manifest,
    load_knowledge_documents,
    preview_chunks,
    save_index_manifest,
    summarize_chunks,
)

REQUIRED_FACTS = [
    "成员人数上限：10 人",
    "已激活成员",
    "外部协作者不占用正式成员名额",
    "基础版最多允许 20 个外部协作者",
    "超过套餐月度额度的 10% 但未超过 50%，则退还 50%",
    "超过套餐月度额度的 50%，则不支持退款",
    "必须先完成红字发票流程",
    "工作区冻结 7 天",
    "再保留 23 天的只读归档期",
    "API 调用额度超出后不会立刻停用",
    "私有化部署不是标准 SaaS 套餐默认内容，需要单独签约",
    "如果企业合同中未单独约定",
]

CHILD_KEYS = {
    "source",
    "chunk_id",
    "parent_id",
    "chapter",
    "section",
    "title",
    "plan",
}


def _load():
    return load_and_split()


def _parent(manifest, chapter, section):
    for parent in manifest["parents"].values():
        if parent["chapter"] == chapter and parent["section"] == section:
            return parent
    raise AssertionError(f"未找到父块: {chapter} / {section}")


def test_load_structure_and_metadata():
    loaded = load_knowledge_documents()
    assert len(loaded) == 1
    assert Path(loaded[0].metadata["source"]).name == "knowledge.txt"

    _, chunks, manifest = _load()
    assert chunks
    for chunk in chunks:
        assert CHILD_KEYS <= set(chunk.metadata)
        parent = manifest["parents"][chunk.metadata["parent_id"]]
        assert chunk.metadata["chunk_id"] in parent["child_ids"]
        assert parent["source"] == chunk.metadata["source"] == "knowledge.txt"
    assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))
    assert len(manifest["children"]) == len(chunks)
    assert all(parent["child_ids"] for parent in manifest["parents"].values())


def test_children_are_not_heading_only_and_within_size():
    _, chunks, _ = _load()
    for chunk in chunks:
        meaningful = documents.meaningful_length(
            chunk.page_content, chunk.metadata["chapter"], chunk.metadata["section"]
        )
        assert meaningful >= config.MIN_CHILD_CHARS
        assert len(chunk.page_content) <= config.CHUNK_SIZE


def test_refund_and_invoice_parents_preserve_conditions():
    _, _, manifest = _load()
    refund = _parent(manifest, "六、退款规则", "2. 基础版与专业版")
    assert "未超过套餐月度额度的 10%" in refund["text"]
    assert "超过套餐月度额度的 10% 但未超过 50%" in refund["text"]
    assert "套餐升级" in refund["text"]
    assert "红字发票流程" in refund["text"]
    assert "基础版" in refund["plan"] and "专业版" in refund["plan"]

    trial = _parent(manifest, "六、退款规则", "1. 试用版")
    assert "试用版为免费版本，不涉及退款" in trial["text"]

    invoice = _parent(manifest, "七、发票规则", "4. 专票补充规则")
    assert "必须先完成红字发票流程" in invoice["text"]


def test_preamble_and_retention_parents():
    _, _, manifest = _load()
    assert "以合同为准" in _parent(manifest, "文档说明", "文档说明")["text"]
    retention = _parent(manifest, "五、数据保留与删除规则", "1. 工作区到期后的数据保留")
    assert "如果企业合同中未单独约定" in retention["text"]


def test_plan_name_is_recorded_in_section_metadata():
    _, chunks, _ = _load()
    basic = [
        chunk
        for chunk in chunks
        if chunk.metadata["chapter"] == "二、套餐说明"
        and chunk.metadata["section"] == "2. 基础版"
    ]
    assert basic
    assert {chunk.metadata["plan"] for chunk in basic} == {"基础版"}


def test_all_business_facts_survive():
    _, chunks, manifest = _load()
    corpus = "\n".join(parent["text"] for parent in manifest["parents"].values())
    source_lines = [
        line.strip()
        for line in config.KNOWLEDGE_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and set(line.strip()) > {"="}
    ]
    assert [line for line in source_lines if line not in corpus] == []
    assert [fact for fact in REQUIRED_FACTS if fact not in corpus] == []
    assert all(chunk.page_content.strip() for chunk in chunks)


def test_split_is_stable():
    _, first, first_manifest = _load()
    _, second, second_manifest = _load()
    assert [c.page_content for c in first] == [c.page_content for c in second]
    assert first_manifest["parents"] == second_manifest["parents"]


def test_manifest_describes_source_and_roundtrips(tmp_path):
    _, chunks, manifest = _load()
    assert manifest["strategy"] == config.CHUNKING_STRATEGY
    assert manifest["collection"] == config.COLLECTION_NAME
    assert manifest["embedding_model"] == config.EMBED_MODEL_NAME
    assert manifest["embedding_dim"] == config.EMBED_DIM
    assert manifest["chunk_size"] == config.CHUNK_SIZE
    assert manifest["chunk_overlap"] == config.CHUNK_OVERLAP
    assert manifest["min_child_chars"] == config.MIN_CHILD_CHARS
    assert manifest["chunk_count"] == len(chunks)
    assert manifest["parent_count"] == len(manifest["parents"])
    assert len(manifest["source_sha256"]) == 64

    path = save_index_manifest(manifest, tmp_path / "index_manifest.json")
    loaded = load_index_manifest(path)
    assert loaded["parents"] == manifest["parents"]
    assert loaded["children"] == manifest["children"]


def test_manifest_rejects_invalid_content_and_missing_file(tmp_path):
    _, _, manifest = _load()

    def drop_parent_text(value):
        parent_id = next(iter(value["parents"]))
        del value["parents"][parent_id]["text"]

    def append_child_id(value):
        parent_id = next(iter(value["parents"]))
        value["parents"][parent_id]["child_ids"].append(999)

    cases = [
        (lambda value: value.update(collection="other"), "collection"),
        (lambda value: value.update(embedding_model="other-model"), "embedding_model"),
        (lambda value: value.update(embedding_dim="abc"), "embedding_dim"),
        (lambda value: value.update(parent_count=999), "parent_count"),
        (lambda value: value.update(source_sha256="0" * 64), "源文件"),
        (drop_parent_text, "text"),
        (append_child_id, "child_ids"),
    ]
    for mutate, expected in cases:
        tampered = json.loads(json.dumps(manifest))
        mutate(tampered)
        with pytest.raises(ManifestError, match=expected):
            documents.validate_index_manifest(tampered)

    with pytest.raises(FileNotFoundError, match="索引清单"):
        load_index_manifest(tmp_path / "missing.json")


def test_summary_and_preview(capsys):
    _, chunks, _ = _load()
    summary = summarize_chunks(chunks)
    assert summary["count"] == len(chunks)
    assert summary["parent_count"] == len({c.metadata["parent_id"] for c in chunks})
    assert summary["sources"] == {"knowledge.txt": len(chunks)}

    preview_chunks(chunks[:2])
    output = capsys.readouterr().out
    for chunk in chunks[:2]:
        assert chunk.metadata["parent_id"] in output
        assert chunk.metadata["chapter"] in output
        assert chunk.metadata["section"] in output
