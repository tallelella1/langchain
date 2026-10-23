"""步骤 5 / 2.1：入库记录、读回核对、清单重建与错误映射（离线）。"""

import pytest

import config
import ingest
from ingest import build_records, verify_readback
from rag import documents
from rag.documents import load_and_split


def _chunks():
    _, chunks, _ = load_and_split()
    return chunks[:3]


def _vectors(chunks):
    return [[0.01 * index] * config.EMBED_DIM for index in range(len(chunks))]


def test_build_records_success_and_validation():
    chunks = _chunks()
    records = build_records(chunks, _vectors(chunks))
    assert [row["id"] for row in records] == [0, 1, 2]
    assert [row["text"] for row in records] == [c.page_content for c in chunks]
    assert {row["source"] for row in records} == {"knowledge.txt"}

    with pytest.raises(ValueError, match="数量"):
        build_records(chunks, _vectors(chunks)[:2])
    bad_vectors = _vectors(chunks)
    bad_vectors[1] = [0.1, 0.2]
    with pytest.raises(Exception, match="维度"):
        build_records(chunks, bad_vectors)


def test_verify_readback_success_and_failures():
    chunks = _chunks()
    rows = [
        {
            "id": chunk.metadata["chunk_id"],
            "text": chunk.page_content,
            "source": chunk.metadata["source"],
            "chunk_id": chunk.metadata["chunk_id"],
        }
        for chunk in chunks
    ]
    verify_readback(chunks, rows)

    tampered = [dict(row) for row in rows]
    tampered[1]["text"] = "被篡改"
    with pytest.raises(AssertionError, match="文本"):
        verify_readback(chunks, tampered)
    with pytest.raises(AssertionError, match="数量"):
        verify_readback(chunks, [])


def test_run_ingest_reports_missing_config(monkeypatch, capsys):
    def fail_validation(**kwargs):
        raise RuntimeError("缺少 Embedding 配置: BAILIAN_API_KEY")

    monkeypatch.setattr(config, "validate_config", fail_validation)
    assert ingest.run_ingest() == 1
    output = capsys.readouterr().out
    assert "入库失败" in output
    assert "BAILIAN_API_KEY" in output


def test_write_manifest_only_is_deprecated(capsys):
    assert ingest.write_manifest_only() == 1
    output = capsys.readouterr().out
    assert "已停用" in output
    assert "版本化索引" in output


def test_ingest_main_maps_unexpected_error(monkeypatch, capsys):
    monkeypatch.setattr(
        ingest,
        "run_ingest",
        lambda: (_ for _ in ()).throw(ConnectionError("socket closed")),
    )
    assert ingest.main([]) == 1
    output = capsys.readouterr().out
    assert "未预期错误" in output
    assert "socket closed" in output
