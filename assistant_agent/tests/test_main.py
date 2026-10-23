"""步骤 8 / 2.1：终端证据、来源与错误映射（离线）。"""

import config
import main
from main import format_evidence, format_sources
from rag import vector_store

HITS = [
    {
        "id": 3,
        "chunk_id": 3,
        "text": "基础版正式成员上限为 10 人。",
        "source": "knowledge.txt",
        "score": 0.91234,
        "metric": "COSINE",
    },
    {
        "id": 4,
        "chunk_id": 4,
        "text": "外部协作者不占用正式成员名额。",
        "source": "knowledge.txt",
        "score": 0.8567,
        "metric": "COSINE",
    },
]

PARENT_HITS = [
    {
        **HITS[0],
        "text": "六、退款规则\n\n2. 基础版与专业版\n完整规则。",
        "parent_id": "knowledge.txt#p006",
        "chapter": "六、退款规则",
        "section": "2. 基础版与专业版",
        "matched_chunk_id": 30,
        "matched_chunk_ids": [30, 31],
        "context_length": 500,
    }
]


def test_evidence_and_sources_for_child_hits_and_empty():
    evidence = format_evidence(HITS)
    assert "[片段 1]" in evidence and "[片段 2]" in evidence
    assert "score=0.9123" in evidence
    assert "source=knowledge.txt" in evidence
    assert "chunk_id=3" in evidence
    assert HITS[0]["text"] in evidence
    assert "未召回" in format_evidence([])

    sources = [{"citation": 1, **HITS[0]}, {"citation": 2, **HITS[1]}]
    output = format_sources(sources)
    assert "knowledge.txt" in output
    assert "chunk_id=3" in output and "chunk_id=4" in output
    assert format_sources([]) == "（本次未使用知识片段）"


def test_parent_evidence_and_sources_show_mapping():
    evidence = format_evidence(PARENT_HITS)
    assert "parent_id=knowledge.txt#p006" in evidence
    assert "六、退款规则" in evidence
    assert "2. 基础版与专业版" in evidence
    assert "matched_chunk_id=30" in evidence
    assert PARENT_HITS[0]["text"] in evidence

    output = format_sources([{"citation": 1, **PARENT_HITS[0]}])
    assert "parent_id=knowledge.txt#p006" in output
    assert "matched_chunk_id=30" in output


def test_unexpected_service_error_maps_to_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(config, "validate_config", lambda **kwargs: None)

    def boom():
        raise ConnectionError("milvus dropped")

    monkeypatch.setattr(vector_store, "inspect_target", boom)
    assert main.main(["基础版多少人？"]) == 7
    output = capsys.readouterr().out
    assert "服务异常" in output
    assert "ConnectionError" in output
