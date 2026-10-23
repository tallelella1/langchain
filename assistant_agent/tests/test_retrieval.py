"""步骤 6 / 2.1：检索参数、归一化、父块补全与异常（离线替身）。"""

import json

import pytest

import config
from rag import documents, retrieval


def _retrieve_no_expand(*args, **kwargs):
    kwargs["expand_parents"] = False
    return retrieval.retrieve(*args, **kwargs)


class FakeEmbedder:
    def __init__(self):
        self.questions = []

    def embed_query(self, text):
        self.questions.append(text)
        return [0.5] * config.EMBED_DIM


class FakeClient:
    def __init__(self, hits=None, error=None):
        self.hits = hits if hits is not None else []
        self.error = error
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.hits


def _hit(
    record_id=3,
    text="基础版成员上限 10 人。",
    source="knowledge.txt",
    chunk_id=None,
    distance=0.87,
):
    return {
        "id": record_id,
        "distance": distance,
        "entity": {
            "text": text,
            "source": source,
            "chunk_id": record_id if chunk_id is None else chunk_id,
        },
    }


PARENT_MANIFEST = {
    "source": "knowledge.txt",
    "parents": {
        "knowledge.txt#p000": {
            "parent_id": "knowledge.txt#p000",
            "source": "knowledge.txt",
            "chapter": "六、退款规则",
            "section": "2. 基础版与专业版",
            "title": "2. 基础版与专业版",
            "plan": "基础版、专业版",
            "text": "退款主规则与升级、红字发票条件。",
            "child_ids": [0, 1],
        },
        "knowledge.txt#p001": {
            "parent_id": "knowledge.txt#p001",
            "source": "knowledge.txt",
            "chapter": "七、发票规则",
            "section": "4. 专票补充规则",
            "title": "4. 专票补充规则",
            "plan": "",
            "text": "专票补充规则：先完成红字发票流程。",
            "child_ids": [2],
        },
    },
    "children": {
        "knowledge.txt#0": {
            "chunk_id": 0,
            "parent_id": "knowledge.txt#p000",
            "source": "knowledge.txt",
        },
        "knowledge.txt#1": {
            "chunk_id": 1,
            "parent_id": "knowledge.txt#p000",
            "source": "knowledge.txt",
        },
        "knowledge.txt#2": {
            "chunk_id": 2,
            "parent_id": "knowledge.txt#p001",
            "source": "knowledge.txt",
        },
    },
}


def test_rejects_invalid_question_and_k():
    with pytest.raises(ValueError, match="问题"):
        _retrieve_no_expand("   ", 5, embedder=FakeEmbedder(), client=FakeClient())
    for k in (0, -1):
        with pytest.raises(ValueError, match="k"):
            _retrieve_no_expand("问题", k, embedder=FakeEmbedder(), client=FakeClient())


def test_embeds_question_once_and_searches_cosine():
    embedder = FakeEmbedder()
    client = FakeClient(hits=[[_hit()]])
    hits = _retrieve_no_expand("基础版多少人？", 5, embedder=embedder, client=client)
    assert embedder.questions == ["基础版多少人？"]
    call = client.calls[0]
    assert call["limit"] == 5
    assert call["search_params"]["metric_type"] == "COSINE"
    assert call["data"] == [[0.5] * config.EMBED_DIM]
    assert hits[0]["id"] == 3 and hits[0]["metric"] == "COSINE"


def test_normalizes_nested_flat_and_top_level_shapes():
    nested = _retrieve_no_expand(
        "问题", 5, embedder=FakeEmbedder(), client=FakeClient([[_hit(0), _hit(1)]])
    )
    assert [hit["id"] for hit in nested] == [0, 1]
    assert nested[0] == {
        "id": 0,
        "chunk_id": 0,
        "text": "基础版成员上限 10 人。",
        "source": "knowledge.txt",
        "score": 0.87,
        "metric": "COSINE",
    }

    flat = retrieval.normalize_hits([_hit(0), _hit(1), _hit(2)])
    assert [hit["id"] for hit in flat] == [0, 1, 2]
    top_level = retrieval.normalize_hits(
        [[{"id": 7, "distance": 0.5, "text": "原文", "source": "knowledge.txt", "chunk_id": 7}]]
    )
    assert top_level[0]["text"] == "原文"
    assert retrieval.normalize_hits([[]]) == []
    assert retrieval.normalize_hits([]) == []


def test_missing_field_and_bad_shape_are_reported():
    broken = _hit()
    del broken["entity"]["source"]
    with pytest.raises(retrieval.RetrievalError, match="source"):
        retrieval.normalize_hits([[broken]])
    with pytest.raises(retrieval.RetrievalError, match="结构"):
        retrieval.normalize_hits([[object()]])


def test_returns_fewer_than_k_without_fabricating():
    hits = _retrieve_no_expand(
        "问题", 5, embedder=FakeEmbedder(), client=FakeClient([[_hit(0), _hit(1)]])
    )
    assert [hit["id"] for hit in hits] == [0, 1]
    assert _retrieve_no_expand(
        "问题", 5, embedder=FakeEmbedder(), client=FakeClient([[]])
    ) == []


def test_search_and_embedding_errors_are_reported():
    with pytest.raises(retrieval.RetrievalError, match="检索"):
        _retrieve_no_expand(
            "问题", 5, embedder=FakeEmbedder(), client=FakeClient(error=ConnectionError("down"))
        )

    class BadEmbedder:
        def embed_query(self, text):
            return [0.5, 0.6]

    with pytest.raises(Exception, match="维度"):
        _retrieve_no_expand("问题", 5, embedder=BadEmbedder(), client=FakeClient())


def test_parent_expansion_dedups_same_parent_and_keeps_order():
    raw = [
        [
            _hit(0, text="子块 A", distance=0.9),
            _hit(1, text="子块 B", distance=0.8),
            _hit(2, text="专票子块", distance=0.7),
        ]
    ]
    hits = retrieval.retrieve(
        "退款？", 5, embedder=FakeEmbedder(), client=FakeClient(raw), manifest=PARENT_MANIFEST
    )
    assert [hit["parent_id"] for hit in hits] == [
        "knowledge.txt#p000",
        "knowledge.txt#p001",
    ]
    first = hits[0]
    assert first["text"] == PARENT_MANIFEST["parents"]["knowledge.txt#p000"]["text"]
    assert first["matched_chunk_ids"] == [0, 1]
    assert first["matched_chunk_id"] == 0
    assert first["matched_text"] == "子块 A"
    assert first["score"] == 0.9
    assert first["context_length"] == len(first["text"])


def test_parent_expansion_failures_are_hard_errors():
    def child_missing(value):
        pass  # hit chunk_id=9 不在映射中

    def parent_missing(value):
        value["children"]["knowledge.txt#0"]["parent_id"] = "missing-parent"

    def source_mismatch(value):
        value["parents"]["knowledge.txt#p000"]["source"] = "other.txt"

    def text_missing(value):
        del value["parents"]["knowledge.txt#p000"]["text"]

    cases = [
        (None, 9, "parent_id"),
        (parent_missing, 0, "父块"),
        (source_mismatch, 0, "来源"),
        (text_missing, 0, "text"),
    ]
    for mutate, chunk_id, expected in cases:
        manifest = json.loads(json.dumps(PARENT_MANIFEST))
        if mutate:
            mutate(manifest)
        with pytest.raises(retrieval.RetrievalError, match=expected):
            retrieval.retrieve(
                "问题",
                5,
                embedder=FakeEmbedder(),
                client=FakeClient([[_hit(chunk_id, chunk_id=chunk_id)]]),
                manifest=manifest,
            )


def test_manifest_load_failure_is_reported_before_embedding(monkeypatch):
    def broken_manifest():
        raise documents.ManifestError("索引清单损坏")

    monkeypatch.setattr(retrieval, "load_index_manifest", broken_manifest)
    embedder = FakeEmbedder()
    with pytest.raises(retrieval.RetrievalError, match="索引清单"):
        retrieval.retrieve(
            "问题", 5, embedder=embedder, client=FakeClient([[_hit(0)]]), manifest=None
        )
    assert embedder.questions == []
