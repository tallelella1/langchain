"""步骤 3/5：记录校验、写入、身份与空集合边界（离线替身）。"""

import pytest

import config
from rag import vector_store


def _record(**overrides):
    record = {
        "id": 0,
        "vector": [0.1] * config.EMBED_DIM,
        "text": "基础版成员人数上限为 10 人。",
        "source": "knowledge.txt",
        "chunk_id": 0,
    }
    record.update(overrides)
    return record


def test_validate_record_accepts_and_rejects():
    vector_store.validate_record(_record())

    cases = [
        ({"vector": [0.1, 0.2]}, vector_store.DimensionMismatchError, "维度"),
        ({"text": "   "}, vector_store.RecordValidationError, "text"),
        ({"text": "知" * 2731}, vector_store.RecordValidationError, "text"),
        ({"id": -1}, vector_store.RecordValidationError, "id"),
    ]
    for overrides, error, expected in cases:
        with pytest.raises(error, match=expected):
            vector_store.validate_record(_record(**overrides))

    missing = _record()
    del missing["source"]
    with pytest.raises(vector_store.RecordValidationError, match="source"):
        vector_store.validate_record(missing)


class FakeClient:
    def __init__(self, insert_count=None):
        self.calls = []
        self.insert_count = insert_count

    def insert(self, **kwargs):
        self.calls.append(kwargs)
        count = len(kwargs["data"]) if self.insert_count is None else self.insert_count
        return {"insert_count": count}


class BoomClient:
    def insert(self, **kwargs):
        raise ConnectionError("milvus dropped")


def test_insert_records_success():
    client = FakeClient()
    result = vector_store.insert_records(
        client, [_record(id=0), _record(id=1, chunk_id=1)], collection_name="docs_test"
    )
    assert result["insert_count"] == 2
    assert client.calls[0]["collection_name"] == "docs_test"
    assert [row["id"] for row in client.calls[0]["data"]] == [0, 1]


def test_insert_records_rejections():
    with pytest.raises(vector_store.RecordValidationError, match="重复"):
        vector_store.insert_records(
            FakeClient(), [_record(id=1), _record(id=1, chunk_id=1)]
        )
    with pytest.raises(vector_store.VectorStoreError, match="写入数量"):
        vector_store.insert_records(
            FakeClient(insert_count=1), [_record(id=0), _record(id=1, chunk_id=1)]
        )
    with pytest.raises(vector_store.VectorStoreError, match="写入 Milvus 失败"):
        vector_store.insert_records(BoomClient(), [_record()])


def test_embedding_identity():
    assert vector_store.embedding_identity() == {
        "embedding_model": config.EMBED_MODEL_NAME,
        "embedding_dim": str(config.EMBED_DIM),
    }
    vector_store.validate_embedding_identity(
        {"properties": vector_store.embedding_identity()}
    )
    with pytest.raises(vector_store.SchemaMismatchError, match="embedding_model"):
        vector_store.validate_embedding_identity(
            {"properties": {"embedding_model": "other-model", "embedding_dim": "1024"}}
        )
    with pytest.raises(vector_store.SchemaMismatchError, match="--stamp"):
        vector_store.validate_embedding_identity({"properties": {}})


def test_prepare_ingest_uses_authoritative_query_even_if_stats_stale(monkeypatch):
    class FakeIngestClient:
        def has_collection(self, collection_name):
            return True

        def close(self):
            raise AssertionError("成功路径不应关闭连接")

    monkeypatch.setattr(vector_store, "open_target", lambda **kwargs: FakeIngestClient())
    monkeypatch.setattr(
        vector_store,
        "describe_target",
        lambda client, collection_name=None: {"row_count": 5},
    )
    monkeypatch.setattr(vector_store, "ensure_loaded", lambda client, name=None: "Loaded")
    monkeypatch.setattr(vector_store, "_collection_is_empty", lambda client, name: True)

    _, summary = vector_store.prepare_ingest_target()
    assert summary["row_count"] == 5
    assert summary["created"] is False


def test_prepare_ingest_rejects_authoritative_nonempty(monkeypatch):
    class FakeIngestClient:
        def has_collection(self, collection_name):
            return True

        def close(self):
            pass

    monkeypatch.setattr(vector_store, "open_target", lambda **kwargs: FakeIngestClient())
    monkeypatch.setattr(
        vector_store,
        "describe_target",
        lambda client, collection_name=None: {"row_count": 0},
    )
    monkeypatch.setattr(vector_store, "ensure_loaded", lambda client, name=None: "Loaded")
    monkeypatch.setattr(vector_store, "_collection_is_empty", lambda client, name: False)

    with pytest.raises(vector_store.CollectionNotEmptyError, match="实际存在记录"):
        vector_store.prepare_ingest_target()
