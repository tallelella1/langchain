"""步骤 9：真实 Milvus 集成验证（使用独立实验集合，结束后删除）。

Milvus 不可达时自动跳过；不会触碰配置中的正式集合 docs。
"""

from __future__ import annotations

import logging
import uuid

import pytest
from pymilvus import MilvusClient

import config
from rag import vector_store


def _milvus_available() -> bool:
    try:
        client = MilvusClient(uri=config.MILVUS_URI, timeout=3)
        client.list_databases()
        client.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _milvus_available(), reason=f"Milvus 不可达: {config.MILVUS_URI}"
)


def _vector(index: int) -> list[float]:
    vector = [0.0] * config.EMBED_DIM
    vector[index] = 1.0
    return vector


def test_milvus_schema_write_read_search_and_non_empty_guard():
    collection = f"docs_it_{uuid.uuid4().hex[:8]}"
    client = vector_store.open_target(create_database=True)
    try:
        client.create_collection(
            collection_name=collection,
            schema=vector_store.build_schema(),
            index_params=vector_store.build_index_params(client),
            consistency_level=vector_store.CONSISTENCY_LEVEL,
            properties=vector_store.embedding_identity(),
        )
        vector_store.ensure_loaded(client, collection)

        description = client.describe_collection(collection_name=collection)
        fields = vector_store.validate_schema(description)
        assert fields["vector"]["params"]["dim"] == config.EMBED_DIM
        index = vector_store.validate_index(client, collection)
        assert index["metric_type"] == "COSINE"
        assert vector_store.count_records(client, collection) == 0

        records = [
            {
                "id": 0,
                "vector": _vector(0),
                "text": "基础版正式成员上限为 10 人。",
                "source": "knowledge.txt",
                "chunk_id": 0,
            },
            {
                "id": 1,
                "vector": _vector(1),
                "text": "已开具专票需先完成红字发票流程。",
                "source": "knowledge.txt",
                "chunk_id": 1,
            },
        ]
        result = vector_store.insert_records(client, records, collection_name=collection)
        assert result["insert_count"] == 2
        vector_store.flush(client, collection)

        rows = vector_store.fetch_records_by_ids(
            client, [0, 1], collection_name=collection
        )
        assert [row["text"] for row in rows] == [r["text"] for r in records]
        assert [row["chunk_id"] for row in rows] == [0, 1]
        assert vector_store.count_records(client, collection) == 2

        hits = vector_store.search_raw(
            client, _vector(0), 2, collection_name=collection
        )
        assert hits[0][0]["id"] == 0
        assert hits[0][0]["distance"] > hits[0][1]["distance"]

        # 服务端维度冲突必须明确失败；这是预期负向测试，压低 ERROR 日志。
        previous_disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            with pytest.raises(Exception):
                client.insert(
                    collection_name=collection,
                    data=[
                        {
                            "id": 99,
                            "vector": [0.1, 0.2, 0.3],
                            "text": "wrong dimension",
                            "source": "knowledge.txt",
                            "chunk_id": 99,
                        }
                    ],
                )
        finally:
            logging.disable(previous_disable)
        assert vector_store.count_records(client, collection) == 2

        with pytest.raises(vector_store.CollectionNotEmptyError):
            vector_store.prepare_ingest_target(collection)
    finally:
        try:
            client.drop_collection(collection_name=collection)
        finally:
            client.close()
