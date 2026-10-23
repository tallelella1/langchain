"""步骤 4：Embedding/对话模型初始化与响应校验（离线，不发请求）。"""

import math

import pytest
from langchain_openai import OpenAIEmbeddings

import config
from rag import models


def test_init_embedding_model_and_missing_env(monkeypatch):
    monkeypatch.setenv("BAILIAN_API_KEY", "test-key")
    monkeypatch.setenv("BAILIAN_API_URL", "https://example.invalid/compatible-mode/v1")
    embedder = models.init_embedding_model()
    assert isinstance(embedder, OpenAIEmbeddings)
    assert embedder.model == config.EMBED_MODEL_NAME
    assert embedder.dimensions == config.EMBED_DIM
    assert embedder.check_embedding_ctx_length is False
    assert str(embedder.openai_api_base).rstrip("/").endswith("/compatible-mode/v1")
    assert embedder.chunk_size == config.EMBED_BATCH_SIZE

    monkeypatch.delenv("BAILIAN_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="BAILIAN_API_KEY"):
        models.init_embedding_model()


def test_validate_embedding_response_accepts_and_rejects():
    vectors = [[0.1] * config.EMBED_DIM, [0.2] * config.EMBED_DIM]
    assert models.validate_embedding_response(vectors, expected_count=2) == {
        "count": 2,
        "dim": config.EMBED_DIM,
    }

    cases = [
        ([], 1, "数量"),
        ([[0.1] * 3], 1, "维度"),
        ([[float("nan")] * config.EMBED_DIM], 1, "有限"),
        ([["x"] * config.EMBED_DIM], 1, "数值"),
    ]
    for bad_vectors, count, expected in cases:
        with pytest.raises(models.EmbeddingResponseError, match=expected):
            models.validate_embedding_response(bad_vectors, expected_count=count)


def test_embed_documents_and_question_checks():
    class ShortEmbedder:
        def embed_documents(self, texts):
            return [[0.0] * config.EMBED_DIM]

    with pytest.raises(models.EmbeddingResponseError, match="数量"):
        models.embed_documents(ShortEmbedder(), ["a", "b"])

    class QueryEmbedder:
        def embed_query(self, text):
            return [0.1] * config.EMBED_DIM

    vector = models.embed_question(QueryEmbedder(), "基础版多少人？")
    assert len(vector) == config.EMBED_DIM
    assert all(math.isfinite(value) for value in vector)
