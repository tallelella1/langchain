"""步骤 4/9：真实 Embedding 调用（会收费），默认跳过。

运行：
    RUN_LIVE_EMBEDDING=1 python -m pytest tests/test_embedding_live.py -q
"""

from __future__ import annotations

import os

import pytest

import config
from rag import models

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_EMBEDDING") != "1",
    reason="真实 Embedding 会调用远端并可能计费；设置 RUN_LIVE_EMBEDDING=1 启用",
)


def test_live_embedding_count_and_dimension():
    summary = models.verify_embedding_model()
    assert summary["document_count"] == 2
    assert summary["document_dim"] == config.EMBED_DIM
    assert summary["query_dim"] == config.EMBED_DIM
    assert summary["self_similarity"] > 0.9
    print(
        f"model={summary['model']} dim={summary['document_dim']} "
        f"self_similarity={summary['self_similarity']:.4f}"
    )
