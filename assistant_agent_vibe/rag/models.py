"""步骤 4：Embedding 与对话模型初始化，以及向量响应校验。

- Embedding 与对话模型分开配置、分开初始化。
- import 阶段不校验环境变量、不创建客户端；真正调用前才检查。
- 本模块只负责“建立模型对象/校验响应”，不写数据库。

用法：
    python -m rag.models            # 真实调用一次 Embedding 并校验数量与 1024 维
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Sequence

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config  # noqa: E402  （sys.path 调整后导入）

EMBED_TIMEOUT = 60
LLM_TIMEOUT = 60


class EmbeddingResponseError(RuntimeError):
    """Embedding 服务返回的数量、维度或数值不符合预期。"""


def init_embedding_model() -> Embeddings:
    """用统一配置建立文档与查询共用的 Embedding 实例（不含网络请求）。"""
    config.validate_embedding_config()
    kwargs: dict[str, Any] = {
        "model": config.EMBED_MODEL_NAME,
        "api_key": config.EMBED_API_KEY,
        "base_url": config.EMBED_BASE_URL,
        # 非 OpenAI 服务不一定接受本地 tokenize 后的 token 数组，发送原始文本。
        "check_embedding_ctx_length": False,
        "chunk_size": config.EMBED_BATCH_SIZE,
        "timeout": EMBED_TIMEOUT,
        # 不做无依据的自动重试；失败由入口明确报告。
        "max_retries": 0,
    }
    if config.EMBED_SEND_DIMENSIONS:
        kwargs["dimensions"] = config.EMBED_DIM
    return init_embeddings(provider=config.EMBED_PROVIDER, **kwargs)


def init_chat_model_instance() -> BaseChatModel:
    """建立对话模型实例；服务商与地址来自独立配置。"""
    config.validate_llm_config()
    return init_chat_model(
        model=config.LLM_MODEL_NAME,
        model_provider=config.LLM_PROVIDER,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        temperature=0,
        timeout=LLM_TIMEOUT,
        max_retries=0,
    )


def validate_embedding_response(
    vectors: Sequence[Sequence[float]],
    *,
    expected_count: int | None = None,
    expected_dim: int | None = None,
) -> dict[str, int]:
    """校验向量数量、维度和数值；不符合时抛出 EmbeddingResponseError。"""
    dim = config.EMBED_DIM if expected_dim is None else expected_dim
    if not isinstance(vectors, list) or not vectors:
        raise EmbeddingResponseError(f"Embedding 返回数量不符: 期望至少 1，实际 {len(vectors)}")
    if expected_count is not None and len(vectors) != expected_count:
        raise EmbeddingResponseError(
            f"Embedding 返回数量不符: 期望 {expected_count}，实际 {len(vectors)}"
        )
    for index, vector in enumerate(vectors):
        if len(vector) != dim:
            raise EmbeddingResponseError(
                f"Embedding 向量维度不符: 第 {index} 条期望 {dim}，实际 {len(vector)}"
            )
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EmbeddingResponseError(
                    f"Embedding 向量数值无效: 第 {index} 条包含非数值 {value!r}"
                )
            if not math.isfinite(float(value)):
                raise EmbeddingResponseError(
                    f"Embedding 向量数值无效: 第 {index} 条包含非有限值"
                )
    return {"count": len(vectors), "dim": dim}


def embed_documents(embedder: Embeddings, texts: Sequence[str]) -> list[list[float]]:
    """批量向量化文档；写入前校验数量与维度。"""
    texts = list(texts)
    if not texts:
        raise ValueError("待向量化文本为空")
    if any(not text.strip() for text in texts):
        raise ValueError("待向量化文本包含空字符串")
    vectors = embedder.embed_documents(texts)
    validate_embedding_response(vectors, expected_count=len(texts))
    return vectors


def embed_question(embedder: Embeddings, question: str) -> list[float]:
    """向量化当前问题；只调用一次查询接口。"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("查询问题不能为空")
    vector = embedder.embed_query(question)
    validate_embedding_response([vector], expected_count=1)
    return vector


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if not norm_left or not norm_right:
        raise EmbeddingResponseError("无法计算相似度: 向量范数为 0")
    return dot / (norm_left * norm_right)


def verify_embedding_model(sample_texts: Sequence[str] | None = None) -> dict[str, Any]:
    """真实调用文档与查询接口，返回数量和维度证据（会调用远端并可能计费）。"""
    texts = list(
        sample_texts
        or [
            "基础版正式成员上限为 10 人，按已激活成员计算。",
            "已开具专票的订单需要先完成红字发票流程才能退款。",
        ]
    )
    embedder = init_embedding_model()
    document_vectors = embed_documents(embedder, texts)
    query_vector = embed_question(embedder, texts[0])
    return {
        "provider": config.EMBED_PROVIDER,
        "base_url": config.EMBED_BASE_URL,
        "model": config.EMBED_MODEL_NAME,
        "document_count": len(document_vectors),
        "document_dim": len(document_vectors[0]),
        "query_count": 1,
        "query_dim": len(query_vector),
        "self_similarity": _cosine(query_vector, document_vectors[0]),
    }


def main() -> int:
    config.validate_embedding_config()
    summary = verify_embedding_model()
    print(f"Embedding 服务商: {summary['provider']} / {summary['base_url']}")
    print(f"模型: {summary['model']}")
    print(f"文档接口: {summary['document_count']} 条，维度 {summary['document_dim']}")
    print(f"查询接口: {summary['query_count']} 条，维度 {summary['query_dim']}")
    print(f"同一文本的查询-文档余弦相似度: {summary['self_similarity']:.4f}")
    if summary["document_dim"] != config.EMBED_DIM or summary["query_dim"] != config.EMBED_DIM:
        print(f"维度与 EMBED_DIM={config.EMBED_DIM} 不符")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
