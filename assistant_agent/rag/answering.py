"""步骤 7：无工具 Agent 的问答编排。

流程：检索（或接收已检索结果）→ 组装上下文 → create_agent(tools=[]) →
提取最终消息 → 保留本次来源与引用检查。没有命中时不调用模型。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config  # noqa: E402
from rag import models, prompts, retrieval  # noqa: E402

INSUFFICIENT_ANSWER = (
    "材料不足：知识库中没有找到能回答该问题的内容。"
    "请补充问题背景，或联系人工客服。"
)


class AnswerGenerationError(RuntimeError):
    """对话模型初始化、调用或返回内容失败。"""


def create_answer_agent(chat_model: Any = None):
    """创建保留课件路径的无工具 Agent。"""
    if chat_model is None:
        chat_model = models.init_chat_model_instance()
    return create_agent(
        model=chat_model,
        tools=[],
        system_prompt=prompts.SYSTEM_PROMPT,
    )


def message_text(message: BaseMessage | Any) -> str:
    """提取最终消息文本；兼容字符串与内容块列表。"""
    content = getattr(message, "content", message)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content)


def _empty_result() -> dict[str, Any]:
    return {
        "answer": INSUFFICIENT_ANSWER,
        "sources": [],
        "hits": [],
        "citations": [],
        "invalid_citations": [],
        "insufficient": True,
    }


def generate_answer(
    question: str,
    *,
    k: int = config.TOP_K,
    hits: Sequence[dict[str, Any]] | None = None,
    retriever: Callable[[str, int], list[dict[str, Any]]] | None = None,
    agent: Any = None,
) -> dict[str, Any]:
    """生成答案；返回 answer、本次 sources、hits 与引用检查结果。"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("查询问题不能为空")

    if hits is None:
        retrieve_fn = retriever or retrieval.retrieve
        hits = retrieve_fn(question, k)
    hits = list(hits)
    if not hits:
        return _empty_result()

    if agent is None:
        try:
            agent = create_answer_agent()
        except Exception as exc:
            raise AnswerGenerationError(
                f"对话模型初始化失败: {type(exc).__name__}: {exc}"
            ) from exc

    user_content = prompts.build_user_content(question, hits)
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": user_content}]})
    except Exception as exc:
        raise AnswerGenerationError(
            f"生成回答失败: {type(exc).__name__}: {exc}"
        ) from exc

    messages = result.get("messages") if isinstance(result, dict) else None
    if not messages:
        raise AnswerGenerationError("Agent 未返回 messages")
    answer = message_text(messages[-1])
    if not answer.strip():
        raise AnswerGenerationError("Agent 返回空答案")

    citations, invalid = prompts.extract_citations(answer, len(hits))
    sources = [
        {"citation": index, **hit} for index, hit in enumerate(hits, start=1)
    ]
    return {
        "answer": answer,
        "sources": sources,
        "hits": hits,
        "citations": citations,
        "invalid_citations": invalid,
        "insufficient": False,
    }
