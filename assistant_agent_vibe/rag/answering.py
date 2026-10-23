"""步骤 7：无工具 Agent 的问答编排。

流程：检索（或接收已检索结果）→ 组装上下文 → create_agent(tools=[]) →
提取最终消息 → 保留本次来源与引用检查。没有命中时不调用模型。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, ToolMessage

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
        "model_calls": 0,
        "usage": None,
    }


def generate_answer(
    question: str,
    *,
    k: int = config.TOP_K,
    hits: Sequence[dict[str, Any]] | None = None,
    retriever: Callable[[str, int], list[dict[str, Any]]] | None = None,
    agent: Any = None,
    history: Sequence[dict[str, Any]] | None = None,
    resolved_query: str | None = None,
) -> dict[str, Any]:
    """生成答案；返回 answer、本次 sources、hits 与引用检查结果。"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("查询问题不能为空")

    if hits is None:
        retrieve_fn = retriever or retrieval.retrieve
        hits = retrieve_fn(question, k)
    hits = list(hits)
    if not hits:
        result = _empty_result()
        result["question"] = question
        return result

    if agent is None:
        try:
            agent = create_answer_agent()
        except Exception as exc:
            raise AnswerGenerationError(
                f"对话模型初始化失败: {type(exc).__name__}: {exc}"
            ) from exc

    user_content = prompts.build_user_content(
        question,
        hits,
        resolved_query=resolved_query,
        history=history,
    )
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
    usage = getattr(messages[-1], "usage_metadata", None)
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "hits": hits,
        "citations": citations,
        "invalid_citations": invalid,
        "insufficient": False,
        "model_calls": 1,
        "usage": usage,
    }


def stream_answer(
    question: str,
    *,
    k: int = config.TOP_K,
    hits: Sequence[dict[str, Any]] | None = None,
    retriever: Callable[[str, int], list[dict[str, Any]]] | None = None,
    agent: Any = None,
    history: Sequence[dict[str, Any]] | None = None,
    resolved_query: str | None = None,
) -> Iterator[dict[str, Any]]:
    """流式生成答案：start → delta… → sources → done；失败只发 error。"""
    if not isinstance(question, str) or not question.strip():
        yield {"event": "error", "message": "查询问题不能为空"}
        return
    yield {
        "event": "start",
        "question": question,
        "resolved_query": resolved_query,
    }

    if hits is None:
        try:
            retrieve_fn = retriever or retrieval.retrieve
            hits = retrieve_fn(question, k)
        except Exception as exc:  # noqa: BLE001
            yield {"event": "error", "message": f"检索失败: {type(exc).__name__}: {exc}"}
            return
    hits = list(hits)
    if not hits:
        yield {"event": "sources", "sources": []}
        yield {
            "event": "done",
            "answer": INSUFFICIENT_ANSWER,
            "sources": [],
            "citations": [],
            "invalid_citations": [],
            "insufficient": True,
            "model_calls": 0,
            "usage": None,
        }
        return

    if agent is None:
        try:
            agent = create_answer_agent()
        except Exception as exc:  # noqa: BLE001
            yield {
                "event": "error",
                "message": f"对话模型初始化失败: {type(exc).__name__}: {exc}",
            }
            return

    user_content = prompts.build_user_content(
        question,
        hits,
        resolved_query=resolved_query,
        history=history,
    )
    parts: list[str] = []
    try:
        for event in agent.stream(
            {"messages": [{"role": "user", "content": user_content}]},
            stream_mode="messages",
        ):
            if not isinstance(event, tuple) or len(event) != 2:
                continue
            message = event[0]
            if isinstance(message, ToolMessage):
                # 工具调用之前的模型文本不是最终答案。
                parts.clear()
                continue
            text = message_text(message)
            if not text:
                continue
            parts.append(text)
            yield {"event": "delta", "text": text}
        answer = "".join(parts).strip()
        if not answer:
            raise AnswerGenerationError("Agent 流式返回空答案")
    except Exception as exc:  # noqa: BLE001
        yield {
            "event": "error",
            "message": f"生成回答失败: {type(exc).__name__}: {exc}",
        }
        return

    citations, invalid = prompts.extract_citations(answer, len(hits))
    sources = [
        {"citation": index, **hit} for index, hit in enumerate(hits, start=1)
    ]
    yield {"event": "sources", "sources": sources}
    yield {
        "event": "done",
        "answer": answer,
        "sources": sources,
        "citations": citations,
        "invalid_citations": invalid,
        "insufficient": False,
        "model_calls": 1,
        "usage": None,
    }
