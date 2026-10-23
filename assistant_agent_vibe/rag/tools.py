"""只读检索工具与工具型 Agent（P2.3）。

- ``search_knowledge`` 只读调用既有 ``retrieval.retrieve``，不写数据库；
- 工具返回带 [片段 N] 编号的原文与来源，编号与最终 sources 一致；
- 工具失败、无结果和调用预算耗尽都明确反馈，不静默兜底；
- 只有评估集证明工具型 Agent 在正确率/调用次数/延迟上可接受时才作为入口。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterator, Sequence

from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config  # noqa: E402
from rag import answering, conversation, models, prompts, retrieval  # noqa: E402


class ToolRetrievalError(RuntimeError):
    """工具参数或执行状态错误。"""


def source_key(hit: dict[str, Any]) -> str:
    """工具来源去重键：父块优先，避免同一父块经不同子块重复编号。"""
    parent_id = hit.get("parent_id")
    if isinstance(parent_id, str) and parent_id:
        return f"parent:{parent_id}"
    return f"chunk:{retrieval.hit_key(hit)}"


class RetrievalToolState:
    """记录本次问题的工具调用、去重后的命中与引用编号。"""

    def __init__(self, *, max_calls: int = config.TOOL_MAX_CALLS) -> None:
        if max_calls <= 0:
            raise ValueError("max_calls 必须是正整数")
        self.max_calls = max_calls
        self.calls = 0
        self.errors: list[str] = []
        self._hits: list[dict[str, Any]] = []
        self._by_key: dict[str, dict[str, Any]] = {}

    @property
    def hits(self) -> list[dict[str, Any]]:
        return list(self._hits)

    def record(self, hits: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        for hit in hits:
            key = source_key(hit)
            if key in self._by_key:
                continue
            citation = len(self._hits) + 1
            entry = {**hit, "citation": citation}
            self._by_key[key] = entry
            self._hits.append(entry)
        return [self._by_key[source_key(hit)] for hit in hits]

    def sources(self) -> list[dict[str, Any]]:
        return list(self._hits)


def format_tool_hits(hits: Sequence[dict[str, Any]]) -> str:
    if not hits:
        return "未找到与问题相关的知识片段。"
    blocks = []
    for hit in hits:
        header = (
            f"[片段 {hit['citation']}]\n"
            f"来源：{hit['source']}；"
            f"定位：{hit.get('location', '')}；"
            f"章节：{hit.get('chapter', '')} / {hit.get('section', '')}"
        )
        blocks.append(f"{header}\n{hit['text']}")
    return "\n\n".join(blocks)


def create_retrieval_tool(
    state: RetrievalToolState,
    *,
    k: int = config.TOP_K,
    strategy: str | None = None,
    rerank_strategy: str | None = None,
    retriever: Any = retrieval.retrieve,
):
    """创建只读检索工具；闭包持有本次问题的调用预算与命中编号。"""

    @tool
    def search_knowledge(query: str) -> str:
        """只读检索 atguigu 客服知识库。

        输入：一个自然语言问题（可包含套餐、退款、发票等关键词）。
        输出：带 [片段 N] 编号的原文、来源与章节定位；没有结果时明确说明。
        本工具不写数据库，不返回知识库之外的内容。
        """
        if not isinstance(query, str) or not query.strip():
            state.errors.append("查询问题不能为空")
            return "检索失败：查询问题不能为空。"
        if state.calls >= state.max_calls:
            message = (
                f"检索调用已达上限 {state.max_calls} 次；"
                "请基于已返回片段回答，或明确说明材料不足。"
            )
            state.errors.append(message)
            return message
        state.calls += 1
        try:
            hits = retriever(
                query,
                k,
                strategy=strategy,
                rerank_strategy=rerank_strategy,
            )
        except Exception as exc:  # noqa: BLE001
            message = f"检索失败: {type(exc).__name__}: {exc}"
            state.errors.append(message)
            return message + "；不能据此编造答案。"
        if not hits:
            return "未找到与问题相关的知识片段；不能据此编造答案。"
        recorded = state.record(hits)
        return format_tool_hits(recorded)

    return search_knowledge


def build_tool_user_content(
    question: str, resolved_query: str | None = None
) -> str:
    blocks = ["【当前问题】", question]
    if resolved_query and resolved_query.strip() != question.strip():
        blocks.extend(["", "【结合历史理解的检索问题】", resolved_query])
    blocks.extend(
        [
            "",
            "请先调用 search_knowledge 检索知识库，再仅依据工具返回的片段回答。",
            "工具没有结果或失败时明确说明，不得凭记忆作答。",
        ]
    )
    return "\n".join(blocks)


def create_tool_agent(
    state: RetrievalToolState,
    *,
    chat_model: Any = None,
    k: int = config.TOP_K,
    strategy: str | None = None,
    rerank_strategy: str | None = None,
    retriever: Any = retrieval.retrieve,
):
    if chat_model is None:
        chat_model = models.init_chat_model_instance()
    search_tool = create_retrieval_tool(
        state,
        k=k,
        strategy=strategy,
        rerank_strategy=rerank_strategy,
        retriever=retriever,
    )
    return create_agent(
        model=chat_model,
        tools=[search_tool],
        system_prompt=prompts.TOOL_SYSTEM_PROMPT,
    )


def _messages(result: Any) -> list[Any]:
    if not isinstance(result, dict):
        raise answering.AnswerGenerationError("Agent 未返回状态字典")
    messages = result.get("messages")
    if not messages:
        raise answering.AnswerGenerationError("Agent 未返回 messages")
    return list(messages)


def _insufficient_tool_answer(
    question: str,
    *,
    state: RetrievalToolState,
    reason: str,
) -> dict[str, Any]:
    return {
        "question": question,
        "answer": answering.INSUFFICIENT_ANSWER,
        "sources": [],
        "hits": [],
        "citations": [],
        "invalid_citations": [],
        "insufficient": True,
        "model_calls": state.calls + 1 if state.calls else 0,
        "tool_calls": state.calls,
        "tool_errors": list(state.errors),
        "usage": None,
        "reason": reason,
    }


def generate_tool_answer(
    question: str,
    *,
    state: RetrievalToolState | None = None,
    agent: Any = None,
    chat_model: Any = None,
    k: int = config.TOP_K,
    strategy: str | None = None,
    rerank_strategy: str | None = None,
    retriever: Any = retrieval.retrieve,
    resolved_query: str | None = None,
    recursion_limit: int = config.AGENT_RECURSION_LIMIT,
) -> dict[str, Any]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("查询问题不能为空")
    state = state or RetrievalToolState()
    if agent is None:
        agent = create_tool_agent(
            state,
            chat_model=chat_model,
            k=k,
            strategy=strategy,
            rerank_strategy=rerank_strategy,
            retriever=retriever,
        )
    payload = {
        "messages": [
            {
                "role": "user",
                "content": build_tool_user_content(question, resolved_query),
            }
        ]
    }
    try:
        result = agent.invoke(payload, config={"recursion_limit": recursion_limit})
    except Exception as exc:  # noqa: BLE001
        raise answering.AnswerGenerationError(
            f"工具型 Agent 调用失败: {type(exc).__name__}: {exc}"
        ) from exc
    messages = _messages(result)
    answer = answering.message_text(messages[-1]).strip()
    if not answer:
        raise answering.AnswerGenerationError("Agent 返回空答案")

    if state.calls == 0:
        if "材料不足" in answer or "未检索" in answer:
            return _insufficient_tool_answer(
                question, state=state, reason="Agent 未调用检索工具"
            )
        raise answering.AnswerGenerationError(
            "Agent 未调用检索工具，回答没有知识库证据"
        )

    if state.errors and not state.hits:
        return _insufficient_tool_answer(
            question, state=state, reason="检索工具执行失败"
        )

    sources = state.sources()
    citations, invalid = prompts.extract_citations(answer, len(sources))
    usage = getattr(messages[-1], "usage_metadata", None)
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "hits": state.hits,
        "citations": citations,
        "invalid_citations": invalid,
        "insufficient": False,
        "model_calls": state.calls + 1,
        "tool_calls": state.calls,
        "tool_errors": list(state.errors),
        "usage": usage,
    }


def stream_tool_answer(
    question: str,
    *,
    state: RetrievalToolState | None = None,
    agent: Any = None,
    chat_model: Any = None,
    k: int = config.TOP_K,
    strategy: str | None = None,
    rerank_strategy: str | None = None,
    retriever: Any = retrieval.retrieve,
    resolved_query: str | None = None,
    recursion_limit: int = config.AGENT_RECURSION_LIMIT,
) -> Iterator[dict[str, Any]]:
    if not isinstance(question, str) or not question.strip():
        yield {"event": "error", "message": "查询问题不能为空"}
        return
    state = state or RetrievalToolState()
    yield {
        "event": "start",
        "question": question,
        "resolved_query": resolved_query,
    }
    if agent is None:
        try:
            agent = create_tool_agent(
                state,
                chat_model=chat_model,
                k=k,
                strategy=strategy,
                rerank_strategy=rerank_strategy,
                retriever=retriever,
            )
        except Exception as exc:  # noqa: BLE001
            yield {
                "event": "error",
                "message": f"工具型 Agent 初始化失败: {type(exc).__name__}: {exc}",
            }
            return
    payload = {
        "messages": [
            {
                "role": "user",
                "content": build_tool_user_content(question, resolved_query),
            }
        ]
    }
    parts: list[str] = []
    try:
        for event in agent.stream(
            payload,
            config={"recursion_limit": recursion_limit},
            stream_mode="messages",
        ):
            if not isinstance(event, tuple) or len(event) != 2:
                continue
            message = event[0]
            if isinstance(message, ToolMessage):
                parts.clear()
                continue
            text = answering.message_text(message)
            if text:
                parts.append(text)
                yield {"event": "delta", "text": text}
        answer = "".join(parts).strip()
        if not answer:
            raise answering.AnswerGenerationError("Agent 流式返回空答案")
    except Exception as exc:  # noqa: BLE001
        yield {
            "event": "error",
            "message": f"工具型 Agent 生成失败: {type(exc).__name__}: {exc}",
        }
        return

    if state.calls == 0 or (state.errors and not state.hits):
        yield {
            "event": "error",
            "message": "工具未提供可用知识片段，不能完成有证据的回答",
        }
        return

    sources = state.sources()
    citations, invalid = prompts.extract_citations(answer, len(sources))
    yield {"event": "sources", "sources": sources}
    yield {
        "event": "done",
        "answer": answer,
        "sources": sources,
        "citations": citations,
        "invalid_citations": invalid,
        "insufficient": False,
        "model_calls": state.calls + 1,
        "tool_calls": state.calls,
        "tool_errors": list(state.errors),
        "usage": None,
    }


def stream_tool_turn(
    question: str,
    *,
    store: conversation.ConversationStore,
    session_id: str,
    k: int = config.TOP_K,
    strategy: str | None = None,
    rerank_strategy: str | None = None,
    chat_model: Any = None,
    retriever: Any = retrieval.retrieve,
    recursion_limit: int = config.AGENT_RECURSION_LIMIT,
    state: RetrievalToolState | None = None,
    agent: Any = None,
) -> Iterator[dict[str, Any]]:
    """工具型 Agent 的流式会话；只在 done 后写入会话，失败不发 done。"""
    try:
        history = store.history(session_id)
        resolved, rewrite_calls = conversation.resolve_query(
            question, history, chat_model=chat_model
        )
    except Exception as exc:  # noqa: BLE001
        yield {"event": "error", "message": f"{type(exc).__name__}: {exc}"}
        return
    yielded_done = False
    saw_error = False
    for event in stream_tool_answer(
        question,
        state=state,
        agent=agent,
        chat_model=chat_model,
        k=k,
        strategy=strategy,
        rerank_strategy=rerank_strategy,
        retriever=retriever,
        resolved_query=resolved,
        recursion_limit=recursion_limit,
    ):
        kind = event.get("event")
        if kind == "start":
            event = {**event, "resolved_query": resolved, "session_id": session_id}
        elif kind == "error":
            saw_error = True
        elif kind == "done":
            yielded_done = True
            event = {
                **event,
                "resolved_query": resolved,
                "session_id": session_id,
                "history_turns": len(history),
                "model_calls": int(event.get("model_calls", 0)) + rewrite_calls,
            }
            store.append(
                session_id,
                question=question,
                resolved_query=resolved,
                answer=event.get("answer", ""),
                citations=event.get("citations", []),
            )
        yield event
    if not yielded_done and not saw_error:
        yield {"event": "error", "message": "流式输出未正常完成"}


def answer_tool_turn(
    question: str,
    *,
    store: conversation.ConversationStore,
    session_id: str,
    k: int = config.TOP_K,
    strategy: str | None = None,
    rerank_strategy: str | None = None,
    chat_model: Any = None,
    retriever: Any = retrieval.retrieve,
    recursion_limit: int = config.AGENT_RECURSION_LIMIT,
    state: RetrievalToolState | None = None,
    agent: Any = None,
) -> dict[str, Any]:
    """工具型 Agent 的一轮会话：先改写指代，再由 Agent 自主检索。"""
    history = store.history(session_id)
    resolved, rewrite_calls = conversation.resolve_query(
        question, history, chat_model=chat_model
    )
    result = generate_tool_answer(
        question,
        state=state,
        agent=agent,
        chat_model=chat_model,
        k=k,
        strategy=strategy,
        rerank_strategy=rerank_strategy,
        retriever=retriever,
        resolved_query=resolved,
        recursion_limit=recursion_limit,
    )
    result["session_id"] = session_id
    result["resolved_query"] = resolved
    result["history_turns"] = len(history)
    result["model_calls"] += rewrite_calls
    store.append(
        session_id,
        question=question,
        resolved_query=resolved,
        answer=result["answer"],
        citations=result.get("citations", []),
    )
    return result
