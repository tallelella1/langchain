"""多轮会话与检索问题改写（P2.2）。

约定：
- 历史只用于理解“那专业版呢”之类的指代，不能替代本次知识检索；
- 最终回答仍使用**原始当前问题**；改写后的问题单独记录在 ``resolved_query``；
- 会话保存在进程内，按 ``session_id`` 隔离；最多保留最近 ``max_turns`` 轮；
- 历史不包含凭据；API 多进程部署需要外部会话存储（属于 P3）。

流式事件顺序：``start`` → ``delta`` … → ``sources`` → ``done``；
失败时只发 ``error``，不发 ``done``。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence

import config
from rag import answering, models, retrieval

REWRITE_SYSTEM_PROMPT = """你负责把多轮客服对话中的当前问题改写为独立、可检索的问题。
规则：
1. 只根据历史补全指代和省略信息，不回答问题，不添加历史之外的事实；
2. 保留用户原始意图、套餐名、数字和条件；
3. 如果当前问题已经独立，原样输出；
4. 只输出改写后的问题本身，不要解释、不要引号、不要编号。"""


class ConversationError(RuntimeError):
    """会话参数、历史或问题改写失败。"""


@dataclass
class Turn:
    question: str
    resolved_query: str
    answer: str
    citations: list[int] = field(default_factory=list)


class ConversationStore:
    """进程内会话存储；按 session_id 隔离，超出轮次时丢弃最旧轮次。"""

    def __init__(self, max_turns: int = 6) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns 必须是正整数")
        self.max_turns = max_turns
        self._turns: dict[str, list[Turn]] = {}
        self._lock = threading.Lock()

    def history(self, session_id: str) -> list[dict[str, Any]]:
        self._validate_session(session_id)
        with self._lock:
            turns = self._turns.get(session_id, [])
            return [
                {
                    "question": turn.question,
                    "resolved_query": turn.resolved_query,
                    "answer": turn.answer,
                    "citations": list(turn.citations),
                }
                for turn in turns
            ]

    def append(
        self,
        session_id: str,
        *,
        question: str,
        resolved_query: str,
        answer: str,
        citations: Sequence[int] = (),
    ) -> None:
        self._validate_session(session_id)
        if not isinstance(question, str) or not question.strip():
            raise ConversationError("会话问题不能为空")
        with self._lock:
            turns = self._turns.setdefault(session_id, [])
            turns.append(
                Turn(
                    question=question.strip(),
                    resolved_query=resolved_query.strip() or question.strip(),
                    answer=answer,
                    citations=list(citations),
                )
            )
            if len(turns) > self.max_turns:
                del turns[: len(turns) - self.max_turns]

    def clear(self, session_id: str) -> None:
        self._validate_session(session_id)
        with self._lock:
            self._turns.pop(session_id, None)

    def session_count(self) -> int:
        with self._lock:
            return len(self._turns)

    @staticmethod
    def _validate_session(session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ConversationError("session_id 不能为空")


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in content
            if isinstance(block, (str, dict))
        )
    return str(content or "")


def build_rewrite_content(question: str, history: Sequence[dict[str, Any]]) -> str:
    lines = ["最近对话（只用于理解指代，不是知识事实）："]
    for turn in history:
        lines.append(f"用户：{turn.get('question', '')}")
        resolved = turn.get("resolved_query", "")
        if resolved and resolved != turn.get("question"):
            lines.append(f"检索问题：{resolved}")
    lines.extend(["", f"当前问题：{question}", "", "只输出改写后的当前问题。"])
    return "\n".join(lines)


def resolve_query(
    question: str,
    history: Sequence[dict[str, Any]],
    *,
    chat_model: Any = None,
) -> tuple[str, int]:
    """结合历史把当前问题改写为独立检索问题；返回（问题，模型调用次数）。"""
    if not isinstance(question, str) or not question.strip():
        raise ConversationError("当前问题不能为空")
    question = question.strip()
    if not history:
        return question, 0
    if chat_model is None:
        chat_model = models.init_chat_model_instance()
    try:
        message = chat_model.invoke(
            [
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": build_rewrite_content(question, history)},
            ]
        )
    except Exception as exc:  # noqa: BLE001
        raise ConversationError(
            f"问题改写失败: {type(exc).__name__}: {exc}"
        ) from exc
    resolved = _message_text(message).strip().strip("\"'“”‘’")
    if not resolved:
        raise ConversationError("问题改写返回空结果")
    return resolved, 1


def answer_turn(
    question: str,
    *,
    store: ConversationStore,
    session_id: str,
    k: int = config.TOP_K,
    strategy: str | None = None,
    rerank_strategy: str | None = None,
    chat_model: Any = None,
    retriever: Callable[..., list[dict[str, Any]]] = retrieval.retrieve,
    answerer: Callable[..., dict[str, Any]] = answering.generate_answer,
) -> dict[str, Any]:
    """一轮非流式问答：改写 → 检索 → 生成 → 记录会话。"""
    history = store.history(session_id)
    resolved, rewrite_calls = resolve_query(question, history, chat_model=chat_model)
    hits = retriever(
        resolved,
        k,
        strategy=strategy,
        rerank_strategy=rerank_strategy,
    )
    result = answerer(
        question,
        k=k,
        hits=hits,
        history=history,
        resolved_query=resolved,
    )
    result["session_id"] = session_id
    result["resolved_query"] = resolved
    result["history_turns"] = len(history)
    result["model_calls"] = result.get("model_calls", 1) + rewrite_calls
    store.append(
        session_id,
        question=question,
        resolved_query=resolved,
        answer=result["answer"],
        citations=result.get("citations", []),
    )
    return result


def stream_turn(
    question: str,
    *,
    store: ConversationStore,
    session_id: str,
    k: int = config.TOP_K,
    strategy: str | None = None,
    rerank_strategy: str | None = None,
    chat_model: Any = None,
    retriever: Callable[..., list[dict[str, Any]]] = retrieval.retrieve,
    streamer: Callable[..., Iterator[dict[str, Any]]] = answering.stream_answer,
) -> Iterator[dict[str, Any]]:
    """一轮流式问答；只在收到 done 后写入会话。"""
    try:
        history = store.history(session_id)
        resolved, rewrite_calls = resolve_query(
            question, history, chat_model=chat_model
        )
        hits = retriever(
            resolved,
            k,
            strategy=strategy,
            rerank_strategy=rerank_strategy,
        )
    except Exception as exc:  # noqa: BLE001
        yield {"event": "error", "message": f"{type(exc).__name__}: {exc}"}
        return

    yielded_done = False
    saw_error = False
    for event in streamer(
        question,
        k=k,
        hits=hits,
        history=history,
        resolved_query=resolved,
    ):
        if event.get("event") == "start":
            event = {**event, "resolved_query": resolved, "session_id": session_id}
        elif event.get("event") == "error":
            saw_error = True
        elif event.get("event") == "done":
            yielded_done = True
            event = {
                **event,
                "resolved_query": resolved,
                "session_id": session_id,
                "history_turns": len(history),
                "model_calls": int(event.get("model_calls", 1)) + rewrite_calls,
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
