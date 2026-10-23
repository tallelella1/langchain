"""P2.2：会话隔离、指代改写、流式事件与错误区分（离线替身）。"""

import pytest
from langchain_core.messages import AIMessageChunk

from rag import answering, conversation, prompts

HITS = [
    {
        "id": 3,
        "chunk_id": 3,
        "chunk_key": "knowledge#c0003",
        "text": "专业版成员上限为 50 人。",
        "source": "knowledge.txt",
        "score": 0.9,
        "metric": "COSINE",
    }
]


class FakeChatModel:
    def __init__(self, answer="专业版多少钱？"):
        self.answer = answer
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return type("Message", (), {"content": self.answer})()


def fake_retriever(queries, hits=None):
    def retrieve(question, k, **kwargs):
        queries.append((question, k, kwargs))
        return hits if hits is not None else HITS

    return retrieve


def fake_answerer(question, **kwargs):
    return {
        "question": question,
        "answer": "答案 [片段 1]",
        "sources": [],
        "hits": kwargs["hits"],
        "citations": [1],
        "invalid_citations": [],
        "insufficient": False,
        "model_calls": 1,
    }


def test_store_isolates_sessions_and_limits_turns():
    store = conversation.ConversationStore(max_turns=2)
    store.append("a", question="问题 1", resolved_query="问题 1", answer="答案 1")
    store.append("b", question="问题 2", resolved_query="问题 2", answer="答案 2")
    assert [turn["question"] for turn in store.history("a")] == ["问题 1"]
    assert [turn["question"] for turn in store.history("b")] == ["问题 2"]

    store.append("a", question="问题 3", resolved_query="问题 3", answer="答案 3")
    store.append("a", question="问题 4", resolved_query="问题 4", answer="答案 4")
    assert [turn["question"] for turn in store.history("a")] == ["问题 3", "问题 4"]
    store.clear("a")
    assert store.history("a") == []
    with pytest.raises(conversation.ConversationError, match="session_id"):
        store.history("  ")


def test_resolve_query_skips_model_without_history_and_rewrites_with_history():
    model = FakeChatModel()
    assert conversation.resolve_query("基础版多少钱？", [], chat_model=model) == (
        "基础版多少钱？",
        0,
    )
    assert model.calls == []

    history = [{"question": "基础版多少钱？", "resolved_query": "基础版多少钱？", "answer": "99 元"}]
    resolved, calls = conversation.resolve_query(
        "那专业版呢？", history, chat_model=model
    )
    assert resolved == "专业版多少钱？"
    assert calls == 1
    prompt = str(model.calls[0])
    assert "基础版多少钱" in prompt and "那专业版呢" in prompt
    assert "99 元" not in prompt  # 历史回答不作为事实进入改写


def test_resolve_query_rejects_empty_model_output():
    with pytest.raises(conversation.ConversationError, match="空结果"):
        conversation.resolve_query(
            "那专业版呢？",
            [{"question": "基础版多少钱？", "resolved_query": "基础版多少钱？"}],
            chat_model=FakeChatModel(answer="  "),
        )


def test_answer_turn_uses_resolved_query_for_retrieval_but_original_for_answer():
    store = conversation.ConversationStore()
    store.append(
        "s1",
        question="基础版多少钱？",
        resolved_query="基础版多少钱？",
        answer="99 元",
    )
    queries = []
    model = FakeChatModel()
    result = conversation.answer_turn(
        "那专业版呢？",
        store=store,
        session_id="s1",
        chat_model=model,
        retriever=fake_retriever(queries),
        answerer=fake_answerer,
    )
    assert queries[0][0] == "专业版多少钱？"
    assert result["question"] == "那专业版呢？"
    assert result["resolved_query"] == "专业版多少钱？"
    assert result["model_calls"] == 2
    assert len(store.history("s1")) == 2
    assert store.history("s1")[-1]["resolved_query"] == "专业版多少钱？"


class StreamAgent:
    def __init__(self, chunks=("基础", "版 [片段 1]"), error=None):
        self.chunks = chunks
        self.error = error

    def stream(self, payload, stream_mode=None):
        if self.error:
            raise self.error
        for text in self.chunks:
            yield (AIMessageChunk(content=text), {"langgraph_node": "model"})


def test_stream_answer_event_order_and_citations():
    events = list(
        answering.stream_answer("专业版多少人？", hits=HITS, agent=StreamAgent())
    )
    kinds = [event["event"] for event in events]
    assert kinds[0] == "start"
    assert kinds[-2:] == ["sources", "done"]
    assert "".join(event["text"] for event in events if event["event"] == "delta") == (
        "基础版 [片段 1]"
    )
    assert events[-1]["answer"] == "基础版 [片段 1]"
    assert events[-1]["citations"] == [1]
    assert events[-1]["model_calls"] == 1


def test_stream_error_does_not_send_done():
    events = list(
        answering.stream_answer(
            "问题", hits=HITS, agent=StreamAgent(error=ConnectionError("down"))
        )
    )
    assert [event["event"] for event in events] == ["start", "error"]
    assert "done" not in [event["event"] for event in events]


def test_stream_without_hits_is_insufficient_and_never_calls_agent():
    class NeverAgent:
        def stream(self, payload, stream_mode=None):
            raise AssertionError("不应调用模型")

    events = list(
        answering.stream_answer("火星基地？", hits=[], agent=NeverAgent())
    )
    assert [event["event"] for event in events] == ["start", "sources", "done"]
    assert events[-1]["insufficient"] is True
    assert events[-1]["model_calls"] == 0


def test_stream_turn_persists_only_on_done_and_reports_error():
    store = conversation.ConversationStore()
    queries = []

    def streamer(question, **kwargs):
        yield {"event": "start", "question": question}
        yield {"event": "delta", "text": "答案"}
        yield {"event": "sources", "sources": []}
        yield {
            "event": "done",
            "answer": "答案 [片段 1]",
            "citations": [1],
            "invalid_citations": [],
            "insufficient": False,
            "model_calls": 1,
        }

    events = list(
        conversation.stream_turn(
            "基础版多少钱？",
            store=store,
            session_id="s1",
            chat_model=FakeChatModel("基础版多少钱？"),
            retriever=fake_retriever(queries),
            streamer=streamer,
        )
    )
    assert events[0]["event"] == "start"
    assert events[-1]["event"] == "done"
    assert len(store.history("s1")) == 1
    assert store.history("s1")[0]["answer"] == "答案 [片段 1]"

    def broken_streamer(question, **kwargs):
        yield {"event": "start", "question": question}
        yield {"event": "error", "message": "模型中断"}

    events = list(
        conversation.stream_turn(
            "第二个问题",
            store=store,
            session_id="s1",
            chat_model=FakeChatModel("第二个问题"),
            retriever=fake_retriever(queries),
            streamer=broken_streamer,
        )
    )
    assert events[-1]["event"] == "error"
    assert len(store.history("s1")) == 1


def test_user_content_includes_history_for_reference_but_not_answers():
    history = [
        {
            "question": "基础版多少钱？",
            "resolved_query": "基础版多少钱？",
            "answer": "99 元",
        }
    ]
    content = prompts.build_user_content(
        "那专业版呢？", HITS, resolved_query="专业版多少钱？", history=history
    )
    assert "那专业版呢" in content
    assert "专业版多少钱" in content
    assert "基础版多少钱" in content
    assert "99 元" not in content
    assert "不是知识事实" in content
