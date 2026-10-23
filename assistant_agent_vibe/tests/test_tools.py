"""P2.3：只读检索工具、调用预算、失败反馈与工具型 Agent 结果（离线替身）。"""

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from rag import answering, conversation, tools

HITS = [
    {
        "id": 3,
        "chunk_id": 3,
        "chunk_key": "knowledge#c0003",
        "parent_id": "knowledge#p001",
        "text": "基础版正式成员上限为 10 人。",
        "source": "knowledge.txt",
        "location": "四、成员与权限规则 / 1. 成员数计算口径",
        "chapter": "四、成员与权限规则",
        "section": "1. 成员数计算口径",
        "score": 0.9,
        "metric": "COSINE",
    }
]


class FakeRetriever:
    def __init__(self, result=None, error=None):
        self.result = [] if result is None else result
        self.error = error
        self.calls = []

    def __call__(self, question, k, **kwargs):
        self.calls.append((question, k, kwargs))
        if self.error:
            raise self.error
        return self.result


def test_tool_returns_numbered_hits_and_dedups_citations():
    retriever = FakeRetriever(HITS)
    state = tools.RetrievalToolState(max_calls=3)
    search = tools.create_retrieval_tool(state, retriever=retriever)
    output = search.invoke({"query": "基础版多少人？"})
    assert "[片段 1]" in output
    assert "基础版正式成员上限为 10 人" in output
    assert "knowledge.txt" in output
    assert state.calls == 1 and len(state.hits) == 1

    output_again = search.invoke({"query": "基础版成员上限？"})
    assert "[片段 1]" in output_again and "[片段 2]" not in output_again
    assert len(state.hits) == 1 and state.calls == 2


def test_tool_dedups_same_parent_across_calls():
    other_child = {
        **HITS[0],
        "id": 4,
        "chunk_id": 4,
        "chunk_key": "knowledge#c0004",
        "matched_chunk_id": 4,
    }
    results = iter([[HITS[0]], [other_child]])

    def retriever(question, k, **kwargs):
        return next(results)

    state = tools.RetrievalToolState(max_calls=3)
    search = tools.create_retrieval_tool(state, retriever=retriever)
    assert "[片段 1]" in search.invoke({"query": "第一次"})
    assert "[片段 2]" not in search.invoke({"query": "第二次"})
    assert len(state.hits) == 1


def test_tool_no_result_and_failure_and_budget():
    state = tools.RetrievalToolState(max_calls=1)
    search = tools.create_retrieval_tool(state, retriever=FakeRetriever([]))
    assert "未找到" in search.invoke({"query": "火星基地？"})
    assert state.calls == 1
    assert "上限" in search.invoke({"query": "再查一次"})
    assert state.calls == 1

    failing_state = tools.RetrievalToolState(max_calls=2)
    failing = tools.create_retrieval_tool(
        failing_state, retriever=FakeRetriever(error=ConnectionError("down"))
    )
    output = failing.invoke({"query": "基础版多少人？"})
    assert "检索失败" in output and "ConnectionError" in output
    assert failing_state.errors
    assert failing_state.hits == []


def test_tool_rejects_empty_query():
    state = tools.RetrievalToolState()
    search = tools.create_retrieval_tool(state, retriever=FakeRetriever(HITS))
    assert "不能为空" in search.invoke({"query": "  "})
    assert state.calls == 0


class FakeAgent:
    def __init__(self, answer="基础版上限 10 人。[片段 1]", configs=None):
        self.answer = answer
        self.configs = configs if configs is not None else []
        self.calls = []

    def invoke(self, payload, config=None):
        self.calls.append(payload)
        self.configs.append(config)
        return {"messages": [AIMessage(content=self.answer)]}


def test_generate_tool_answer_uses_tool_sources_and_citations():
    retriever = FakeRetriever(HITS)
    state = tools.RetrievalToolState()
    search = tools.create_retrieval_tool(state, retriever=retriever)
    search.invoke({"query": "基础版多少人？"})
    agent = FakeAgent()
    result = tools.generate_tool_answer(
        "基础版多少人？", state=state, agent=agent, retriever=retriever
    )
    assert result["answer"] == "基础版上限 10 人。[片段 1]"
    assert result["citations"] == [1]
    assert result["invalid_citations"] == []
    assert result["tool_calls"] == 1
    assert result["model_calls"] == 2
    assert result["sources"][0]["citation"] == 1
    assert agent.configs[0]["recursion_limit"] > 0


def test_generate_tool_answer_rejects_ungrounded_answer_without_tool_call():
    agent = FakeAgent("基础版上限 10 人。")
    with pytest.raises(answering.AnswerGenerationError, match="未调用检索工具"):
        tools.generate_tool_answer(
            "基础版多少人？",
            state=tools.RetrievalToolState(),
            agent=agent,
            retriever=FakeRetriever(HITS),
        )

    insufficient = tools.generate_tool_answer(
        "火星基地专线？",
        state=tools.RetrievalToolState(),
        agent=FakeAgent("材料不足，无法确认。"),
        retriever=FakeRetriever(HITS),
    )
    assert insufficient["insufficient"] is True
    assert insufficient["sources"] == []


def test_generate_tool_answer_masks_answer_when_tool_failed():
    retriever = FakeRetriever(error=ConnectionError("down"))
    state = tools.RetrievalToolState()
    search = tools.create_retrieval_tool(state, retriever=retriever)
    search.invoke({"query": "基础版多少人？"})
    result = tools.generate_tool_answer(
        "基础版多少人？",
        state=state,
        agent=FakeAgent("基础版永远免费。[片段 9]"),
        retriever=retriever,
    )
    assert result["insufficient"] is True
    assert "材料不足" in result["answer"]
    assert result["sources"] == []
    assert result["tool_errors"]


class StreamAgent:
    def __init__(self, chunks=("基础版上限 10 人。", "[片段 1]")):
        self.chunks = chunks

    def stream(self, payload, config=None, stream_mode=None):
        for text in self.chunks:
            yield (AIMessageChunk(content=text), {"langgraph_node": "model"})


def test_stream_tool_answer_emits_sources_and_done():
    retriever = FakeRetriever(HITS)
    state = tools.RetrievalToolState()
    search = tools.create_retrieval_tool(state, retriever=retriever)
    search.invoke({"query": "基础版多少人？"})
    events = list(
        tools.stream_tool_answer(
            "基础版多少人？", state=state, agent=StreamAgent(), retriever=retriever
        )
    )
    kinds = [event["event"] for event in events]
    assert kinds[0] == "start"
    assert kinds[-2:] == ["sources", "done"]
    assert events[-1]["answer"] == "基础版上限 10 人。[片段 1]"
    assert events[-1]["tool_calls"] == 1


def test_stream_tool_answer_errors_without_tool_call():
    events = list(
        tools.stream_tool_answer(
            "基础版多少人？",
            state=tools.RetrievalToolState(),
            agent=StreamAgent(),
            retriever=FakeRetriever(HITS),
        )
    )
    assert events[-1]["event"] == "error"
    assert "done" not in [event["event"] for event in events]


def test_answer_tool_turn_records_conversation_and_rewrite_calls():
    class RewriteModel:
        def invoke(self, messages):
            return AIMessage(content="专业版支持多少成员？")

    retriever = FakeRetriever(HITS)
    state = tools.RetrievalToolState()
    search = tools.create_retrieval_tool(state, retriever=retriever)
    search.invoke({"query": "专业版支持多少成员？"})
    store = conversation.ConversationStore()
    store.append(
        "s1",
        question="基础版呢？",
        resolved_query="基础版成员上限？",
        answer="10 人",
    )
    result = tools.answer_tool_turn(
        "那专业版呢？",
        store=store,
        session_id="s1",
        chat_model=RewriteModel(),
        state=state,
        agent=FakeAgent("专业版上限 50 人。[片段 1]"),
        retriever=retriever,
    )
    assert result["resolved_query"] == "专业版支持多少成员？"
    assert result["model_calls"] == 3  # 1 次改写 + 1 次工具 + 1 次回答
    assert len(store.history("s1")) == 2
