"""步骤 7：无工具 Agent 编排、请求内容、拒答与引用检查（离线替身）。"""

import pytest
from langchain_core.messages import AIMessage

from rag import answering
from rag.prompts import SYSTEM_PROMPT

HITS = [
    {
        "id": 3,
        "chunk_id": 3,
        "text": "基础版正式成员上限为 10 人，按已激活成员计算。",
        "source": "knowledge.txt",
        "score": 0.9,
        "metric": "COSINE",
    }
]


class FakeAgent:
    def __init__(self, answer="基础版上限 10 人。[片段 1]"):
        self.answer = answer
        self.calls = []
        self.system_prompt = SYSTEM_PROMPT

    def invoke(self, payload):
        self.calls.append(payload)
        return {"messages": [AIMessage(content=self.answer)]}


def test_no_hits_returns_insufficient_without_calling_agent():
    agent = FakeAgent()
    result = answering.generate_answer("火星基地专线？", hits=[], agent=agent)
    assert result["insufficient"] is True
    assert "材料不足" in result["answer"]
    assert result["sources"] == []
    assert agent.calls == []


def test_question_and_context_enter_request_and_sources_are_kept():
    agent = FakeAgent()
    question = "基础版最多多少个正式成员？"
    result = answering.generate_answer(question, hits=HITS, agent=agent)

    payload_text = str(agent.calls[0])
    assert question in payload_text
    assert HITS[0]["text"] in payload_text
    assert result["answer"] == "基础版上限 10 人。[片段 1]"
    assert result["sources"][0]["citation"] == 1
    assert result["sources"][0]["chunk_id"] == 3
    assert result["citations"] == [1]
    assert result["invalid_citations"] == []


def test_invalid_citation_is_flagged():
    result = answering.generate_answer("问题", hits=HITS, agent=FakeAgent("结论 [片段 5]"))
    assert result["citations"] == []
    assert result["invalid_citations"] == [5]


def test_retriever_used_when_hits_not_given():
    calls = []

    def fake_retriever(question, k):
        calls.append((question, k))
        return HITS

    result = answering.generate_answer(
        "问题", k=3, retriever=fake_retriever, agent=FakeAgent()
    )
    assert calls == [("问题", 3)]
    assert result["hits"] == HITS


def test_generation_errors_are_reported():
    class EmptyAgent(FakeAgent):
        def invoke(self, payload):
            return {"messages": [AIMessage(content="")]}

    class BrokenAgent(FakeAgent):
        def invoke(self, payload):
            raise ConnectionError("model gateway down")

    with pytest.raises(answering.AnswerGenerationError, match="空答案"):
        answering.generate_answer("问题", hits=HITS, agent=EmptyAgent())
    with pytest.raises(answering.AnswerGenerationError, match="生成回答失败"):
        answering.generate_answer("问题", hits=HITS, agent=BrokenAgent())
    with pytest.raises(ValueError, match="问题"):
        answering.generate_answer("  ", hits=HITS, agent=FakeAgent())


def test_content_block_answer_is_extracted():
    class BlockAgent(FakeAgent):
        def invoke(self, payload):
            return {"messages": [AIMessage(content=[{"type": "text", "text": "分块答案 [片段 1]"}])]}

    result = answering.generate_answer("问题", hits=HITS, agent=BlockAgent())
    assert result["answer"] == "分块答案 [片段 1]"
