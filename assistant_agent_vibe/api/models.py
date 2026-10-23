"""API 请求/响应结构（P2.4）。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

import config


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="当前问题原文")
    session_id: str | None = Field(default=None, description="会话 ID；省略时自动生成")
    k: int = Field(default=config.TOP_K, ge=1, description="召回片段数")
    mode: Literal["fixed", "tool"] = Field(
        default="fixed", description="fixed=先检索再回答；tool=Agent 自主检索"
    )
    strategy: Literal["vector", "hybrid"] | None = Field(
        default=None, description="检索策略；省略时使用配置"
    )
    rerank: Literal["none", "llm"] | None = Field(
        default=None, description="重排策略；llm 会额外调用模型"
    )

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question 不能为空")
        return value

    @field_validator("session_id")
    @classmethod
    def session_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("session_id 不能为空字符串")
        return value


class AskResponse(BaseModel):
    session_id: str
    question: str
    resolved_query: str
    mode: Literal["fixed", "tool"]
    answer: str
    sources: list[dict[str, Any]]
    citations: list[int]
    invalid_citations: list[int]
    insufficient: bool
    model_calls: int
    tool_calls: int = 0
    tool_errors: list[str] = Field(default_factory=list)
    usage: dict[str, Any] | None = None


class IngestRequest(BaseModel):
    force: bool = Field(default=False, description="内容未变化也构建新版本")


class IngestTaskResponse(BaseModel):
    task_id: str
    status: Literal["pending", "running", "succeeded", "failed"]
    force: bool
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    collection: str | None = None
    index_version: int | None = None
    record_count: int | None = None
    reused: bool | None = None
    error: str | None = None


class ReadyResponse(BaseModel):
    ready: bool
    checks: dict[str, Any]
    errors: list[str]
