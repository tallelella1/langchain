from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

JobStatus = Literal["queued", "running", "succeeded", "failed"]


class Registration(BaseModel):
    username: str = Field(min_length=3, max_length=50, pattern=r"^[A-Za-z0-9_]+$")
    password: str = Field(min_length=8, max_length=128)


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    username: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class JobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    filename: str
    file_size: int
    sha256: str
    status: JobStatus
    attempt_count: int
    summary: dict | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @field_validator("created_at", "started_at", "finished_at")
    @classmethod
    def utc_timestamp(cls, value):
        # SQLite drops tzinfo; all persisted timestamps are UTC.
        return value.replace(tzinfo=timezone.utc) if value and value.tzinfo is None else value


class JobPage(BaseModel):
    items: list[JobRead]
    total: int
    page: int
    page_size: int
