from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DATACLEAN_", env_file=ROOT / ".env", extra="ignore"
    )

    database_url: str = "sqlite:///./data/dataclean.db"
    storage_dir: Path = ROOT / "data" / "files"
    jwt_secret: str = Field(min_length=32)
    task_backend: Literal["local", "celery"] = "local"
    broker_url: str = "redis://localhost:6379/0"
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    token_minutes: int = Field(default=60, ge=1)
    lease_seconds: int = Field(default=300, ge=15)
    poll_seconds: float = Field(default=2, ge=0.1)

    @field_validator("jwt_secret")
    @classmethod
    def real_secret(cls, value: str) -> str:
        if value.startswith("replace-with-"):
            raise ValueError("请在 .env 中设置随机 DATACLEAN_JWT_SECRET")
        return value
