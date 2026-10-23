"""步骤 1：配置路径、参数校验与缺失变量提示（离线）。"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import config

PROJECT_DIR = Path(__file__).resolve().parents[1]
PARENT_DIR = PROJECT_DIR.parent


def _resolve_from(cwd: Path) -> str:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_DIR)
    result = subprocess.run(
        [sys.executable, "-c", "import config; print(config.KNOWLEDGE_FILE)"],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return result.stdout.strip()


def test_knowledge_path_is_absolute_and_cwd_independent():
    from_project = _resolve_from(PROJECT_DIR)
    from_parent = _resolve_from(PARENT_DIR)
    assert from_project == from_parent == str(config.KNOWLEDGE_FILE)
    assert config.KNOWLEDGE_FILE.is_absolute()
    assert config.KNOWLEDGE_FILE.name == "knowledge.txt"


def test_missing_or_non_utf8_knowledge_file_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "KNOWLEDGE_FILE", tmp_path / "missing.txt")
    with pytest.raises(FileNotFoundError, match="知识库文件不存在"):
        config._validate_knowledge_file()

    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfe\x00bad")
    monkeypatch.setattr(config, "KNOWLEDGE_FILE", bad)
    with pytest.raises(ValueError, match="UTF-8"):
        config._validate_knowledge_file()


def test_invalid_params_are_reported(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_OVERLAP", config.CHUNK_SIZE)
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        config._validate_params()
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 80)
    monkeypatch.setattr(config, "TOP_K", 0)
    with pytest.raises(ValueError, match="TOP_K"):
        config._validate_params()


def test_missing_env_names_without_values(monkeypatch):
    monkeypatch.delenv("BAILIAN_API_KEY", raising=False)
    monkeypatch.delenv("BAILIAN_API_URL", raising=False)
    assert "BAILIAN_API_KEY" in config.missing_env_vars(
        config.EMBED_API_KEY_ENV, config.EMBED_BASE_URL_ENV
    )
    with pytest.raises(RuntimeError) as excinfo:
        config.validate_embedding_config()
    assert "BAILIAN_API_KEY" in str(excinfo.value)
    assert "http" not in str(excinfo.value)


def test_baseline_configuration():
    assert config.EMBED_PROVIDER == "openai"
    assert config.EMBED_MODEL_NAME == "text-embedding-v4"
    assert config.EMBED_DIM == 1024
    assert config.EMBED_API_KEY_ENV == "BAILIAN_API_KEY"
    assert config.EMBED_BASE_URL_ENV == "BAILIAN_API_URL"
    assert (config.CHUNK_SIZE, config.CHUNK_OVERLAP, config.TOP_K) == (220, 80, 5)
    assert config.CHUNKING_STRATEGY == "section-parent-v2"
    assert config.SCHEMA_VERSION == 2
