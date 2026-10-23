"""项目内路径规范化（避免 documents/sources 循环导入）。"""

from __future__ import annotations

from pathlib import Path

import config


def normalize_source(path: str | Path) -> str:
    """把来源路径规范为项目内相对路径（统一使用 /）；项目外保留绝对路径。"""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(config.PROJECT_DIR).as_posix()
    except ValueError:
        return resolved.as_posix()


def is_inside_project(path: str | Path) -> bool:
    try:
        Path(path).resolve().relative_to(config.PROJECT_DIR)
        return True
    except ValueError:
        return False
