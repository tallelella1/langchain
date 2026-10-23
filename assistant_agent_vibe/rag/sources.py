"""知识来源清单（P1.3 / P2.1）。

- 默认只有 ``knowledge.txt``；``data/sources.json`` 记录显式加入的来源；
- ``asset/load`` 中的学习素材不会被自动扫描入库，必须显式 add；
- 删除来源即从清单移除并重建新集合，旧集合按版本保留，不原地删除记录；
- 注册表只保存相对路径与格式，不保存内容或密钥。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import config
from rag import loaders
from rag.paths import is_inside_project, normalize_source

REGISTRY_VERSION = 1


class SourceError(RuntimeError):
    """来源清单缺失、重复、格式不支持或路径越界。"""


def _format_of(source: str) -> str:
    suffix = Path(source).suffix.lower()
    if suffix not in loaders.LOADERS:
        raise SourceError(
            f"不支持的来源格式: {suffix or '无扩展名'}；"
            f"支持 {', '.join(loaders.SUPPORTED_SUFFIXES)}"
        )
    return suffix.lstrip(".")


def resolve_source(source: str | Path, *, require_inside: bool = True) -> Path:
    path = Path(source)
    if not path.is_absolute():
        path = config.PROJECT_DIR / path
    path = path.resolve()
    # 注册表来源必须位于项目内；显式绝对路径（测试/离线构建）单独放行。
    if require_inside and not Path(source).is_absolute() and not is_inside_project(path):
        raise SourceError(f"来源必须位于项目目录内: {path}")
    if not path.is_file():
        raise SourceError(f"来源文件不存在: {path}")
    _format_of(path.name)
    return path


def make_entry(source: str | Path, *, require_inside: bool = True) -> dict[str, str]:
    path = resolve_source(source, require_inside=require_inside)
    normalized = normalize_source(path)
    return {"source": normalized, "format": _format_of(normalized)}


def default_entries() -> list[dict[str, str]]:
    return [make_entry(config.DEFAULT_SOURCE)]


def load_entries() -> list[dict[str, str]]:
    path = config.SOURCE_REGISTRY_FILE
    if not path.is_file():
        return default_entries()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceError(f"来源清单无法读取: {path}；{exc}") from exc
    if not isinstance(payload, dict) or payload.get("version") != REGISTRY_VERSION:
        raise SourceError(f"来源清单版本无效: {path}")
    raw = payload.get("sources")
    if not isinstance(raw, list) or not raw:
        raise SourceError(f"来源清单不能为空: {path}")
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("source"), str):
            raise SourceError(f"来源清单条目无效: {item!r}")
        entry = make_entry(item["source"])
        if entry["source"] in seen:
            raise SourceError(f"来源清单存在重复路径: {entry['source']}")
        seen.add(entry["source"])
        entries.append(entry)
    return entries


def save_entries(entries: Iterable[dict[str, str]]) -> Path:
    entries = list(entries)
    if not entries:
        raise SourceError("来源清单不能为空；如需删除知识库请显式维护旧集合")
    path = config.SOURCE_REGISTRY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": REGISTRY_VERSION, "sources": entries}
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temp_path, path)
    return path


def add_source(source: str | Path) -> list[dict[str, str]]:
    entries = load_entries()
    entry = make_entry(source)
    if any(item["source"] == entry["source"] for item in entries):
        raise SourceError(f"来源已存在: {entry['source']}")
    entries.append(entry)
    save_entries(entries)
    return entries


def remove_source(source: str | Path) -> list[dict[str, str]]:
    entries = load_entries()
    normalized = normalize_source(resolve_source(source))
    remaining = [item for item in entries if item["source"] != normalized]
    if len(remaining) == len(entries):
        raise SourceError(f"来源不在清单中: {normalized}")
    save_entries(remaining)
    return remaining


def entries_from_paths(paths: Iterable[str | Path]) -> list[dict[str, str]]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    entries = [make_entry(path) for path in paths]
    seen: set[str] = set()
    for entry in entries:
        if entry["source"] in seen:
            raise SourceError(f"来源路径重复: {entry['source']}")
        seen.add(entry["source"])
    if not entries:
        raise SourceError("至少需要一个知识来源")
    return entries


def summary(entries: Iterable[dict[str, str]]) -> dict[str, Any]:
    entries = list(entries)
    return {
        "count": len(entries),
        "sources": entries,
        "formats": sorted({entry["format"] for entry in entries}),
    }
