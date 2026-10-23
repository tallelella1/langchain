"""Milvus 连接、集合边界与显式 schema。

边界约定：
- 只创建缺失的 database/collection，任何路径都不执行 drop。
- 问答入口使用 inspect_target()，只做读取检查，不创建、不加载、不写入。
- 入库入口使用 prepare_ingest_target()，已有非空集合时抛出 CollectionNotEmptyError。
"""

from __future__ import annotations

from typing import Any, Sequence

from pymilvus import DataType, MilvusClient

import config

TEXT_MAX_LENGTH = 8192  # 字节，不是字符
SOURCE_MAX_LENGTH = 1024  # 字节
INDEX_TYPE = "FLAT"
METRIC_TYPE = "COSINE"
CONSISTENCY_LEVEL = "Strong"
CONNECT_TIMEOUT = 10
LOAD_TIMEOUT = 30


class VectorStoreError(RuntimeError):
    """Milvus 配置或状态不符合预期。"""


class MilvusConnectionError(VectorStoreError):
    pass


class DatabaseNotFoundError(VectorStoreError):
    pass


class CollectionNotFoundError(VectorStoreError):
    pass


class CollectionNotEmptyError(VectorStoreError):
    def __init__(self, message: str, details: dict[str, Any]):
        super().__init__(message)
        self.details = details


class SchemaMismatchError(VectorStoreError):
    pass


class DimensionMismatchError(VectorStoreError):
    pass


# =========================
# 连接与 database 边界
# =========================


def _open_client(db_name: str | None = None) -> MilvusClient:
    try:
        return MilvusClient(
            uri=config.MILVUS_URI,
            db_name=db_name,
            timeout=CONNECT_TIMEOUT,
        )
    except Exception as exc:
        raise MilvusConnectionError(
            f"连接 Milvus 失败: {config.MILVUS_URI}；{type(exc).__name__}: {exc}"
        ) from exc


def open_target(*, create_database: bool) -> MilvusClient:
    """连接目标 database；create_database=True 时只创建缺失的目标库。"""
    client = _open_client()
    try:
        databases = client.list_databases()
    except Exception as exc:
        client.close()
        raise MilvusConnectionError(
            f"连接 Milvus 失败: {config.MILVUS_URI}；{type(exc).__name__}: {exc}"
        ) from exc

    if config.DB_NAME not in databases:
        if not create_database:
            client.close()
            raise DatabaseNotFoundError(
                f"数据库不存在: {config.DB_NAME}；现有数据库: {databases}"
            )
        try:
            client.create_database(config.DB_NAME)
            databases = client.list_databases()
        except Exception as exc:
            client.close()
            raise VectorStoreError(
                f"创建数据库失败: {config.DB_NAME}；{type(exc).__name__}: {exc}"
            ) from exc
        if config.DB_NAME not in databases:
            client.close()
            raise VectorStoreError(
                f"创建数据库后仍不存在: {config.DB_NAME}；现有数据库: {databases}"
            )

    try:
        client.using_database(config.DB_NAME)
        client.list_collections()  # 强制一次往返，确认已切换目标库
        return client
    except Exception as exc:
        client.close()
        raise VectorStoreError(
            f"切换数据库失败: {config.DB_NAME}；{type(exc).__name__}: {exc}"
        ) from exc


# =========================
# 显式 schema 与索引
# =========================


def build_schema():
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=False)
    schema.add_field(
        field_name="vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=config.EMBED_DIM,
    )
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=TEXT_MAX_LENGTH,
    )
    schema.add_field(
        field_name="source",
        datatype=DataType.VARCHAR,
        max_length=SOURCE_MAX_LENGTH,
    )
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64)
    return schema


def build_index_params(client: MilvusClient):
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type=INDEX_TYPE,
        metric_type=METRIC_TYPE,
    )
    return index_params


def validate_vector_dim(vector: Sequence[float], *, expected: int | None = None) -> None:
    expected = config.EMBED_DIM if expected is None else expected
    actual = len(vector)
    if actual != expected:
        raise DimensionMismatchError(
            f"向量维度不符: 期望 {expected}，实际 {actual}"
        )


# =========================
# 只读检查
# =========================


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _type_name(field_type: Any) -> str:
    try:
        return DataType(field_type).name
    except ValueError:
        return str(field_type)


def _field_map(description: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {field["name"]: field for field in description.get("fields", [])}


def validate_schema(description: dict[str, Any]) -> dict[str, dict[str, Any]]:
    fields = _field_map(description)
    problems: list[str] = []

    expected_types = {
        "id": DataType.INT64,
        "vector": DataType.FLOAT_VECTOR,
        "text": DataType.VARCHAR,
        "source": DataType.VARCHAR,
        "chunk_id": DataType.INT64,
    }
    for name, expected_type in expected_types.items():
        field = fields.get(name)
        if field is None:
            problems.append(f"缺少字段 {name}")
            continue
        actual_type = _type_name(field.get("type"))
        if actual_type != expected_type.name:
            problems.append(f"字段 {name} 类型为 {actual_type}，期望 {expected_type.name}")

    id_field = fields.get("id")
    if id_field is not None and not id_field.get("is_primary"):
        problems.append("字段 id 不是主键")
    if description.get("auto_id") is not False:
        problems.append(f"auto_id 期望 False，实际 {description.get('auto_id')}")
    if description.get("enable_dynamic_field") is True:
        problems.append("enable_dynamic_field 期望 False，实际 True")

    vector_params = fields.get("vector", {}).get("params", {})
    dim = _to_int(vector_params.get("dim"))
    if dim != config.EMBED_DIM:
        problems.append(f"向量维度为 {dim}，期望 {config.EMBED_DIM}")

    for name, expected_length in (
        ("text", TEXT_MAX_LENGTH),
        ("source", SOURCE_MAX_LENGTH),
    ):
        params = fields.get(name, {}).get("params", {})
        actual_length = _to_int(params.get("max_length"))
        if actual_length != expected_length:
            problems.append(
                f"字段 {name} max_length 为 {actual_length}，期望 {expected_length}"
            )

    if problems:
        raise SchemaMismatchError("集合 schema 与配置不符: " + "；".join(problems))
    return fields


def validate_index(client: MilvusClient, collection_name: str) -> dict[str, Any]:
    index_names = client.list_indexes(collection_name=collection_name)
    if not index_names:
        raise SchemaMismatchError(f"集合缺少索引: {collection_name}")

    index_name = "vector" if "vector" in index_names else index_names[0]
    description = client.describe_index(
        collection_name=collection_name,
        index_name=index_name,
    )
    problems: list[str] = []
    field_name = description.get("field_name")
    index_type = str(description.get("index_type", "")).upper()
    metric_type = str(description.get("metric_type", "")).upper()

    if field_name != "vector":
        problems.append(f"索引字段为 {field_name}，期望 vector")
    if index_type != INDEX_TYPE:
        problems.append(f"索引类型为 {index_type or '未知'}，期望 {INDEX_TYPE}")
    if metric_type != METRIC_TYPE:
        problems.append(f"距离度量为 {metric_type or '未知'}，期望 {METRIC_TYPE}")

    if problems:
        raise SchemaMismatchError("集合索引与配置不符: " + "；".join(problems))
    return {
        "name": index_name,
        "field": field_name,
        "index_type": index_type,
        "metric_type": metric_type,
    }


def load_state_name(client: MilvusClient, collection_name: str) -> str:
    result = client.get_load_state(collection_name=collection_name)
    state = result.get("state") if isinstance(result, dict) else result
    return getattr(state, "name", str(state))


def ensure_loaded(client: MilvusClient, collection_name: str) -> str:
    state = load_state_name(client, collection_name)
    if state.lower() == "loaded":
        return state
    client.load_collection(collection_name=collection_name, timeout=LOAD_TIMEOUT)
    state = load_state_name(client, collection_name)
    if state.lower() != "loaded":
        raise VectorStoreError(f"集合加载失败: {collection_name}，状态 {state}")
    return state


def describe_target(client: MilvusClient) -> dict[str, Any]:
    """读取并校验目标集合；只读，不创建、不加载、不写入。"""
    collection_name = config.COLLECTION_NAME
    description = client.describe_collection(collection_name=collection_name)
    fields = validate_schema(description)
    index = validate_index(client, collection_name)
    state = load_state_name(client, collection_name)
    stats = client.get_collection_stats(collection_name=collection_name)
    row_count = _to_int(stats.get("row_count")) or 0

    return {
        "uri": config.MILVUS_URI,
        "database": config.DB_NAME,
        "collection": collection_name,
        "databases": client.list_databases(),
        "auto_id": description.get("auto_id"),
        "enable_dynamic_field": description.get("enable_dynamic_field"),
        "fields": [
            {
                "name": field["name"],
                "type": _type_name(field.get("type")),
                "is_primary": bool(field.get("is_primary", False)),
                "params": field.get("params", {}),
            }
            for field in fields.values()
        ],
        "index": index,
        "load_state": state,
        "row_count": row_count,
    }


def inspect_target() -> dict[str, Any]:
    """问答入口使用：只读检查，不执行初始化写操作。"""
    client = open_target(create_database=False)
    try:
        if not client.has_collection(config.COLLECTION_NAME):
            raise CollectionNotFoundError(
                f"集合不存在: {config.DB_NAME}.{config.COLLECTION_NAME}"
            )
        return describe_target(client)
    finally:
        client.close()


def _collection_is_empty(client: MilvusClient) -> bool:
    rows = client.query(
        collection_name=config.COLLECTION_NAME,
        filter="",
        output_fields=["id"],
        limit=1,
        consistency_level=CONSISTENCY_LEVEL,
    )
    return len(rows) == 0


# =========================
# 入库边界
# =========================


def prepare_ingest_target() -> tuple[MilvusClient, dict[str, Any]]:
    """创建缺失的 database/collection；已有非空集合时停止，不覆盖数据。"""
    client = open_target(create_database=True)
    try:
        created = False
        if not client.has_collection(config.COLLECTION_NAME):
            client.create_collection(
                collection_name=config.COLLECTION_NAME,
                schema=build_schema(),
                index_params=build_index_params(client),
                consistency_level=CONSISTENCY_LEVEL,
            )
            created = True

        summary = describe_target(client)
        if not created and summary["row_count"] > 0:
            summary["created"] = created
            raise CollectionNotEmptyError(
                f"集合已有 {summary['row_count']} 行，停止入库: "
                f"{config.DB_NAME}.{config.COLLECTION_NAME}",
                summary,
            )

        summary["load_state"] = ensure_loaded(client, config.COLLECTION_NAME)

        # stats 只是辅助；加载后再用实际查询确认空集合。
        if not created and not _collection_is_empty(client):
            summary["created"] = created
            raise CollectionNotEmptyError(
                f"集合实际存在记录，停止入库: {config.DB_NAME}.{config.COLLECTION_NAME}",
                summary,
            )

        summary["created"] = created
        return client, summary
    except Exception:
        client.close()
        raise
