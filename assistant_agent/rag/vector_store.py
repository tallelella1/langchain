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


class RecordValidationError(VectorStoreError):
    pass


class CollectionNotLoadedError(VectorStoreError):
    pass


def _name(collection_name: str | None) -> str:
    return config.COLLECTION_NAME if collection_name is None else collection_name


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


def embedding_identity() -> dict[str, str]:
    """写入集合属性的 Embedding 身份，用于检测同维度换模型。"""
    return {
        "embedding_model": config.EMBED_MODEL_NAME,
        "embedding_dim": str(config.EMBED_DIM),
    }


def validate_embedding_identity(description: dict[str, Any]) -> None:
    """核对集合属性中的 Embedding 模型标识；缺失或多模型混用都明确失败。"""
    properties = description.get("properties") or {}
    model = properties.get("embedding_model")
    dim = properties.get("embedding_dim")
    if not model or not dim:
        raise SchemaMismatchError(
            "集合缺少 embedding_model/embedding_dim 标识；"
            "如确认集合由当前模型生成，请运行 python ingest.py --stamp"
        )
    if model != config.EMBED_MODEL_NAME:
        raise SchemaMismatchError(
            f"集合 embedding_model 为 {model}，当前配置为 {config.EMBED_MODEL_NAME}；"
            "换模型必须重新向量化全部文档，不能混用同一索引"
        )
    if str(dim) != str(config.EMBED_DIM):
        raise SchemaMismatchError(
            f"集合 Embedding 维度为 {dim}，当前配置为 {config.EMBED_DIM}"
        )


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
    try:
        result = client.get_load_state(collection_name=collection_name)
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(
            f"读取集合加载状态失败: {config.DB_NAME}.{collection_name}；"
            f"{type(exc).__name__}: {exc}"
        ) from exc
    state = result.get("state") if isinstance(result, dict) else result
    return getattr(state, "name", str(state))


def ensure_loaded(client: MilvusClient, collection_name: str) -> str:
    state = load_state_name(client, collection_name)
    if state.lower() == "loaded":
        return state
    try:
        client.load_collection(collection_name=collection_name, timeout=LOAD_TIMEOUT)
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(
            f"加载集合失败: {config.DB_NAME}.{collection_name}；"
            f"{type(exc).__name__}: {exc}"
        ) from exc
    state = load_state_name(client, collection_name)
    if state.lower() != "loaded":
        raise VectorStoreError(f"集合加载失败: {collection_name}，状态 {state}")
    return state


def describe_target(
    client: MilvusClient, collection_name: str | None = None
) -> dict[str, Any]:
    """读取并校验目标集合；只读，不创建、不加载、不写入。"""
    collection_name = _name(collection_name)
    try:
        description = client.describe_collection(collection_name=collection_name)
        validate_embedding_identity(description)
        fields = validate_schema(description)
        index = validate_index(client, collection_name)
        state = load_state_name(client, collection_name)
        stats = client.get_collection_stats(collection_name=collection_name)
        databases = client.list_databases()
        row_count = _to_int(stats.get("row_count")) or 0
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(
            f"读取集合信息失败: {config.DB_NAME}.{collection_name}；"
            f"{type(exc).__name__}: {exc}"
        ) from exc

    return {
        "uri": config.MILVUS_URI,
        "database": config.DB_NAME,
        "collection": collection_name,
        "databases": databases,
        "auto_id": description.get("auto_id"),
        "enable_dynamic_field": description.get("enable_dynamic_field"),
        "properties": description.get("properties") or {},
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


def inspect_target(collection_name: str | None = None) -> dict[str, Any]:
    """问答入口使用：只读检查，不执行初始化写操作。"""
    collection_name = _name(collection_name)
    client = open_target(create_database=False)
    try:
        try:
            exists = client.has_collection(collection_name)
        except Exception as exc:
            raise VectorStoreError(
                f"检查集合失败: {config.DB_NAME}.{collection_name}；"
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not exists:
            raise CollectionNotFoundError(
                f"集合不存在: {config.DB_NAME}.{collection_name}"
            )
        return describe_target(client, collection_name)
    finally:
        client.close()


def _collection_is_empty(client: MilvusClient, collection_name: str) -> bool:
    try:
        rows = client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["id"],
            limit=1,
            consistency_level=CONSISTENCY_LEVEL,
        )
    except Exception as exc:
        raise VectorStoreError(
            f"检查集合是否为空失败: {config.DB_NAME}.{collection_name}；"
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return len(rows) == 0


# =========================
# 入库边界
# =========================


def prepare_ingest_target(
    collection_name: str | None = None,
) -> tuple[MilvusClient, dict[str, Any]]:
    """创建缺失的 database/collection；已有非空集合时停止，不覆盖数据。"""
    collection_name = _name(collection_name)
    client = open_target(create_database=True)
    try:
        try:
            exists = client.has_collection(collection_name)
        except Exception as exc:
            raise VectorStoreError(
                f"检查集合失败: {config.DB_NAME}.{collection_name}；"
                f"{type(exc).__name__}: {exc}"
            ) from exc

        created = False
        if not exists:
            try:
                client.create_collection(
                    collection_name=collection_name,
                    schema=build_schema(),
                    index_params=build_index_params(client),
                    consistency_level=CONSISTENCY_LEVEL,
                    properties=embedding_identity(),
                )
            except Exception as exc:
                raise VectorStoreError(
                    f"创建集合失败: {config.DB_NAME}.{collection_name}；"
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            created = True

        summary = describe_target(client, collection_name)
        summary["load_state"] = ensure_loaded(client, collection_name)

        # get_collection_stats 只用于展示；空集合以 Strong 一致性实际查询为准。
        if not created and not _collection_is_empty(client, collection_name):
            summary["created"] = created
            raise CollectionNotEmptyError(
                f"集合实际存在记录，停止入库: {config.DB_NAME}.{collection_name}",
                summary,
            )

        summary["created"] = created
        return client, summary
    except VectorStoreError:
        client.close()
        raise
    except Exception as exc:
        client.close()
        raise VectorStoreError(
            f"准备入库目标失败: {config.DB_NAME}.{collection_name}；"
            f"{type(exc).__name__}: {exc}"
        ) from exc


def stamp_collection_identity(collection_name: str | None = None) -> dict[str, Any]:
    """维护：为已有集合写入当前 Embedding 身份属性；不改动数据。"""
    collection_name = _name(collection_name)
    client = open_query_client(collection_name)
    try:
        try:
            description = client.describe_collection(collection_name=collection_name)
        except Exception as exc:
            raise VectorStoreError(
                f"读取集合信息失败: {config.DB_NAME}.{collection_name}；"
                f"{type(exc).__name__}: {exc}"
            ) from exc
        validate_schema(description)
        validate_index(client, collection_name)
        try:
            client.alter_collection_properties(
                collection_name=collection_name,
                properties=embedding_identity(),
            )
        except Exception as exc:
            raise VectorStoreError(
                f"写入集合 Embedding 标识失败: {config.DB_NAME}.{collection_name}；"
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return describe_target(client, collection_name)
    finally:
        client.close()


# =========================
# 查询入口（只读检查）
# =========================


def open_query_client(collection_name: str | None = None) -> MilvusClient:
    """打开问答用的只读连接；不创建 database/collection，不加载、不写入。"""
    collection_name = _name(collection_name)
    client = open_target(create_database=False)
    try:
        try:
            exists = client.has_collection(collection_name)
        except Exception as exc:
            raise VectorStoreError(
                f"检查集合失败: {config.DB_NAME}.{collection_name}；"
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not exists:
            raise CollectionNotFoundError(
                f"集合不存在: {config.DB_NAME}.{collection_name}；请先运行 python ingest.py"
            )
        return client
    except Exception:
        client.close()
        raise


def assert_loaded(client: MilvusClient, collection_name: str | None = None) -> str:
    """确认集合已加载；未加载时给出明确维护提示，不擅自执行加载。"""
    collection_name = _name(collection_name)
    state = load_state_name(client, collection_name)
    if state.lower() != "loaded":
        raise CollectionNotLoadedError(
            f"集合未加载: {config.DB_NAME}.{collection_name}，当前状态 {state}；"
            "请先运行 python ingest.py --load"
        )
    return state


def load_existing(collection_name: str | None = None) -> dict[str, Any]:
    """维护操作：加载已存在的集合，不写入数据、不删除集合。"""
    collection_name = _name(collection_name)
    client = open_query_client(collection_name)
    try:
        state = ensure_loaded(client, collection_name)
        summary = describe_target(client, collection_name)
        summary["load_state"] = state
        return summary
    finally:
        client.close()


def validate_query_target(
    client: MilvusClient, collection_name: str | None = None
) -> dict[str, Any]:
    """问答前核对字段、索引与加载状态；全部为只读检查。"""
    collection_name = _name(collection_name)
    description = client.describe_collection(collection_name=collection_name)
    validate_embedding_identity(description)
    validate_schema(description)
    validate_index(client, collection_name)
    assert_loaded(client, collection_name)
    return description


# =========================
# 记录写入、读取与原始搜索
# =========================


def _utf8_bytes(value: str) -> int:
    return len(value.encode("utf-8"))


def validate_record(record: dict[str, Any]) -> None:
    """写入前检查字段、ID、文本字节长度与向量维度。"""
    for field in ("id", "vector", "text", "source", "chunk_id"):
        if field not in record:
            raise RecordValidationError(f"记录缺少字段 {field}")

    for field in ("id", "chunk_id"):
        value = record[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RecordValidationError(f"记录 {field} 必须是非负整数，实际 {value!r}")

    text = record["text"]
    if not isinstance(text, str) or not text.strip():
        raise RecordValidationError("记录 text 不能为空")
    if _utf8_bytes(text) > TEXT_MAX_LENGTH:
        raise RecordValidationError(
            f"记录 text 超过 {TEXT_MAX_LENGTH} 字节（UTF-8 实际 {_utf8_bytes(text)}）"
        )

    source = record["source"]
    if not isinstance(source, str) or not source.strip():
        raise RecordValidationError("记录 source 不能为空")
    if _utf8_bytes(source) > SOURCE_MAX_LENGTH:
        raise RecordValidationError(
            f"记录 source 超过 {SOURCE_MAX_LENGTH} 字节（UTF-8 实际 {_utf8_bytes(source)}）"
        )

    validate_vector_dim(record["vector"])


def insert_records(
    client: MilvusClient,
    records: Sequence[dict[str, Any]],
    collection_name: str | None = None,
) -> dict[str, Any]:
    """校验后写入；写入数量不符时明确失败。"""
    collection_name = _name(collection_name)
    records = list(records)
    if not records:
        raise RecordValidationError("待写入记录为空")
    seen: set[int] = set()
    for record in records:
        validate_record(record)
        if record["id"] in seen:
            raise RecordValidationError(f"记录 id 重复: {record['id']}")
        seen.add(record["id"])

    try:
        result = client.insert(collection_name=collection_name, data=records)
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(
            f"写入 Milvus 失败: {config.DB_NAME}.{collection_name}；"
            f"{type(exc).__name__}: {exc}"
        ) from exc
    insert_count = _to_int(result.get("insert_count"))
    if insert_count != len(records):
        raise VectorStoreError(
            f"写入数量不符: 期望 {len(records)}，实际 {insert_count}；{result}"
        )
    return result


def flush(client: MilvusClient, collection_name: str | None = None) -> None:
    collection_name = _name(collection_name)
    try:
        client.flush(collection_name=collection_name)
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(
            f"刷新集合失败: {config.DB_NAME}.{collection_name}；"
            f"{type(exc).__name__}: {exc}"
        ) from exc


def fetch_records_by_ids(
    client: MilvusClient,
    ids: Sequence[int],
    collection_name: str | None = None,
    output_fields: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """按 ID 读取实际记录（Strong 一致性），用于入库后的可见性核对。"""
    collection_name = _name(collection_name)
    ids = [int(value) for value in ids]
    if not ids:
        return []
    filter_expr = "id in [" + ", ".join(str(value) for value in ids) + "]"
    try:
        rows = client.query(
            collection_name=collection_name,
            filter=filter_expr,
            output_fields=list(output_fields or ("id", "text", "source", "chunk_id")),
            limit=len(ids),
            consistency_level=CONSISTENCY_LEVEL,
        )
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(
            f"按 ID 读取记录失败: {config.DB_NAME}.{collection_name}；"
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return sorted(rows, key=lambda row: row.get("id", 0))


def count_records(client: MilvusClient, collection_name: str | None = None) -> int:
    """用 count(*) 统计实际记录数，不用一次受限 query 的长度代替。"""
    collection_name = _name(collection_name)
    try:
        rows = client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["count(*)"],
            consistency_level=CONSISTENCY_LEVEL,
        )
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(
            f"统计记录数失败: {config.DB_NAME}.{collection_name}；"
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not rows:
        return 0
    value = rows[0].get("count(*)")
    count = _to_int(value)
    if count is None:
        raise VectorStoreError(f"无法解析 count(*) 结果: {rows}")
    return count


def search_raw(
    client: MilvusClient,
    query_vector: Sequence[float],
    k: int,
    collection_name: str | None = None,
) -> list[list[dict[str, Any]]]:
    """按 COSINE 搜索并返回 PyMilvus 原始嵌套结果。"""
    collection_name = _name(collection_name)
    if k <= 0:
        raise ValueError("k 必须大于 0")
    validate_vector_dim(query_vector)
    try:
        return client.search(
            collection_name=collection_name,
            data=[list(query_vector)],
            anns_field="vector",
            search_params={"metric_type": METRIC_TYPE, "params": {}},
            limit=k,
            output_fields=["id", "text", "source", "chunk_id"],
            consistency_level=CONSISTENCY_LEVEL,
        )
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(
            f"搜索集合失败: {config.DB_NAME}.{collection_name}；"
            f"{type(exc).__name__}: {exc}"
        ) from exc
