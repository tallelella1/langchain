"""步骤 3：建立 Milvus 连接与集合边界。

默认行为：检查/创建目标 database 与 collection，打印字段、维度、主键、索引和加载状态。
已有非空集合时停止，不覆盖数据；任何路径都不执行 drop。

用法：
    python ingest.py            # 建立/检查入库目标边界
    python ingest.py --inspect  # 只读检查，不创建、不加载、不写入
    python ingest.py --verify   # 真实写入/读取/搜索/删除验证，并测试非空与维度保护
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config
from rag import vector_store

VERIFY_ID = 2**63 - 1  # 避免与真实切片 id（从 0 开始）冲突
WRONG_DIM_ID = VERIFY_ID - 1


def print_summary(summary: dict) -> None:
    print(f"Milvus URI: {summary['uri']}")
    print(f"数据库: {summary['database']}；现有数据库: {summary['databases']}")
    print(f"集合: {summary['collection']}")
    print(
        f"auto_id: {summary['auto_id']}；"
        f"enable_dynamic_field: {summary['enable_dynamic_field']}"
    )
    print("字段:")
    for field in summary["fields"]:
        primary = " primary_key" if field["is_primary"] else ""
        print(f"  {field['name']}: {field['type']}{primary} {field['params']}")
    index = summary["index"]
    print(
        f"索引: {index['index_type']} / {index['metric_type']} "
        f"on {index['field']} (name={index['name']})"
    )
    print(f"加载状态: {summary['load_state']}")
    print(f"行数(get_collection_stats，可能滞后): {summary['row_count']}")
    print(f"本次创建: {summary.get('created', False)}")


def verify_roundtrip(client) -> None:
    """真实写入、读取、搜索、删除；并验证非空与维度冲突保护。"""
    collection = config.COLLECTION_NAME
    vector = [1.0] + [0.0] * (config.EMBED_DIM - 1)
    record = {
        "id": VERIFY_ID,
        "vector": vector,
        "text": "step-3 verify: real insert/read/search record",
        "source": "verification",
        "chunk_id": -1,
    }

    try:
        # 客户端侧维度保护
        try:
            vector_store.validate_vector_dim([0.0, 0.0, 0.0])
        except vector_store.DimensionMismatchError as exc:
            print(f"客户端维度保护: pass ({exc})")
        else:
            raise AssertionError("客户端维度保护未生效")

        result = client.insert(collection_name=collection, data=[record])
        if result.get("insert_count") != 1:
            raise AssertionError(f"写入数量异常: {result}")
        print(f"写入验证: pass ({result})")

        rows = client.query(
            collection_name=collection,
            filter=f"id == {VERIFY_ID}",
            output_fields=["id", "text", "source", "chunk_id"],
            consistency_level=vector_store.CONSISTENCY_LEVEL,
        )
        if len(rows) != 1 or rows[0]["text"] != record["text"]:
            raise AssertionError(f"读取验证失败: {rows}")
        print(f"读取验证: pass (id={rows[0]['id']}, chunk_id={rows[0]['chunk_id']})")

        hits = client.search(
            collection_name=collection,
            data=[vector],
            anns_field="vector",
            search_params={
                "metric_type": vector_store.METRIC_TYPE,
                "params": {},
            },
            limit=1,
            output_fields=["id"],
            consistency_level=vector_store.CONSISTENCY_LEVEL,
        )
        top = hits[0][0] if hits and hits[0] else None
        if not top or top.get("id") != VERIFY_ID:
            raise AssertionError(f"搜索验证失败: {hits}")
        print(f"搜索验证: pass (id={top['id']}, score={top.get('distance')})")

        # 集合此刻非空，入库准备必须停止
        try:
            vector_store.prepare_ingest_target()
        except vector_store.CollectionNotEmptyError as exc:
            print(f"非空保护: pass ({exc})")
        else:
            raise AssertionError("非空保护未生效")

        # 服务端维度冲突保护：错误维度必须被拒绝。
        # 这是预期内的负向测试，临时压低 pymilvus 的 ERROR 日志，避免误读为失败。
        previous_disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            try:
                client.insert(
                    collection_name=collection,
                    data=[
                        {
                            "id": WRONG_DIM_ID,
                            "vector": [0.1, 0.2, 0.3],
                            "text": "wrong dimension",
                            "source": "verification",
                            "chunk_id": -2,
                        }
                    ],
                )
            except Exception as exc:
                message = str(exc).replace("\n", " ")[:300]
                print(f"服务端维度冲突: pass ({type(exc).__name__}: {message})")
            else:
                raise AssertionError("服务端接受了错误维度的向量")
        finally:
            logging.disable(previous_disable)
    finally:
        try:
            client.delete(
                collection_name=collection,
                ids=[VERIFY_ID, WRONG_DIM_ID],
            )
            print("验证记录清理: pass")
        except Exception as exc:
            print(f"验证记录清理失败: {type(exc).__name__}: {exc}")

    rows = client.query(
        collection_name=collection,
        filter="",
        output_fields=["id"],
        limit=2,
        consistency_level=vector_store.CONSISTENCY_LEVEL,
    )
    if rows:
        raise AssertionError(f"清理后集合仍非空: {rows}")
    print("清理后集合为空: pass")


def main() -> int:
    parser = argparse.ArgumentParser(description="建立 Milvus 连接与集合边界")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--inspect",
        action="store_true",
        help="只读检查，不创建 database/collection，也不加载集合",
    )
    group.add_argument(
        "--verify",
        action="store_true",
        help="在空集合上执行真实写入/读取/搜索/删除验证",
    )
    args = parser.parse_args()

    try:
        if args.inspect:
            print("=== 只读检查 ===")
            print_summary(vector_store.inspect_target())
            return 0

        client, summary = vector_store.prepare_ingest_target()
        try:
            print("=== 入库目标边界 ===")
            print_summary(summary)
            if args.verify:
                print("\n=== 真实读写验证 ===")
                verify_roundtrip(client)
        finally:
            client.close()
        return 0
    except vector_store.CollectionNotEmptyError as exc:
        print(f"\n入库停止: {exc}")
        print_summary(exc.details)
        return 2
    except vector_store.VectorStoreError as exc:
        print(f"\nMilvus 检查失败: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
