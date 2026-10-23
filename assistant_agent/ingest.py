"""独立知识入库入口（问答入口不会触发入库）。

默认行为：校验配置 → 检查/创建空集合 → 切分 knowledge.txt → 文档 Embedding
→ 数量/维度/字节长度校验 → 写入 Milvus → 读回核对 → 自向量搜索抽查。

用法：
    python ingest.py            # 完整入库（已有非空集合时停止）
    python ingest.py --inspect  # 只读检查集合边界，不创建、不加载、不写入
    python ingest.py --load     # 维护：加载已存在的集合（不写入）
    python ingest.py --verify   # 步骤 3：空集合上真实写入/读取/搜索/删除验证
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config
from rag import documents, models, vector_store

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
    print(f"Embedding 标识: {summary.get('properties', {})}")
    print(f"行数(get_collection_stats，可能滞后): {summary['row_count']}")
    print(f"本次创建: {summary.get('created', False)}")


# =========================
# 入库记录与读回核对（可离线测试）
# =========================


def build_records(
    chunks: Sequence[Any], vectors: Sequence[Sequence[float]]
) -> list[dict[str, Any]]:
    """把切片与向量组装成 Milvus 记录；首版单文件 id 即切片序号。"""
    if len(chunks) != len(vectors):
        raise ValueError(
            f"切片与向量数量不符: {len(chunks)} 个切片，{len(vectors)} 个向量"
        )
    records: list[dict[str, Any]] = []
    for chunk, vector in zip(chunks, vectors):
        record = {
            "id": chunk.metadata["chunk_id"],
            "vector": list(vector),
            "text": chunk.page_content,
            "source": chunk.metadata["source"],
            "chunk_id": chunk.metadata["chunk_id"],
        }
        vector_store.validate_record(record)
        records.append(record)
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("切片 id 重复，拒绝写入")
    return records


def verify_readback(chunks: Sequence[Any], rows: Sequence[dict[str, Any]]) -> None:
    """核对实际读回的记录与预期切片：数量、ID、正文、来源、切片号。"""
    expected = {chunk.metadata["chunk_id"]: chunk for chunk in chunks}
    if len(rows) != len(expected):
        raise AssertionError(f"读回数量不符: 期望 {len(expected)}，实际 {len(rows)}")
    actual_ids = {row.get("id") for row in rows}
    if actual_ids != set(expected):
        raise AssertionError(
            f"读回 ID 不符: 缺少 {sorted(set(expected) - actual_ids)}，"
            f"多出 {sorted(actual_ids - set(expected))}"
        )
    for row in rows:
        chunk = expected[row["id"]]
        if row.get("text") != chunk.page_content:
            raise AssertionError(f"读回文本不符: id={row['id']}")
        if row.get("source") != chunk.metadata["source"]:
            raise AssertionError(f"读回来源不符: id={row['id']}")
        if row.get("chunk_id") != chunk.metadata["chunk_id"]:
            raise AssertionError(f"读回切片号不符: id={row['id']}")


# =========================
# 步骤 5：完整入库
# =========================


def run_ingest() -> int:
    client = None
    try:
        config.validate_config(require_embedding=True)
        client, summary = vector_store.prepare_ingest_target()
        print("=== 入库目标边界 ===")
        print_summary(summary)

        source_documents, chunks, manifest = documents.load_and_split()
        print("\n=== 输入与切分 ===")
        print(f"输入文件: {config.KNOWLEDGE_FILE}")
        print(
            f"切分策略: {manifest['strategy']}；"
            f"chunk_size={config.CHUNK_SIZE}, chunk_overlap={config.CHUNK_OVERLAP}"
        )
        print(
            f"加载 Document: {len(source_documents)}；"
            f"子块: {len(chunks)}；父块: {manifest['parent_count']}"
        )
        if not chunks:
            raise RuntimeError("切分结果为空，停止入库")
        texts = [chunk.page_content for chunk in chunks]

        print("\n=== 文档 Embedding ===")
        print(
            f"服务商: {config.EMBED_PROVIDER} / {config.EMBED_BASE_URL}；"
            f"模型: {config.EMBED_MODEL_NAME}；期望维度: {config.EMBED_DIM}"
        )
        embedder = models.init_embedding_model()
        vectors = models.embed_documents(embedder, texts)
        print(f"文档接口返回 {len(vectors)} 条向量，维度 {len(vectors[0])}")

        records = build_records(chunks, vectors)
        print(
            f"记录校验通过: {len(records)} 条；"
            f"text 最大 {max(len(r['text'].encode('utf-8')) for r in records)} 字节；"
            f"source 最大 {max(len(r['source'].encode('utf-8')) for r in records)} 字节"
        )

        print("\n=== 写入 Milvus ===")
        result = vector_store.insert_records(client, records)
        print(f"写入响应: {result}")
        vector_store.flush(client)

        rows = vector_store.fetch_records_by_ids(client, [r["id"] for r in records])
        verify_readback(chunks, rows)
        print(f"读回核对通过: {len(rows)} 条记录的 id/text/source/chunk_id 一致")

        k = min(3, len(records))
        hits = vector_store.search_raw(client, vectors[0], k)
        top = hits[0][0] if hits and hits[0] else None
        if not top:
            raise vector_store.VectorStoreError("用文档向量搜索没有返回结果")
        top_id = top.get("id")
        print(
            f"文档向量搜索抽查: top1 id={top_id}, score={top.get('distance')}, "
            f"期望包含 id={records[0]['id']}"
        )
        returned_ids = {hit.get("id") for hit in hits[0]}
        if records[0]["id"] not in returned_ids:
            raise vector_store.VectorStoreError(
                f"文档向量未召回自身切片: 期望 id={records[0]['id']}，实际 {returned_ids}"
            )

        total = vector_store.count_records(client)
        if total != len(records):
            raise vector_store.VectorStoreError(
                f"集合实际记录数 {total} 与切片数 {len(records)} 不符"
            )

        manifest["created_at"] = datetime.now().isoformat(timespec="seconds")
        try:
            manifest_path = documents.save_index_manifest(manifest)
        except Exception as exc:
            print(f"\n入库数据已写入，但索引清单写入失败: {type(exc).__name__}: {exc}")
            print(
                "查询入口会因清单缺失或与集合不匹配而拒绝父块补全；"
                "请运行 python ingest.py --write-manifest 重建清单（不写 Milvus 数据）。"
                "若集合未加载，先运行 python ingest.py --load。"
            )
            return 1
        print(f"索引清单已写入: {manifest_path}")

        print(f"\n入库完成: {total} 条记录，集合 {config.DB_NAME}.{config.COLLECTION_NAME}")
        print(
            f"父块补全可用: {manifest['parent_count']} 个父块；"
            "查询 Embedding 链路与真实问题验收请运行 python main.py '问题'"
        )
        return 0
    except vector_store.CollectionNotEmptyError:
        raise
    except Exception as exc:
        print(f"\n入库失败: {type(exc).__name__}: {exc}")
        print(
            "若集合已有部分写入，本次实验集合需按显式维护流程处理；"
            "问答入口不会自动删除或重建集合。"
        )
        return 1
    finally:
        if client is not None:
            client.close()


def write_manifest_only() -> int:
    """维护：用当前源文件重建父块清单，核对集合计数后原子写入；不写 Milvus。"""
    client = None
    try:
        config.validate_config(require_embedding=True)
        _, _, manifest = documents.load_and_split()
        client = vector_store.open_query_client()
        vector_store.validate_query_target(client)
        total = vector_store.count_records(client)
        if total != manifest["chunk_count"]:
            print(
                f"重建清单停止: 集合记录数 {total} 与当前切分子块数 "
                f"{manifest['chunk_count']} 不符；请核对切分配置或重新入库。"
            )
            return 1
        manifest["created_at"] = datetime.now().isoformat(timespec="seconds")
        path = documents.save_index_manifest(manifest)
        print(
            f"索引清单已重建: {path}；子块 {manifest['chunk_count']}，"
            f"父块 {manifest['parent_count']}"
        )
        return 0
    except vector_store.VectorStoreError as exc:
        print(f"重建索引清单失败: {exc}")
        return 1
    except Exception as exc:
        print(f"重建索引清单失败（未预期错误）: {type(exc).__name__}: {exc}")
        return 1
    finally:
        if client is not None:
            client.close()


# =========================
# 步骤 3：真实读写验证（独立的临时记录，结束后删除）
# =========================


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Milvus 集合边界与知识入库")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--inspect",
        action="store_true",
        help="只读检查，不创建 database/collection，也不加载集合",
    )
    group.add_argument(
        "--load",
        action="store_true",
        help="维护：加载已存在的集合，不写入数据",
    )
    group.add_argument(
        "--stamp",
        action="store_true",
        help="维护：为已有集合写入当前 Embedding 模型标识（不改数据）",
    )
    group.add_argument(
        "--write-manifest",
        action="store_true",
        help="维护：核对集合计数后用当前源文件重建索引清单（不写 Milvus 数据）",
    )
    group.add_argument(
        "--verify",
        action="store_true",
        help="在空集合上执行真实写入/读取/搜索/删除验证",
    )
    args = parser.parse_args(argv)

    try:
        if args.inspect:
            print("=== 只读检查 ===")
            print_summary(vector_store.inspect_target())
            return 0

        if args.load:
            print("=== 加载已有集合 ===")
            print_summary(vector_store.load_existing())
            return 0

        if args.stamp:
            print("=== 写入集合 Embedding 标识 ===")
            print_summary(vector_store.stamp_collection_identity())
            return 0

        if args.write_manifest:
            print("=== 重建索引清单 ===")
            return write_manifest_only()

        if args.verify:
            client, summary = vector_store.prepare_ingest_target()
            try:
                print("=== 入库目标边界 ===")
                print_summary(summary)
                print("\n=== 真实读写验证 ===")
                verify_roundtrip(client)
            finally:
                client.close()
            return 0

        return run_ingest()
    except vector_store.CollectionNotEmptyError as exc:
        print(f"\n入库停止: {exc}")
        print_summary(exc.details)
        return 2
    except vector_store.VectorStoreError as exc:
        print(f"\nMilvus 检查失败: {exc}")
        return 1
    except Exception as exc:
        print(f"\nMilvus 检查失败（未预期错误）: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
