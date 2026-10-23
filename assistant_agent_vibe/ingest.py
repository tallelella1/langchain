"""版本化知识入库入口（问答入口不会触发入库）。

默认行为：读取来源清单 → 计算内容/配置摘要 → 未变化则复用 → 在新集合完成
解析、切分、向量化、写入、读回核对与固定问题抽查 → 原子切换索引清单。

用法：
    python ingest.py                       # 构建/更新当前版本（未变化则复用）
    python ingest.py --force               # 强制构建新版本
    python ingest.py --add-source doc.md   # 加入来源并重建
    python ingest.py --remove-source doc.md
    python ingest.py --source a.md --source b.pdf   # 临时来源，不改清单
    python ingest.py --rollback            # 切回上一版本
    python ingest.py --prune --keep 1 --yes
    python ingest.py --inspect             # 只读检查当前集合
"""

from __future__ import annotations

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
from rag import documents, indexer, sources, vector_store

VERIFY_ID = 2**63 - 1  # 旧版 --verify 使用的临时记录 ID
WRONG_DIM_ID = VERIFY_ID - 1


def print_summary(summary: dict) -> None:
    print(f"Milvus URI: {summary['uri']}")
    print(f"数据库: {summary['database']}；现有数据库: {summary['databases']}")
    print(f"集合: {summary['collection']}；schema_version: {summary.get('schema_version', 1)}")
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
    print(f"集合属性: {summary.get('properties', {})}")
    print(f"行数(get_collection_stats，可能滞后): {summary['row_count']}")


# =========================
# 旧版 v1 记录工具（保留离线测试与兼容）
# =========================


def build_records(
    chunks: Sequence[Any], vectors: Sequence[Sequence[float]]
) -> list[dict[str, Any]]:
    """把切片与向量组装成旧版 v1 记录；仅用于兼容测试。"""
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
        vector_store.validate_record(record, schema_version=1)
        records.append(record)
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("切片 id 重复，拒绝写入")
    return records


def verify_readback(chunks: Sequence[Any], rows: Sequence[dict[str, Any]]) -> None:
    """旧版读回核对：数量、ID、正文、来源、切片号。"""
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
# 版本化构建
# =========================


def active_collection() -> str:
    try:
        return documents.active_collection()
    except Exception:
        return config.COLLECTION_NAME


def run_build(
    *,
    force: bool = False,
    entries: Sequence[dict[str, str]] | None = None,
    verify: bool = True,
) -> int:
    try:
        config.validate_config(require_embedding=True)
        result = indexer.build_index(entries=entries, force=force, verify=verify)
        if result.reused:
            print(
                f"索引未变化，复用当前集合: {result.collection}；"
                f"记录数 {result.record_count}。需要重建请加 --force。"
            )
            return 0
        print("=== 版本化索引构建完成 ===")
        print(f"当前集合: {result.collection}")
        print(f"上一集合: {result.previous_collection or '无'}")
        print(f"记录数: {result.record_count}")
        print(f"文档数: {len(result.manifest['documents'])}")
        print(f"index_version: {result.manifest['index_version']}")
        print(
            f"来源: {', '.join(result.manifest['sources'])}；"
            f"schema_version={result.manifest['schema_version']}"
        )
        print(f"索引清单: {config.INDEX_MANIFEST_FILE}")
        return 0
    except indexer.BuildError as exc:
        print(f"\n入库失败: {exc}")
        print("当前索引清单未被替换；旧版本仍可用。")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"\n入库失败: {type(exc).__name__}: {exc}")
        print("当前索引清单未被替换；请检查来源、Milvus 与模型配置。")
        return 1


def run_ingest() -> int:
    """默认入口：构建/更新当前版本。"""
    return run_build()


def write_manifest_only() -> int:
    """已停用：v2 清单必须随新集合一起原子切换，不能只改清单。"""
    print(
        "重建清单已停用：版本化索引要求清单与集合一起切换。"
        "请运行 python ingest.py 构建新版本；未变化时会自动复用。"
    )
    return 1


# =========================
# 旧版空集合真实读写验证（保留维护能力）
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

        try:
            vector_store.prepare_ingest_target()
        except vector_store.CollectionNotEmptyError as exc:
            print(f"非空保护: pass ({exc})")
        else:
            raise AssertionError("非空保护未生效")

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
            except Exception as exc:  # noqa: BLE001
                message = str(exc).replace("\n", " ")[:300]
                print(f"服务端维度冲突: pass ({type(exc).__name__}: {message})")
            else:
                raise AssertionError("服务端接受了错误维度的向量")
        finally:
            logging.disable(previous_disable)
    finally:
        try:
            client.delete(collection_name=collection, ids=[VERIFY_ID, WRONG_DIM_ID])
            print("验证记录清理: pass")
        except Exception as exc:  # noqa: BLE001
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
    parser = argparse.ArgumentParser(description="版本化知识入库与集合维护")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--inspect", action="store_true", help="只读检查当前集合")
    group.add_argument("--load", action="store_true", help="维护：加载当前集合")
    group.add_argument("--stamp", action="store_true", help="维护：写入当前 Embedding 标识")
    group.add_argument("--write-manifest", action="store_true", help="已停用（兼容参数）")
    group.add_argument("--verify", action="store_true", help="旧版空集合读写验证")
    group.add_argument("--rollback", action="store_true", help="切回上一版本索引")
    group.add_argument("--prune", action="store_true", help="清理归档旧集合")
    parser.add_argument("--force", action="store_true", help="内容未变化也构建新版本")
    parser.add_argument("--source", action="append", default=None, help="临时来源（可重复）")
    parser.add_argument("--add-source", action="append", default=None, help="加入来源清单并构建")
    parser.add_argument("--remove-source", action="append", default=None, help="移除来源并构建")
    parser.add_argument("--no-verify", action="store_true", help="跳过固定问题抽查（仅调试）")
    parser.add_argument("--keep", type=int, default=1, help="--prune 保留的旧版本数")
    parser.add_argument("--yes", action="store_true", help="确认执行 --prune 删除")
    args = parser.parse_args(argv)

    try:
        if args.inspect:
            print("=== 只读检查 ===")
            print_summary(vector_store.inspect_target(active_collection()))
            return 0

        if args.load:
            print("=== 加载当前集合 ===")
            print_summary(vector_store.load_existing(active_collection()))
            return 0

        if args.stamp:
            print("=== 写入集合 Embedding 标识 ===")
            print_summary(vector_store.stamp_collection_identity(active_collection()))
            return 0

        if args.write_manifest:
            return write_manifest_only()

        if args.verify:
            client, summary = vector_store.prepare_ingest_target()
            try:
                print("=== 旧版空集合读写验证 ===")
                print_summary(summary)
                verify_roundtrip(client)
            finally:
                client.close()
            return 0

        if args.rollback:
            manifest = indexer.rollback()
            print(
                f"已回滚到: {manifest['collection']}；"
                f"来源标记 stale_sources={manifest.get('stale_sources', False)}"
            )
            return 0

        if args.prune:
            deleted = indexer.prune_old_collections(
                keep=args.keep, yes=args.yes
            )
            print(f"已删除旧集合: {deleted or '无'}")
            return 0

        for source in args.add_source or []:
            sources.add_source(source)
            print(f"已加入来源清单: {source}")
        for source in args.remove_source or []:
            sources.remove_source(source)
            print(f"已移除来源清单: {source}")

        entries = None
        if args.source:
            entries = sources.entries_from_paths(args.source)
        if args.force or entries is not None or args.no_verify:
            return run_build(
                force=args.force, entries=entries, verify=not args.no_verify
            )
        return run_ingest()
    except (sources.SourceError, indexer.BuildError) as exc:
        print(f"\n维护操作失败: {exc}")
        return 1
    except vector_store.VectorStoreError as exc:
        print(f"\nMilvus 检查失败: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"\n维护操作失败（未预期错误）: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
