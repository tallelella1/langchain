"""步骤 8：单次终端问答入口。

展示顺序：本次知识范围 → 召回片段（编号/分数/来源/chunk_id/原文）→ 答案与引用。
只读取集合；不触发入库、不创建/加载集合、不删除任何数据。

用法：
    python main.py "基础版最多多少个正式成员？"
    python main.py --k 3 "已开专票，能直接退款吗？"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config
from rag import answering, retrieval, vector_store


def format_evidence(hits: Sequence[dict[str, Any]]) -> str:
    """召回片段的编号、分数、来源、chunk_id 与完整原文。"""
    if not hits:
        return "未召回相关片段。"
    blocks = []
    for index, hit in enumerate(hits, start=1):
        if hit.get("parent_id"):
            meta = (
                f"source={hit['source']} | parent_id={hit['parent_id']} "
                f"| chapter={hit.get('chapter', '')} "
                f"| section={hit.get('section', '')} "
                f"| matched_chunk_id="
                f"{hit.get('matched_chunk_id', hit['chunk_id'])} "
                f"| context_len={hit.get('context_length', len(hit['text']))}"
            )
        else:
            meta = (
                f"source={hit['source']} | chunk_id={hit['chunk_id']} "
                f"| id={hit['id']}"
            )
        blocks.append(
            f"[片段 {index}] score={hit['score']:.4f} | {meta}\n{hit['text']}"
        )
    return "\n\n".join(blocks)


def format_sources(sources: Sequence[dict[str, Any]]) -> str:
    """本次提供给模型的来源：引用编号 → 文件、父块与实际命中切片。"""
    if not sources:
        return "（本次未使用知识片段）"
    lines = []
    for source in sources:
        if source.get("parent_id"):
            lines.append(
                f"[片段 {source['citation']}] {source['source']}；"
                f"{source.get('chapter', '')} / {source.get('section', '')}；"
                f"parent_id={source['parent_id']}；"
                f"matched_chunk_id="
                f"{source.get('matched_chunk_id', source['chunk_id'])}"
            )
        else:
            lines.append(
                f"[片段 {source['citation']}] {source['source']}"
                f"#chunk_id={source['chunk_id']} (id={source['id']})"
            )
    return "\n".join(lines)


def run_question(question: str, k: int) -> int:
    config.validate_config(require_embedding=True)

    summary = vector_store.inspect_target()
    print("=== 本次知识范围 ===")
    print(
        f"数据库/集合: {summary['database']}.{summary['collection']}；"
        f"加载状态: {summary['load_state']}；"
        f"记录数(统计可能滞后): {summary['row_count']}"
    )

    hits = retrieval.retrieve(question, k)
    print("\n=== 召回片段 ===")
    print(format_evidence(hits))

    result = answering.generate_answer(question, hits=hits)
    print("\n=== 答案 ===")
    print(result["answer"])
    print("\n=== 本次依据 ===")
    print(format_sources(result["sources"]))
    if result["citations"]:
        print(f"有效引用: {result['citations']}")
    if result["invalid_citations"]:
        print(
            f"提示: 回答引用了不存在的片段编号 {result['invalid_citations']}，"
            "请结合上方原文人工核对。"
        )
    if result["insufficient"]:
        print("（未调用对话模型：检索没有返回可用片段）")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="atguigu 助手单次知识问答")
    parser.add_argument("question", nargs="*", help="当前问题（建议用引号包裹）")
    parser.add_argument("--k", type=int, default=config.TOP_K, help="召回片段数")
    args = parser.parse_args(argv)

    question = " ".join(args.question).strip()
    if not question:
        try:
            question = input("请输入问题: ").strip()
        except EOFError:
            question = ""
    if not question:
        print("参数错误: 问题不能为空")
        return 2

    try:
        return run_question(question, args.k)
    except ValueError as exc:
        print(f"参数错误: {exc}")
        return 2
    except vector_store.CollectionNotFoundError as exc:
        print(f"知识库未就绪: {exc}")
        return 3
    except vector_store.CollectionNotLoadedError as exc:
        print(f"知识库未就绪: {exc}")
        return 3
    except vector_store.VectorStoreError as exc:
        print(f"知识库检查失败: {exc}")
        return 3
    except retrieval.RetrievalError as exc:
        print(f"检索失败: {exc}")
        return 4
    except answering.AnswerGenerationError as exc:
        print(f"回答生成失败: {exc}")
        return 5
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"配置或服务错误: {exc}")
        return 6
    except Exception as exc:
        print(f"服务异常: {type(exc).__name__}: {exc}")
        return 7
    except KeyboardInterrupt:
        print("\n已取消")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
