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
from rag import answering, conversation, documents, retrieval, tools, vector_store


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
                f"| location={hit.get('location', '')} "
                f"| matched_chunk_id="
                f"{hit.get('matched_chunk_id', hit['chunk_id'])} "
                f"| matched_chunk_key="
                f"{hit.get('matched_chunk_key', hit.get('chunk_key', ''))} "
                f"| context_len={hit.get('context_length', len(hit['text']))}"
            )
        else:
            meta = (
                f"source={hit['source']} | chunk_id={hit['chunk_id']} "
                f"| chunk_key={hit.get('chunk_key', '')} "
                f"| location={hit.get('location', '')} | id={hit['id']}"
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
                f"location={source.get('location', '')}；"
                f"parent_id={source['parent_id']}；"
                f"matched_chunk_id="
                f"{source.get('matched_chunk_id', source['chunk_id'])}；"
                f"matched_chunk_key="
                f"{source.get('matched_chunk_key', source.get('chunk_key', ''))}"
            )
        else:
            lines.append(
                f"[片段 {source['citation']}] {source['source']}"
                f"#chunk_id={source['chunk_id']}"
                f" (chunk_key={source.get('chunk_key', '')}; "
                f"id={source['id']}; location={source.get('location', '')})"
            )
    return "\n".join(lines)


def print_scope(strategy: str | None, rerank_strategy: str | None) -> None:
    collection = documents.active_collection()
    summary = vector_store.inspect_target(collection)
    print("=== 本次知识范围 ===")
    print(
        f"数据库/集合: {summary['database']}.{summary['collection']}；"
        f"schema_version: {summary.get('schema_version', 1)}；"
        f"加载状态: {summary['load_state']}；"
        f"记录数(统计可能滞后): {summary['row_count']}"
    )
    print(
        f"检索策略: {strategy or config.RETRIEVAL_STRATEGY}；"
        f"重排: {rerank_strategy or config.RERANK_STRATEGY}"
    )


def print_result(result: dict[str, Any]) -> None:
    print("\n=== 答案 ===")
    print(result["answer"])
    print("\n=== 本次依据 ===")
    print(format_sources(result["sources"]))
    if result.get("resolved_query") and result["resolved_query"] != result.get(
        "question", ""
    ):
        print(f"结合历史理解的检索问题: {result['resolved_query']}")
    if result["citations"]:
        print(f"有效引用: {result['citations']}")
    if result["invalid_citations"]:
        print(
            f"提示: 回答引用了不存在的片段编号 {result['invalid_citations']}，"
            "请结合上方原文人工核对。"
        )
    if result["insufficient"]:
        print("（未调用对话模型：检索没有返回可用片段）")


def run_question(
    question: str,
    k: int,
    *,
    strategy: str | None = None,
    rerank_strategy: str | None = None,
    session_id: str | None = None,
    mode: str = "fixed",
) -> int:
    config.validate_config(require_embedding=True)
    print_scope(strategy, rerank_strategy)

    if mode == "tool":
        store = conversation.ConversationStore()
        result = tools.answer_tool_turn(
            question,
            store=store,
            session_id=session_id or "cli-tool",
            k=k,
            strategy=strategy,
            rerank_strategy=rerank_strategy,
        )
        print("\n=== 工具检索片段 ===")
        print(format_evidence(result["hits"]))
    elif session_id:
        store = conversation.ConversationStore()
        result = conversation.answer_turn(
            question,
            store=store,
            session_id=session_id,
            k=k,
            strategy=strategy,
            rerank_strategy=rerank_strategy,
        )
        print("\n=== 召回片段 ===")
        print(format_evidence(result["hits"]))
    else:
        hits = retrieval.retrieve(
            question,
            k,
            strategy=strategy,
            rerank_strategy=rerank_strategy,
        )
        print("\n=== 召回片段 ===")
        print(format_evidence(hits))
        result = answering.generate_answer(question, hits=hits)
    print_result(result)
    return 0


def run_stream(
    question: str,
    k: int,
    *,
    strategy: str | None = None,
    rerank_strategy: str | None = None,
    session_id: str = "default",
    mode: str = "fixed",
) -> int:
    config.validate_config(require_embedding=True)
    store = conversation.ConversationStore()
    stream_fn = tools.stream_tool_turn if mode == "tool" else conversation.stream_turn
    print("=== 流式答案 ===")
    for event in stream_fn(
        question,
        store=store,
        session_id=session_id,
        k=k,
        strategy=strategy,
        rerank_strategy=rerank_strategy,
    ):
        kind = event.get("event")
        if kind == "delta":
            print(event["text"], end="", flush=True)
        elif kind == "sources":
            print("\n\n=== 本次依据 ===")
            print(format_sources(event["sources"]))
        elif kind == "done":
            print()
            if event["citations"]:
                print(f"有效引用: {event['citations']}")
            if event["invalid_citations"]:
                print(
                    f"提示: 回答引用了不存在的片段编号 "
                    f"{event['invalid_citations']}，请人工核对。"
                )
            if event.get("resolved_query") and event["resolved_query"] != question:
                print(f"结合历史理解的检索问题: {event['resolved_query']}")
            return 0
        elif kind == "error":
            print(f"\n流式回答失败: {event['message']}")
            return 5
        elif kind == "start":
            if event.get("resolved_query") and event["resolved_query"] != question:
                print(f"结合历史理解的检索问题: {event['resolved_query']}")
    print("\n流式回答失败: 未收到完成事件")
    return 5


def run_chat(
    k: int,
    *,
    strategy: str | None = None,
    rerank_strategy: str | None = None,
    session_id: str = "default",
    stream: bool = False,
    mode: str = "fixed",
) -> int:
    config.validate_config(require_embedding=True)
    print_scope(strategy, rerank_strategy)
    store = conversation.ConversationStore()
    stream_fn = tools.stream_tool_turn if mode == "tool" else conversation.stream_turn
    turn_fn = tools.answer_tool_turn if mode == "tool" else conversation.answer_turn
    print(f"\n多轮会话: {session_id}；输入空行退出。")
    while True:
        try:
            question = input("请输入问题: ").strip()
        except EOFError:
            break
        if not question:
            break
        try:
            if stream:
                print("=== 流式答案 ===")
                for event in stream_fn(
                    question,
                    store=store,
                    session_id=session_id,
                    k=k,
                    strategy=strategy,
                    rerank_strategy=rerank_strategy,
                ):
                    kind = event.get("event")
                    if kind == "delta":
                        print(event["text"], end="", flush=True)
                    elif kind == "sources":
                        print("\n\n=== 本次依据 ===")
                        print(format_sources(event["sources"]))
                    elif kind == "done":
                        print()
                        if event.get("citations"):
                            print(f"有效引用: {event['citations']}")
                        if event.get("invalid_citations"):
                            print(
                                f"提示: 回答引用了不存在的片段编号 "
                                f"{event['invalid_citations']}，请人工核对。"
                            )
                    elif kind == "error":
                        print(f"\n流式回答失败: {event['message']}")
            else:
                result = turn_fn(
                    question,
                    store=store,
                    session_id=session_id,
                    k=k,
                    strategy=strategy,
                    rerank_strategy=rerank_strategy,
                )
                print_result(result)
        except (conversation.ConversationError, answering.AnswerGenerationError) as exc:
            print(f"回答失败: {exc}")
        except (retrieval.RetrievalError, vector_store.VectorStoreError) as exc:
            print(f"知识库或检索失败: {exc}")
        print()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="atguigu 助手单次知识问答")
    parser.add_argument("question", nargs="*", help="当前问题（建议用引号包裹）")
    parser.add_argument("--k", type=int, default=config.TOP_K, help="召回片段数")
    parser.add_argument(
        "--strategy",
        choices=("vector", "hybrid"),
        default=config.RETRIEVAL_STRATEGY,
        help="检索策略",
    )
    parser.add_argument(
        "--rerank",
        choices=("none", "llm"),
        default=config.RERANK_STRATEGY,
        help="重排策略；llm 会额外调用一次对话模型",
    )
    parser.add_argument(
        "--mode",
        choices=("fixed", "tool"),
        default="fixed",
        help="fixed=先检索再回答；tool=Agent 自主调用检索工具",
    )
    parser.add_argument("--chat", action="store_true", help="进入多轮对话")
    parser.add_argument("--stream", action="store_true", help="流式输出")
    parser.add_argument("--session", default="default", help="会话 ID")
    args = parser.parse_args(argv)

    if args.chat:
        try:
            return run_chat(
                args.k,
                strategy=args.strategy,
                rerank_strategy=args.rerank,
                session_id=args.session,
                stream=args.stream,
                mode=args.mode,
            )
        except KeyboardInterrupt:
            print("\n已取消")
            return 130
        except Exception as exc:  # noqa: BLE001
            print(f"服务异常: {type(exc).__name__}: {exc}")
            return 7

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
        if args.stream:
            return run_stream(
                question,
                args.k,
                strategy=args.strategy,
                rerank_strategy=args.rerank,
                session_id=args.session,
                mode=args.mode,
            )
        return run_question(
            question,
            args.k,
            strategy=args.strategy,
            rerank_strategy=args.rerank,
            session_id=args.session if args.session != "default" else None,
            mode=args.mode,
        )
    except ValueError as exc:
        print(f"参数错误: {exc}")
        return 2
    except vector_store.CollectionNotFoundError as exc:
        print(f"知识库未就绪: {exc}")
        return 3
    except vector_store.CollectionNotLoadedError as exc:
        print(f"知识库未就绪: {exc}")
        return 3
    except documents.ManifestError as exc:
        print(f"索引清单不可用: {exc}")
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
