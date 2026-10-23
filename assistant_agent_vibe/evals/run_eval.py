"""固定评估集：检索与答案指标、参数对比（P1.4）。

用法：
    python -m evals.run_eval --strategy vector --k 5
    python -m evals.run_eval --strategy hybrid --k 5 --answer
    python -m evals.run_eval --compare evals/results/a.json evals/results/b.json

默认只做检索评估：调用一次查询 Embedding 和 Milvus，不调用对话模型。
``--answer`` 会为每道题再调用一次对话模型，可能产生费用。
结果写入 ``evals/results/<时间>-<策略>.json`` 与同名 Markdown。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config  # noqa: E402
from rag import answering, documents, prompts, retrieval, tools  # noqa: E402

CASES_FILE = Path(__file__).resolve().parent / "cases.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

ORDER_INFO_TERMS = ("套餐", "版本", "支付", "付款", "使用量", "额度", "升级", "开票", "专票")
INSUFFICIENT_TERMS = ("材料不足", "无法确认", "无法判断", "没有提到", "未提及", "不知道", "缺少")
SUPPORT_ASSERTIONS = ("默认包含", "一定支持", "肯定支持", "永久免费")


def load_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    path = Path(path or CASES_FILE)
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"评估集缺少 cases: {path}")
    seen: set[str] = set()
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("评估用例缺少 id")
        if case_id in seen:
            raise ValueError(f"评估用例 id 重复: {case_id}")
        seen.add(case_id)
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"{case_id} 缺少 question")
        facts = case.get("expected_facts", [])
        if not isinstance(facts, list) or any(
            not isinstance(fact, str) or not fact for fact in facts
        ):
            raise ValueError(f"{case_id} 的 expected_facts 必须是非空字符串列表")
    return cases


def select_cases(cases: Iterable[dict[str, Any]], ids: str | None) -> list[dict[str, Any]]:
    cases = list(cases)
    if not ids:
        return cases
    wanted = [value.strip() for value in ids.split(",") if value.strip()]
    known = {case["id"] for case in cases}
    unknown = [value for value in wanted if value not in known]
    if unknown:
        raise ValueError(f"未知用例: {', '.join(unknown)}")
    return [case for case in cases if case["id"] in wanted]


def coverage(text: str, facts: Sequence[str]) -> tuple[list[str], list[str]]:
    covered = [fact for fact in facts if fact in text]
    missing = [fact for fact in facts if fact not in text]
    return covered, missing


def support_coverage(
    hits: Sequence[dict[str, Any]], sections: Sequence[str]
) -> tuple[list[str], list[str]]:
    """按父块 section 的组合判断支持段落是否被召回。"""
    if not sections:
        return [], []
    recalled = {
        f"{hit.get('chapter', '')} / {hit.get('section', '')}".strip()
        for hit in hits
        if hit.get("section")
    }
    covered = [section for section in sections if section in recalled]
    missing = [section for section in sections if section not in recalled]
    return covered, missing


def behavior_ok(case: dict[str, Any], answer: str) -> bool | None:
    """只对声明了 behavior 的拒答类问题检查；其他问题返回 None。"""
    behavior = case.get("behavior")
    if behavior not in {"insufficient", "missing_order_info"}:
        return None
    text = answer.strip()
    if not any(term in text for term in INSUFFICIENT_TERMS):
        return False
    # 断言必须看整句：Q11 的正确回答会写“不是所有企业版合同默认包含”，
    # 不能仅因出现“默认包含”就判失败。
    negations = ("不", "非", "未", "无", "没")
    for sentence in re.split(r"[。；\n]", text):
        if any(term in sentence for term in SUPPORT_ASSERTIONS):
            if not any(neg in sentence for neg in negations):
                return False
    if behavior == "missing_order_info":
        matched = sum(1 for term in ORDER_INFO_TERMS if term in text)
        if matched < 3:
            return False
        if re.search(r"(可以退|退还|退款)\s*\d+(?:\.\d+)?\s*元", text):
            return False
    return True


def evaluate_retrieval(
    cases: Sequence[dict[str, Any]],
    *,
    k: int,
    strategy: str = "vector",
    expand_parents: bool = True,
    answer: bool = False,
    retrieval_options: dict[str, Any] | None = None,
    retriever: Callable[..., list[dict[str, Any]]] = retrieval.retrieve,
    answerer: Callable[..., dict[str, Any]] = answering.generate_answer,
) -> dict[str, Any]:
    if k <= 0:
        raise ValueError("k 必须是正整数")
    options = dict(retrieval_options or {})
    results: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        error = None
        hits: list[dict[str, Any]] = []
        try:
            hits = retriever(
                case["question"],
                k,
                strategy=strategy,
                expand_parents=expand_parents,
                **options,
            )
        except TypeError as exc:
            if "strategy" not in str(exc):
                raise
            hits = retriever(
                case["question"], k, expand_parents=expand_parents, **options
            )
        except Exception as exc:  # noqa: BLE001 - 评估需要保留错误证据
            error = f"{type(exc).__name__}: {exc}"
        retrieval_ms = (time.perf_counter() - started) * 1000

        text = "\n".join(hit.get("text", "") for hit in hits)
        covered, missing = coverage(text, case.get("expected_facts", []))
        support_covered, support_missing = support_coverage(
            hits, case.get("support_sections", [])
        )
        formatted = prompts.format_context(hits) if hits else ""
        result: dict[str, Any] = {
            "id": case["id"],
            "question": case["question"],
            "answerable": bool(case.get("answerable", True)),
            "behavior": case.get("behavior", "answer"),
            "error": error,
            "hits": [
                {
                    "id": hit.get("id"),
                    "chunk_id": hit.get("chunk_id"),
                    "chunk_key": hit.get("chunk_key"),
                    "parent_id": hit.get("parent_id"),
                    "source": hit.get("source"),
                    "chapter": hit.get("chapter"),
                    "section": hit.get("section"),
                    "score": hit.get("score"),
                    "metric": hit.get("metric"),
                    "semantic_score": hit.get("semantic_score"),
                    "bm25_score": hit.get("bm25_score"),
                    "rrf_score": hit.get("rrf_score"),
                    "rerank_score": hit.get("rerank_score"),
                }
                for hit in hits
            ],
            "fact_coverage": len(covered),
            "fact_total": len(case.get("expected_facts", [])),
            "missing_facts": missing,
            "support_coverage": len(support_covered),
            "support_total": len(case.get("support_sections", [])),
            "missing_supports": support_missing,
            "context_chars": len(text),
            "formatted_context_chars": len(formatted),
            "retrieval_ms": round(retrieval_ms, 2),
            "model_calls": 1 if options.get("rerank_strategy") == "llm" else 0,
        }

        if answer and not error:
            answer_started = time.perf_counter()
            try:
                answer_result = answerer(case["question"], k=k, hits=hits)
                result["answer"] = answer_result["answer"]
                result["citations"] = answer_result.get("citations", [])
                result["invalid_citations"] = answer_result.get(
                    "invalid_citations", []
                )
                result["insufficient"] = answer_result.get("insufficient", False)
                result["answer_fact_coverage"] = len(
                    [f for f in case.get("expected_facts", []) if f in answer_result["answer"]]
                )
                result["behavior_ok"] = behavior_ok(case, answer_result["answer"])
                result["model_calls"] += answer_result.get("model_calls", 1)
            except Exception as exc:  # noqa: BLE001
                result["answer_error"] = f"{type(exc).__name__}: {exc}"
                result["behavior_ok"] = False
            result["answer_ms"] = round((time.perf_counter() - answer_started) * 1000, 2)
        results.append(result)
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "strategy": strategy,
        "k": k,
        "expand_parents": expand_parents,
        "answer_mode": answer,
        "retrieval_options": options,
        "embedding_model": config.EMBED_MODEL_NAME,
        "llm_model": config.LLM_MODEL_NAME if answer else None,
        "cases_file": str(CASES_FILE.relative_to(PROJECT_DIR).as_posix()),
        "summary": summarize(results),
        "results": results,
    }


def evaluate_tool_agent(
    cases: Sequence[dict[str, Any]],
    *,
    k: int,
    strategy: str = "vector",
    rerank_strategy: str = "none",
    answerer: Callable[..., dict[str, Any]] = tools.generate_tool_answer,
) -> dict[str, Any]:
    """评估检索工具型 Agent：检索由 Agent 自主发起，记录调用次数与延迟。"""
    if k <= 0:
        raise ValueError("k 必须是正整数")
    results: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        error = None
        answer_result: dict[str, Any] = {}
        try:
            answer_result = answerer(
                case["question"],
                k=k,
                strategy=strategy,
                rerank_strategy=rerank_strategy,
            )
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        elapsed = (time.perf_counter() - started) * 1000
        hits = list(answer_result.get("hits") or [])
        text = "\n".join(hit.get("text", "") for hit in hits)
        covered, missing = coverage(text, case.get("expected_facts", []))
        support_covered, support_missing = support_coverage(
            hits, case.get("support_sections", [])
        )
        row: dict[str, Any] = {
            "id": case["id"],
            "question": case["question"],
            "answerable": bool(case.get("answerable", True)),
            "behavior": case.get("behavior", "answer"),
            "error": error,
            "hits": [
                {
                    "id": hit.get("id"),
                    "chunk_id": hit.get("chunk_id"),
                    "chunk_key": hit.get("chunk_key"),
                    "parent_id": hit.get("parent_id"),
                    "source": hit.get("source"),
                    "chapter": hit.get("chapter"),
                    "section": hit.get("section"),
                    "score": hit.get("score"),
                    "metric": hit.get("metric"),
                }
                for hit in hits
            ],
            "fact_coverage": len(covered),
            "fact_total": len(case.get("expected_facts", [])),
            "missing_facts": missing,
            "support_coverage": len(support_covered),
            "support_total": len(case.get("support_sections", [])),
            "missing_supports": support_missing,
            "context_chars": len(text),
            "formatted_context_chars": len(text),
            "retrieval_ms": round(elapsed, 2),
            "model_calls": answer_result.get("model_calls", 0),
            "tool_calls": answer_result.get("tool_calls", 0),
            "tool_errors": answer_result.get("tool_errors", []),
        }
        if answer_result:
            row["answer"] = answer_result.get("answer", "")
            row["citations"] = answer_result.get("citations", [])
            row["invalid_citations"] = answer_result.get("invalid_citations", [])
            row["insufficient"] = answer_result.get("insufficient", False)
            row["answer_fact_coverage"] = len(
                [f for f in case.get("expected_facts", []) if f in row["answer"]]
            )
            row["behavior_ok"] = behavior_ok(case, row["answer"])
            row["answer_ms"] = round(elapsed, 2)
        else:
            row["behavior_ok"] = False
            row["answer_error"] = error
        results.append(row)
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "agent": "tool",
        "strategy": strategy,
        "k": k,
        "expand_parents": True,
        "answer_mode": True,
        "retrieval_options": {"rerank_strategy": rerank_strategy},
        "embedding_model": config.EMBED_MODEL_NAME,
        "llm_model": config.LLM_MODEL_NAME,
        "cases_file": str(CASES_FILE.relative_to(PROJECT_DIR).as_posix()),
        "summary": summarize(results),
        "results": results,
    }


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in results if row["answerable"]]
    total_facts = sum(row["fact_total"] for row in answerable)
    covered_facts = sum(row["fact_coverage"] for row in answerable)
    total_supports = sum(row["support_total"] for row in answerable)
    covered_supports = sum(row["support_coverage"] for row in answerable)
    retrieval_ms = [row["retrieval_ms"] for row in results]
    summary: dict[str, Any] = {
        "cases": len(results),
        "answerable_cases": len(answerable),
        "errors": sum(1 for row in results if row["error"]),
        "fact_recall": round(covered_facts / total_facts, 4) if total_facts else None,
        "all_facts_cases": sum(
            1
            for row in answerable
            if row["fact_total"] and row["fact_coverage"] == row["fact_total"]
        ),
        "support_recall": (
            round(covered_supports / total_supports, 4) if total_supports else None
        ),
        "avg_context_chars": round(
            sum(row["context_chars"] for row in results) / len(results), 1
        ),
        "avg_formatted_context_chars": round(
            sum(row["formatted_context_chars"] for row in results) / len(results), 1
        ),
        "avg_retrieval_ms": round(sum(retrieval_ms) / len(retrieval_ms), 2)
        if retrieval_ms
        else None,
        "model_calls": sum(row["model_calls"] for row in results),
        "tool_calls": sum(row.get("tool_calls", 0) for row in results),
    }
    if any("answer" in row or "answer_error" in row for row in results):
        answered = [row for row in results if "answer" in row]
        total_answer_facts = sum(row["fact_total"] for row in answered)
        covered_answer_facts = sum(
            row.get("answer_fact_coverage", 0) for row in answered
        )
        summary.update(
            {
                "answer_cases": len(answered),
                "answer_fact_recall": (
                    round(covered_answer_facts / total_answer_facts, 4)
                    if total_answer_facts
                    else None
                ),
                "answer_errors": sum(1 for row in results if "answer_error" in row),
                "behavior_checked": sum(
                    1 for row in results if row.get("behavior_ok") is not None
                ),
                "behavior_passed": sum(
                    1 for row in results if row.get("behavior_ok") is True
                ),
                "avg_answer_ms": round(
                    sum(row.get("answer_ms", 0) for row in answered) / len(answered), 2
                )
                if answered
                else None,
            }
        )
    return summary


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}{suffix}"
    return f"{value}{suffix}"


def markdown_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    rerank = result.get("retrieval_options", {}).get("rerank_strategy", "none")
    agent = result.get("agent", "fixed")
    lines = [
        f"# 固定评估结果：{agent} / {result['strategy']} / rerank={rerank}",
        "",
        f"- 时间: {result['timestamp']}",
        f"- Top-K: {result['k']}；父块补全: {result['expand_parents']}",
        f"- Embedding: {result['embedding_model']}",
        f"- 对话模型: {result['llm_model'] or '未调用'}",
        f"- 检索参数: `{json.dumps(result['retrieval_options'], ensure_ascii=False)}`",
        "",
        "## 汇总",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| 用例数 | {summary['cases']} |",
        f"| 事实召回率 | {_fmt(summary['fact_recall'])} |",
        f"| 全部事实命中用例 | {summary['all_facts_cases']}/{summary['answerable_cases']} |",
        f"| 支持段落召回率 | {_fmt(summary['support_recall'])} |",
        f"| 平均召回原文长度 | {summary['avg_context_chars']} |",
        f"| 平均格式化上下文长度 | {summary['avg_formatted_context_chars']} |",
        f"| 平均检索延迟(ms) | {summary['avg_retrieval_ms']} |",
        f"| 检索错误 | {summary['errors']} |",
        f"| 模型调用次数 | {summary['model_calls']} |",
        f"| 检索工具调用次数 | {summary.get('tool_calls', 0)} |",
    ]
    if "answer_fact_recall" in summary:
        lines.extend(
            [
                f"| 答案事实召回率 | {_fmt(summary['answer_fact_recall'])} |",
                f"| 拒答行为通过 | {summary['behavior_passed']}/{summary['behavior_checked']} |",
                f"| 答案错误 | {summary['answer_errors']} |",
                f"| 平均回答延迟(ms) | {summary['avg_answer_ms']} |",
            ]
        )
    lines.extend(["", "## 明细", "", "| 用例 | 事实 | 支持段落 | 原文长度 | 检索(ms) | 命中父块/切片 |", "| --- | --- | --- | --- | --- | --- |"])
    for row in result["results"]:
        hits = "；".join(
            hit.get("parent_id") or hit.get("chunk_key") or str(hit.get("chunk_id"))
            for hit in row["hits"][:5]
        ) or "-"
        lines.append(
            f"| {row['id']} | {row['fact_coverage']}/{row['fact_total']} | "
            f"{row['support_coverage']}/{row['support_total']} | "
            f"{row['context_chars']} | {row['retrieval_ms']} | {hits} |"
        )
        if row["missing_facts"]:
            lines.append(f"|  | 缺失事实: {'、'.join(row['missing_facts'])} | | | | |")
        if row.get("answer_error"):
            lines.append(f"|  | 答案错误: {row['answer_error']} | | | | |")
        elif row.get("behavior_ok") is False:
            lines.append("|  | 拒答行为未通过 | | | | |")
    return "\n".join(lines) + "\n"


def save_result(result: dict[str, Any], path: str | Path | None = None) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        rerank = result.get("retrieval_options", {}).get("rerank_strategy", "none")
        agent = result.get("agent", "fixed")
        suffix = "-answer" if result["answer_mode"] else ""
        path = RESULTS_DIR / f"{stamp}-{agent}-{result['strategy']}-{rerank}{suffix}.json"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = path.with_suffix(".md")
    md_path.write_text(markdown_report(result), encoding="utf-8")
    return path, md_path


def compare_results(left: dict[str, Any], right: dict[str, Any]) -> str:
    """比较两次评估的汇总指标；数值差 = 右 - 左。"""

    def summary_of(value: dict[str, Any]) -> dict[str, Any]:
        return value.get("summary", value)

    left_summary = summary_of(left)
    right_summary = summary_of(right)
    keys = [
        "fact_recall",
        "all_facts_cases",
        "support_recall",
        "avg_context_chars",
        "avg_formatted_context_chars",
        "avg_retrieval_ms",
        "model_calls",
        "answer_fact_recall",
        "behavior_passed",
        "avg_answer_ms",
    ]
    lines = [
        "# 评估对比",
        "",
        f"- 左: {left.get('strategy', '?')} / k={left.get('k', '?')} / {left.get('timestamp', '?')}",
        f"- 右: {right.get('strategy', '?')} / k={right.get('k', '?')} / {right.get('timestamp', '?')}",
        "",
        "| 指标 | 左 | 右 | 差值(右-左) |",
        "| --- | --- | --- | --- |",
    ]
    for key in keys:
        a = left_summary.get(key)
        b = right_summary.get(key)
        if a is None and b is None:
            continue
        delta = "-"
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta = f"{b - a:+.4f}"
        lines.append(f"| {key} | {_fmt(a)} | {_fmt(b)} | {delta} |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Atguigu Assistant 固定评估集")
    parser.add_argument("--cases", default=str(CASES_FILE), help="评估用例 JSON")
    parser.add_argument("--k", type=int, default=config.TOP_K, help="召回片段数")
    parser.add_argument(
        "--strategy",
        default="vector",
        choices=("vector", "hybrid"),
        help="检索策略",
    )
    parser.add_argument("--ids", help="只运行指定用例，逗号分隔，如 Q05,Q06")
    parser.add_argument("--no-expand-parents", action="store_true", help="不按父块补全")
    parser.add_argument(
        "--rerank",
        default=config.RERANK_STRATEGY,
        choices=("none", "llm"),
        help="重排策略；llm 每道题额外调用一次对话模型",
    )
    parser.add_argument("--answer", action="store_true", help="调用对话模型生成答案")
    parser.add_argument(
        "--agent",
        default="fixed",
        choices=("fixed", "tool"),
        help="fixed=先检索再回答；tool=Agent 自主调用只读检索工具",
    )
    parser.add_argument("--out", help="结果 JSON 路径")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("LEFT", "RIGHT"),
        help="比较两份结果 JSON，不调用服务",
    )
    args = parser.parse_args(argv)

    if args.compare:
        left = json.loads(Path(args.compare[0]).read_text(encoding="utf-8"))
        right = json.loads(Path(args.compare[1]).read_text(encoding="utf-8"))
        print(compare_results(left, right))
        return 0

    cases = select_cases(load_cases(args.cases), args.ids)
    if args.k <= 0:
        print("参数错误: --k 必须是正整数")
        return 2
    try:
        if args.agent == "tool":
            result = evaluate_tool_agent(
                cases,
                k=args.k,
                strategy=args.strategy,
                rerank_strategy=args.rerank,
            )
        else:
            result = evaluate_retrieval(
                cases,
                k=args.k,
                strategy=args.strategy,
                expand_parents=not args.no_expand_parents,
                answer=args.answer,
                retrieval_options={"rerank_strategy": args.rerank},
            )
    except Exception as exc:  # noqa: BLE001
        print(f"评估失败: {type(exc).__name__}: {exc}")
        return 1
    path, md_path = save_result(result, args.out)
    print(markdown_report(result))
    print(f"结果已写入: {path}")
    print(f"报告已写入: {md_path}")
    if result["summary"]["errors"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
