"""P1.4：评估指标与拒答判定（离线）。"""

from evals import run_eval

Q11 = {"id": "Q11", "behavior": "insufficient"}
Q12 = {"id": "Q12", "behavior": "missing_order_info"}


def test_behavior_ok_handles_negated_assertions():
    assert run_eval.behavior_ok(
        Q11,
        "材料不足。知识库没有提到火星基地专线部署，"
        "企业版可以选配私有化部署，但不是所有合同默认包含，需要单独签约。",
    )
    assert not run_eval.behavior_ok(Q11, "支持，默认包含火星基地专线部署。")


def test_behavior_ok_requires_order_information_and_no_amount():
    good = (
        "材料不足，无法判断。请提供套餐版本、支付成功时间、"
        "AI 问答使用量占额度比例、是否升级、是否已开专票。"
    )
    assert run_eval.behavior_ok(Q12, good)
    assert not run_eval.behavior_ok(Q12, "可以退 100 元。")
    assert not run_eval.behavior_ok(Q12, "材料不足。请提供套餐。")


def test_coverage_and_support_coverage():
    covered, missing = run_eval.coverage("10% 和 50%", ["10%", "50%", "红字发票"])
    assert covered == ["10%", "50%"]
    assert missing == ["红字发票"]

    hits = [
        {"chapter": "六、退款规则", "section": "2. 基础版与专业版"},
        {"chapter": "七、发票规则", "section": "4. 专票补充规则"},
    ]
    covered, missing = run_eval.support_coverage(
        hits,
        ["六、退款规则 / 2. 基础版与专业版", "八、企业版专属支持规则 / 2. 私有化部署"],
    )
    assert covered == ["六、退款规则 / 2. 基础版与专业版"]
    assert missing == ["八、企业版专属支持规则 / 2. 私有化部署"]


def test_compare_results_reports_deltas():
    left = {
        "strategy": "vector",
        "k": 5,
        "timestamp": "a",
        "summary": {"fact_recall": 0.8, "avg_retrieval_ms": 100.0, "model_calls": 0},
    }
    right = {
        "strategy": "hybrid",
        "k": 5,
        "timestamp": "b",
        "summary": {"fact_recall": 1.0, "avg_retrieval_ms": 120.0, "model_calls": 1},
    }
    report = run_eval.compare_results(left, right)
    assert "fact_recall" in report
    assert "+0.2000" in report
    assert "+20.0000" in report
