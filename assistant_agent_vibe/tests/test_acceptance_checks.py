"""步骤 9：验收脚本的失败判定（离线，不调用模型）。"""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "live_acceptance.py"
_SPEC = importlib.util.spec_from_file_location("live_acceptance_under_test", SCRIPT)
acceptance = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(acceptance)


def _result(case_id, *, missing=None, answer_text=None, invalid=None, error=None):
    result = {
        "case": {"id": case_id, "question": "问题", "facts": []},
        "error": error,
        "hits": [],
        "covered": [],
        "missing": missing or [],
    }
    if answer_text is not None:
        result["answer"] = {
            "answer": answer_text,
            "invalid_citations": invalid or [],
            "citations": [],
        }
    return result


def test_expectation_ok_cases():
    assert acceptance.expectation_ok("Q11", "材料不足，资料中没有提到该部署方式。")
    assert not acceptance.expectation_ok("Q11", "支持，默认包含火星基地专线部署。")
    assert acceptance.expectation_ok("Q12", "无法判断，请提供套餐、支付时间和使用量。")
    assert not acceptance.expectation_ok("Q12", "可以退 100 元。")


def test_case_failed_objective_cases():
    assert acceptance.case_failed(_result("Q03", missing=["10%"]), strict=False)
    assert acceptance.case_failed(
        _result("Q01", answer_text="答案 [片段 9]", invalid=[9]), strict=False
    )
    assert acceptance.case_failed(_result("Q01", error="boom"), strict=False)
    assert not acceptance.case_failed(
        _result("Q01", answer_text="10 人 [片段 1]"), strict=False
    )


def test_strict_mode_checks_q11_behavior():
    assert acceptance.case_failed(
        _result("Q11", answer_text="支持，默认包含。"), strict=True
    )
    assert not acceptance.case_failed(
        _result("Q11", answer_text="材料不足，无法确认。"), strict=True
    )
