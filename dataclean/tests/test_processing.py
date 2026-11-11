import csv
import io
import json

import pytest

from app.config import ROOT
from app.processing import DataValidationError, process_csv


def run_csv(tmp_path, content):
    source = tmp_path / "input.csv"
    source.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
    result = tmp_path / "results"
    result.mkdir()
    return process_csv(source, result), result


def test_example_cleaning_and_exact_statistics(tmp_path):
    summary, result = run_csv(tmp_path, (ROOT / "examples" / "sales.csv").read_bytes())
    assert summary == {"total_rows": 8, "valid_rows": 4, "invalid_rows": 3, "duplicate_rows": 1, "total_quantity": 7, "total_amount": "2078.40"}
    report = json.loads((result / "report.json").read_text(encoding="utf-8"))
    assert report["summary"] == summary
    assert sum(int(row["quantity"]) for row in csv.DictReader((result / "cleaned.csv").open(encoding="utf-8-sig"))) == 7
    with (result / "rejected.csv").open(encoding="utf-8-sig") as handle:
        errors = list(csv.DictReader(handle))
    assert len(errors) == 3
    assert "quantity" in errors[0]["reason"] and "日期无效" in errors[1]["reason"]


@pytest.mark.parametrize("content,expected", [
    (b"", "为空"),
    ("order_id,product\nx,Mouse\n", "缺少"),
    ("order_id,order_id\nx,y\n", "重复列名"),
    ("order_id,order_date,product,quantity,unit_price\nx,2026-10-01,Mouse,1,1,extra\n", "列数不匹配"),
    ("order_id,order_date,product,quantity,unit_price\nx,2026-10-01,Mouse,1\n", "列数不匹配"),
    ("order_id,order_date,product,quantity,unit_price\n\"unterminated", "CSV 格式"),
    (b"\xff\xfe\x00", "UTF-8"),
    ("order_id,order_date,product,quantity,unit_price\n", "无有效数据行"),
])
def test_invalid_files_fail_explicitly(tmp_path, content, expected):
    with pytest.raises(DataValidationError, match=expected):
        run_csv(tmp_path, content)


def test_row_validation_formula_escape_and_large_money(tmp_path):
    content = "\ufefforder_id,order_date,product,quantity,unit_price\n" + "\n".join([
        "A1,2026-10-01,=HYPERLINK(1),1000000,1000000.00",
        "A2,2026-10-01,Mouse,1,0.10",
        "A3,2026-10-01,Mouse,1,NaN",
        "A4,2026-10-01,Mouse,1,inf",
        "A5,2026-10-01,Mouse,1,0.001",
        "A6,2026-10-01,Mouse,1.5,1.00",
        "A7,2026-10-01,Mouse,1000001,1.00",
        "A8,2026-10-01,Mouse,1,1000000.01",
        "A9,2026-10-01,Mouse,1,-1",
        "A10,2026-10-01,Mouse,1,1e2",
        "A11,2026-02-30,Mouse,1,1.00",
    ])
    summary, result = run_csv(tmp_path, content)
    assert summary["invalid_rows"] == 9
    assert summary["total_amount"] == "1000000000000.10"
    with (result / "cleaned.csv").open(encoding="utf-8-sig") as handle:
        assert list(csv.DictReader(handle))[0]["product"].startswith("'=HYPERLINK")
    with (result / "statistics.csv").open(encoding="utf-8-sig") as handle:
        assert list(csv.DictReader(handle))[0]["product"].startswith("'=")


def test_duplicate_id_keeps_first_valid_row_and_trims_whitespace(tmp_path):
    summary, result = run_csv(tmp_path, "order_id,order_date,product,quantity,unit_price\nA1,2026-10-01,Mouse,-1,1\n A1 ,2026-10-01, Mouse ,2,0.10\nA1,2026-10-01,Mouse,9,1\n")
    assert summary["invalid_rows"] == summary["duplicate_rows"] == summary["valid_rows"] == 1
    assert summary["total_amount"] == "0.20"
