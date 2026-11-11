"""纯离线处理：金额用整数分，pandas 负责去重及分组。"""
import csv
import json
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

COLUMNS = ["order_id", "order_date", "product", "quantity", "unit_price"]
ARTIFACTS = {
    "cleaned": "cleaned.csv", "report": "report.json",
    "statistics": "statistics.csv", "rejected": "rejected.csv",
}


class DataValidationError(ValueError):
    pass


def spreadsheet_safe(value):
    """Prevent a user-supplied cell from being evaluated as a spreadsheet formula."""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def write_csv(path: Path, columns: list[str], rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([spreadsheet_safe(cell) for cell in row])


def money(cents: int) -> str:
    return f"{cents // 100}.{cents % 100:02d}"


def process_csv(source: Path, destination: Path) -> dict:
    valid, rejected = [], []
    try:
        with source.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            header = next(reader, None)
            if header is None:
                raise DataValidationError("CSV 文件为空")
            header = [item.strip() for item in header]
            if len(set(header)) != len(header):
                raise DataValidationError("CSV 包含重复列名")
            missing = set(COLUMNS) - set(header)
            extra = set(header) - set(COLUMNS)
            if missing or extra:
                raise DataValidationError(f"列名不符合要求；缺少：{', '.join(sorted(missing)) or '无'}；多余：{', '.join(sorted(extra)) or '无'}")
            total = 0
            for record_number, cells in enumerate(reader, start=2):
                total += 1
                if len(cells) != len(header):
                    raise DataValidationError(f"第 {record_number} 条 CSV 记录列数不匹配")
                row = {key: value.strip() for key, value in zip(header, cells)}
                errors = []
                for key in COLUMNS:
                    if not row[key]:
                        errors.append(f"{key} 缺失")
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["order_date"]):
                    errors.append("order_date 必须为 YYYY-MM-DD")
                else:
                    try:
                        date.fromisoformat(row["order_date"])
                    except ValueError:
                        errors.append("order_date 日期无效")
                if not re.fullmatch(r"[0-9]{1,7}", row["quantity"]) or not 1 <= int(row["quantity"] or "0") <= 1_000_000:
                    errors.append("quantity 必须为 1 到 1000000 的整数")
                price = None
                if re.fullmatch(r"[0-9]{1,7}(?:\.[0-9]{1,2})?", row["unit_price"]):
                    try:
                        price = Decimal(row["unit_price"])
                    except InvalidOperation:
                        pass
                if price is None or not 0 <= price <= 1_000_000:
                    errors.append("unit_price 必须为 0 到 1000000、最多两位小数的金额")
                if errors:
                    rejected.append([record_number, *[row[key] for key in COLUMNS], "; ".join(errors)])
                else:
                    valid.append({**row, "quantity": int(row["quantity"]), "price_cents": int(price * 100)})
    except (UnicodeError, csv.Error) as exc:
        raise DataValidationError("CSV 必须为 UTF-8 编码且符合 CSV 格式") from exc

    if not valid:
        raise DataValidationError(f"CSV 无有效数据行（共 {total} 行，异常 {len(rejected)} 行）")
    frame = pd.DataFrame(valid)
    # order_id is a sales record identifier; keep its first valid occurrence.
    duplicates = int(frame.duplicated(subset=["order_id"]).sum())
    frame = frame.drop_duplicates(subset=["order_id"], keep="first").copy()
    # Python integers avoid accumulation overflow and binary floating-point rounding.
    frame["amount_cents"] = pd.Series(
        [int(q) * int(p) for q, p in zip(frame["quantity"], frame["price_cents"])],
        index=frame.index, dtype=object,
    )
    grouped = frame.groupby("product", sort=True).agg(
        orders=("order_id", "size"), quantity=("quantity", "sum"), amount_cents=("amount_cents", "sum")
    )
    products = [
        {"product": product, "orders": int(row.orders), "quantity": int(row.quantity), "total_amount": money(int(row.amount_cents))}
        for product, row in grouped.iterrows()
    ]
    days = [
        {"date": day, "total_amount": money(int(value))}
        for day, value in frame.groupby("order_date", sort=True)["amount_cents"].sum().items()
    ]
    summary = {
        "total_rows": total, "valid_rows": len(frame), "invalid_rows": len(rejected),
        "duplicate_rows": duplicates, "total_quantity": int(frame["quantity"].sum()),
        "total_amount": money(sum(frame["amount_cents"])),
    }
    write_csv(destination / "cleaned.csv", [*COLUMNS, "total_amount"], (
        [row.order_id, row.order_date, row.product, row.quantity, money(int(row.price_cents)), money(int(row.amount_cents))]
        for row in frame.itertuples()
    ))
    write_csv(destination / "rejected.csv", ["record_number", *COLUMNS, "reason"], rejected)
    write_csv(destination / "statistics.csv", ["product", "orders", "quantity", "total_amount"], (
        [item["product"], item["orders"], item["quantity"], item["total_amount"]] for item in products
    ))
    (destination / "report.json").write_text(
        json.dumps({"summary": summary, "by_product": products, "by_date": days}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
