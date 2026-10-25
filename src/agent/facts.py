from datetime import date, datetime, timedelta
import csv
import re

from src.config import settings


REFERENCE_DATE = date(2026, 10, 20)


def _holidays() -> set[date]:
    path = settings.data_dir / "holidays.csv"
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {date.fromisoformat(row["date"]) for row in csv.DictReader(handle)}


def add_business_days(start: date, days: int) -> date:
    holidays = _holidays()
    current = start
    added = 0
    while added < days:
        current += timedelta(days=1)
        if current.weekday() != 6 and current not in holidays:
            added += 1
    return current


def refund_facts(text: str) -> list[str]:
    lowered = text.casefold()
    payment = next(
        (
            method
            for method in ("credit card", "debit card", "card", "upi", "net banking", "netbanking")
            if method in lowered
        ),
        None,
    )
    date_match = re.search(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(2026)\b", text)
    if not payment or not date_match:
        return []
    started = date(
        2026,
        datetime.strptime(date_match.group(0), "%d %B %Y").month,
        int(date_match.group(1)),
    )
    if payment in ("credit card", "debit card", "card"):
        low, high, label = 5, 7, "card"
    elif payment == "upi":
        low, high, label = 1, 3, "UPI"
    else:
        low, high, label = 3, 5, "net banking"
    return [
        f"{label} refund started on {started.isoformat()}",
        f"expected refund window: {add_business_days(started, low).isoformat()} "
        f"to {add_business_days(started, high).isoformat()}",
        "business days are Monday through Saturday and skip listed holidays",
    ]


def order_facts(order: dict | None, text: str) -> list[str]:
    if not order:
        return []
    facts: list[str] = []
    if order.get("items"):
        item_text = " ".join(
            f"{item.get('title', '')} {item.get('return_category', '')}"
            for item in order["items"]
        ).casefold()
        if "refund" in text.casefold() and order.get("payment_method"):
            if order["payment_method"] == "wallet":
                facts.append("wallet refund is instant within 2 hours")
        if "return" in text.casefold() and order.get("delivered_on"):
            if "fashion" in item_text:
                days = 30 if "2026-10-01" <= (order.get("placed_at") or "")[:10] <= "2026-10-15" else 15
                deadline = date.fromisoformat(order["delivered_on"]) + timedelta(days=days)
                facts.append(f"return deadline: {deadline.isoformat()}")
                facts.append(f"return window: {days} calendar days")
    return facts
