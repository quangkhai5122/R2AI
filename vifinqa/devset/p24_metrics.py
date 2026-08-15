"""Deterministic standard-statement metric resolver for P2.4 authoring.

The resolver is deliberately read-only and report-scoped.  It returns exact
table/row/column references, never a value detached from its source cell.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..extraction.build_store import Store
from ..utils.viet_text import fuzz_token_set, tokens


def _norm(value: object) -> str:
    return " ".join(tokens(unicodedata.normalize("NFC", str(value or "")).lower()))


@dataclass(frozen=True)
class MetricDef:
    labels: tuple[str, ...]
    codes: tuple[str, ...] = ()
    reject: tuple[str, ...] = ()


METRICS: dict[str, MetricDef] = {
    "net_revenue": MetricDef(("doanh thu thuan ve ban hang va cung cap dich vu", "doanh thu thuan"), ("10",)),
    "gross_profit": MetricDef(("loi nhuan gop ve ban hang va cung cap dich vu", "loi nhuan gop"), ("20",)),
    "cogs": MetricDef(("gia von hang ban va dich vu cung cap", "gia von hang ban"), ("11",)),
    "pbt": MetricDef(("tong loi nhuan ke toan truoc thue", "loi nhuan truoc thue"), ("50", "01")),
    "pat": MetricDef(("loi nhuan sau thue thu nhap doanh nghiep", "loi nhuan sau thue tndn"), ("60",)),
    "current_assets": MetricDef(("tai san ngan han",), ("100",), ("khac",)),
    "cash": MetricDef(("tien va cac khoan tuong duong tien",), ("110",), ("dau nam", "cuoi nam")),
    "inventory": MetricDef(("hang ton kho",), ("140",), ("du phong", "bien dong", "tang", "giam")),
    "total_assets": MetricDef(("tong cong tai san", "tong tai san co", "tong tai san"), ("270",)),
    "current_liabilities": MetricDef(("no ngan han",), ("310",)),
    "total_liabilities": MetricDef(("no phai tra",), ("300",), ("khac",)),
    "equity": MetricDef(("von chu so huu",), ("400",), ("nguon von",)),
    "cfo": MetricDef(("luu chuyen tien thuan tu hoat dong kinh doanh",), ("20",)),
    "interest_expense": MetricDef(("chi phi lai vay",), ("23", "06", "6")),
    "basic_eps": MetricDef(("lai co ban tren co phieu",), ("70",)),
    "long_term_borrowings": MetricDef(("vay va no thue tai chinh dai han", "vay dai han"), ("338",)),
    "bonus_welfare": MetricDef(("quy khen thuong phuc loi",), ("322",)),
}


def _canon_code(value: object) -> str:
    text = str(value or "").strip().lstrip("0")
    if text.endswith(".0"):
        text = text[:-2]
    return text or "0"


class StandardMetricResolver:
    def __init__(self, store_dir: Path | str):
        self.store = Store(Path(store_dir))
        self.cache: dict[str, Any] = {}

    def _cells(self, ticker: str):
        ticker = ticker.upper()
        if ticker not in self.cache:
            self.cache[ticker] = self.store.cells_of(ticker).copy()
        return self.cache[ticker]

    def candidates(self, ticker: str, year: int, doc_type: str, metric: str) -> list[dict[str, Any]]:
        definition = METRICS[metric]
        report_id = f"{ticker.upper()}_financial_statements_{int(year)}_{doc_type}"
        frame = self._cells(ticker)
        frame = frame[frame.report_id == report_id]
        labels = tuple(_norm(x) for x in definition.labels)
        rejects = tuple(_norm(x) for x in definition.reject)
        codes = {_canon_code(x) for x in definition.codes}
        rows: dict[tuple[int, int], tuple[float, Any]] = {}
        for row in frame.itertuples():
            label = _norm(row.label)
            if any(term and term in label for term in rejects):
                continue
            fuzzy = max(fuzz_token_set(label, target) for target in labels)
            contains = max((len(set(target.split()) & set(label.split())) / max(1, len(set(target.split()))) for target in labels), default=0)
            code_match = _canon_code(row.row_code) in codes
            if fuzzy < 62 and not (code_match and contains >= 0.55):
                continue
            score = float(fuzzy) + 18.0 * contains + (12.0 if code_match else 0.0)
            key = (int(row.table_pos), int(row.row))
            if key not in rows or score > rows[key][0]:
                rows[key] = (score, row)
        results: list[dict[str, Any]] = []
        for (table_pos, row_no), (row_score, exemplar) in rows.items():
            same = frame[(frame.table_pos.astype(int) == table_pos) & (frame.row.astype(int) == row_no)]
            values = []
            for cell in same.itertuples():
                col_name = _norm(cell.col_name)
                if "ma so" in col_name or "thuyet minh" in col_name:
                    continue
                header_score = 0.0
                if re.search(rf"\b{int(year)}\b", col_name): header_score += 40.0
                if any(x in col_name for x in ("nam nay", "so cuoi nam", "cuoi nam")): header_score += 30.0
                if "nam truoc" in col_name or "so dau nam" in col_name or "dau nam" in col_name: header_score -= 30.0
                header_score -= 0.01 * int(cell.col)
                values.append((header_score, cell))
            if not values:
                continue
            header_score, cell = max(values, key=lambda item: item[0])
            results.append({
                "metric": metric, "report_id": report_id, "table_pos": table_pos,
                "row": row_no, "col": int(cell.col), "label": str(exemplar.label),
                "code": str(exemplar.row_code), "col_name": str(cell.col_name),
                "value": float(cell.value), "unit_scale": float(cell.unit_scale),
                "score": round(row_score + header_score - 0.03 * table_pos, 4),
            })
        return sorted(results, key=lambda x: x["score"], reverse=True)

    def resolve(self, ticker: str, year: int, doc_type: str, metric: str) -> dict[str, Any]:
        hits = self.candidates(ticker, year, doc_type, metric)
        if not hits:
            raise LookupError(f"no {metric} for {ticker} {year} {doc_type}")
        return hits[0]
