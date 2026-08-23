"""Source-fact loader for the G3B program-first corpus."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from ..extraction.build_store import Store
from ..finance.metrics import METRICS, metric_keys
from ..utils.viet_text import fuzz_token_set
from .common import normalize_question

CODE_TO_METRIC = {
    "10": "net_revenue",
    "11": "cost_of_goods_sold",
    "20": "gross_profit",
    "50": "pretax_profit",
    "60": "net_profit",
    "270": "total_assets",
    "300": "liabilities",
    "400": "equity",
}
DISPLAY = {
    "net_revenue": "doanh thu thuần",
    "cost_of_goods_sold": "giá vốn hàng bán",
    "gross_profit": "lợi nhuận gộp",
    "pretax_profit": "lợi nhuận trước thuế",
    "net_profit": "lợi nhuận sau thuế",
    "total_assets": "tổng tài sản",
    "liabilities": "nợ phải trả",
    "equity": "vốn chủ sở hữu",
}


@dataclass(frozen=True)
class Fact:
    fact_id: str
    ticker: str
    report_year: int
    period_year: int
    doc_type: str
    scope: str
    report_id: str
    table_pos: int
    table_line: int
    row: int
    col: int
    row_code: str
    metric_key: str
    label: str
    col_name: str
    value: float
    unit_scale: float
    base_value: float
    source_kind: str
    period_semantics: str


def _row_code(value: object) -> str:
    raw = str(value or "").strip()
    return raw[:-2] if raw.endswith(".0") else raw


def scope_label(doc_type: str) -> str:
    return {
        "consolidated": "hợp nhất",
        "separate": "riêng của công ty mẹ",
    }.get(doc_type, doc_type)


def period_year(report_year: int, col_name: str) -> tuple[int | None, str]:
    value = str(col_name)
    normalized = normalize_question(value)
    years = [
        int(match)
        for match in re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", value)
    ]
    opening = bool(re.search(
        r"(?:01[./-]01|dau nam|dau ky|opening)", normalized
    ))
    if opening and report_year in years:
        return report_year - 1, "opening"
    if report_year in years:
        return report_year, "current"
    if report_year - 1 in years:
        return report_year - 1, "prior"
    return None, "unknown"


def _progress(values):
    try:
        from tqdm.auto import tqdm
        return tqdm(values, desc="G3B source facts", unit="ticker")
    except ImportError:
        return values


def load_source_facts(store_dir: Path | str) -> list[Fact]:
    store = Store(Path(store_dir), cache_size=3)
    note_keys = {
        key
        for key, metric in METRICS.items()
        if not metric.is_derived
        and not metric.codes
        and metric.statement == "other"
    }
    facts: list[Fact] = []
    tickers = sorted({str(value) for value in store.reports.ticker})
    for ticker in _progress(tickers):
        cells = store.cells_of(ticker)
        tables = store.tables_of(ticker)
        if not len(cells) or not len(tables):
            continue
        table_meta = {
            (str(row.report_id), int(row.table_pos)): row
            for row in tables.itertuples()
        }
        best: dict[tuple, tuple[float, Fact]] = {}
        for cell in cells.itertuples():
            label = str(cell.label or "").strip()
            if not bool(cell.unit_known) or len(label) < 5:
                continue
            value = float(cell.value)
            scale = float(cell.unit_scale)
            if (
                not math.isfinite(value)
                or not math.isfinite(scale)
                or scale <= 0
                or abs(value * scale) < 1
            ):
                continue
            report_year = int(cell.year)
            fact_year, semantics = period_year(
                report_year, str(cell.col_name)
            )
            if fact_year not in {report_year, report_year - 1}:
                continue
            key = (str(cell.report_id), int(cell.table_pos))
            meta = table_meta.get(key)
            if meta is None:
                continue

            code = _row_code(cell.row_code)
            metric_key = CODE_TO_METRIC.get(code)
            source_kind = "statement"
            if metric_key:
                score = max(
                    float(fuzz_token_set(label, variant))
                    for variant in METRICS[metric_key].variants
                )
                if score < 82:
                    continue
            else:
                matches = [
                    name
                    for name in metric_keys([label], expand_derived=False)
                    if name in note_keys
                ]
                if not matches or int(cell.table_pos) < 4:
                    continue
                metric_key = sorted(matches)[0]
                score = max(
                    float(fuzz_token_set(label, variant))
                    for variant in METRICS[metric_key].variants
                )
                if score < 88:
                    continue
                source_kind = "note_table"

            fact = Fact(
                fact_id=(
                    f"{cell.report_id}|{int(cell.table_pos)}|"
                    f"{int(cell.row)}|{int(cell.col)}|"
                    f"{metric_key}|{fact_year}"
                ),
                ticker=str(cell.ticker),
                report_year=report_year,
                period_year=int(fact_year),
                doc_type=str(cell.doc_type),
                scope=scope_label(str(cell.doc_type)),
                report_id=str(cell.report_id),
                table_pos=int(cell.table_pos),
                table_line=int(meta.line_no),
                row=int(cell.row),
                col=int(cell.col),
                row_code=code,
                metric_key=metric_key,
                label=label,
                col_name=str(cell.col_name),
                value=value,
                unit_scale=scale,
                base_value=value * scale,
                source_kind=source_kind,
                period_semantics=semantics,
            )
            identity = (
                fact.report_id,
                fact.metric_key,
                fact.period_year,
                fact.source_kind,
            )
            rank = score - 0.0001 * fact.table_pos
            if identity not in best or rank > best[identity][0]:
                best[identity] = (rank, fact)
        facts.extend(item[1] for item in best.values())
    return sorted(facts, key=lambda item: item.fact_id)


def statement_facts(
    facts: list[Fact], *, current: bool = True
) -> list[Fact]:
    return [
        fact
        for fact in facts
        if fact.source_kind == "statement"
        and (
            fact.period_year == fact.report_year
            if current
            else fact.period_year == fact.report_year - 1
        )
    ]
