"""Resolve one Fact (ticker, year, doc_type, metric) to one concrete table cell.

This is the missing link between `router.decompose` (which says WHICH facts a
question needs) and `codegen.formulas` (which says HOW to combine them).
Composite questions — growth, difference, ratio, ranking — are ~50% of the test
set and scored 0.000 with the lookup-only rule engine.

A resolved fact carries full provenance so the generated pandas query is a
single expression addressing exactly the located row/column, and so the caller
can compute a confidence for arbitration against the LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..retrieval.shortlist import _period_kind, build_shortlist
from ..utils.viet_text import norm

# a column header explicitly naming the wanted year is the strongest evidence
_YEAR_STRONG = 3
_YEAR_WEAK = 1


@dataclass
class ResolvedFact:
    ticker: str
    year: int | None
    metric: str
    var: str                 # df variable holding the cell
    report_id: str
    table_pos: int
    row: int
    label: str
    code: str
    col: int
    col_name: str
    value: float             # raw cell value (NOT unit-converted)
    unit_scale: float        # multiply to get VND
    score: float             # label-match score of the chosen row
    year_evidence: int       # 3 = header names the year, 1 = positional guess

    @property
    def value_vnd(self) -> float:
        return float(self.value) * float(self.unit_scale)

    def expr(self) -> str:
        """Single pandas expression returning this cell's raw value."""
        label = re.sub(r"\s+", " ", str(self.label)).strip()
        return (f"float({self.var}.loc[({self.var}['row'] == {self.row}) "
                f"& {self.var}['label'].str.strip().eq({label!r}) "
                f"& ({self.var}['col'] == {self.col}), 'value'].iloc[0])")

    def expr_vnd(self) -> str:
        return f"({self.expr()} * {self.unit_scale:g})"


def _tables_for(tables: list[dict], ticker: str, year: int | None) -> list[dict]:
    """Tables belonging to this fact's company (and, if possible, its year).

    report_id looks like TICKER_financial_statements_YYYY_{consolidated,separate}
    so the ticker/year filter is exact rather than fuzzy.
    """
    if not ticker:
        return tables
    same_ticker = [t for t in tables
                   if str(t["report_id"]).split("_")[0].upper() == ticker.upper()]
    if not same_ticker:
        return []
    if year is None:
        return same_ticker
    same_year = [t for t in same_ticker if f"_{year}_" in f"_{t['report_id']}_"
                 or str(t.get("report_year")) == str(year)]
    # the FY-Y figure also lives in the Y+1 report's prior-year column
    if not same_year:
        same_year = [t for t in same_ticker
                     if str(t.get("report_year")) == str(year + 1)]
    return same_year or same_ticker


def resolve_fact(fact, tables: list[dict], metric_variants: list[str],
                 encoder=None, min_score: float = 62.0,
                 question: str = "") -> ResolvedFact | None:
    """Locate the cell for one Fact. Returns None when nothing clears min_score."""
    scoped = _tables_for(tables, fact.ticker, fact.year)
    if not scoped:
        return None
    variants = [v for v in (metric_variants or [fact.metric]) if v]
    cands = build_shortlist(scoped, variants, [fact.year] if fact.year else [],
                            top_n=6, encoder=encoder, min_score=min_score,
                            question=question)
    if not cands:
        return None
    best = cands[0]
    year_ev = _year_evidence(best.col_name, fact.year, best.report_id)
    return ResolvedFact(
        ticker=fact.ticker, year=fact.year, metric=fact.metric,
        var=best.var, report_id=best.report_id, table_pos=best.table_pos,
        row=best.row, label=best.label, code=best.code, col=best.col,
        col_name=best.col_name,
        value=best.value, unit_scale=best.unit_scale, score=best.score,
        year_evidence=year_ev)


def _year_evidence(col_name: str, year: int | None,
                   report_id: str = "") -> int:
    if year is None:
        return _YEAR_WEAK
    cn = str(col_name)
    if re.search(rf"31\s*/\s*12\s*/\s*{year}", cn) or re.search(rf"(?<!\d){year}(?!\d)", cn):
        return _YEAR_STRONG
    found = re.search(r"(?:financial_statements_|_)(20\d{2})(?:_|$)",
                      str(report_id))
    report_year = int(found.group(1)) if found else None
    kind = _period_kind(cn)
    if report_year == year and kind == "current":
        return _YEAR_STRONG
    if report_year == year + 1 and kind == "prior":
        return _YEAR_STRONG
    return _YEAR_WEAK


def resolve_all(facts, tables: list[dict], metric_variants: list[str],
                encoder=None, min_score: float = 62.0,
                question: str = ""):
    """(resolved list, confidence 0..100). Confidence is driven by the WEAKEST
    fact: a composite answer is only as trustworthy as its worst operand."""
    out = []
    for f in facts:
        r = resolve_fact(f, tables, metric_variants, encoder, min_score,
                         question=question)
        if r is None:
            return out, 0.0
        out.append(r)
    if not out:
        return out, 0.0
    weakest = min(r.score for r in out)
    year_ok = all(r.year_evidence == _YEAR_STRONG for r in out)
    conf = min(99.0, weakest + (8.0 if year_ok else -10.0))
    # distinct facts must not collapse onto the same cell (a classic failure of
    # multi-entity questions where only one company's tables were retrieved)
    keys = {(r.report_id, r.table_pos, r.row, r.label, r.col) for r in out}
    if len(keys) < len(out):
        conf = min(conf, 35.0)
    return out, max(0.0, conf)


def distinct_cells(resolved: list[ResolvedFact]) -> bool:
    return len({(r.report_id, r.table_pos, r.row, norm(r.label), r.col)
                for r in resolved}) == len(resolved)
