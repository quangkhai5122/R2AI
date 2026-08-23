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

import math
import re
from dataclasses import dataclass

from ..finance.metrics import get_metric, metric_context_matches
from ..retrieval.shortlist import (
    _period_kind,
    build_shortlist,
    candidate_matches_requirement,
    requirement_linking_variants,
)
from ..utils.viet_text import norm, tokens

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


def resolve_requirement(requirement: dict, tables: list[dict], encoder=None,
                        min_score: float = 62.0, question: str = "",
                        ambiguity_gap: float = 3.0) -> ResolvedFact | None:
    """Resolve one canonical requirement, refusing fuzzy or ambiguous rows.

    This stricter path is intended for deterministic projections such as
    argmax(metric by year). It accepts only canonical row identity, strong
    period evidence and one unambiguous value for the requested report year.
    """
    ticker = str(requirement.get("ticker") or "").upper()
    raw_year = requirement.get("year")
    year = int(raw_year) if raw_year is not None else None
    metric_key = str(requirement.get("metric_key") or "")
    if not ticker or year is None or not metric_key:
        return None

    doc_type = str(requirement.get("doc_type") or "")
    scoped = _tables_for_exact_year(tables, ticker, year, doc_type)
    if not scoped:
        return None
    variants = requirement_linking_variants(requirement)
    if not variants:
        return None
    candidates = build_shortlist(
        scoped, variants, [year], top_n=16, encoder=encoder,
        min_score=min(35.0, min_score), question=question,
    )
    exact = [
        candidate for candidate in candidates
        if candidate_matches_requirement(candidate, requirement)
        and _canonical_label_is_exact(candidate.label, candidate.code, metric_key)
        and _candidate_context_is_exact(candidate.var, scoped, metric_key)
        and _year_evidence(candidate.col_name, year, candidate.report_id) == _YEAR_STRONG
    ]
    if not exact:
        return None

    # Prefer the requested year's own filing. Use the following filing's
    # prior-year column only when the requested filing has no exact candidate.
    current = [candidate for candidate in exact
               if _report_year(candidate.report_id) == year]
    exact = current or [candidate for candidate in exact
                        if _report_year(candidate.report_id) == year + 1]
    if not exact:
        return None
    best = exact[0]

    # Duplicate renderings of the same cell/value are harmless. Competing
    # exact-looking rows with different labels or values are not.
    for candidate in exact[1:]:
        if candidate.score < best.score - ambiguity_gap:
            break
        same_identity = (
            norm(candidate.label) == norm(best.label)
            and _clean_code(candidate.code) == _clean_code(best.code)
        )
        same_value = math.isclose(
            candidate.value * candidate.unit_scale,
            best.value * best.unit_scale,
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
        if not (same_identity and same_value):
            return None

    return ResolvedFact(
        ticker=ticker, year=year,
        metric=str(requirement.get("metric_label") or metric_key),
        var=best.var, report_id=best.report_id, table_pos=best.table_pos,
        row=best.row, label=best.label, code=best.code, col=best.col,
        col_name=best.col_name, value=best.value, unit_scale=best.unit_scale,
        score=best.score, year_evidence=_YEAR_STRONG,
    )


def _tables_for_exact_year(tables: list[dict], ticker: str, year: int,
                           doc_type: str = "") -> list[dict]:
    out = []
    for table in tables:
        report_id = str(table.get("report_id") or "")
        if report_id.split("_")[0].upper() != ticker:
            continue
        if doc_type in {"consolidated", "separate", "aggregated"}:
            if not re.search(rf"_{re.escape(doc_type)}(?:_|$)", report_id):
                continue
        report_year = _report_year(report_id)
        if report_year is None:
            raw = table.get("report_year")
            report_year = int(raw) if raw is not None else None
        if report_year in {year, year + 1}:
            out.append(table)
    return out


def _report_year(report_id: str) -> int | None:
    found = re.search(r"(?:financial_statements_|_)(20\d{2})(?:_|$)",
                      str(report_id))
    return int(found.group(1)) if found else None


def _clean_code(code: str) -> str:
    return re.sub(r"\.0$", "", str(code or "").strip())


def _canonical_label_is_exact(label: str, code: str, metric_key: str) -> bool:
    """Reject child/detail rows that merely contain a parent metric alias."""
    raw_label = str(label).strip()
    # OCR can fold an empty current-period cell into the label and leave the
    # prior-period value as the first numeric candidate.
    if re.search(r"(?:^|\s)[-\u2013\u2014]\s*$", raw_label):
        return False

    metric = get_metric(metric_key)
    clean_code = _clean_code(code)
    if clean_code.isdigit() and metric.codes:
        return clean_code in metric.codes

    label_norm = norm(label)
    if any(value in label_norm for value in metric.forbidden_phrases):
        return False
    label_tokens = tokens(label_norm)
    while label_tokens and (
        label_tokens[0].isdigit()
        or len(label_tokens[0]) == 1
        or re.fullmatch(r"[ivxlcdm]+", label_tokens[0])
    ):
        label_tokens.pop(0)
    clean_label = " ".join(label_tokens)
    variants = list(dict.fromkeys([
        metric.label,
        *metric.row_aliases,
        *(value for value in metric.variants if len(tokens(value)) >= 3),
    ]))
    return any(
        clean_label == variant or clean_label.startswith(f"{variant} ")
        for variant in variants
    )


def _candidate_context_is_exact(var: str, tables: list[dict],
                                metric_key: str) -> bool:
    table = next((table for table in tables if table.get("var") == var), None)
    return bool(table) and metric_context_matches(
        metric_key, str(table.get("context") or ""))


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
