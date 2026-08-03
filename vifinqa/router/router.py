"""Structured lookup router: lock (ticker, year, doc_type) -> report_ids."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

from ..extraction.build_store import Store
from .entities import StockMap, parse_question, Parsed


@dataclass
class Route:
    qid: int
    question: str
    tickers: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    doc_type: str = "consolidated"
    unit_scale: float = 1.0
    unit_name: str = "đồng"
    is_percent: bool = False
    output_type: str = "number"
    growth: bool = False
    metric_norm: str = ""
    report_ids: list[str] = field(default_factory=list)
    confidence: str = "high"           # high | medium | low
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def route_question(qid: int, question: str, stock: StockMap, store: Store) -> Route:
    p: Parsed = parse_question(question, stock)
    r = Route(qid=qid, question=question, tickers=p.tickers, years=list(p.years),
              doc_type=p.doc_type, unit_scale=p.unit_scale, unit_name=p.unit_name,
              is_percent=p.is_percent, output_type=p.output_type,
              growth=p.growth, metric_norm=p.metric_norm)

    if not p.tickers:
        r.confidence = "low"
        r.notes.append("no ticker found")
        return r
    if p.ticker_source == "low_conf_fuzzy":
        r.confidence = "low"
        r.notes.append(f"fuzzy ticker score={p.ticker_score:.0f}")
    elif p.ticker_source == "fuzzy":
        r.confidence = "medium"

    # years: default = most recent available; growth -> include previous year
    if not r.years:
        avail = store.years_of(r.tickers[0])
        if avail:
            r.years = [avail[-1]]
            r.notes.append("no year in question -> latest available")
            r.confidence = "low"
    if r.growth and len(r.years) == 1:
        r.years.append(r.years[0] - 1)
        r.notes.append("growth question -> added prior year")

    for t in r.tickers:
        for y in r.years:
            exact = store.find_reports(t, y, r.doc_type, allow_fallback=False)
            matches = exact or store.find_reports(t, y, r.doc_type)
            if not matches:
                r.notes.append(f"missing report {t}/{y}/{r.doc_type}")
                # try adjacent year (a FY-{y} figure often sits in the {y+1} report's
                # prior-year column)
                alt = store.find_reports(t, y + 1, r.doc_type)
                if alt:
                    r.report_ids.extend(alt)
                    r.notes.append(f"fallback to {t}/{y+1} (prior-year column)")
                continue
            r.report_ids.extend(matches)
            if not exact:
                r.notes.append(f"doc_type fallback for {t}/{y}")
    # dedupe, keep order
    seen = set()
    r.report_ids = [x for x in r.report_ids if not (x in seen or seen.add(x))]
    if not r.report_ids:
        r.confidence = "low"
        r.notes.append("no report locked")
    return r
