"""Canonical evidence requirements shared by retrieval and code generation."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from ..finance.metrics import get_metric, metric_evidence_components, metric_keys
from ..utils.viet_text import norm


@dataclass(frozen=True)
class EvidenceRequirement:
    requirement_id: str
    ticker: str
    year: int | None
    doc_type: str
    metric_key: str
    metric_label: str
    metric_variants: tuple[str, ...]
    statement: str

    def to_dict(self) -> dict:
        value = asdict(self)
        value["metric_variants"] = list(self.metric_variants)
        return value


def build_evidence_requirements(
    question: str,
    tickers: list[str],
    years: list[int],
    doc_type: str,
    metric_variants: list[str] | None = None,
    plan: dict | None = None,
) -> list[EvidenceRequirement]:
    """Expand a question into concrete (entity, period, operand) requirements.

    Derived canonical metrics are expanded to their atomic components. This is
    deliberately a cross product: count, temporal and nested-ranking questions
    need every named entity/period to have every operand available before they
    can be answered safely.
    """
    specific_sources = [str(value) for value in (metric_variants or []) if value]
    texts = [question, *specific_sources]
    for fact in (plan or {}).get("facts", []):
        fact_metric = str(fact.get("metric") or "")
        texts.append(fact_metric)
        if fact_metric:
            specific_sources.append(fact_metric)
    root_keys = metric_keys(texts, expand_derived=False)
    if not root_keys:
        return []

    operands: list[tuple[str, int]] = []
    for root_key in root_keys:
        for operand in metric_evidence_components(root_key):
            if operand not in operands:
                operands.append(operand)

    entities = list(dict.fromkeys(str(t).upper() for t in tickers if t)) or [""]
    periods = list(dict.fromkeys(int(y) for y in years if y is not None)) or [None]
    out = []
    seen = set()
    for ticker in entities:
        for year in periods:
            for key, year_offset in operands:
                metric = get_metric(key)
                specific = [
                    norm(source) for source in specific_sources
                    if key in metric_keys([source], expand_derived=False)
                ]
                variants = tuple(dict.fromkeys(
                    value for value in (*specific, *metric.variants) if value
                ))
                evidence_year = None if year is None else year + year_offset
                req_id = (f"{ticker or '*'}|"
                          f"{evidence_year if evidence_year is not None else '*'}|{key}")
                if req_id in seen:
                    continue
                seen.add(req_id)
                out.append(EvidenceRequirement(
                    requirement_id=req_id,
                    ticker=ticker,
                    year=evidence_year,
                    doc_type=doc_type,
                    metric_key=key,
                    metric_label=metric.label,
                    metric_variants=variants,
                    statement=metric.statement,
                ))
    return out


def evidence_coverage(requirements: list[dict], candidates: list[dict]) -> dict:
    """Summarize which requirements are covered by the selected table set."""
    required = [str(r.get("requirement_id") or "") for r in requirements]
    required = list(dict.fromkeys(r for r in required if r))
    covered = {
        str(req_id)
        for candidate in candidates
        for req_id in candidate.get("requirement_hits", [])
        if req_id
    }
    missing = [req_id for req_id in required if req_id not in covered]
    return {
        "required": len(required),
        "covered": len(required) - len(missing),
        "complete": bool(required) and not missing,
        "missing": missing,
    }
