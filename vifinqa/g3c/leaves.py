"""Label-blind atomic query formation for G3C.

Only question text, the frozen clean route, the canonical metric registry, and
store metadata are accepted.  G3B family labels, gold cells, question-ID lists,
and evaluator records are deliberately outside this API.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from typing import Any

from ..finance.metrics import (
    METRICS,
    extract_metric_qualifiers,
    get_metric,
    metric_keys,
)
from ..router.decompose import detect_op
from ..utils.viet_text import norm
from .common import canonical_json_sha256

_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
_REPORT_YEAR_RE = re.compile(
    r"bao cao(?:\s+(?:tai chinh|hop nhat|rieng|cua cong ty me))*"
    r".{0,45}?nam\s+(20\d{2})"
)


@dataclass(frozen=True)
class AtomicLeaf:
    leaf_id: str
    ticker: str
    period_year: int | None
    report_year: int | None
    doc_type: str
    metric_key: str
    metric_label: str
    aliases: tuple[str, ...]
    role: str
    operation: str
    qualifiers: tuple[tuple[str, str], ...]
    report_ids: tuple[str, ...]
    question_fragment: str

    def to_dict(self) -> dict:
        value = asdict(self)
        value["aliases"] = list(self.aliases)
        value["qualifiers"] = dict(self.qualifiers)
        value["report_ids"] = list(self.report_ids)
        return value

    @property
    def hard_key(self) -> tuple[str, int | None, str]:
        return (self.ticker, self.report_year, self.doc_type)


def decompose_atomic_leaves(question: str, route: Any, store: Any) -> list[AtomicLeaf]:
    """Create deterministic evidence requirements without reading any labels."""
    question_norm = norm(question)
    tickers = _dedupe(_route_value(route, "tickers", []))
    years = sorted(set(
        int(year) for year in (_route_value(route, "years", []) or [])
        if year is not None and 2000 <= int(year) <= 2099
    ))
    text_years = sorted({int(value) for value in _YEAR_RE.findall(question_norm)})
    if text_years:
        years = text_years
    operation = _operation(question_norm)
    metric_key_list = metric_keys([question_norm], expand_derived=True)
    if not metric_key_list:
        metric_key_list = list(_route_value(route, "metric_keys", []) or [])
    metric_key_list = [key for key in metric_key_list if key in METRICS]

    base_scope = str(_route_value(route, "doc_type", "consolidated"))
    scopes = _scopes(question_norm, base_scope)
    report_year_match = _REPORT_YEAR_RE.search(question_norm)
    explicit_report_year = (
        int(report_year_match.group(1)) if report_year_match else None
    )

    period_years = list(years) or [None]
    if operation == "cagr" and len(period_years) > 2:
        period_years = [period_years[0], period_years[-1]]
    prior_period = _is_prior_period(question_norm, years)
    if prior_period:
        period_years = [min(years)] if years else [None]
        explicit_report_year = max(years) if years else explicit_report_year
    elif len(period_years) != 1:
        # A generic "report ... year X" match in a multi-year question is
        # usually the first endpoint, not a container report for every period.
        # Bind each ordinary multi-year leaf to its own exact-year report.
        explicit_report_year = None

    if not tickers:
        tickers = [""]
    if not metric_key_list:
        fallback = norm(str(_route_value(route, "metric_norm", "")))
        raw = AtomicLeaf(
            leaf_id="",
            ticker=tickers[0],
            period_year=period_years[0],
            report_year=explicit_report_year or period_years[0],
            doc_type=scopes[0],
            metric_key="",
            metric_label=fallback,
            aliases=(fallback,) if fallback else (),
            role="value",
            operation=operation,
            qualifiers=tuple(sorted(
                extract_metric_qualifiers(question_norm, ()).to_dict().items()
            )),
            report_ids=(),
            question_fragment=question_norm,
        )
        return [_finalize(raw, store)]

    leaves: list[AtomicLeaf] = []
    for ticker in tickers:
        for scope in scopes:
            for period_year in period_years:
                report_year = explicit_report_year or period_year
                for metric_index, key in enumerate(metric_key_list):
                    metric = get_metric(key)
                    qualifiers = extract_metric_qualifiers(
                        question_norm, [key]
                    ).to_dict()
                    if prior_period:
                        qualifiers["period"] = "opening"
                    role = _metric_role(
                        operation, metric_index, len(metric_key_list)
                    )
                    if operation == "cagr" and len(period_years) == 2:
                        role = "base" if period_year == period_years[0] else "end"
                    elif operation == "scope_delta":
                        role = "minuend" if scope == "consolidated" else "subtrahend"
                    leaves.append(_finalize(AtomicLeaf(
                        leaf_id="",
                        ticker=ticker,
                        period_year=period_year,
                        report_year=report_year,
                        doc_type=scope,
                        metric_key=key,
                        metric_label=metric.label,
                        aliases=metric.variants,
                        role=role,
                        operation=operation,
                        qualifiers=tuple(sorted(qualifiers.items())),
                        report_ids=(),
                        question_fragment=_question_fragment(
                            metric.label, ticker, period_year, report_year, scope,
                            qualifiers, role,
                        ),
                    ), store))
    return _unique_leaves(leaves)


def serialize_leaf_query(leaf: AtomicLeaf, original_question: str) -> str:
    qualifiers = ", ".join(
        f"{key}={value}" for key, value in leaf.qualifiers if value
    ) or "none"
    aliases = " | ".join(leaf.aliases[:4])
    return (
        f"Vietnamese source question (context only): "
        f"{norm(original_question)}\n"
        f"Atomic retrieval target (only this fact): "
        f"{leaf.question_fragment}\n"
        f"Canonical metric: {leaf.metric_key or 'unregistered'} / "
        f"{leaf.metric_label}\n"
        f"Aliases: {aliases}\n"
        f"Role: {leaf.role}; operation: {leaf.operation}; "
        f"qualifiers: {qualifiers}. Ignore other facts, entities, and "
        f"periods from the source question."
    )


def hard_constraint_violations(
    leaf: AtomicLeaf, candidate: dict
) -> list[str]:
    violations = []
    if candidate.get("ticker") != leaf.ticker:
        violations.append("ticker")
    if candidate.get("report_id") not in set(leaf.report_ids):
        violations.append("report_id")
    report_year = candidate.get("report_year")
    if report_year is not None and leaf.report_year is not None:
        if int(report_year) != int(leaf.report_year):
            violations.append("report_year")
    doc_type = candidate.get("doc_type")
    if doc_type is not None and doc_type != leaf.doc_type:
        violations.append("doc_type")
    return violations


def _route_value(route: Any, name: str, default: Any) -> Any:
    if isinstance(route, dict):
        return route.get(name, default)
    return getattr(route, name, default)


def _operation(question_norm: str) -> str:
    if "hop nhat" in question_norm and "rieng" in question_norm and (
        "hop nhat tru rieng" in question_norm
        or "tinh hop nhat tru rieng" in question_norm
    ):
        return "scope_delta"
    if "diem phan tram" in question_norm:
        return "percentage_point_change"
    if _is_prior_period(question_norm, []):
        return "prior_period_lookup"
    if "bang bao nhieu lan" in question_norm:
        return "ratio"
    detected = detect_op(question_norm)
    if detected == "average" and "bien loi nhuan" in question_norm:
        return "nested_margin_average"
    return detected


def _scopes(question_norm: str, default: str) -> list[str]:
    has_consolidated = "hop nhat" in question_norm
    has_separate = "bao cao rieng" in question_norm or "so lieu rieng" in question_norm
    if has_consolidated and has_separate:
        return ["consolidated", "separate"]
    if has_separate:
        return ["separate"]
    if has_consolidated:
        return ["consolidated"]
    return [default]


def _is_prior_period(question_norm: str, years: list[int]) -> bool:
    return (
        "so dau ky" in question_norm
        or "so dau nam" in question_norm
        or ("tuong ung cuoi nam" in question_norm and len(years) >= 2)
    )


def _metric_role(operation: str, index: int, count: int) -> str:
    if count < 2:
        return "value"
    if operation in (
        "ratio", "margin", "nested_margin_average",
        "percentage_point_change",
    ):
        return "numerator" if index == 0 else "denominator"
    return "value"


def _question_fragment(
    metric: str,
    ticker: str,
    period_year: int | None,
    report_year: int | None,
    scope: str,
    qualifiers: dict[str, str],
    role: str,
) -> str:
    period = "unspecified" if period_year is None else str(period_year)
    report = "unspecified" if report_year is None else str(report_year)
    qualifier_text = ", ".join(
        f"{key}={value}" for key, value in sorted(qualifiers.items()) if value
    ) or "none"
    return (
        f"metric={metric}; ticker={ticker}; period={period}; "
        f"report_year={report}; scope={scope}; role={role}; "
        f"qualifiers={qualifier_text}"
    )


def _finalize(leaf: AtomicLeaf, store: Any) -> AtomicLeaf:
    report_ids: tuple[str, ...] = ()
    if leaf.ticker and leaf.report_year is not None:
        report_ids = tuple(store.find_reports(
            leaf.ticker, int(leaf.report_year), leaf.doc_type,
            allow_fallback=False,
        ))
    body = {
        "ticker": leaf.ticker,
        "period_year": leaf.period_year,
        "report_year": leaf.report_year,
        "doc_type": leaf.doc_type,
        "metric_key": leaf.metric_key,
        "role": leaf.role,
        "operation": leaf.operation,
        "qualifiers": dict(leaf.qualifiers),
        "report_ids": list(report_ids),
    }
    return replace(
        leaf,
        leaf_id=f"leaf-{canonical_json_sha256(body)[:16]}",
        report_ids=report_ids,
    )


def _unique_leaves(leaves: list[AtomicLeaf]) -> list[AtomicLeaf]:
    seen: set[str] = set()
    output = []
    for leaf in leaves:
        if leaf.leaf_id not in seen:
            seen.add(leaf.leaf_id)
            output.append(leaf)
    return output


def _dedupe(values: list[Any]) -> list[Any]:
    return list(dict.fromkeys(values))
