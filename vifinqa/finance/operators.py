"""Canonical operator vocabulary for the existing typed Selection v2 IR.

This module documents and fingerprints the compiler's operator surface.  It
does not evaluate formulas and therefore cannot diverge into a second solver.
"""
from __future__ import annotations

import hashlib
import json

from .schemas import OperatorSpec


def _op(name: str, arity: str, category: str, outputs: tuple[str, ...],
        description: str) -> OperatorSpec:
    return OperatorSpec(name, arity, category, outputs, description)


_NUMERIC = ("money", "number", "ratio", "percent", "percentage_point")
OPERATORS: dict[str, OperatorSpec] = {
    spec.name: spec for spec in (
        _op("lookup", "1", "value", _NUMERIC, "Return one grounded value."),
        _op("abs", "1", "unary", _NUMERIC, "Absolute value."),
        _op("negate", "1", "unary", _NUMERIC, "Signed negation."),
        _op("not", "1", "boolean", ("bool",), "Boolean negation."),
        _op("add", "2+", "aggregate", _NUMERIC, "Add compatible values."),
        _op("sum", "2+", "aggregate", _NUMERIC, "Sum compatible values."),
        _op("average", "2+", "aggregate", _NUMERIC, "Arithmetic mean."),
        _op("min", "2+", "aggregate", _NUMERIC, "Minimum value."),
        _op("max", "2+", "aggregate", _NUMERIC, "Maximum value."),
        _op("median", "2+", "aggregate", _NUMERIC, "Median value."),
        _op("subtract", "2", "arithmetic", _NUMERIC, "Signed difference."),
        _op("multiply", "2", "arithmetic", _NUMERIC, "Typed product."),
        _op("divide", "2", "arithmetic", _NUMERIC, "Typed quotient."),
        _op("ratio", "2", "arithmetic", ("ratio",), "Numerator/denominator."),
        _op("growth_percent", "2", "finance", ("percent",), "Signed growth percent."),
        _op("growth_pct", "2", "finance", ("percent",), "Alias of growth_percent."),
        _op("cagr_percent", "2", "finance", ("percent",), "Compound annual growth."),
        _op("percentage_point", "2", "finance", ("percentage_point",), "Percentage-point difference."),
        _op("percentage_point_change", "2", "finance", ("percentage_point",), "Percentage-point change."),
        _op("apply_percent_change", "2", "finance", _NUMERIC, "Apply a signed percent change."),
        _op("increase_percent", "2", "finance", _NUMERIC, "Increase by a percent."),
        _op("decrease_percent", "2", "finance", _NUMERIC, "Decrease by a percent."),
        _op("power", "2", "arithmetic", ("number", "ratio"), "Grounded exponentiation."),
        *(_op(name, "2", "comparison", ("bool",), "Typed comparison.")
          for name in ("gt", "ge", "lt", "le", "eq", "ne")),
        _op("and", "2+", "boolean", ("bool",), "Boolean conjunction."),
        _op("or", "2+", "boolean", ("bool",), "Boolean disjunction."),
        _op("count_true", "1+", "aggregate", ("count",), "Count true conditions."),
        _op("if_else", "3", "control", _NUMERIC + ("year",), "Grounded conditional."),
        _op("argmax_project", "2+", "ranking", _NUMERIC + ("year",), "Project the result at maximum score."),
        _op("argmin_project", "2+", "ranking", _NUMERIC + ("year",), "Project the result at minimum score."),
    )
}

LEGACY_TO_TYPED = {
    "difference": "subtract",
    "ratio_times": "ratio",
    "margin": "divide",
    "growth_pct": "growth_percent",
    "cagr": "cagr_percent",
    "ranking": "argmax_project",
    "count": "count_true",
}


def get_operator(name: str) -> OperatorSpec:
    return OPERATORS[name]


def operator_registry_fingerprint() -> str:
    payload = {name: spec.to_dict() for name, spec in sorted(OPERATORS.items())}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

