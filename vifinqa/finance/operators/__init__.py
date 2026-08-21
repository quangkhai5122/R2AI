"""Package form of the canonical operator contract.

Kept import-compatible with :mod:`vifinqa.finance.operators` on Windows and
other platforms.  The small compatibility method on CanonicalMetric lets the
clean manifest serialize the frozen registry without changing its source
definition.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from ..metrics import CanonicalMetric
from ..schemas import OperatorSpec


if not hasattr(CanonicalMetric, "to_dict"):
    CanonicalMetric.to_dict = lambda self: asdict(self)  # type: ignore[attr-defined]


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
        *(_op(name, "2+", "aggregate", _NUMERIC, "Typed aggregate.")
          for name in ("add", "sum", "average", "min", "max", "median")),
        _op("subtract", "2", "arithmetic", _NUMERIC, "Signed difference."),
        _op("multiply", "2", "arithmetic", _NUMERIC, "Typed product."),
        _op("divide", "2", "arithmetic", _NUMERIC, "Typed quotient."),
        _op("ratio", "2", "arithmetic", ("ratio",), "Numerator/denominator."),
        _op("growth_percent", "2", "finance", ("percent",), "Signed growth percent."),
        _op("growth_pct", "2", "finance", ("percent",), "Alias of growth_percent."),
        _op("cagr_percent", "2", "finance", ("percent",), "Compound annual growth."),
        _op("percentage_point", "2", "finance", ("percentage_point",), "Percentage-point difference."),
        _op("percentage_point_change", "2", "finance", ("percentage_point",), "Percentage-point change."),
        *(_op(name, "2", "finance", _NUMERIC, "Apply a percent change.")
          for name in ("apply_percent_change", "increase_percent", "decrease_percent")),
        _op("power", "2", "arithmetic", ("number", "ratio"), "Grounded exponentiation."),
        *(_op(name, "2", "comparison", ("bool",), "Typed comparison.")
          for name in ("gt", "ge", "lt", "le", "eq", "ne")),
        _op("and", "2+", "boolean", ("bool",), "Boolean conjunction."),
        _op("or", "2+", "boolean", ("bool",), "Boolean disjunction."),
        _op("count_true", "1+", "aggregate", ("count",), "Count true conditions."),
        _op("if_else", "3", "control", _NUMERIC + ("year",), "Grounded conditional."),
        _op("argmax_project", "2+", "ranking", _NUMERIC + ("year",), "Project at maximum score."),
        _op("argmin_project", "2+", "ranking", _NUMERIC + ("year",), "Project at minimum score."),
    )
}

LEGACY_TO_TYPED = {
    "difference": "subtract", "ratio_times": "ratio", "margin": "divide",
    "growth_pct": "growth_percent", "cagr": "cagr_percent",
    "ranking": "argmax_project", "count": "count_true",
}


def get_operator(name: str) -> OperatorSpec:
    return OPERATORS[name]


def operator_registry_fingerprint() -> str:
    payload = {name: spec.to_dict() for name, spec in sorted(OPERATORS.items())}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


__all__ = ["OPERATORS", "LEGACY_TO_TYPED", "get_operator", "operator_registry_fingerprint"]
