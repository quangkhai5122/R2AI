"""Typed, JSON-safe contracts shared by routing, retrieval and codegen.

The schema deliberately carries semantic identity only.  It contains no
leaderboard IDs, question-specific exceptions or learned weights.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from ..utils.viet_text import norm


@dataclass(frozen=True)
class MetricQualifiers:
    stock_flow: str = ""
    gross_net: str = ""
    maturity: str = ""
    period: str = ""
    granularity: str = ""
    sign: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalMetric:
    key: str
    label: str
    aliases: tuple[str, ...] = ()
    codes: tuple[str, ...] = ()
    statement: str = "other"
    required_phrases: tuple[str, ...] = ()
    forbidden_phrases: tuple[str, ...] = ()
    qualifier_phrases: tuple[str, ...] = ()
    components: tuple[str, ...] = ()
    qualifiers: MetricQualifiers = MetricQualifiers()

    @property
    def variants(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            value for value in (norm(self.label), *(norm(x) for x in self.aliases))
            if value
        ))

    @property
    def is_derived(self) -> bool:
        return bool(self.components)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MetricMatch:
    metric: CanonicalMetric
    alias: str
    start: int
    end: int


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    arity: str
    category: str
    output_types: tuple[str, ...]
    description: str

    def to_dict(self) -> dict:
        return asdict(self)

