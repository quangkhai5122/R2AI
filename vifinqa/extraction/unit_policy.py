"""Conservative unit provenance repair shared by parsing and codegen.

Some reports contain an early summary table in billion VND followed by audited
statements whose heading ends in a bare VND token. Older stores inherited the
early multiplier through the sticky-unit fallback. This module recognises only
that narrow terminal-currency pattern and provides an auditable runtime repair
so frozen stores do not need to be rebuilt.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from ..utils.viet_text import strip_diacritics


POLICY_VERSION = "terminal_bare_vnd_v1"
RESOLUTION_STORED = "stored_unit"
RESOLUTION_OVERRIDE = "terminal_bare_vnd_overrides_sticky"
RESOLUTION_CONFIRMED = "terminal_bare_vnd_confirms_vnd"

_TERMINAL_VND = re.compile(r"\bvnd\s*[\]\[(){}.,:;_-]*\s*$")
_TERMINAL_MULTIPLIED_VND = re.compile(
    r"\b(?:nghin\s+ty|ngan\s+ty|tram\s+ty|ty|trieu|nghin|ngan|"
    r"trillion|billion|million|thousand)\s+vnd\s*[\]\[(){}.,:;_-]*\s*$"
)


@dataclass(frozen=True)
class UnitResolution:
    stored_scale: float
    effective_scale: float
    stored_source: str
    effective_source: str
    reason: str
    terminal_bare_vnd: bool
    changed: bool


def has_terminal_bare_vnd(context: str) -> bool:
    """True only for an unqualified VND token at the end of table context."""
    text = re.sub(r"\s+", " ", strip_diacritics(str(context or "")).lower()).strip()
    if not text or not _TERMINAL_VND.search(text):
        return False
    return _TERMINAL_MULTIPLIED_VND.search(text) is None


def resolve_stored_table_unit(
    unit_scale: float | None,
    unit_source: str | None,
    context: str,
) -> UnitResolution:
    """Resolve an effective scale without mutating the frozen table store.

    Only sticky metadata can be overridden. Explicit/header/bare declarations
    remain authoritative. A missing/non-finite stored scale is represented as
    VND downstream, matching the existing extraction contract.
    """
    try:
        stored = float(unit_scale) if unit_scale is not None else 1.0
    except (TypeError, ValueError):
        stored = 1.0
    if not math.isfinite(stored) or stored <= 0:
        stored = 1.0

    source = str(unit_source or "none")
    terminal = has_terminal_bare_vnd(context)
    if source == "sticky" and terminal:
        changed = not math.isclose(stored, 1.0, rel_tol=0.0, abs_tol=0.0)
        return UnitResolution(
            stored_scale=stored,
            effective_scale=1.0,
            stored_source=source,
            effective_source="terminal_vnd",
            reason=RESOLUTION_OVERRIDE if changed else RESOLUTION_CONFIRMED,
            terminal_bare_vnd=True,
            changed=changed,
        )
    return UnitResolution(
        stored_scale=stored,
        effective_scale=stored,
        stored_source=source,
        effective_source=source,
        reason=RESOLUTION_STORED,
        terminal_bare_vnd=terminal,
        changed=False,
    )
