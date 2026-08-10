"""Structured selection: the LLM PICKS cells, we WRITE the pandas.

WHY (audited on submission #12, Qwen 7B, 183 generated queries):
    35%  no column filter at all -> .iloc[0] grabs the wrong period
    15%  forgot to divide by ANSWER_SCALE -> raw VND for a "tỷ đồng" question
    90%  str.contains without regex=False -> parentheses become regex groups
    only 35% used the ['col'] == N condition the shortlist already handed them
Those 175 new answers moved the leaderboard by ~2 questions (~2% accuracy).

The shortlist ALREADY knows var/label/code/col/col_name/value/unit_scale — the
cell is located before the model is even called. Asking the model to re-derive
that addressing in pandas is where it fails. So instead it emits a tiny JSON:

    {"op": "difference", "operands": [1, 3]}
    {"op": "lookup", "operands": [2]}
    {"op": "growth_pct", "operands": [4, 5]}     # [end, base]

`operands` are 1-based indices into the rendered shortlist. We synthesise the
expression with the correct column filter and unit conversion, reusing the same
arithmetic as the deterministic engine. This makes three whole error classes
structurally impossible, and shrinks the generated text from ~256 tokens to ~30.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .units import check_answer_unit, percent_from_cell, cell_is_already_percent

# Ops the synthesiser can execute from selected cells.  ``argmax``/``argmin``
# are typed projections: they return the year attached to the winning cell,
# rather than returning the cell value.  ``ranking_*`` keeps its historical
# numeric behaviour except when the requested output type is ``year``; that
# compatibility path lets us safely replay the P2.1 responses already on disk.
SELECT_OPS = {"lookup", "sum", "average", "difference", "growth_pct",
              "ratio", "margin", "ratio_times", "ranking_max", "ranking_min",
              "argmax", "argmin", "count", "percentage_point"}

# Fixed-arity operators must receive exactly this many operands.  The old
# lower-bound-only check silently accepted e.g. ``lookup`` with five operands
# and then discarded four of them.
EXACT_ARITY = {"lookup": 1, "difference": 2, "growth_pct": 2, "ratio": 2,
               "margin": 2, "ratio_times": 2, "percentage_point": 2}
MIN_ARITY = {"sum": 1, "average": 1, "count": 1,
             "ranking_max": 2, "ranking_min": 2,
             "argmax": 2, "argmin": 2}

# Fail closed only for unmistakable unit explosions.  The lower, heuristic
# ranges in units.PLAUSIBLE remain confidence warnings until P2.4 provides
# enough labelled examples for calibration.
_HARD_ABS_LIMIT = {
    "percent": 1_000_000.0,
    "percentage_point": 10_000.0,
    "ratio": 1_000_000.0,
}

_JSON_OBJ = re.compile(r"\{[^{}]*\}", re.S)


@dataclass
class Selection:
    op: str
    operands: list[int]          # 1-based indices into the shortlist
    note: str = ""

    def valid_for(self, n_candidates: int) -> str | None:
        if self.op not in SELECT_OPS:
            return f"unknown op {self.op!r}"
        if not self.operands:
            return "no operands"
        exact = EXACT_ARITY.get(self.op)
        if exact is not None and len(self.operands) != exact:
            return (f"op {self.op} needs exactly {exact} operands, "
                    f"got {len(self.operands)}")
        minimum = MIN_ARITY.get(self.op)
        if minimum is not None and len(self.operands) < minimum:
            return (f"op {self.op} needs at least {minimum} operands, "
                    f"got {len(self.operands)}")
        if len(set(self.operands)) != len(self.operands):
            return "duplicate operands are not allowed"
        bad = [i for i in self.operands if not (1 <= i <= n_candidates)]
        if bad:
            return f"operand index out of range: {bad}"
        return None


def parse_selection(text: str) -> Selection | None:
    """Tolerant JSON extraction: models wrap output in prose or code fences."""
    if not text:
        return None
    blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, re.S) or [text]
    for block in blocks:
        for m in [block] + _JSON_OBJ.findall(block):
            try:
                obj = json.loads(m.strip())
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(obj, dict):
                continue
            op = str(obj.get("op", "")).strip().lower()
            raw = obj.get("operands", obj.get("idx", obj.get("rows")))
            if isinstance(raw, (int, float)):
                raw = [raw]
            if not isinstance(raw, list):
                continue
            ops = []
            for x in raw:
                try:
                    ops.append(int(x))
                except (TypeError, ValueError):
                    continue
            if op and ops:
                return Selection(op=op, operands=ops,
                                 note=str(obj.get("note", ""))[:120])
    return None


def synthesize(sel: Selection, candidates, route: dict):
    """(answer, pandas_query, error). candidates = shortlist Candidate objects."""
    err = sel.valid_for(len(candidates))
    if err:
        return None, "", err
    picks = [candidates[i - 1] for i in sel.operands]
    try:
        stable_cells = [_stable_cell_key(candidate) for candidate in picks]
    except ValueError as exc:
        return None, "", f"ValueError: {exc}"
    # Fact-aware allocation can expose the same physical cell more than once
    # under different shortlist indices/fact slots.  Counting or ranking those
    # aliases as independent observations silently inflates counts and can make
    # a requested fact year look grounded even though both operands dereference
    # the same submitted CSV cell.  Fail closed for every operation.
    if len(set(stable_cells)) != len(stable_cells):
        return None, "", "duplicate stable cells are not allowed"
    q_scale = float(route.get("unit_scale", 1.0) or 1.0)
    output_type = route.get("output_type", "number")

    compatibility_error = _output_compatibility_error(sel.op, output_type)
    if compatibility_error:
        return None, "", compatibility_error

    def vnd(c):
        return float(c.value) * float(c.unit_scale)

    def expr(c):
        return cell_value_expr(c)

    def expr_vnd(c):
        # Parenthesise the scaled leaf before composing ratios.  Without these
        # parentheses ``num * scale / den * scale`` multiplies by the
        # denominator scale instead of dividing by it.
        return f"({expr(c)} * {float(c.unit_scale):g})"

    try:
        answer, query = _apply(sel.op, picks, q_scale, output_type,
                               vnd, expr, expr_vnd, route)
    except ZeroDivisionError:
        return None, "", "division by zero"
    except (ValueError, IndexError) as e:
        return None, "", f"{type(e).__name__}: {e}"
    if answer is None:
        return None, "", f"op {sel.op} not synthesisable"
    if answer != answer or abs(answer) == float("inf"):
        return None, "", "non-finite answer"
    hard_limit = _HARD_ABS_LIMIT.get(output_type)
    if hard_limit is not None and abs(float(answer)) > hard_limit:
        return (None, "",
                f"{output_type} magnitude {float(answer):g} exceeds hard "
                f"limit {hard_limit:g}")
    return round(float(answer), 2), query, None


def _apply(op, picks, q_scale, output_type, vnd, expr, expr_vnd, route):
    if op == "lookup":
        c = picks[0]
        if output_type == "percent":
            ans = percent_from_cell(c.value, c.label, c.col_name)
            mul = "" if cell_is_already_percent(c.label, c.col_name, c.value) else " * 100"
            return ans, f"round({expr(c)}{mul}, 2)"
        return vnd(c) / q_scale, f"round({expr_vnd(c)} / {q_scale:g}, 2)"

    if op == "difference":
        a, b = picks[0], picks[1]
        if output_type in ("percent", "percentage_point"):
            return (vnd(a) - vnd(b)), f"round({expr_vnd(a)} - {expr_vnd(b)}, 2)"
        return ((vnd(a) - vnd(b)) / q_scale,
                f"round(({expr_vnd(a)} - {expr_vnd(b)}) / {q_scale:g}, 2)")

    if op == "percentage_point":
        a, b = picks[0], picks[1]
        return (float(a.value) - float(b.value),
                f"round({expr(a)} - {expr(b)}, 2)")

    if op == "growth_pct":
        end, base = picks[0], picks[1]
        if vnd(base) == 0:
            raise ZeroDivisionError
        return ((vnd(end) - vnd(base)) / abs(vnd(base)) * 100.0,
                f"round(({expr_vnd(end)} - {expr_vnd(base)}) "
                f"/ abs({expr_vnd(base)}) * 100, 2)")

    if op in ("ratio", "margin", "ratio_times"):
        num, den = picks[0], picks[1]
        if vnd(den) == 0:
            raise ZeroDivisionError
        if op == "ratio_times" or output_type == "ratio":
            return vnd(num) / vnd(den), f"round({expr_vnd(num)} / {expr_vnd(den)}, 2)"
        return (vnd(num) / vnd(den) * 100.0,
                f"round({expr_vnd(num)} / {expr_vnd(den)} * 100, 2)")

    if op == "count":
        # The model selected the cells satisfying the question's simple
        # condition; count those selections.  Each term still dereferences its
        # exact submitted cell, so the answer is grounded and replayable rather
        # than a naked constant.
        terms = [f"(1 + 0 * {expr(c)})" for c in picks]
        return float(len(picks)), f"round(sum([{', '.join(terms)}]), 2)"

    if op in ("argmax", "argmin") or (
            output_type == "year" and op in ("ranking_max", "ranking_min")):
        years = [_candidate_year(c, route) for c in picks]
        if any(y is None for y in years):
            raise ValueError("year projection needs an unambiguous year per operand")
        if len(set(years)) != len(years):
            raise ValueError("year projection operands must map to distinct years")
        vals = [vnd(c) for c in picks]
        want_max = op in ("argmax", "ranking_max")
        chosen_value = max(vals) if want_max else min(vals)
        chosen_index = vals.index(chosen_value)
        fn = "max" if want_max else "min"
        exprs = [expr_vnd(c) for c in picks]
        values = ", ".join(exprs)
        # Python's list.index and min/max both choose the first tied operand,
        # matching the deterministic calculation above.
        query = (f"float({years!r}[[{values}].index("
                 f"{fn}({values}))])")
        return float(years[chosen_index]), query

    if op in ("sum", "average", "ranking_max", "ranking_min"):
        vals = [vnd(c) for c in picks]
        exprs = [expr_vnd(c) for c in picks]
        if op == "sum":
            return sum(vals) / q_scale, f"round(({' + '.join(exprs)}) / {q_scale:g}, 2)"
        if op == "average":
            return (sum(vals) / len(vals) / q_scale,
                    f"round(({' + '.join(exprs)}) / {len(exprs)} / {q_scale:g}, 2)")
        fn = "max" if op == "ranking_max" else "min"
        chosen = max(vals) if op == "ranking_max" else min(vals)
        return chosen / q_scale, f"round({fn}({', '.join(exprs)}) / {q_scale:g}, 2)"

    return None, ""


def cell_value_expr(candidate) -> str:
    """Return an expression addressing one stable tidy-CSV cell.

    ``Candidate.row`` and ``Candidate.col`` are the exact coordinates emitted
    into the submitted CSV.  Label substring matching is deliberately avoided:
    a label such as ``Các khoản tương đương tiền`` also occurs inside ``Tiền và
    các khoản tương đương tiền`` and made ``.iloc[0]`` read the wrong row.
    """
    var, row, col = _stable_cell_key(candidate)
    return (f"float({var}.loc[({var}['row'] == {row}) & "
            f"({var}['col'] == {col}), 'value'].iloc[0])")


def _stable_cell_key(candidate) -> tuple[str, int, int]:
    """Return the exact submitted-cell identity used by generated queries."""
    var = str(getattr(candidate, "var", ""))
    if not re.fullmatch(r"df\d+", var):
        raise ValueError(f"invalid dataframe variable {var!r}")
    try:
        row = int(getattr(candidate, "row"))
        col = int(getattr(candidate, "col"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("candidate is missing stable row/col coordinates") from exc
    return var, row, col


def _output_compatibility_error(op: str, output_type: str) -> str | None:
    if output_type == "year" and op not in {
            "argmax", "argmin", "ranking_max", "ranking_min"}:
        return f"output_type=year is incompatible with op {op}"
    if output_type == "count" and op != "count":
        return f"output_type=count is incompatible with op {op}"
    if op == "count" and output_type != "count":
        return f"op count requires output_type=count, got {output_type}"
    if op in {"argmax", "argmin"} and output_type != "year":
        return f"op {op} requires output_type=year, got {output_type}"
    if op == "ratio_times" and output_type != "ratio":
        return f"op ratio_times requires output_type=ratio, got {output_type}"
    if op == "percentage_point" and output_type != "percentage_point":
        return ("op percentage_point requires output_type=percentage_point, "
                f"got {output_type}")
    return None


def _candidate_year(candidate, route: dict) -> int | None:
    """Resolve the period represented by a concrete selected cell.

    ``fact_year`` is authoritative because a report for Y+1 can contain the Y
    comparative column.  ``report_year`` must therefore never be used as the
    projected answer.  The column header and a single routed year are safe
    fallbacks for legacy/global candidates.
    """
    value = getattr(candidate, "fact_year", None)
    try:
        if value is not None:
            year = int(value)
            if 1900 <= year <= 2100:
                return year
    except (TypeError, ValueError):
        pass

    matches = re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)",
                         str(getattr(candidate, "col_name", "")))
    if len(set(matches)) == 1:
        return int(matches[0])

    years = []
    for raw in route.get("years") or []:
        try:
            year = int(raw)
        except (TypeError, ValueError):
            continue
        if year not in years:
            years.append(year)
    return years[0] if len(years) == 1 else None


def confidence(sel: Selection, candidates, answer: float, route: dict) -> float:
    """Confidence for arbitration: driven by the picked cells' match scores."""
    picks = [candidates[i - 1] for i in sel.operands
             if 1 <= i <= len(candidates)]
    if not picks:
        return 0.0
    base = min(float(c.score) for c in picks)
    if check_answer_unit(answer, route.get("output_type", "number")):
        base -= 20.0
    return max(0.0, min(95.0, base))
