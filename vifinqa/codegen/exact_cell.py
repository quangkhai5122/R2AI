"""Exact-cell resolver for note tables with row and column semantics.

Many Vietnamese financial-note tables put the entity in the row label and the
requested metric in a column header (for example, "Tỷ lệ biểu quyết"). The
normal shortlist only scores row labels, so it can miss or confuse these cells.
This resolver combines row, column, and table context and remains fail-closed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..finance.metrics_v2 import best_row_profile
from ..utils.viet_text import label_metric_score, norm
from .rule_codegen import _distinct_fragment


@dataclass(frozen=True)
class ExactCell:
    var: str
    report_id: str
    table_pos: int
    label: str
    col: int
    col_name: str
    value: float
    unit_scale: float
    score: float
    row_score: float
    col_score: float
    context_score: float

    def expr(self, q_scale: float, output_type: str) -> str:
        label = _distinct_fragment(self.label)
        if output_type == "percent":
            scale = "" if _already_percent(self.label, self.col_name) else " * 100"
            return (f"round(float({self.var}.loc[{self.var}['label'].str.contains(" 
                    f"{label!r}, case=False, regex=False, na=False) "
                    f"& ({self.var}['col'] == {self.col}), 'value'].iloc[0]){scale}, 2)")
        return (f"round(float({self.var}.loc[{self.var}['label'].str.contains(" 
                f"{label!r}, case=False, regex=False, na=False) "
                f"& ({self.var}['col'] == {self.col}), 'value'].iloc[0]) "
                f"* {self.unit_scale:g} / {q_scale:g}, 2)")


def _already_percent(label: str, col_name: str) -> bool:
    text = norm(f"{label} {col_name}")
    return "%" in str(label) or "phan tram" in text or "ty le" in text


def _context(table: dict) -> str:
    raw = str(table.get("context") or "")
    if raw:
        return norm(raw)
    try:
        grid = json.loads(table.get("grid_json") or "[]")
    except (TypeError, ValueError):
        grid = []
    if not isinstance(grid, list):
        return ""
    return norm(" ".join(" ".join(str(x) for x in row) for row in grid[:4]
                         if isinstance(row, list)))


def _entity_terms(route: dict) -> list[str]:
    text = norm(route.get("question", ""))
    # Company/ticker names are already removed from metric_norm by the router;
    # retain named-note targets that are not the issuer itself.
    terms = []
    for phrase in (route.get("metric_variants") or []):
        p = norm(phrase)
        if len(p.split()) >= 2 and any(w in p for w in ("chu ", "visorutex", "bao viet", "nhan tho")):
            terms.append(p)
    return terms


def _column_query(route: dict) -> list[str]:
    text = norm(route.get("question", ""))
    variants = [norm(x) for x in (route.get("metric_variants") or []) if norm(x)]
    out = []
    for phrase in variants:
        for token in ("ty le bieu quyet", "ty le loi ich", "gia tri con lai",
                      "hao mon luy ke", "nguyen gia", "phai tra", "so da nop",
                      "so phat sinh"):
            if token in phrase or token in text:
                out.append(token)
    if "bieu quyet" in text:
        out.append("ty le bieu quyet")
    if "loi ich" in text and "bieu quyet" not in text:
        out.append("ty le loi ich")
    if "phai tra" in text or "phai nop" in text:
        out.append("phai tra")
    if any(x in text for x in ("cuoi nam", "cuoi ky", "den ngay", "tai ngay")):
        out.append("31/12")
        out.extend(str(y) for y in (route.get("years") or []))
    if "gia tri con lai" in text:
        out.append("gia tri con lai")
    # Remuneration tables often encode the metric only in the section/context;
    # the requested year is then the value-column discriminator.
    if not out and any(key in (route.get("metric_profile_keys") or [])
                       for key in ("employee_remuneration", "named_board_remuneration")):
        out.extend(str(y) for y in (route.get("years") or []))
    return list(dict.fromkeys(out))


def _entity_anchor(route: dict) -> str:
    text = norm(" ".join(str(x) for x in (route.get("metric_variants") or [])))
    for phrase in ("chu thi binh", "visorutex"):
        if phrase in text:
            return phrase
    return ""


def _row_query(route: dict) -> list[str]:
    variants = [norm(x) for x in (route.get("metric_variants") or []) if norm(x)]
    out = list(variants)
    text = norm(route.get("question", ""))
    if "thue thu nhap" in text:
        out.extend(("thue thu nhap doanh nghiep", "thue tndn"))
    return list(dict.fromkeys(out))


def _column_score(col_name: str, wanted: list[str]) -> float:
    if not wanted:
        return 0.0
    cn = norm(col_name)
    scores = []
    for phrase in wanted:
        phrase = norm(phrase)
        if phrase == "31/12" and re.search(r"31\s*/\s*12", cn):
            scores.append(100.0)
        elif re.fullmatch(r"20\d{2}", phrase) and re.search(rf"(?<!\d){phrase}(?!\d)", cn):
            scores.append(100.0)
        else:
            scores.append(label_metric_score(cn, phrase))
    return max(scores, default=0.0)


def _row_score(label: str, wanted: list[str]) -> float:
    if not wanted:
        return 0.0
    return max(label_metric_score(label, phrase) for phrase in wanted)


def _column_year_matches(col_name: str, years: list[int]) -> bool:
    if not years:
        return True
    found = {int(x) for x in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", str(col_name))}
    return not found or bool(found & {int(y) for y in years})


def _semantic_row_guard(question: str, label: str) -> bool:
    q, row = norm(question), norm(label)
    if "phai tra" in q and "tra truoc" in row:
        return False
    if "tra truoc" in q and "phai tra" in row:
        return False
    if "du phong" not in q and "du phong" in row:
        return False
    if "gia goc" in q and "du phong" in row:
        return False
    return True


def _period_bonus(question: str, label: str, col_name: str = "") -> float:
    q, row, col = norm(question), norm(label), norm(col_name)
    if any(x in q for x in ("cuoi nam", "cuoi ky", "den ngay", "tai ngay")):
        if any(x in row for x in ("so cuoi", "cuoi nam", "cuoi ky")):
            return 18.0
        if "31/12" in col or "cuoi nam" in col or "cuoi ky" in col:
            return 18.0
        return -8.0
    if any(x in q for x in ("dau nam", "dau ky")):
        if any(x in row for x in ("so dau", "dau nam", "dau ky")):
            return 18.0
        if "dau nam" in col or "dau ky" in col:
            return 18.0
        return -8.0
    return 0.0


def resolve_exact_cell(route: dict, tables: list[dict], min_score: float = 72.0) -> ExactCell | None:
    """Resolve an exact note cell, or return None without guessing."""
    if not route.get("metric_profile_keys"):
        return None
    if (route.get("plan") or {}).get("op", "lookup") != "lookup":
        return None
    output_type = route.get("output_type", "number")
    row_wanted, col_wanted = _row_query(route), _column_query(route)
    if not row_wanted or not col_wanted:
        return None
    years = route.get("years") or []
    candidates = []
    for table in tables:
        try:
            import pandas as pd
            df = pd.read_csv(__import__("io").StringIO(table["csv_text"]))
        except Exception:
            continue
        context = _context(table)
        for label in df["label"].dropna().astype(str).unique():
            if not _semantic_row_guard(route.get("question", ""), label):
                continue
            rs = _row_score(label, row_wanted)
            sub = df[df["label"].astype(str) == label]
            for rec in sub.itertuples():
                col_name = str(rec.col_name)
                if not _column_year_matches(col_name, years):
                    continue
                cs = _column_score(col_name, col_wanted)
                if cs < 28:
                    continue
                # A note table can encode the requested metric in its column
                # header while the row carries only an entity or period. The
                # combined schema view is used solely for the profile contract;
                # row/column scores still prevent broad context-only matches.
                schema_text = f"{context} {label} {col_name}"
                profile_text = schema_text
                profile_keys = route.get("metric_profile_keys") or []
                if (any(key in profile_keys for key in
                        ("employee_remuneration", "named_board_remuneration"))
                        and any(token in context for token in
                                ("nhan su chu chot", "luong", "thuong", "tro cap"))):
                    profile_text += " thu lao thanh vien hoi dong quan tri"
                if ("current_income_tax_payable" in profile_keys
                        and ("phai tra" in norm(col_name)
                             or any(x in norm(route.get("question", ""))
                                    for x in ("phai tra", "phai nop")))):
                    profile_text += " thue thu nhap doanh nghiep phai nop"
                anchor = _entity_anchor(route)
                if anchor and anchor not in norm(label):
                    continue
                profile, bonus, _ = best_row_profile(
                    route, profile_text, str(getattr(rec, "code", "")), col_name,
                    qualifier_text=f"{label} {col_name}")
                if profile is None:
                    continue
                context_score = max((label_metric_score(
                    f"{context} {col_name}", phrase) for phrase in row_wanted), default=0.0)
                entity_score = max(rs, context_score)
                if entity_score < 28:
                    continue
                # Prefer exact table context and explicit requested year.
                year_bonus = 10.0 if any(str(y) in col_name for y in years) else 0.0
                period_bonus = _period_bonus(route.get("question", ""), label, col_name)
                anchor_bonus = 15.0 if anchor and anchor in norm(label) else 0.0
                score = (0.38 * entity_score + 0.32 * cs
                         + 0.10 * context_score + bonus + year_bonus
                         + period_bonus + anchor_bonus)
                candidates.append(ExactCell(
                    var=table["var"], report_id=table["report_id"],
                    table_pos=int(table["table_pos"]), label=label,
                    col=int(rec.col), col_name=str(rec.col_name),
                    value=float(rec.value), unit_scale=float(rec.unit_scale),
                    score=score, row_score=rs, col_score=cs,
                    context_score=context_score))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c.score, reverse=True)
    best = candidates[0]
    runner = candidates[1].score if len(candidates) > 1 else 0.0
    same_cell = False
    if len(candidates) > 1:
        other = candidates[1]
        same_cell = (norm(other.label) == norm(best.label)
                     and other.col == best.col
                     and abs(other.value - best.value) <= 0.011)
    if best.score < min_score or (runner and best.score - runner < 6.0 and not same_cell):
        return None
    return best
