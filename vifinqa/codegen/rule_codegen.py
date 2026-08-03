"""Deterministic rule-based Text-to-Pandas fallback / fast path.

Good for single-fact lookups (the vast majority of ViFinQA questions):
fuzzy-match the metric phrase against row labels of the retrieved tables,
pick the year column, emit a reproducible one-line pandas expression.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..utils.viet_text import label_metric_score, norm
from ..retrieval.serialize import df_roundtrip


@dataclass
class RuleAnswer:
    ok: bool
    answer: float = 0.0
    pandas_query: str = ""
    var: str = ""
    confidence: float = 0.0
    detail: str = ""


def _year_col_score(col_name: str, year: int) -> int:
    cn = str(col_name)
    if re.search(rf"31\s*/\s*12\s*/\s*{year}", cn):
        return 3
    if str(year) in cn:
        return 2
    if norm(cn) in ("nam nay", "cuoi nam") and year:  # weak
        return 1
    return 0


def try_rule_answer(route: dict, tables: list[dict], min_label: float = 78.0) -> RuleAnswer:
    """tables: [{var, report_id, table_pos, report_year, csv_text, unit_scale}]"""
    metric = route.get("metric_norm", "")
    years = route.get("years") or []
    q_scale = float(route.get("unit_scale", 1.0))
    output_type = route.get("output_type") or (
        "percent" if route.get("is_percent") else "number"
    )
    if output_type != "number" or route.get("growth") or not metric:
        return RuleAnswer(ok=False, detail="needs multi-step reasoning -> LLM")
    target_year = years[0] if years else None

    best = None  # (score, var, label, col, value, unit_scale, exact_label)
    for t in tables:
        df = df_roundtrip(t["csv_text"])
        if not len(df):
            continue
        report_year = t.get("report_year")
        for label in df["label"].unique():
            if not label or len(str(label)) < 4:
                continue
            s_lab = label_metric_score(str(label), metric)
            if s_lab < min_label:
                continue
            sub = df[df["label"] == label]
            # choose column
            col_best = None  # (col_score, col, value, unit_scale)
            for r in sub.itertuples():
                cs = _year_col_score(r.col_name, target_year) if target_year else 0
                if cs == 0 and target_year and report_year:
                    # positional heuristic: first numeric col = report_year,
                    # second = report_year - 1
                    cols = sorted(sub["col"].unique())
                    if target_year == report_year and r.col == cols[0]:
                        cs = 1
                    elif target_year == report_year - 1 and len(cols) > 1 and r.col == cols[1]:
                        cs = 1
                if col_best is None or cs > col_best[0]:
                    col_best = (cs, int(r.col), float(r.value), float(r.unit_scale))
            if col_best is None or (target_year and col_best[0] == 0):
                continue
            score = s_lab + 10 * col_best[0]
            # prefer tables whose unit was detected confidently
            if t.get("unit_source") in ("explicit", "header"):
                score += 5
            # magnitude sanity: a money line item asked in trieu/ty dong is
            # almost never < 1e6 absolute VND -> likely a wrong-unit table
            if q_scale >= 1e6 and abs(col_best[2] * col_best[3]) < 1e6:
                score -= 25
            if best is None or score > best[0]:
                best = (score, t["var"], str(label), col_best[1], col_best[2],
                        col_best[3], s_lab)
    if best is None:
        return RuleAnswer(ok=False, detail="no confident label/column match")

    _score, var, label, col, value, us, s_lab = best
    answer = round(value * us / q_scale, 2)
    frag = _distinct_fragment(label)
    query = (f"round(float({var}.loc[{var}['label'].str.contains({frag!r}, case=False, "
             f"regex=False, na=False) & ({var}['col'] == {col}), 'value'].iloc[0]) "
             f"* {us:g} / {q_scale:g}, 2)")
    return RuleAnswer(ok=True, answer=answer, pandas_query=query, var=var,
                      confidence=min(99.0, _score), detail=f"label='{label}' fuzz={s_lab:.0f}")


def _distinct_fragment(label: str, max_len: int = 40) -> str:
    """A stable substring of the label for str.contains (avoid regex chars)."""
    s = re.sub(r"\s+", " ", str(label)).strip()
    return s[:max_len]
