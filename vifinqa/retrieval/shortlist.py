"""Row-level schema linking: turn N wide tables into a short list of candidate
cells, so the LLM picks from ~8 options instead of searching ~40 rows x 4 tables.

This is the text-to-SQL "schema linking" step (CHESS / DIN-SQL) applied to
pandas: narrowing BEFORE generation is what makes small models reliable.

Scoring per row label = max over metric variants of:
    lexical  : label_metric_score (token coverage + order-insensitive ratio)
    semantic : cosine(BGE-M3 label, BGE-M3 metric)   [optional, see dense.py]
plus small bonuses for a matching VAS code and for the qualifier
(ngan han / dai han / hop nhat) agreeing with the question.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from ..finance.metrics import code_expectation, expand_metric_variants
from ..router.metric_phrase import has_qualifier
from ..utils.viet_text import label_metric_score, norm
from .serialize import df_roundtrip

_QUALIFIERS = ("ngan han", "dai han", "hop nhat", "rieng")


@dataclass
class Candidate:
    var: str
    report_id: str
    table_pos: int
    row: int
    label: str
    code: str
    col: int
    col_name: str
    value: float
    unit_scale: float
    score: float
    lexical: float
    semantic: float

    def to_dict(self) -> dict:
        return asdict(self)


def build_shortlist(tables: list[dict], metric_variants: list[str],
                    years: list[int] | None = None, top_n: int = 8,
                    encoder=None, min_score: float = 35.0) -> list[Candidate]:
    """tables: bundle tables ({var, report_id, table_pos, csv_text, ...}).

    Returns the best `top_n` (label, column) cells across all tables, sorted by
    descending score. One entry per (table, label, chosen column).
    """
    metric_variants = expand_metric_variants(m for m in metric_variants if m)
    if not metric_variants:
        return []
    want_qual = _qualifiers_in(metric_variants[0])

    sem_lookup = {}
    if encoder is not None:
        all_labels = []
        for t in tables:
            df = df_roundtrip(t["csv_text"])
            all_labels.extend(str(l) for l in df["label"].unique()
                              if isinstance(l, str) and len(l) > 3)
        if all_labels:
            sem_lookup = encoder.similarity(metric_variants, sorted(set(all_labels)))

    out: list[Candidate] = []
    for t in tables:
        df = df_roundtrip(t["csv_text"])
        if not len(df):
            continue
        for label in df["label"].unique():
            if not isinstance(label, str) or len(label) <= 3:
                continue
            lex = max(label_metric_score(label, m) for m in metric_variants)
            sem = float(sem_lookup.get(label, 0.0)) * 100.0
            score = max(lex, sem) + 0.25 * min(lex, sem)
            score += _qualifier_bonus(label, want_qual)
            if score < min_score:
                continue
            # Prefer the TIGHTEST label covering the metric: extra qualifier
            # tokens change the meaning ("Lợi nhuận sau thuế" vs "... chưa phân
            # phối"), yet token-coverage alone scores both 100.
            score -= _extra_token_penalty(label, metric_variants)

            sub = df[df["label"] == label]
            pick = _pick_column(sub, years, t.get("report_year"))
            if pick is None:
                continue
            row_i, col, col_name, value, unit_scale, code = pick
            # A column header naming the asked year is hard evidence. A header
            # naming a DIFFERENT year is evidence of the WRONG period, which
            # silently corrupts multi-company comparisons (one company read at
            # 2015, another at 2014) - so it is punished much harder than a
            # header with no year at all.
            ys = _year_status(col_name, years)
            score += {"match": 10.0, "none": -6.0, "other": -30.0}[ys]
            # VAS line code is far more stable than OCR'd Vietnamese text:
            # it separates "Lợi nhuận sau thuế" (code 60, income statement) from
            # "Lợi nhuận sau thuế chưa phân phối" (code 421, balance sheet).
            score += _code_bonus(code, metric_variants, label)
            if score < min_score:      # re-check: the adjustments can sink a row
                continue
            out.append(Candidate(
                var=t["var"], report_id=t["report_id"], table_pos=int(t["table_pos"]),
                row=int(row_i), label=str(label), code=str(code), col=int(col),
                col_name=str(col_name), value=float(value),
                unit_scale=float(unit_scale), score=round(float(score), 1),
                lexical=round(float(lex), 1), semantic=round(float(sem), 1)))
    out.sort(key=lambda c: -c.score)
    return _dedupe(out)[:top_n]


def _dedupe(cands: list[Candidate]) -> list[Candidate]:
    seen, out = set(), []
    for c in cands:
        key = (c.report_id, c.table_pos, norm(c.label), c.col)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _extra_token_penalty(label: str, metric_variants: list[str],
                         per_token: float = 3.0, cap: float = 18.0) -> float:
    from ..utils.viet_text import tokens
    lt = set(tokens(label))
    best = min((len(lt - set(tokens(m))) for m in metric_variants), default=0)
    return min(cap, per_token * max(0, best - 1))


def _year_status(col_name: str, years: list[int] | None) -> str:
    """'match' | 'other' (header names a different year) | 'none'."""
    if not years:
        return "match"
    cn = str(col_name)
    if any(re.search(rf"(?<!\d){y}(?!\d)", cn) for y in years):
        return "match"
    other = re.search(r"(?<!\d)(19|20)\d{2}(?!\d)", cn)
    return "other" if other else "none"


# Thong tu 200 line codes: the OCR-stable identity of a financial line item.
def _code_bonus(code: str, metric_variants: list[str], label: str = "") -> float:
    expected, known_mismatch = code_expectation(metric_variants, label)
    if known_mismatch:
        return -12.0
    if not expected:
        return 0.0
    c = str(code or "").strip().rstrip(".0") or str(code or "").strip()
    c = re.sub(r"\.0$", "", str(code or "").strip())
    if not c:
        return 0.0
    if c in expected:
        return 12.0
    return -10.0 if c.isdigit() else 0.0


def _qualifiers_in(metric: str) -> set[str]:
    return {q for q in _QUALIFIERS if q in metric}


def _qualifier_bonus(label: str, want: set[str]) -> float:
    """A wrong maturity qualifier means a semantically WRONG row, so the penalty
    must outweigh the lexical gain those extra shared tokens give
    ("...ngắn hạn" vs "...dài hạn" differ by one token but are different lines).
    """
    have = _qualifiers_in(norm(label))
    if not want and not have:
        return 0.0
    if want & have:
        return 8.0
    if want and have:
        return -25.0       # "ngan han" asked but row says "dai han"
    return -3.0 if want else 0.0


_CURRENT_PERIOD_HEADERS = (
    "nam nay", "ky nay", "so cuoi nam", "cuoi nam", "cuoi ky",
    "tai ngay ket thuc", "nam ket thuc ngay", "nam tai chinh ket thuc ngay",
    "tong cong", "current year", "current period", "ending balance", "total",
)
_PRIOR_PERIOD_HEADERS = (
    "nam truoc", "ky truoc", "so dau nam", "dau nam", "dau ky",
    "prior year", "previous year", "beginning balance",
)
_METADATA_HEADERS = (
    "thuyet minh", "ma so", "chi tieu", "noi dung", "dien giai", "stt",
    "thuyetminh", "maso", "chitieu", "item", "code", "note",
)


def _period_kind(col_name: str) -> str:
    """Classify an original table header as current/prior/metadata/unknown."""
    cn = norm(str(col_name or ""))
    if not cn or cn == "nan":
        return "metadata"
    if any(marker in cn for marker in _PRIOR_PERIOD_HEADERS):
        return "prior"
    if any(marker in cn for marker in _CURRENT_PERIOD_HEADERS):
        return "current"
    if any(marker == cn or marker in cn for marker in _METADATA_HEADERS):
        return "metadata"
    return "unknown"


def _column_score(col_name: str, years: list[int] | None,
                  report_year: int | None) -> int:
    """Rank value columns using explicit years and financial-period headers."""
    cn = str(col_name or "")
    requested = {int(y) for y in (years or [])}

    # A literal requested year beats every inferred current/prior convention.
    for y in requested:
        if re.search(rf"31\s*/\s*12\s*/\s*{y}", cn):
            return 100
        if re.search(rf"(?<!\d){y}(?!\d)", cn):
            return 95
    if re.search(r"(?<!\d)(19|20)\d{2}(?!\d)", cn):
        return -40

    kind = _period_kind(cn)
    if kind == "metadata":
        return -100

    current_wanted = not requested or report_year is None or report_year in requested
    prior_wanted = (report_year is not None and
                    any(int(report_year) == y + 1 for y in requested))
    if kind == "current":
        return 80 if current_wanted and not prior_wanted else 20
    if kind == "prior":
        return 80 if prior_wanted else 10
    return 30


def _pick_column(sub, years: list[int] | None,
                 report_year: int | None = None):
    """Choose the requested financial period, excluding code/note columns."""
    best = None
    for r in sub.itertuples():
        cn = str(r.col_name)
        cs = _column_score(cn, years, report_year)
        # For equally informative headers, retain the original left-to-right
        # convention among actual value columns.
        rank = (cs, -int(r.col))
        if best is None or rank > best[0]:
            best = (rank, r.Index, int(r.col), cn, float(r.value),
                    float(r.unit_scale), str(getattr(r, "code", "")))
    if best is None:
        return None
    _rank, idx, col, cn, val, us, code = best
    row_i = int(sub.loc[idx, "row"]) if "row" in sub.columns else 0
    return row_i, col, cn, val, us, code


def render_shortlist(cands: list[Candidate], unit_name: str = "") -> str:
    """Compact text block for the prompt."""
    if not cands:
        return "(no candidate row matched the metric — search the tables yourself)"
    lines = ["idx | var | label | code | col | col_name | value | unit_scale"]
    for i, c in enumerate(cands, 1):
        lines.append(f"{i} | {c.var} | {c.label[:60]} | {c.code or '-'} | {c.col} | "
                     f"{c.col_name[:22] or '-'} | {c.value:g} | {c.unit_scale:g}")
    return "\n".join(lines)
