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
    # Provenance shown to the selection model. ``fact_slot`` is empty for the
    # legacy/global shortlist and F1/F2/... for a fact-aware shortlist.
    ticker: str = ""
    report_year: int | None = None
    fact_slot: str = ""
    fact_ticker: str = ""
    fact_year: int | None = None
    fact_metric: str = ""
    fact_role: str = "value"

    def to_dict(self) -> dict:
        return asdict(self)


def build_shortlist(tables: list[dict], metric_variants: list[str],
                    years: list[int] | None = None, top_n: int = 8,
                    encoder=None, min_score: float = 35.0,
                    facts: list[dict] | None = None) -> list[Candidate]:
    """Build a global or fact-aware shortlist of concrete table cells.

    ``facts`` activates per-(ticker, year, metric) groups. Each non-empty fact
    gets a fair share of the prompt budget, so a high-scoring company/report
    cannot starve the other operands of a composite question. The legacy call
    shape remains valid and preserves its ranking for zero/one requested year.
    """
    facts = [f for f in (facts or []) if isinstance(f, dict)]
    requested_years = list(dict.fromkeys(years or []))
    if not facts and len(requested_years) <= 1:
        cands = _build_shortlist_legacy(
            tables, metric_variants, requested_years, top_n, encoder, min_score,
        )
        return _attach_metadata(cands, tables)

    groups: list[list[Candidate]] = []
    if facts:
        for idx, fact in enumerate(facts, 1):
            scoped = _tables_for_fact(tables, fact)
            variants = _variants_for_fact(fact, metric_variants)
            fact_year = _as_year(fact.get("year"))
            group = _build_shortlist_legacy(
                scoped, variants, [fact_year] if fact_year is not None else [],
                top_n, encoder, min_score,
            )
            groups.append(_attach_metadata(
                group, scoped, fact=fact, fact_slot=f"F{idx}",
            ))
    else:
        # Preserve a separate candidate for each requested period even if both
        # values are columns in the same physical table.
        for idx, year in enumerate(requested_years, 1):
            group = _build_shortlist_legacy(
                tables, metric_variants, [year], top_n, encoder, min_score,
            )
            groups.append(_attach_metadata(
                group, tables, fact={"year": year}, fact_slot=f"Y{idx}",
            ))

    # Grow beyond the historical 12 candidates only when necessary to avoid
    # silently omitting entire fact groups; still bound prompt size at 24.
    effective_n = max(top_n, min(len(groups), 24))
    return _allocate_per_fact(groups, effective_n)


def _build_shortlist_legacy(tables: list[dict], metric_variants: list[str],
                            years: list[int] | None = None, top_n: int = 8,
                            encoder=None, min_score: float = 35.0) -> list[Candidate]:
    """tables: bundle tables ({var, report_id, table_pos, csv_text, ...}).

    Returns the best `top_n` (label, column) cells across all tables, sorted by
    descending score. One entry per (table, label, chosen column).
    """
    metric_variants = [m for m in metric_variants if m]
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
            pick = _pick_column(sub, years)
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
            score += _code_bonus(code, metric_variants)
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


def _attach_metadata(cands: list[Candidate], tables: list[dict],
                     fact: dict | None = None,
                     fact_slot: str = "") -> list[Candidate]:
    """Attach provenance that used to be lost before rendering the prompt."""
    by_key = {
        (str(t.get("report_id", "")), int(t.get("table_pos", -1)),
         str(t.get("var", ""))): t
        for t in tables
    }
    for cand in cands:
        table = by_key.get((cand.report_id, cand.table_pos, cand.var), {})
        cand.ticker = str(table.get("ticker") or
                          _ticker_from_report(cand.report_id))
        cand.report_year = _as_year(table.get("report_year"))
        cand.fact_slot = fact_slot
        cand.fact_ticker = str((fact or {}).get("ticker") or "")
        cand.fact_year = _as_year((fact or {}).get("year"))
        cand.fact_metric = str((fact or {}).get("metric") or "")
        cand.fact_role = str((fact or {}).get("role") or "value")
    return cands


def _tables_for_fact(tables: list[dict], fact: dict) -> list[dict]:
    ticker = str(fact.get("ticker") or "").upper()
    year = _as_year(fact.get("year"))
    out = []
    for table in tables:
        table_ticker = str(table.get("ticker") or
                           _ticker_from_report(table.get("report_id", ""))).upper()
        if ticker and table_ticker != ticker:
            continue
        report_year = _as_year(table.get("report_year"))
        # A FY-Y value can be the current column of report Y or the comparison
        # column of report Y+1.
        if year is not None and report_year is not None \
                and report_year not in (year, year + 1):
            continue
        out.append(table)
    return out


def _variants_for_fact(fact: dict, global_variants: list[str]) -> list[str]:
    metric = str(fact.get("metric") or "").strip()
    variants = [str(v) for v in global_variants if v]
    if not metric:
        return variants
    # Ratio/margin plans split the original phrase into two different metrics;
    # do not contaminate each fact with the combined numerator/denominator text.
    if any(norm(metric) == norm(v) for v in variants):
        return list(dict.fromkeys([metric, *variants]))
    return [metric]


def _allocate_per_fact(groups: list[list[Candidate]], top_n: int) -> list[Candidate]:
    """Reserve an even quota for each non-empty fact, then fill by score."""
    nonempty = [(idx, group) for idx, group in enumerate(groups) if group]
    if not nonempty or top_n <= 0:
        return []
    quota = max(1, top_n // len(nonempty))
    chosen: list[Candidate] = []
    chosen_ids = set()
    group_of = {}
    for idx, group in nonempty:
        for cand in group:
            group_of[id(cand)] = idx
        for cand in group[:quota]:
            chosen.append(cand)
            chosen_ids.add(id(cand))
    remaining = sorted(
        (cand for _idx, group in nonempty for cand in group
         if id(cand) not in chosen_ids),
        key=lambda c: -c.score,
    )
    chosen.extend(remaining[:max(0, top_n - len(chosen))])
    chosen.sort(key=lambda c: (group_of[id(c)], -c.score))
    return chosen[:top_n]


def _ticker_from_report(report_id: str) -> str:
    return str(report_id or "").split("_", 1)[0]


def _as_year(value) -> int | None:
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    return year if 1900 <= year <= 2100 else None


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
VAS_CODE_HINTS: list[tuple[str, set[str]]] = [
    ("doanh thu thuan", {"10"}),
    ("doanh thu ban hang va cung cap dich vu", {"01", "1"}),
    ("gia von hang ban", {"11"}),
    ("loi nhuan gop", {"20"}),
    ("chi phi tai chinh", {"22"}),
    ("chi phi ban hang", {"25"}),
    ("chi phi quan ly doanh nghiep", {"26"}),
    ("loi nhuan thuan tu hoat dong kinh doanh", {"30"}),
    ("loi nhuan truoc thue", {"50"}),
    ("loi nhuan sau thue", {"60"}),
    ("lai co ban tren co phieu", {"70"}),
    ("tien va cac khoan tuong duong tien", {"110"}),
    ("hang ton kho", {"140", "141"}),
    ("tai san ngan han", {"100"}),
    ("tai san dai han", {"200"}),
    ("tong tai san", {"270"}),
    ("tong cong tai san", {"270"}),
    ("no phai tra", {"300"}),
    ("no ngan han", {"310"}),
    ("no dai han", {"330"}),
    ("von chu so huu", {"400", "410"}),
    ("von gop cua chu so huu", {"411"}),
    ("loi nhuan sau thue chua phan phoi", {"421"}),
]


def _expected_codes(metric_variants: list[str]) -> set[str]:
    best: set[str] = set()
    best_len = 0
    for phrase, codes in VAS_CODE_HINTS:
        for m in metric_variants:
            mn = norm(m)
            if phrase in mn and len(phrase) > best_len:
                best, best_len = codes, len(phrase)
    return best


def _code_bonus(code: str, metric_variants: list[str]) -> float:
    expected = _expected_codes(metric_variants)
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


def _pick_column(sub, years: list[int] | None):
    """Choose the column matching the asked year; fall back to the first one."""
    best = None
    cols = sorted(sub["col"].unique())
    for r in sub.itertuples():
        cs = 0
        cn = str(r.col_name)
        for y in (years or []):
            if re.search(rf"31\s*/\s*12\s*/\s*{y}", cn):
                cs = max(cs, 3)
            elif re.search(rf"(?<!\d){y}(?!\d)", cn):
                cs = max(cs, 2)
        if cs == 0 and cols and int(r.col) == int(cols[0]):
            cs = 1                       # positional default = current period
        if best is None or cs > best[0]:
            best = (cs, r.Index, int(r.col), cn, float(r.value),
                    float(r.unit_scale), str(getattr(r, "code", "")))
    if best is None:
        return None
    _cs, idx, col, cn, val, us, code = best
    row_i = int(sub.loc[idx, "row"]) if "row" in sub.columns else 0
    return row_i, col, cn, val, us, code


def render_shortlist(cands: list[Candidate], unit_name: str = "") -> str:
    """Compact text block for the prompt."""
    if not cands:
        return "(no candidate row matched the metric — search the tables yourself)"
    lines = [
        "idx | fact | ticker | report_year | report_id | var | label | code | "
        "col | col_name | value | unit_scale"
    ]
    for i, c in enumerate(cands, 1):
        lines.append(
            f"{i} | {c.fact_slot or '-'} | {c.ticker or '-'} | "
            f"{c.report_year or '-'} | {c.report_id} | {c.var} | "
            f"{c.label[:60]} | {c.code or '-'} | {c.col} | "
            f"{c.col_name[:22] or '-'} | {c.value:g} | {c.unit_scale:g}"
        )
    return "\n".join(lines)
