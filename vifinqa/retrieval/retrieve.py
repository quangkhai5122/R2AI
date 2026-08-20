"""Retrieval = structured lookup (router) + BM25 inside the locked report(s).

Writes retrieval.jsonl: one record per question with the route and ranked
candidate tables.
"""
from __future__ import annotations

from pathlib import Path

from ..config import RETRIEVE_DEPTH
from ..extraction.build_store import Store
from ..finance.metrics import expand_metric_variants
from ..router.entities import StockMap
from ..router.decompose import split_ratio_metric
from ..router.router import route_question
from ..utils.io import read_jsonl, write_jsonl
from ..utils.viet_text import fuzz_token_set, norm, tokens
from .bm25 import BM25
from .serialize import table_doc_tokens, grid_of, tidy_csv_text
from .shortlist import build_shortlist

_LABEL_BOOST = 1.5
_LABEL_THRESHOLD = 88.0
_ROW_SCORE_WEIGHT = 0.18

def retrieve_for_route(route, store: Store, depth: int = RETRIEVE_DEPTH,
                       row_rerank: bool = False,
                       row_score_weight: float = _ROW_SCORE_WEIGHT) -> list[dict]:
    if not route.report_ids or not route.tickers:
        return []
    cands = []
    for ticker in route.tickers:
        tdf = store.tables_of(ticker, route.report_ids)
        if not len(tdf):
            continue
        metas = tdf.to_dict("records")
        grids = [grid_of(m) for m in metas]
        docs = [table_doc_tokens(m, g) for m, g in zip(metas, grids)]
        bm25 = BM25(docs)
        base_variants = _base_metric_variants(route)
        variants = _retrieval_metric_variants(route)
        expanded = bool(set(variants) - set(base_variants))
        rank_variants = variants if expanded else base_variants
        scores = _combined_bm25_scores(
            bm25, base_variants, variants, getattr(route, "years", []) or []
        )
        order = sorted(range(len(metas)), key=lambda i: -scores[i])[: max(depth * 3, 60)]
        row_scores = _row_scores(route, metas, order, rank_variants) if row_rerank else {}
        for i in order:
            m, g = metas[i], grids[i]
            s = scores[i]
            # number-aware-lite boost: a row label closely matching the metric
            best_lab = 0.0
            for row in g[1:60]:
                for cell in row[:3]:
                    if cell and len(cell) > 3:
                        best_lab = max(
                            best_lab,
                            max((fuzz_token_set(cell, v) for v in rank_variants),
                                default=0.0),
                        )
                        break
            if best_lab >= _LABEL_THRESHOLD:
                s *= _LABEL_BOOST
            key = (m["report_id"], int(m["table_pos"]))
            row_score = row_scores.get(key, 0.0)
            bm25_score = float(s)
            final_score = bm25_score + row_score_weight * row_score
            cands.append({
                "report_id": m["report_id"], "ticker": m["ticker"],
                "table_pos": int(m["table_pos"]), "page": int(m["page"]),
                "unit_scale": (None if m["unit_scale"] is None or m["unit_scale"] != m["unit_scale"]
                               else float(m["unit_scale"])),
                "unit_source": m["unit_source"],
                "n_rows": int(m["n_rows"]), "score": round(float(final_score), 4),
                "bm25_score": round(bm25_score, 4),
                "row_score": round(float(row_score), 1),
                "label_match": round(best_lab, 1),
            })
    cands.sort(key=lambda c: -c["score"])
    return _apply_quota(cands, route, depth)


def _base_metric_variants(route) -> list[str]:
    return _dedupe_variants(
        list(getattr(route, "metric_variants", None) or [getattr(route, "metric_norm", "")])
    )


def _retrieval_metric_variants(route) -> list[str]:
    """Metric phrases used only for table retrieval/rerank.

    Composite questions often ask one final metric after filtering/ranking on
    several other metrics. Pulling those atomic fact metrics and formula parts
    into retrieval improves table recall without changing the router's primary
    metric that downstream codegen sees.
    """
    out: list[str] = []
    for v in (getattr(route, "metric_variants", None) or [getattr(route, "metric_norm", "")]):
        out.append(str(v or ""))
    plan = getattr(route, "plan", None) or {}
    facts = plan.get("facts", []) if isinstance(plan, dict) else []
    for fact in facts:
        out.append(str(fact.get("metric") or ""))
    for v in list(out):
        num, den = split_ratio_metric(v)
        out.extend([num, den])
    return expand_metric_variants(
        _dedupe_variants(out),
        question=getattr(route, "question", "") or "",
    )


def _combined_bm25_scores(bm25: BM25, base_variants: list[str],
                          variants: list[str], years: list[int]) -> list[float]:
    """Legacy BM25 foundation, with component-metric recall as a bounded boost."""
    if not bm25.docs:
        return []
    year_tokens = [str(y) for y in years]
    base_tokens: list[str] = []
    for v in base_variants:
        base_tokens.extend(tokens(v))
    base_tokens.extend(year_tokens)
    base = bm25.scores(base_tokens)
    extras = [v for v in variants if v and v not in set(base_variants)]
    if not extras:
        return base
    if not base_tokens:
        return bm25.scores(year_tokens)
    best_extra = [0.0] * len(bm25.docs)
    for v in extras[:32]:
        q_tokens = tokens(v) + year_tokens
        scores = bm25.scores(q_tokens)
        for i, s in enumerate(scores):
            if s > best_extra[i]:
                best_extra[i] = s
    return [b + 0.35 * e for b, e in zip(base, best_extra)]


def _dedupe_variants(variants: list[str]) -> list[str]:
    seen, out = set(), []
    for v in variants:
        v = " ".join(norm(v).split())
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _row_scores(route, metas: list[dict], order: list[int],
                variants: list[str] | None = None) -> dict[tuple[str, int], float]:
    """Best row-level schema-linking score per table.

    BM25 ranks whole tables. For table retrieval we also care whether a table
    contains a concrete row/cell that looks like the asked metric and period.
    `build_shortlist` is already the row linker used by rule/codegen; reusing it
    here makes the submitted top-k closer to the evidence the answer path would
    actually use.
    """
    variants = variants or _retrieval_metric_variants(route)
    if not variants:
        return {}
    blocks = []
    for j, i in enumerate(order, start=1):
        m = metas[i]
        blocks.append({
            "var": f"df{j}",
            "report_id": m["report_id"],
            "table_pos": int(m["table_pos"]),
            "csv_text": tidy_csv_text(m),
        })
    # Ask for more rows than tables: several top rows can live in the same
    # table, and we only need the best row score per table.
    shortlist = build_shortlist(
        blocks, variants, getattr(route, "years", []) or [],
        top_n=min(80, max(12, len(blocks) * 2)),
        min_score=35.0,
        question=getattr(route, "question", "") or "",
    )
    out: dict[tuple[str, int], float] = {}
    for c in shortlist:
        key = (c.report_id, int(c.table_pos))
        out[key] = max(out.get(key, 0.0), float(c.score))
    return out


def _apply_quota(cands: list[dict], route, depth: int) -> list[dict]:
    """Dynamic evidence allocation (P1.4).

    A flat top-k starves composite questions: for "chênh lệch giữa A và B" a
    pure score ranking can return 5 tables that all belong to A. Guarantee at
    least `per_doc` slots for every locked report, then fill by score.
    """
    plan = getattr(route, "plan", None) or {}
    facts = plan.get("facts", [])
    if len(facts) <= 1 or not cands:
        return cands[:depth]
    docs = list(dict.fromkeys(c["report_id"] for c in cands))
    per_doc = max(1, depth // max(1, len(docs)))
    taken, out = {}, []
    for c in cands:
        if len(out) >= depth:
            break
        if taken.get(c["report_id"], 0) < per_doc:
            out.append(c)
            taken[c["report_id"]] = taken.get(c["report_id"], 0) + 1
    for c in cands:                      # fill the remainder by pure score
        if len(out) >= depth:
            break
        if c not in out:
            out.append(c)
    return out[:depth]


def run_retrieval(questions_path: Path, store_dir: Path, code_stock_csv: Path,
                  out_path: Path, depth: int = RETRIEVE_DEPTH, limit: int = 0,
                  row_rerank: bool = False,
                  row_score_weight: float = _ROW_SCORE_WEIGHT) -> None:
    store = Store(store_dir)
    stock = StockMap(code_stock_csv)
    questions = read_jsonl(questions_path)
    if limit:
        questions = questions[:limit]
    out = []
    try:
        from tqdm import tqdm
        it = tqdm(questions, desc="retrieve")
    except ImportError:
        it = questions
    for q in it:
        route = route_question(q["id"], q["question"], stock, store)
        cands = retrieve_for_route(
            route, store, depth, row_rerank=row_rerank,
            row_score_weight=row_score_weight,
        )
        out.append({"id": q["id"], "question": q["question"],
                    "route": route.to_dict(), "candidates": cands})
    write_jsonl(out_path, out)
    n_empty = sum(1 for r in out if not r["candidates"])
    print(f"retrieval done: {len(out)} questions, {n_empty} with no candidates")
