"""Retrieval = structured lookup (router) + BM25 inside the locked report(s).

Writes retrieval.jsonl: one record per question with the route and ranked
candidate tables.
"""
from __future__ import annotations

from pathlib import Path

from ..config import RETRIEVE_DEPTH
from ..extraction.build_store import Store
from ..router.entities import StockMap
from ..router.router import route_question
from ..utils.io import read_jsonl, write_jsonl
from ..utils.viet_text import fuzz_token_set
from .bm25 import BM25
from .serialize import table_doc_tokens, grid_of

_LABEL_BOOST = 1.5
_LABEL_THRESHOLD = 88.0


def retrieve_for_route(route, store: Store, depth: int = RETRIEVE_DEPTH) -> list[dict]:
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
        # query = every metric variant (extractive core + wide + legacy) so a
        # short core phrase does not lose the qualifier tokens
        q_tokens = []
        for v in (getattr(route, "metric_variants", None) or [route.metric_norm]):
            q_tokens.extend(str(v).split())
        q_tokens.extend(str(y) for y in route.years)
        scores = bm25.scores(q_tokens)
        order = sorted(range(len(metas)), key=lambda i: -scores[i])[: max(depth * 3, 60)]
        for i in order:
            m, g = metas[i], grids[i]
            s = scores[i]
            # number-aware-lite boost: a row label closely matching the metric
            best_lab = 0.0
            for row in g[1:60]:
                for cell in row[:3]:
                    if cell and len(cell) > 3:
                        best_lab = max(best_lab, fuzz_token_set(cell, route.metric_norm))
                        break
            if best_lab >= _LABEL_THRESHOLD:
                s *= _LABEL_BOOST
            cands.append({
                "report_id": m["report_id"], "ticker": m["ticker"],
                "table_pos": int(m["table_pos"]), "page": int(m["page"]),
                "unit_scale": (None if m["unit_scale"] is None or m["unit_scale"] != m["unit_scale"]
                               else float(m["unit_scale"])),
                "unit_source": m["unit_source"],
                "n_rows": int(m["n_rows"]), "score": round(float(s), 4),
                "label_match": round(best_lab, 1),
            })
    cands.sort(key=lambda c: -c["score"])
    return _apply_quota(cands, route, depth)


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
                  out_path: Path, depth: int = RETRIEVE_DEPTH, limit: int = 0) -> None:
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
        cands = retrieve_for_route(route, store, depth)
        out.append({"id": q["id"], "question": q["question"],
                    "route": route.to_dict(), "candidates": cands})
    write_jsonl(out_path, out)
    n_empty = sum(1 for r in out if not r["candidates"])
    print(f"retrieval done: {len(out)} questions, {n_empty} with no candidates")
