"""Structured table retrieval with legacy and multi-channel RRF modes.

The default ``legacy`` mode preserves the leaderboard-tested ranking. The
opt-in ``rrf`` mode independently ranks tables with BM25, row-label lexical
matching, row schema linking and optional cached BGE-M3 embeddings, then fuses
those rankings with Reciprocal Rank Fusion.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from types import SimpleNamespace

from ..config import RETRIEVE_DEPTH
from ..extraction.build_store import Store
from ..finance.metrics import code_expectation, expand_metric_variants
from ..router.entities import StockMap
from ..router.decompose import split_ratio_metric
from ..router.router import route_question
from ..utils.io import read_jsonl, write_json, write_jsonl
from ..utils.viet_num import parse_vn_number
from ..utils.viet_text import fuzz_token_set, label_metric_score, norm, tokens
from .bm25 import BM25
from .fusion import reciprocal_rank_fusion
from .serialize import table_doc_tokens, grid_of, tidy_csv_text
from .shortlist import build_shortlist

_LABEL_BOOST = 1.5
_LABEL_THRESHOLD = 88.0
_ROW_SCORE_WEIGHT = 0.18
_RETRIEVAL_MODES = {"legacy", "rrf"}


def retrieve_for_route(route, store: Store, depth: int = RETRIEVE_DEPTH,
                       row_rerank: bool = False,
                       row_score_weight: float = _ROW_SCORE_WEIGHT,
                       retrieval_mode: str = "legacy", encoder=None,
                       candidate_pool: set[tuple[str, int]] | None = None,
                       rrf_k: float = 60.0, pool_factor: int = 5,
                       dense_min_similarity: float = 0.35) -> list[dict]:
    """Rank tables for one route while preserving the legacy default."""
    if retrieval_mode not in _RETRIEVAL_MODES:
        raise ValueError(f"unknown retrieval_mode={retrieval_mode!r}")
    if retrieval_mode == "legacy":
        return _retrieve_legacy(
            route, store, depth, row_rerank=row_rerank,
            row_score_weight=row_score_weight,
        )
    return _retrieve_rrf(
        route, store, depth, encoder=encoder, candidate_pool=candidate_pool,
        rrf_k=rrf_k, pool_factor=pool_factor,
        dense_min_similarity=dense_min_similarity,
    )


def _retrieve_legacy(route, store: Store, depth: int,
                     row_rerank: bool = False,
                     row_score_weight: float = _ROW_SCORE_WEIGHT) -> list[dict]:
    """Original BM25/row-score pipeline kept byte-for-byte in behavior."""
    if not route.report_ids or not route.tickers:
        return []
    cands = []
    for ticker in route.tickers:
        tdf = store.tables_of(ticker, route.report_ids)
        if not len(tdf):
            continue
        metas = tdf.to_dict("records")
        grids = [grid_of(meta) for meta in metas]
        docs = [table_doc_tokens(meta, grid) for meta, grid in zip(metas, grids)]
        bm25 = BM25(docs)
        base_variants = _base_metric_variants(route)
        variants = _retrieval_metric_variants(route)
        expanded = bool(set(variants) - set(base_variants))
        rank_variants = variants if expanded else base_variants
        scores = _combined_bm25_scores(
            bm25, base_variants, variants, getattr(route, "years", []) or []
        )
        order = sorted(range(len(metas)), key=lambda i: -scores[i])[:max(depth * 3, 60)]
        row_scores = _row_scores(route, metas, order, rank_variants) if row_rerank else {}
        for i in order:
            meta, grid = metas[i], grids[i]
            score = scores[i]
            best_label = _legacy_best_label_score(grid, rank_variants)
            if best_label >= _LABEL_THRESHOLD:
                score *= _LABEL_BOOST
            key = (meta["report_id"], int(meta["table_pos"]))
            row_score = row_scores.get(key, 0.0)
            bm25_score = float(score)
            final_score = bm25_score + row_score_weight * row_score
            cands.append(_candidate_record(
                meta,
                score=final_score,
                bm25_score=bm25_score,
                row_score=row_score,
                label_match=best_label,
            ))
    cands.sort(key=lambda cand: -cand["score"])
    return _apply_quota(cands, route, depth)


def _retrieve_rrf(route, store: Store, depth: int, encoder=None,
                  candidate_pool: set[tuple[str, int]] | None = None,
                  rrf_k: float = 60.0, pool_factor: int = 5,
                  dense_min_similarity: float = 0.35) -> list[dict]:
    """Generate independent rankings and fuse them without raw-score mixing."""
    if not route.report_ids or not route.tickers:
        return []
    if pool_factor < 1:
        raise ValueError("pool_factor must be at least 1")

    all_candidates: list[dict] = []
    pool_size = max(60, depth * pool_factor)
    for ticker in route.tickers:
        tdf = store.tables_of(ticker, route.report_ids)
        if not len(tdf):
            continue
        metas = tdf.to_dict("records")
        if candidate_pool is not None:
            metas = [meta for meta in metas
                     if (meta["report_id"], int(meta["table_pos"])) in candidate_pool]
        if not metas:
            continue
        grids = [grid_of(meta) for meta in metas]
        docs = [table_doc_tokens(meta, grid) for meta, grid in zip(metas, grids)]
        variants = _retrieval_metric_variants(route)
        base_variants = _base_metric_variants(route)
        bm25_scores = _combined_bm25_scores(
            BM25(docs), base_variants, variants, getattr(route, "years", []) or []
        )
        label_scores = {
            i: _best_label_score(grid, variants) for i, grid in enumerate(grids)
        }
        dense_scores = _dense_table_scores(grids, variants, encoder)

        bm25_order = _ranked_indices(bm25_scores, pool_size)
        lexical_order = _ranked_mapping(label_scores, pool_size, minimum=0.01)
        dense_order = _ranked_mapping(
            dense_scores, pool_size, minimum=dense_min_similarity
        )
        preliminary = list(dict.fromkeys(bm25_order + lexical_order + dense_order))
        schema_scores = _schema_table_scores(
            route, grids, variants, preliminary,
        )
        schema_order = _ranked_mapping(schema_scores, pool_size, minimum=0.01)

        rankings = {
            "bm25": bm25_order,
            "lexical_row": lexical_order,
            "schema_row": schema_order,
        }
        weights = {"bm25": 1.0, "lexical_row": 0.8, "schema_row": 1.0}
        if dense_order:
            rankings["dense_row"] = dense_order
            weights["dense_row"] = 1.0
        fused, ranks = reciprocal_rank_fusion(
            rankings, rank_constant=rrf_k, weights=weights
        )
        fused_order = sorted(
            fused,
            key=lambda i: (-fused[i], -bm25_scores[i], i),
        )[:pool_size]
        for i in fused_order:
            meta = metas[i]
            key = (meta["report_id"], int(meta["table_pos"]))
            all_candidates.append(_candidate_record(
                meta,
                score=fused[i],
                bm25_score=bm25_scores[i],
                row_score=schema_scores.get(i, 0.0),
                label_match=label_scores.get(i, 0.0),
                dense_score=dense_scores.get(i, 0.0),
                retrieval_mode="rrf",
                channel_ranks=ranks.get(i, {}),
            ))

    all_candidates.sort(
        key=lambda cand: (-cand["score"], -cand["bm25_score"],
                          cand["report_id"], cand["table_pos"])
    )
    return _apply_quota(all_candidates, route, depth)


def _candidate_record(meta: dict, *, score: float, bm25_score: float,
                      row_score: float, label_match: float,
                      dense_score: float = 0.0,
                      retrieval_mode: str = "legacy",
                      channel_ranks: dict | None = None) -> dict:
    unit_scale = meta.get("unit_scale")
    if unit_scale is None or unit_scale != unit_scale:
        unit_scale = None
    else:
        unit_scale = float(unit_scale)
    record = {
        "report_id": meta["report_id"],
        "ticker": meta["ticker"],
        "table_pos": int(meta["table_pos"]),
        "page": int(meta["page"]),
        "unit_scale": unit_scale,
        "unit_source": meta["unit_source"],
        "n_rows": int(meta["n_rows"]),
        "score": round(float(score), 8 if retrieval_mode == "rrf" else 4),
        "bm25_score": round(float(bm25_score), 4),
        "row_score": round(float(row_score), 1),
        "label_match": round(float(label_match), 1),
    }
    if retrieval_mode != "legacy":
        record.update({
            "retrieval_mode": retrieval_mode,
            "fusion_score": round(float(score), 8),
            "dense_score": round(float(dense_score), 6),
            "channel_ranks": dict(sorted((channel_ranks or {}).items())),
        })
    return record


def _ranked_indices(scores: list[float], limit: int) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:limit]


def _ranked_mapping(scores: dict[int, float], limit: int,
                    minimum: float = 0.0) -> list[int]:
    kept = (i for i, score in scores.items() if score >= minimum)
    return sorted(kept, key=lambda i: (-scores[i], i))[:limit]


def _row_labels(grid: list[list[str]], max_rows: int = 120) -> list[str]:
    labels = []
    for row in grid[1:max_rows + 1]:
        for cell in row[:3]:
            cell = str(cell or "").strip()
            if cell and len(cell) > 3 and parse_vn_number(cell) is None:
                labels.append(cell[:160])
                break
    return list(dict.fromkeys(labels))


def _legacy_best_label_score(grid: list[list[str]], variants: list[str]) -> float:
    """Preserve the exact label scan used by the 0.2806 branch."""
    best = 0.0
    for row in grid[1:60]:
        for cell in row[:3]:
            if cell and len(cell) > 3:
                best = max(
                    best,
                    max((fuzz_token_set(cell, variant) for variant in variants),
                        default=0.0),
                )
                break
    return best


def _best_label_score(grid: list[list[str]], variants: list[str]) -> float:
    return max(
        (fuzz_token_set(label, variant)
         for label in _row_labels(grid, max_rows=60)
         for variant in variants),
        default=0.0,
    )


def _row_label_and_code(row: list[str]) -> tuple[str, str]:
    """Extract a compact label/code pair directly from an HTML grid row."""
    labels, code = [], ""
    for cell in row[:6]:
        text = str(cell or "").strip()
        if not text:
            continue
        value = parse_vn_number(text)
        if value is not None:
            if not code and len(text.replace(".", "").replace(",", "")) <= 5:
                if text.replace(".", "", 1).isdigit():
                    code = text
            continue
        labels.append(text)
    return " ".join(labels)[:160], code


def _schema_table_scores(route, grids: list[list[list[str]]],
                         variants: list[str], order: list[int]) -> dict[int, float]:
    """Cheap row-schema channel: metric, VAS code and year evidence.

    This intentionally works on parsed grids instead of rebuilding tidy pandas
    frames. It is used on every question in RRF mode; the richer shortlist
    remains the codegen/schema-linking path.
    """
    if not variants or not order:
        return {}
    expected_codes, known_mismatch = code_expectation(variants)
    years = [int(year) for year in (getattr(route, "years", []) or [])]
    out: dict[int, float] = {}
    for table_index in order:
        grid = grids[table_index]
        header = " ".join(str(cell or "") for cell in (grid[0] if grid else []))
        header_has_year = any(str(year) in header for year in years)
        best = 0.0
        for row in grid[1:121]:
            label, code = _row_label_and_code(row)
            if len(label) <= 3:
                continue
            lexical = max((label_metric_score(label, variant)
                           for variant in variants), default=0.0)
            if lexical <= 0:
                continue
            score = lexical
            if expected_codes:
                clean_code = re.sub(r"\.0$", "", code) if code else ""
                if clean_code in expected_codes:
                    score += 12.0
                elif code and not known_mismatch:
                    score -= 10.0
            if years:
                score += 10.0 if header_has_year else -4.0
            best = max(best, score)
        if best > 0:
            out[table_index] = best
    return out


def _dense_table_scores(grids: list[list[list[str]]], variants: list[str],
                        encoder) -> dict[int, float]:
    """Return best dense row-label similarity per table."""
    if encoder is None or not variants:
        return {}
    labels_by_table = [_row_labels(grid) for grid in grids]
    unique_labels = sorted({label for labels in labels_by_table for label in labels})
    if not unique_labels:
        return {}
    similarities = encoder.similarity(variants, unique_labels)
    return {
        i: max((float(similarities.get(label, 0.0)) for label in labels), default=0.0)
        for i, labels in enumerate(labels_by_table)
    }


def _base_metric_variants(route) -> list[str]:
    return _dedupe_variants(
        list(getattr(route, "metric_variants", None) or
             [getattr(route, "metric_norm", "")])
    )


def _retrieval_metric_variants(route) -> list[str]:
    """Metric phrases used only for table retrieval and reranking."""
    out: list[str] = []
    for variant in (getattr(route, "metric_variants", None) or
                    [getattr(route, "metric_norm", "")]):
        out.append(str(variant or ""))
    plan = getattr(route, "plan", None) or {}
    facts = plan.get("facts", []) if isinstance(plan, dict) else []
    for fact in facts:
        out.append(str(fact.get("metric") or ""))
    for variant in list(out):
        numerator, denominator = split_ratio_metric(variant)
        out.extend([numerator, denominator])
    return expand_metric_variants(
        _dedupe_variants(out),
        question=getattr(route, "question", "") or "",
    )


def _combined_bm25_scores(bm25: BM25, base_variants: list[str],
                          variants: list[str], years: list[int]) -> list[float]:
    """Legacy BM25 foundation, with component-metric recall as a bounded boost."""
    if not bm25.docs:
        return []
    year_tokens = [str(year) for year in years]
    base_tokens: list[str] = []
    for variant in base_variants:
        base_tokens.extend(tokens(variant))
    base_tokens.extend(year_tokens)
    base = bm25.scores(base_tokens)
    extras = [variant for variant in variants
              if variant and variant not in set(base_variants)]
    if not extras:
        return base
    if not base_tokens:
        return bm25.scores(year_tokens)
    best_extra = [0.0] * len(bm25.docs)
    for variant in extras[:32]:
        query_tokens = tokens(variant) + year_tokens
        scores = bm25.scores(query_tokens)
        for i, score in enumerate(scores):
            if score > best_extra[i]:
                best_extra[i] = score
    return [base_score + 0.35 * extra
            for base_score, extra in zip(base, best_extra)]


def _dedupe_variants(variants: list[str]) -> list[str]:
    seen, out = set(), []
    for variant in variants:
        variant = " ".join(norm(variant).split())
        if not variant or variant in seen:
            continue
        seen.add(variant)
        out.append(variant)
    return out


def _row_scores(route, metas: list[dict], order: list[int],
                variants: list[str] | None = None, *,
                top_n_cap: int = 80, multiplier: int = 2,
                minimum_rows: int = 12,
                include_report_year: bool = False) -> dict[tuple[str, int], float]:
    """Best row-level schema-linking score per table."""
    variants = variants or _retrieval_metric_variants(route)
    if not variants or not order:
        return {}
    blocks = []
    for position, i in enumerate(order, start=1):
        meta = metas[i]
        block = {
            "var": f"df{position}",
            "report_id": meta["report_id"],
            "table_pos": int(meta["table_pos"]),
            "csv_text": tidy_csv_text(meta),
        }
        if include_report_year:
            block["report_year"] = int(meta["year"])
        blocks.append(block)
    shortlist = build_shortlist(
        blocks, variants, getattr(route, "years", []) or [],
        top_n=min(top_n_cap, max(minimum_rows, len(blocks) * multiplier)),
        min_score=35.0,
    )
    out: dict[tuple[str, int], float] = {}
    for candidate in shortlist:
        key = (candidate.report_id, int(candidate.table_pos))
        out[key] = max(out.get(key, 0.0), float(candidate.score))
    return out


def _apply_quota(cands: list[dict], route, depth: int) -> list[dict]:
    """Guarantee evidence capacity across locked reports for composite routes."""
    plan = getattr(route, "plan", None) or {}
    facts = plan.get("facts", [])
    if len(facts) <= 1 or not cands:
        return cands[:depth]
    docs = list(dict.fromkeys(candidate["report_id"] for candidate in cands))
    per_doc = max(1, depth // max(1, len(docs)))
    taken, out = {}, []
    for candidate in cands:
        if len(out) >= depth:
            break
        report_id = candidate["report_id"]
        if taken.get(report_id, 0) < per_doc:
            out.append(candidate)
            taken[report_id] = taken.get(report_id, 0) + 1
    for candidate in cands:
        if len(out) >= depth:
            break
        if candidate not in out:
            out.append(candidate)
    return out[:depth]


def _load_dense_encoder(store_dir: Path, model_name: str, cache_dir: Path | None,
                        device: str | None, required: bool):
    from .dense import load_encoder

    cache_dir = Path(cache_dir) if cache_dir else Path(store_dir) / "label_index"
    required_files = [cache_dir / "labels.json", cache_dir / "labels.npy"]
    if not all(path.exists() for path in required_files):
        message = f"dense cache is incomplete at {cache_dir}; run scripts/10_build_label_index.py"
        if required:
            raise SystemExit(message)
        print(f"[dense] disabled ({message})")
        return None, {"enabled": False, "reason": message, "cache_dir": str(cache_dir)}
    encoder = load_encoder(model_name, cache_dir, device)
    if encoder is None:
        message = "dense encoder dependency/model could not be loaded"
        if required:
            raise SystemExit(message)
        return None, {"enabled": False, "reason": message, "cache_dir": str(cache_dir)}
    return encoder, encoder.describe()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_route_source(path: Path | None, questions: list[dict]):
    """Load and validate frozen routes for a retrieval-only ablation."""
    if path is None:
        return {}, {"enabled": False}
    path = Path(path)
    rows = read_jsonl(path)
    by_id: dict[int, dict] = {}
    for row in rows:
        qid = int(row["id"])
        if qid in by_id:
            raise ValueError(f"duplicate route-source id={qid}")
        if not isinstance(row.get("route"), dict):
            raise ValueError(f"route-source id={qid} has no route object")
        by_id[qid] = row
    for question in questions:
        qid = int(question["id"])
        if qid not in by_id:
            raise ValueError(f"route-source is missing id={qid}")
        if by_id[qid].get("question") != question.get("question"):
            raise ValueError(f"route-source question mismatch at id={qid}")
    return by_id, {
        "enabled": True,
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "records": len(rows),
    }


def run_retrieval(questions_path: Path, store_dir: Path, code_stock_csv: Path,
                  out_path: Path, depth: int = RETRIEVE_DEPTH, limit: int = 0,
                  route_source_path: Path | None = None,
                  freeze_candidate_pool: bool = False,
                  row_rerank: bool = False,
                  row_score_weight: float = _ROW_SCORE_WEIGHT,
                  retrieval_mode: str = "legacy", use_dense: bool = False,
                  dense_model: str = "BAAI/bge-m3",
                  dense_cache_dir: Path | None = None,
                  dense_device: str | None = None,
                  dense_required: bool = False,
                  rrf_k: float = 60.0, pool_factor: int = 5,
                  dense_min_similarity: float = 0.35) -> None:
    store = Store(store_dir)
    stock = StockMap(code_stock_csv)
    questions = read_jsonl(questions_path)
    if limit:
        questions = questions[:limit]
    route_source, route_source_info = _load_route_source(
        route_source_path, questions
    )

    if freeze_candidate_pool and not route_source:
        raise ValueError("freeze_candidate_pool requires route_source_path")
    if freeze_candidate_pool and retrieval_mode != "rrf":
        raise ValueError("freeze_candidate_pool requires retrieval_mode='rrf'")

    encoder = None
    dense_info = {"enabled": False, "reason": "not requested"}
    if use_dense and retrieval_mode != "rrf":
        raise ValueError("--use-dense requires --retrieval-mode rrf")
    if use_dense:
        encoder, dense_info = _load_dense_encoder(
            store_dir, dense_model, dense_cache_dir, dense_device, dense_required
        )
    config = {
        "retrieval_mode": retrieval_mode,
        "depth": depth,
        "row_rerank": bool(row_rerank),
        "row_score_weight": row_score_weight,
        "rrf_k": rrf_k,
        "pool_factor": pool_factor,
        "dense_min_similarity": dense_min_similarity,
        "dense": dense_info,
        "route_source": route_source_info,
        "candidate_pool": "frozen_route_source" if freeze_candidate_pool else "locked_reports",
    }

    out = []
    try:
        from tqdm import tqdm
        iterator = tqdm(questions, desc="retrieve")
    except ImportError:
        iterator = questions
    for question in iterator:
        frozen = route_source.get(int(question["id"])) if route_source else None
        if frozen is not None:
            route_dict = frozen["route"]
            route = SimpleNamespace(**route_dict)
        else:
            route = route_question(question["id"], question["question"], stock, store)
            route_dict = route.to_dict()
        candidate_pool = None
        if freeze_candidate_pool:
            candidate_pool = {
                (candidate["report_id"], int(candidate["table_pos"]))
                for candidate in frozen.get("candidates", [])
            }
        cands = retrieve_for_route(
            route, store, depth, row_rerank=row_rerank,
            row_score_weight=row_score_weight, retrieval_mode=retrieval_mode,
            encoder=encoder, candidate_pool=candidate_pool,
            rrf_k=rrf_k, pool_factor=pool_factor,
            dense_min_similarity=dense_min_similarity,
        )
        record = {
            "id": question["id"],
            "question": question["question"],
            "route": route_dict,
            "candidates": cands,
        }
        if retrieval_mode != "legacy":
            record["retrieval_config"] = config
        out.append(record)
    write_jsonl(out_path, out)
    metadata_path = Path(str(out_path) + ".meta.json")
    write_json(metadata_path, {**config, "questions": len(out)})
    n_empty = sum(1 for record in out if not record["candidates"])
    print(f"retrieval done: {len(out)} questions, {n_empty} with no candidates")
    print(f"retrieval metadata -> {metadata_path}")
