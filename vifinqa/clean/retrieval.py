"""Canonical, configurable retrieval for the clean baseline.

The historical retriever remains frozen.  This module reuses its store/router
contracts while adding only source-derived metric/component expansion and an
optional row-aware score.  No question ID is accepted by this API.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import RETRIEVE_DEPTH
from ..extraction.build_store import Store
from ..finance.metrics import (
    expand_metric_variants,
    extract_metric_qualifiers,
    metric_keys,
)
from ..retrieval.bm25 import BM25
from ..retrieval.retrieve import _apply_quota
from ..retrieval.serialize import grid_of, table_doc_tokens, tidy_csv_text
from ..retrieval.shortlist import build_shortlist
from ..router.decompose import split_ratio_metric
from ..router.entities import StockMap
from ..router.router import route_question
from ..utils.io import read_jsonl, write_jsonl
from ..utils.viet_text import fuzz_token_set, norm, tokens
from .profile import CLEAN_PROFILE, canonical_json_sha256


@dataclass(frozen=True)
class CleanRetrievalConfig:
    profile: str = CLEAN_PROFILE
    canonical_registry: bool = True
    component_expansion: bool = True
    canonical_extra_weight: float = 0.35
    label_boost: float = 1.5
    label_threshold: float = 88.0
    row_rerank: bool = False
    row_score_weight: float = 0.18
    max_extra_variants: int = 32

    def validate(self) -> None:
        if self.profile != CLEAN_PROFILE:
            raise ValueError("clean retriever requires profile=clean")
        if not self.canonical_registry:
            raise ValueError("clean retriever requires the canonical registry")
        if not 0.0 <= self.canonical_extra_weight <= 1.0:
            raise ValueError("canonical_extra_weight must be in [0, 1]")
        if not 0.0 <= self.row_score_weight <= 1.0:
            raise ValueError("row_score_weight must be in [0, 1]")
        if not 0.0 <= self.label_threshold <= 100.0:
            raise ValueError("label_threshold must be in [0, 100]")
        if self.max_extra_variants < 0:
            raise ValueError("max_extra_variants must be non-negative")

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        return canonical_json_sha256(self.to_dict())


def canonicalize_route(route) -> tuple[list[str], list[str], dict[str, str]]:
    base = _dedupe(list(getattr(route, "metric_variants", None) or [route.metric_norm]))
    plan = getattr(route, "plan", None) or {}
    phrases = list(base)
    for fact in plan.get("facts", []) if isinstance(plan, dict) else []:
        phrases.append(str(fact.get("metric") or ""))
    for phrase in list(phrases):
        numerator, denominator = split_ratio_metric(phrase)
        phrases.extend([numerator, denominator])
    keys = metric_keys(phrases, expand_derived=False)
    variants = expand_metric_variants(
        _dedupe(phrases), question=getattr(route, "question", "") or ""
    )
    qualifiers = extract_metric_qualifiers(
        getattr(route, "question", "") or "", keys
    ).to_dict()
    route.metric_variants = variants
    return variants, keys, qualifiers


def retrieve_for_route(route, store: Store, depth: int = RETRIEVE_DEPTH,
                       config: CleanRetrievalConfig | None = None) -> list[dict]:
    config = config or CleanRetrievalConfig()
    config.validate()
    if not route.report_ids or not route.tickers:
        return []
    variants, _keys, _qualifiers = canonicalize_route(route)
    base_variants = _dedupe(
        list(getattr(route, "metric_variants", None) or [route.metric_norm])
    )
    candidates: list[dict] = []
    for ticker in route.tickers:
        frame = store.tables_of(ticker, route.report_ids)
        if not len(frame):
            continue
        metas = frame.to_dict("records")
        grids = [grid_of(meta) for meta in metas]
        bm25 = BM25([table_doc_tokens(meta, grid)
                     for meta, grid in zip(metas, grids)])
        scores = _combined_scores(
            bm25, [getattr(route, "metric_norm", "")], variants,
            list(getattr(route, "years", []) or []), config,
        )
        order = sorted(range(len(metas)), key=lambda i: -scores[i])[:max(depth * 3, 60)]
        row_scores = (_row_scores(route, metas, order, variants)
                      if config.row_rerank else {})
        for index in order:
            meta, grid = metas[index], grids[index]
            best_label = 0.0
            for row in grid[1:60]:
                for cell in row[:3]:
                    if cell and len(cell) > 3:
                        best_label = max(
                            best_label,
                            max((fuzz_token_set(cell, phrase) for phrase in variants),
                                default=0.0),
                        )
                        break
            bm25_score = float(scores[index])
            if best_label >= config.label_threshold:
                bm25_score *= config.label_boost
            key = (meta["report_id"], int(meta["table_pos"]))
            row_score = float(row_scores.get(key, 0.0))
            final = bm25_score + config.row_score_weight * row_score
            scale = meta.get("unit_scale")
            candidates.append({
                "report_id": meta["report_id"],
                "ticker": meta["ticker"],
                "table_pos": int(meta["table_pos"]),
                "page": int(meta["page"]),
                "unit_scale": None if scale is None or scale != scale else float(scale),
                "unit_source": meta["unit_source"],
                "n_rows": int(meta["n_rows"]),
                "score": round(final, 4),
                "bm25_score": round(bm25_score, 4),
                "row_score": round(row_score, 1),
                "label_match": round(best_label, 1),
            })
    candidates.sort(key=lambda item: -item["score"])
    return _apply_quota(candidates, route, depth)


def _combined_scores(bm25: BM25, base_variants: list[str], variants: list[str],
                     years: list[int], config: CleanRetrievalConfig) -> list[float]:
    base_variants = _dedupe(base_variants)
    year_tokens = [str(year) for year in years]
    base_tokens = [token for phrase in base_variants for token in tokens(phrase)]
    base = bm25.scores([*base_tokens, *year_tokens])
    if not config.component_expansion or not base_tokens:
        return base
    base_set = set(base_variants)
    extras = [phrase for phrase in variants if phrase not in base_set]
    best = [0.0] * len(bm25.docs)
    for phrase in extras[:config.max_extra_variants]:
        scores = bm25.scores([*tokens(phrase), *year_tokens])
        for index, score in enumerate(scores):
            best[index] = max(best[index], score)
    return [primary + config.canonical_extra_weight * extra
            for primary, extra in zip(base, best)]


def _row_scores(route, metas: list[dict], order: list[int],
                variants: list[str]) -> dict[tuple[str, int], float]:
    blocks = []
    for rank, index in enumerate(order, 1):
        meta = metas[index]
        blocks.append({
            "var": f"df{rank}",
            "report_id": meta["report_id"],
            "table_pos": int(meta["table_pos"]),
            "csv_text": tidy_csv_text(meta),
        })
    shortlist = build_shortlist(
        blocks, variants, list(getattr(route, "years", []) or []),
        top_n=min(80, max(12, len(blocks) * 2)), min_score=35.0,
    )
    out: dict[tuple[str, int], float] = {}
    for candidate in shortlist:
        key = (candidate.report_id, int(candidate.table_pos))
        out[key] = max(out.get(key, 0.0), float(candidate.score))
    return out


def _dedupe(values: list[str]) -> list[str]:
    seen, out = set(), []
    for value in values:
        clean = " ".join(norm(str(value or "")).split())
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out


def run_clean_retrieval(questions_path: Path, store_dir: Path,
                        code_stock_csv: Path, out_path: Path,
                        depth: int = RETRIEVE_DEPTH, limit: int = 0,
                        config: CleanRetrievalConfig | None = None) -> None:
    config = config or CleanRetrievalConfig()
    config.validate()
    store = Store(store_dir)
    stock = StockMap(code_stock_csv)
    questions = read_jsonl(questions_path)
    if limit:
        questions = questions[:limit]
    output = []
    for question in questions:
        route = route_question(question["id"], question["question"], stock, store)
        candidates = retrieve_for_route(route, store, depth, config)
        variants, keys, qualifiers = canonicalize_route(route)
        route_dict = route.to_dict()
        route_dict.update({
            "metric_variants": variants,
            "metric_keys": keys,
            "metric_qualifiers": qualifiers,
            "clean_profile": CLEAN_PROFILE,
            "retrieval_config_sha256": config.fingerprint(),
        })
        output.append({
            "id": question["id"],
            "question": question["question"],
            "route": route_dict,
            "candidates": candidates,
        })
    write_jsonl(out_path, output)

