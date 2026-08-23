"""Deterministic stage construction for the G3C retrieval ladder."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from ..extraction.build_store import Store
from ..retrieval.bm25 import BM25
from ..utils.viet_text import tokens
from .leaves import AtomicLeaf, decompose_atomic_leaves, serialize_leaf_query
from .serialize import (
    candidate_from_meta,
    row_passages,
    table_key,
    table_passage,
)


TableKey = tuple[str, int]


@dataclass
class QuestionState:
    baseline: dict
    leaves: list[AtomicLeaf]
    meta_by_key: dict[TableKey, dict]
    passage_by_key: dict[TableKey, dict]
    eligible_by_leaf: dict[str, list[TableKey]]
    r0_by_key: dict[TableKey, dict]
    r0_rank: list[TableKey]
    lexical_by_leaf: dict[str, list[TableKey]]
    lexical_scores: dict[tuple[str, TableKey], float]
    query_by_leaf: dict[str, str]
    dense_by_leaf: dict[str, list[TableKey]] = field(default_factory=dict)
    dense_scores: dict[tuple[str, TableKey], float] = field(default_factory=dict)
    rerank_by_leaf: dict[str, list[TableKey]] = field(default_factory=dict)
    rerank_scores: dict[tuple[str, TableKey], float] = field(default_factory=dict)
    row_rank_by_leaf: dict[str, list[TableKey]] = field(default_factory=dict)
    row_scores: dict[tuple[str, TableKey], float] = field(default_factory=dict)


def prepare_state(
    baseline: dict, store: Store, config: dict
) -> QuestionState:
    question = str(baseline["question"])
    leaves = decompose_atomic_leaves(question, baseline["route"], store)
    meta_by_key: dict[TableKey, dict] = {}
    eligible_by_leaf: dict[str, list[TableKey]] = {}
    passage_by_key: dict[TableKey, dict] = {}
    label_limit = int(config["retrieval"]["table_label_limit"])

    for leaf in leaves:
        keys = []
        frame = store.tables_of(leaf.ticker, list(leaf.report_ids))
        for meta in frame.to_dict("records"):
            key = table_key(meta)
            meta_by_key.setdefault(key, meta)
            if key not in keys:
                keys.append(key)
        keys.sort()
        eligible_by_leaf[leaf.leaf_id] = keys

    for key, meta in meta_by_key.items():
        passage_by_key[key] = table_passage(meta, label_limit)

    allowed = set().union(
        *(set(values) for values in eligible_by_leaf.values())
    ) if eligible_by_leaf else set()
    r0_by_key = {
        table_key(candidate): deepcopy(candidate)
        for candidate in baseline.get("candidates", [])
        if table_key(candidate) in allowed
    }
    r0_rank = [
        table_key(candidate) for candidate in baseline.get("candidates", [])
        if table_key(candidate) in allowed
    ]

    lexical_by_leaf: dict[str, list[TableKey]] = {}
    lexical_scores: dict[tuple[str, TableKey], float] = {}
    query_by_leaf: dict[str, str] = {}
    for leaf in leaves:
        query = serialize_leaf_query(leaf, question)
        query_by_leaf[leaf.leaf_id] = query
        keys = eligible_by_leaf[leaf.leaf_id]
        documents = [
            tokens(passage_by_key[key]["content"]) for key in keys
        ]
        scorer = BM25(documents)
        scores = scorer.scores(tokens(query))
        ranked = sorted(
            zip(keys, scores), key=lambda item: (-item[1], item[0])
        )
        lexical_by_leaf[leaf.leaf_id] = [key for key, _ in ranked]
        for key, score in ranked:
            lexical_scores[(leaf.leaf_id, key)] = float(score)

    return QuestionState(
        baseline=baseline,
        leaves=leaves,
        meta_by_key=meta_by_key,
        passage_by_key=passage_by_key,
        eligible_by_leaf=eligible_by_leaf,
        r0_by_key=r0_by_key,
        r0_rank=r0_rank,
        lexical_by_leaf=lexical_by_leaf,
        lexical_scores=lexical_scores,
        query_by_leaf=query_by_leaf,
    )


def r0_record(state: QuestionState) -> dict:
    """Return the frozen control without changing route or candidate fields."""
    return deepcopy(state.baseline)


def r0l_record(state: QuestionState, config: dict) -> dict:
    rankings = [state.r0_rank]
    rankings.extend(
        state.lexical_by_leaf[leaf.leaf_id] for leaf in state.leaves
    )
    scores = reciprocal_rank_fusion(
        rankings, int(config["retrieval"]["rrf_k"])
    )
    keys = rank_scores(scores)[:int(config["retrieval"]["depth"])]
    return materialize_record(state, "R0L", keys, scores)


def attach_dense_scores(
    state: QuestionState,
    query_vectors: dict[str, np.ndarray],
    passage_vectors: dict[str, np.ndarray],
    top_n: int,
) -> None:
    for leaf in state.leaves:
        leaf_id = leaf.leaf_id
        query = query_vectors[leaf_id].astype(np.float32)
        scored = []
        for key in state.eligible_by_leaf[leaf_id]:
            passage_id = state.passage_by_key[key]["passage_id"]
            passage = passage_vectors[passage_id].astype(np.float32)
            score = float(np.dot(query, passage))
            state.dense_scores[(leaf_id, key)] = score
            scored.append((key, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        state.dense_by_leaf[leaf_id] = [
            key for key, _ in scored[:top_n]
        ]


def r1_record(state: QuestionState, config: dict) -> dict:
    rankings = [state.r0_rank]
    for leaf in state.leaves:
        rankings.append(state.lexical_by_leaf[leaf.leaf_id])
        rankings.append(state.dense_by_leaf[leaf.leaf_id])
    scores = reciprocal_rank_fusion(
        rankings, int(config["retrieval"]["rrf_k"])
    )
    keys = rank_scores(scores)[:int(config["retrieval"]["depth"])]
    return materialize_record(state, "R1", keys, scores)


def rerank_pools(state: QuestionState, cap: int, rrf_k: int) -> dict[str, list[TableKey]]:
    output = {}
    for leaf in state.leaves:
        leaf_id = leaf.leaf_id
        allowed = set(state.eligible_by_leaf[leaf_id])
        r0 = [key for key in state.r0_rank if key in allowed]
        scores = reciprocal_rank_fusion([
            r0,
            state.lexical_by_leaf[leaf_id],
            state.dense_by_leaf[leaf_id],
        ], rrf_k)
        output[leaf_id] = rank_scores(scores)[:cap]
    return output


def attach_reranker_scores(
    state: QuestionState,
    scores_by_pair: dict[tuple[str, TableKey], float],
) -> None:
    by_leaf: dict[str, list[tuple[TableKey, float]]] = {}
    for leaf in state.leaves:
        leaf_id = leaf.leaf_id
        items = []
        for key in state.eligible_by_leaf[leaf_id]:
            pair_key = (leaf_id, key)
            if pair_key not in scores_by_pair:
                continue
            score = float(scores_by_pair[pair_key])
            state.rerank_scores[pair_key] = score
            items.append((key, score))
        items.sort(key=lambda item: (-item[1], item[0]))
        by_leaf[leaf_id] = items
    state.rerank_by_leaf = {
        leaf_id: [key for key, _ in items]
        for leaf_id, items in by_leaf.items()
    }


def r2_record(state: QuestionState, config: dict) -> dict:
    rankings = [
        state.rerank_by_leaf[leaf.leaf_id] for leaf in state.leaves
    ]
    scores = reciprocal_rank_fusion(
        rankings, int(config["retrieval"]["rrf_k"])
    )
    keys = rank_scores(scores)[:int(config["retrieval"]["depth"])]
    return materialize_record(state, "R2", keys, scores)


def r3_record(state: QuestionState, config: dict) -> dict:
    depth = int(config["retrieval"]["depth"])
    quota = int(config["retrieval"]["per_leaf_quota"])
    rankings = {
        leaf.leaf_id: state.rerank_by_leaf[leaf.leaf_id]
        for leaf in state.leaves
    }
    global_scores = reciprocal_rank_fusion(
        list(rankings.values()), int(config["retrieval"]["rrf_k"])
    )
    keys = select_with_leaf_quota(
        rankings, rank_scores(global_scores), quota, depth
    )
    return materialize_record(state, "R3", keys, global_scores)


def attach_row_scores(
    state: QuestionState,
    store: Store,
    selected_keys: list[TableKey],
    score_pairs,
    config: dict,
) -> None:
    """Lexically prefilter rows globally per leaf, then neural-score a bound."""
    lexical_cap = int(config["retrieval"]["row_lexical_prefilter"])
    neural_cap = int(config["retrieval"]["row_rerank_top_n_per_leaf"])
    row_limit = int(config["retrieval"]["row_limit_per_table"])
    selected = set(selected_keys)
    rows_by_key: dict[TableKey, list[dict]] = {}
    for key in selected:
        meta = state.meta_by_key[key]
        cells = store.cells_of(meta["ticker"], [meta["report_id"]])
        records = cells[
            cells.table_pos == int(meta["table_pos"])
        ].to_dict("records")
        rows_by_key[key] = row_passages(records, meta, row_limit)

    for leaf in state.leaves:
        leaf_id = leaf.leaf_id
        allowed = selected & set(state.eligible_by_leaf[leaf_id])
        row_items: list[tuple[TableKey, dict]] = [
            (key, row)
            for key in sorted(allowed)
            for row in rows_by_key.get(key, [])
        ]
        if not row_items:
            state.row_rank_by_leaf[leaf_id] = (
                state.rerank_by_leaf.get(leaf_id, [])
            )
            continue
        scorer = BM25([
            tokens(row["content"]) for _, row in row_items
        ])
        lexical = scorer.scores(tokens(state.query_by_leaf[leaf_id]))
        order = sorted(
            range(len(row_items)),
            key=lambda index: (-lexical[index], row_items[index][1]["passage_id"]),
        )[:lexical_cap]
        prefiltered = [row_items[index] for index in order]
        neural_scores = score_pairs(
            [
                (state.query_by_leaf[leaf_id], row["content"])
                for _, row in prefiltered
            ],
            kind="row",
        )
        reranked = sorted(
            zip(prefiltered, neural_scores),
            key=lambda item: (
                -float(item[1]), item[0][1]["passage_id"]
            ),
        )[:neural_cap]
        per_table: dict[TableKey, float] = {}
        for (key, _row), score in reranked:
            per_table[key] = max(per_table.get(key, 0.0), float(score))
        for key, score in per_table.items():
            state.row_scores[(leaf_id, key)] = score
        row_rank = sorted(
            per_table, key=lambda key: (-per_table[key], key)
        )
        table_rank = [
            key for key in state.rerank_by_leaf.get(leaf_id, [])
            if key in allowed
        ]
        combined = reciprocal_rank_fusion(
            [table_rank, row_rank],
            int(config["retrieval"]["rrf_k"]),
        )
        state.row_rank_by_leaf[leaf_id] = rank_scores(combined)


def r4_record(state: QuestionState, config: dict) -> dict:
    depth = int(config["retrieval"]["depth"])
    quota = int(config["retrieval"]["per_leaf_quota"])
    rankings = {
        leaf.leaf_id: state.row_rank_by_leaf.get(
            leaf.leaf_id, state.rerank_by_leaf[leaf.leaf_id]
        )
        for leaf in state.leaves
    }
    global_scores = reciprocal_rank_fusion(
        list(rankings.values()), int(config["retrieval"]["rrf_k"])
    )
    keys = select_with_leaf_quota(
        rankings, rank_scores(global_scores), quota, depth
    )
    return materialize_record(state, "R4", keys, global_scores)


def materialize_record(
    state: QuestionState,
    stage: str,
    keys: list[TableKey],
    fused_scores: dict[TableKey, float],
) -> dict:
    candidates = []
    for rank, key in enumerate(keys, 1):
        if key in state.r0_by_key:
            candidate = deepcopy(state.r0_by_key[key])
            candidate.update(candidate_from_meta(state.meta_by_key[key]))
        else:
            candidate = candidate_from_meta(state.meta_by_key[key])
        sources, leaf_ids = _candidate_provenance(state, key)
        dense = [
            state.dense_scores[(leaf_id, key)]
            for leaf_id in leaf_ids
            if (leaf_id, key) in state.dense_scores
        ]
        rerank = [
            state.rerank_scores[(leaf_id, key)]
            for leaf_id in leaf_ids
            if (leaf_id, key) in state.rerank_scores
        ]
        row = [
            state.row_scores[(leaf_id, key)]
            for leaf_id in leaf_ids
            if (leaf_id, key) in state.row_scores
        ]
        candidate["score"] = round(
            float(fused_scores.get(key, 0.0)) * 10000.0, 8
        )
        candidate["g3c"] = {
            "stage": stage,
            "rank": rank,
            "sources": sources,
            "leaf_ids": leaf_ids,
            "dense_score_max": round(max(dense), 8) if dense else None,
            "reranker_score_max": round(max(rerank), 8) if rerank else None,
            "row_reranker_score_max": round(max(row), 8) if row else None,
        }
        candidates.append(candidate)
    output = {
        "id": state.baseline["id"],
        "question": state.baseline["question"],
        "route": deepcopy(state.baseline["route"]),
        "candidates": candidates,
        "g3c": {
            "stage": stage,
            "leaf_count": len(state.leaves),
            "leaves": [leaf.to_dict() for leaf in state.leaves],
            "eligible_table_count": len(state.meta_by_key),
            "hard_constraint_violations": audit_hard_constraints(
                state, candidates
            ),
        },
    }
    return output


def audit_hard_constraints(
    state: QuestionState, candidates: list[dict]
) -> list[dict]:
    allowed: dict[TableKey, list[str]] = {}
    for leaf in state.leaves:
        for key in state.eligible_by_leaf[leaf.leaf_id]:
            allowed.setdefault(key, []).append(leaf.leaf_id)
    violations = []
    for rank, candidate in enumerate(candidates, 1):
        key = table_key(candidate)
        if key not in allowed:
            violations.append({
                "rank": rank,
                "report_id": key[0],
                "table_pos": key[1],
                "reason": "not_eligible_for_any_atomic_leaf",
            })
    return violations


def reciprocal_rank_fusion(
    rankings: Iterable[list[TableKey]], rrf_k: int
) -> dict[TableKey, float]:
    output: dict[TableKey, float] = {}
    for ranking in rankings:
        seen = set()
        for rank, key in enumerate(ranking, 1):
            if key in seen:
                continue
            seen.add(key)
            output[key] = output.get(key, 0.0) + 1.0 / (rrf_k + rank)
    return output


def rank_scores(scores: dict[TableKey, float]) -> list[TableKey]:
    return sorted(scores, key=lambda key: (-scores[key], key))


def select_with_leaf_quota(
    per_leaf_rankings: dict[str, list[TableKey]],
    global_ranking: list[TableKey],
    quota: int,
    depth: int,
) -> list[TableKey]:
    required: set[TableKey] = set()
    for leaf_id in sorted(per_leaf_rankings):
        required.update(per_leaf_rankings[leaf_id][:quota])

    selected: list[TableKey] = []
    seen = set()
    for key in global_ranking:
        if key in required and key not in seen:
            seen.add(key)
            selected.append(key)
            if len(selected) >= depth:
                return selected
    for key in sorted(required):
        if key not in seen:
            seen.add(key)
            selected.append(key)
            if len(selected) >= depth:
                return selected
    for key in global_ranking:
        if key not in seen:
            seen.add(key)
            selected.append(key)
            if len(selected) >= depth:
                break
    return selected


def _candidate_provenance(
    state: QuestionState, key: TableKey
) -> tuple[list[str], list[str]]:
    sources = []
    if key in state.r0_by_key:
        sources.append("r0")
    leaf_ids = []
    for leaf in state.leaves:
        leaf_id = leaf.leaf_id
        if key in state.eligible_by_leaf[leaf_id]:
            leaf_ids.append(leaf_id)
        if key in state.lexical_by_leaf.get(leaf_id, []):
            if "leaf_lexical" not in sources:
                sources.append("leaf_lexical")
        if key in state.dense_by_leaf.get(leaf_id, []):
            if "dense" not in sources:
                sources.append("dense")
        if (leaf_id, key) in state.rerank_scores:
            if "table_reranker" not in sources:
                sources.append("table_reranker")
        if (leaf_id, key) in state.row_scores:
            if "row_reranker" not in sources:
                sources.append("row_reranker")
    return sources, leaf_ids
