"""Deterministic, label-blind workload plan for exact two-T4 execution."""
from __future__ import annotations

import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

from ..extraction.build_store import Store
from ..g3c.common import canonical_json_sha256, write_json
from ..g3c.retrieval import QuestionState, prepare_state
from .common import WORKLOAD_SCHEMA, validate_execution_config


def prepare_states(
    baseline_rows: list[dict],
    store: Store,
    config: dict,
    *,
    description: str = "G3C official prepare",
) -> list[QuestionState]:
    states: list[QuestionState] = []
    for row in _progress(baseline_rows, description):
        state = prepare_state(row, store, config)
        if not state.leaves:
            raise ValueError(f"question {row['id']} produced zero leaves")
        missing = [leaf.to_dict() for leaf in state.leaves if not leaf.report_ids]
        if missing:
            raise ValueError(
                f"question {row['id']} has no exact report for leaves: {missing}"
            )
        states.append(state)
    return states


def prepare_states_total(
    baseline_rows: list[dict],
    store: Store,
    config: dict,
    *,
    description: str = "G3C official total prepare",
) -> tuple[list[QuestionState], dict[int, list[dict]]]:
    """Prepare every row and classify only frozen precondition failures.

    The original G3C runner raises when any atomic leaf has no exact report.
    Official execution totalizes that undefined case with an R0 passthrough;
    it never searches for a nearby report.
    """
    states = []
    unsupported = {}
    for state_index, row in enumerate(_progress(baseline_rows, description)):
        state = prepare_state(row, store, config)
        if not state.leaves:
            raise ValueError(f"question {row['id']} produced zero leaves")
        missing = [leaf.to_dict() for leaf in state.leaves if not leaf.report_ids]
        if missing:
            unsupported[state_index] = missing
        states.append(state)
    return states, unsupported


def r0_unsupported_record(
    state: QuestionState, missing_leaves: list[dict],
) -> dict:
    output = deepcopy(state.baseline)
    output["g3c"] = {
        "stage": "R4",
        "execution": "r0_passthrough_unsupported",
        "fallback_reason": "missing_exact_report_for_atomic_leaf",
        "missing_exact_report_leaves": missing_leaves,
        "leaf_count": len(state.leaves),
        "leaves": [leaf.to_dict() for leaf in state.leaves],
        "eligible_table_count": len(state.meta_by_key),
        "hard_constraint_violations": [],
    }
    return output


def embedding_catalog(
    states: list[QuestionState], *, state_indices: list[int] | None = None,
) -> tuple[dict, dict]:
    passages: dict[str, str] = {}
    queries: dict[str, str] = {}
    active_indices = (
        list(range(len(states))) if state_indices is None else list(state_indices)
    )
    for state_index in active_indices:
        state = states[state_index]
        for passage in state.passage_by_key.values():
            identity = passage["passage_id"]
            content = passage["content"]
            previous = passages.setdefault(identity, content)
            if previous != content:
                raise ValueError(f"passage identity collision: {identity}")
        for leaf in state.leaves:
            identity = f"{state_index}:{leaf.leaf_id}"
            if identity in queries:
                raise ValueError(f"query identity collision: {identity}")
            queries[identity] = state.query_by_leaf[leaf.leaf_id]
    return passages, queries


def exact_embedding_batches(
    states: list[QuestionState], *, batch_size: int,
    state_indices: list[int] | None = None,
) -> list[dict]:
    """Reproduce the frozen passage-then-query sorted batch boundaries."""
    passages, queries = embedding_catalog(states, state_indices=state_indices)
    output: list[dict] = []
    global_index = 0
    for kind, texts in (("table", passages), ("query", queries)):
        ordered = sorted(texts.items())
        for start in range(0, len(ordered), batch_size):
            items = ordered[start:start + batch_size]
            output.append({
                "batch_index": global_index,
                "kind": kind,
                "identities": [identity for identity, _ in items],
                "texts": [text for _, text in items],
                "content_sha256": [
                    canonical_json_sha256(text) for _, text in items
                ],
                "estimated_cost": max(
                    (len(text) for _, text in items), default=0
                ),
            })
            global_index += 1
    return output


def build_workload_plan(
    *,
    questions: list[dict],
    baseline_rows: list[dict],
    store_dir: Path | str,
    config: dict,
    execution: dict,
    output_path: Path | str | None = None,
    promotion_runtime: dict | None = None,
    promotion_vector_count: int | None = None,
    promotion_score_count: int | None = None,
) -> tuple[dict, list[QuestionState]]:
    validate_execution_config(execution)
    if len(questions) != 1012 or len(baseline_rows) != 1012:
        raise ValueError("official workload requires exactly 1,012 rows")
    question_ids = [str(row.get("id")) for row in questions]
    baseline_ids = [str(row.get("id")) for row in baseline_rows]
    if question_ids != baseline_ids or len(set(question_ids)) != 1012:
        raise ValueError("official question/R0 IDs are not an ordered bijection")
    for question, baseline in zip(questions, baseline_rows):
        if set(question) != {"id", "question"}:
            raise ValueError("official question rows may contain only id/question")
        if question["question"] != baseline.get("question"):
            raise ValueError(f"official question mismatch: {question['id']}")

    started = time.perf_counter()
    store = Store(store_dir)
    states, unsupported = prepare_states_total(baseline_rows, store, config)
    supported_indices = [
        index for index in range(len(states)) if index not in unsupported
    ]
    batches = exact_embedding_batches(
        states,
        batch_size=int(config["runtime"]["embedding_batch_size"]),
        state_indices=supported_indices,
    )
    passages, queries = embedding_catalog(
        states, state_indices=supported_indices
    )
    query_counts = Counter(queries.values())
    if any(count != 1 for count in query_counts.values()):
        raise ValueError(
            "cross-question duplicate serialized queries would violate the "
            "independent reranker-shard proof"
        )

    worker_count = int(execution["gpu_workers"])
    embedding_assignment = _greedy_assign(
        [
            (int(batch["batch_index"]), int(batch["estimated_cost"]))
            for batch in batches
        ],
        worker_count,
    )
    embedding_workers = []
    batch_by_index = {int(batch["batch_index"]): batch for batch in batches}
    for worker_index, indices in enumerate(embedding_assignment):
        indices = sorted(indices)
        embedding_workers.append({
            "worker_index": worker_index,
            "batch_indices": indices,
            "batch_count": len(indices),
            "vector_count": sum(
                len(batch_by_index[index]["identities"]) for index in indices
            ),
            "estimated_cost": sum(
                int(batch_by_index[index]["estimated_cost"])
                for index in indices
            ),
        })

    cap = int(config["retrieval"]["rerank_top_n_per_leaf"])
    row_cap = int(config["retrieval"]["row_lexical_prefilter"])
    table_length = int(config["runtime"]["reranker_max_length"])
    row_length = int(config["runtime"]["row_max_length"])
    question_work = []
    empty_eligible_questions = []
    empty_eligible_leaves = 0
    for state_index, state in enumerate(states):
        if state_index in unsupported:
            question_work.append({
                "state_index": state_index,
                "id": str(state.baseline["id"]),
                "leaf_count": len(state.leaves),
                "eligible_table_count": len(state.meta_by_key),
                "table_pair_count": 0,
                "row_pair_upper_bound": 0,
                "estimated_cost": 0,
                "unsupported": True,
            })
            continue
        table_pairs = sum(
            min(cap, len(state.eligible_by_leaf[leaf.leaf_id]))
            for leaf in state.leaves
        )
        empty_leaf_count = sum(
            not state.eligible_by_leaf[leaf.leaf_id] for leaf in state.leaves
        )
        if empty_leaf_count:
            empty_eligible_questions.append(str(state.baseline["id"]))
            empty_eligible_leaves += empty_leaf_count
        row_pair_upper = sum(
            row_cap
            for leaf in state.leaves
            if state.eligible_by_leaf[leaf.leaf_id]
        )
        weight = table_pairs * table_length + row_pair_upper * row_length
        question_work.append({
            "state_index": state_index,
            "id": str(state.baseline["id"]),
            "leaf_count": len(state.leaves),
            "eligible_table_count": len(state.meta_by_key),
            "table_pair_count": table_pairs,
            "row_pair_upper_bound": row_pair_upper,
            "estimated_cost": weight,
            "unsupported": False,
        })
    shard_count = int(execution["rerank_shards"])
    shard_assignment = _greedy_assign(
        [
            (int(item["state_index"]), int(item["estimated_cost"]))
            for item in question_work
        ],
        shard_count,
    )
    work_by_index = {
        int(item["state_index"]): item for item in question_work
    }
    shards = []
    for shard_index, indices in enumerate(shard_assignment):
        indices = sorted(indices)
        items = [work_by_index[index] for index in indices]
        shards.append({
            "shard_index": shard_index,
            "state_indices": indices,
            "question_ids": [item["id"] for item in items],
            "question_count": len(items),
            "leaf_count": sum(item["leaf_count"] for item in items),
            "table_pair_count": sum(item["table_pair_count"] for item in items),
            "row_pair_upper_bound": sum(
                item["row_pair_upper_bound"] for item in items
            ),
            "estimated_cost": sum(item["estimated_cost"] for item in items),
        })

    batch_summary = [
        {
            key: value for key, value in batch.items()
            if key != "texts"
        }
        for batch in batches
    ]
    projection = _runtime_projection(
        official_vector_count=len(passages) + len(queries),
        official_score_upper=sum(
            item["table_pair_count"] + item["row_pair_upper_bound"]
            for item in question_work
        ),
        worker_count=worker_count,
        shard_count=shard_count,
        promotion_runtime=promotion_runtime,
        promotion_vector_count=promotion_vector_count,
        promotion_score_count=promotion_score_count,
    )
    body = {
        "schema_version": WORKLOAD_SCHEMA,
        "question_count": len(states),
        "supported_r4_question_count": len(supported_indices),
        "unsupported_r0_passthrough_count": len(unsupported),
        "unsupported_r0_passthrough_ids": [
            str(states[index].baseline["id"]) for index in sorted(unsupported)
        ],
        "unsupported_reason": "missing_exact_report_for_atomic_leaf",
        "question_ids_sha256": canonical_json_sha256(question_ids),
        "question_texts_sha256": canonical_json_sha256([
            row["question"] for row in questions
        ]),
        "leaf_count": sum(len(state.leaves) for state in states),
        "unique_query_count": len(queries),
        "duplicate_query_text_count": 0,
        "unique_table_passage_count": len(passages),
        "embedding_batch_size": int(
            config["runtime"]["embedding_batch_size"]
        ),
        "embedding_batch_count": len(batches),
        "embedding_batches_sha256": canonical_json_sha256(batch_summary),
        "embedding_workers": embedding_workers,
        "reranker_batch_size": int(
            config["runtime"]["reranker_batch_size"]
        ),
        "rerank_shards": shards,
        "empty_eligible_question_count": len(empty_eligible_questions),
        "empty_eligible_question_ids": empty_eligible_questions,
        "empty_eligible_leaf_count": empty_eligible_leaves,
        "table_pair_count": sum(
            item["table_pair_count"] for item in question_work
        ),
        "row_pair_upper_bound": sum(
            item["row_pair_upper_bound"] for item in question_work
        ),
        "runtime_projection": projection,
        "build_seconds": round(time.perf_counter() - started, 6),
        "semantic_invariants": {
            "embedding_batches_are_frozen_batches": True,
            "embedding_batches_split_across_gpus": False,
            "reranker_questions_split_across_gpus": False,
            "cross_question_score_cache_keys_possible": False,
            "prior_qwen_cache_seeded": False,
            "gold_fields_read": False,
            "unsupported_questions_preserve_r0_rank": True,
            "fallback_report_search_used": False,
        },
    }
    body["workload_fingerprint"] = canonical_json_sha256(body)
    if output_path is not None:
        write_json(output_path, body)
    return body, states


def validate_workload_plan(plan: dict, execution: dict) -> None:
    validate_execution_config(execution)
    if plan.get("schema_version") != WORKLOAD_SCHEMA:
        raise ValueError("unknown official workload schema")
    expected = canonical_json_sha256({
        key: value for key, value in plan.items()
        if key != "workload_fingerprint"
    })
    if plan.get("workload_fingerprint") != expected:
        raise ValueError("official workload fingerprint mismatch")
    if int(plan.get("question_count", 0)) != 1012:
        raise ValueError("official workload question count drift")
    if int(plan.get("duplicate_query_text_count", -1)) != 0:
        raise ValueError("official workload has cross-question query collisions")
    if int(plan.get("supported_r4_question_count", 0)) + int(
        plan.get("unsupported_r0_passthrough_count", 0)
    ) != 1012:
        raise ValueError("official supported/unsupported partition is incomplete")
    all_ids = [
        qid for shard in plan["rerank_shards"]
        for qid in shard["question_ids"]
    ]
    if len(all_ids) != 1012 or len(set(all_ids)) != 1012:
        raise ValueError("rerank shards are not a 1,012-question partition")
    if len(plan["embedding_workers"]) != int(execution["gpu_workers"]):
        raise ValueError("embedding worker-count drift")
    batch_indices = [
        index for worker in plan["embedding_workers"]
        for index in worker["batch_indices"]
    ]
    if sorted(batch_indices) != list(range(int(plan["embedding_batch_count"]))):
        raise ValueError("embedding batches are not an exact partition")


def _greedy_assign(
    weighted_items: list[tuple[int, int]], bins: int,
) -> list[list[int]]:
    output = [[] for _ in range(bins)]
    totals = [0] * bins
    for identity, weight in sorted(
        weighted_items, key=lambda item: (-item[1], item[0])
    ):
        target = min(range(bins), key=lambda index: (totals[index], index))
        output[target].append(identity)
        totals[target] += int(weight)
    return output


def _runtime_projection(
    *,
    official_vector_count: int,
    official_score_upper: int,
    worker_count: int,
    shard_count: int,
    promotion_runtime: dict | None,
    promotion_vector_count: int | None,
    promotion_score_count: int | None,
) -> dict:
    if not promotion_runtime or not promotion_vector_count or not promotion_score_count:
        return {"available": False}
    timings = promotion_runtime["timings"]
    embedding_gpu_hours = (
        float(timings["embedding_seconds"])
        * official_vector_count / promotion_vector_count / 3600.0
    )
    reranker_gpu_hours_upper = (
        float(timings["reranker_seconds"])
        * official_score_upper / promotion_score_count / 3600.0
    )
    return {
        "available": True,
        "basis": "linear_from_55_question_promotion_conservative_row_upper",
        "promotion_total_minutes": round(
            float(timings["total_seconds"]) / 60.0, 3
        ),
        "embedding_gpu_hours": round(embedding_gpu_hours, 3),
        "embedding_two_t4_wall_hours": round(
            embedding_gpu_hours / worker_count, 3
        ),
        "reranker_gpu_hours_upper": round(reranker_gpu_hours_upper, 3),
        "reranker_wall_hours_per_fourth_upper": round(
            reranker_gpu_hours_upper / shard_count, 3
        ),
        "monolithic_two_t4_wall_hours_upper": round(
            (embedding_gpu_hours + reranker_gpu_hours_upper) / worker_count, 3
        ),
        "recommended_runs": [
            "embedding_two_t4",
            "rerank_shards_0_1_two_t4",
            "rerank_shards_2_3_two_t4",
        ],
    }


def _progress(iterable, description: str):
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=description)
    except ImportError:
        return iterable
