from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from vifinqa.g3c.common import read_json
from vifinqa.g3c_official.canary import (
    _semantic_replay_equivalence,
    load_numeric_canary,
)
from vifinqa.g3c_official.closeout import validate_pb_closeout
from vifinqa.g3c_official.common import (
    load_execution_config,
    validate_execution_config,
)
from vifinqa.g3c_official.execution import (
    _validate_vector_chunk,
    _write_vector_chunk,
)
from vifinqa.g3c_official.finalize import audit_official_rows
from vifinqa.g3c_official.workload import (
    _greedy_assign,
    exact_embedding_batches,
)


def test_official_execution_forbids_semantic_shortcuts():
    config = load_execution_config("configs/g3c_official_1012_v1.json")
    for field in (
        "allow_approximate_search",
        "allow_quantization",
        "allow_cache_seed_from_prior_runs",
    ):
        changed = deepcopy(config)
        changed["equivalence_policy"][field] = True
        with pytest.raises(ValueError):
            validate_execution_config(changed)
    changed = deepcopy(config)
    changed["equivalence_policy"]["preserve_batch_sizes"] = False
    with pytest.raises(ValueError):
        validate_execution_config(changed)


def test_pb_closeout_and_numeric_canary_are_hash_bound():
    root = Path(__file__).resolve().parents[1]
    closeout, registry = validate_pb_closeout(
        repo_root=root,
        closeout_path=(
            root / "experiments/g3c_qwen_retrieval_v1/pb_r4_closeout.json"
        ),
        registry_path=root / "experiments/g3c_qwen_retrieval_v1/registry.json",
    )
    assert closeout["status"] == "g3c_complete_pb_r4_frozen"
    assert closeout["promotion_evaluator_consumed"] is True
    assert registry["promotion"]["runs_consumed"] == 1
    canary_dir = root / "artifacts/g3c_v1/official_preflight"
    canary = load_numeric_canary(
        canary_dir / "exact_numeric_canary.json",
        canary_dir / "exact_numeric_canary_vectors.npz",
    )
    assert canary["exact_cached_replay_passed"] is True
    assert canary["cached_replay_equivalence"]["rank_exact"] is True


class _Leaf:
    def __init__(self, leaf_id: str):
        self.leaf_id = leaf_id


class _State:
    def __init__(self, passages, queries):
        self.passage_by_key = {
            (f"r{index}", index): value
            for index, value in enumerate(passages)
        }
        self.leaves = [_Leaf(key) for key in queries]
        self.query_by_leaf = queries


def test_embedding_batches_keep_sorted_frozen_boundaries():
    state = _State(
        [
            {"passage_id": "p3", "content": "three"},
            {"passage_id": "p1", "content": "one"},
            {"passage_id": "p2", "content": "two"},
        ],
        {"leaf-b": "query b", "leaf-a": "query a"},
    )
    batches = exact_embedding_batches([state], batch_size=2)
    assert [(row["kind"], row["identities"]) for row in batches] == [
        ("table", ["p1", "p2"]),
        ("table", ["p3"]),
        ("query", ["0:leaf-a", "0:leaf-b"]),
    ]
    assignment = _greedy_assign(
        [(row["batch_index"], row["estimated_cost"]) for row in batches], 2
    )
    assert sorted(index for group in assignment for index in group) == [0, 1, 2]


def test_vector_chunks_are_atomic_finite_and_assignment_bound(tmp_path):
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    record = _write_vector_chunk(
        chunks_dir=chunks,
        chunk_index=0,
        worker_index=1,
        batch_indices=[4, 9],
        arrays={
            "a": np.array([1, 2], dtype=np.float16),
            "b": np.array([3, 4], dtype=np.float16),
        },
    )
    validated = _validate_vector_chunk(
        chunks,
        chunks / "chunk_0000.json",
        1,
        [{"batch_index": 4}, {"batch_index": 9}],
    )
    assert validated["chunk_fingerprint"] == record["chunk_fingerprint"]
    with pytest.raises(ValueError, match="batch assignment"):
        _validate_vector_chunk(
            chunks,
            chunks / "chunk_0000.json",
            1,
            [{"batch_index": 4}],
        )


def test_semantic_replay_allows_only_tiny_dense_diagnostic_delta():
    base = [{
        "id": 1,
        "candidates": [{
            "report_id": "R",
            "table_pos": 1,
            "score": 100.0,
            "g3c": {
                "dense_score_max": 0.5,
                "reranker_score_max": 0.75,
            },
        }],
    }]
    changed = deepcopy(base)
    changed[0]["candidates"][0]["g3c"]["dense_score_max"] += 5e-7
    assert _semantic_replay_equivalence(base, changed)["passed"] is True
    changed[0]["candidates"][0]["g3c"]["reranker_score_max"] = 0.74
    assert _semantic_replay_equivalence(base, changed)["passed"] is False


def test_official_audit_checks_all_1012_without_gold():
    questions = [
        {"id": index, "question": f"q{index}"}
        for index in range(1, 1013)
    ]
    baseline = [
        {"id": row["id"], "question": row["question"], "route": {}, "candidates": []}
        for row in questions
    ]
    r4 = [
        {
            "id": row["id"],
            "question": row["question"],
            "route": {},
            "candidates": [],
            "g3c": {
                "stage": "R4",
                "hard_constraint_violations": [],
            },
        }
        for row in questions
    ]
    payload = {
        "payload_fingerprint": "a" * 64,
        "candidate_fingerprint": "b" * 64,
    }
    audit = audit_official_rows(
        questions=questions,
        baseline=baseline,
        r4_rows=r4,
        payload=payload,
        expected_fallback_ids=set(),
    )
    assert audit["passed"] is True
    assert audit["question_count"] == 1012
    assert audit["empty_candidate_count"] == 1012
    assert audit["public_score_read"] is False
