"""Build and execute exact numeric canaries from the frozen Promotion run."""
from __future__ import annotations

import os
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np

from ..extraction.build_store import Store
from ..g3c.cache import (
    ScoreCache,
    VectorCache,
    score_cache_key,
    vector_cache_key,
)
from ..g3c.common import (
    canonical_json_sha256,
    config_fingerprint,
    load_config,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)
from ..g3c.modeling import build_embedding_backend, build_reranker_backend
from ..g3c.payload import validate_gpu_payload
from ..g3c.retrieval import (
    attach_dense_scores,
    attach_reranker_scores,
    attach_row_scores,
    r3_record,
    r4_record,
    rerank_pools,
)
from ..g3c.serialize import table_key
from .common import CANARY_SCHEMA, load_fingerprinted
from .workload import exact_embedding_batches, prepare_states


def build_numeric_canary(
    *,
    promotion_payload_dir: Path | str,
    promotion_result_dir: Path | str,
    manifest_output_path: Path | str,
    vectors_output_path: Path | str,
) -> dict:
    promotion_payload_dir = Path(promotion_payload_dir)
    promotion_result_dir = Path(promotion_result_dir)
    manifest_output_path = Path(manifest_output_path)
    vectors_output_path = Path(vectors_output_path)
    payload = validate_gpu_payload(promotion_payload_dir)
    if payload.get("mode") != "promotion" or payload.get("selected_stage") != "R4":
        raise ValueError("numeric canary requires the frozen R4 Promotion payload")
    config = load_config(promotion_payload_dir / payload["paths"]["config"])
    questions = read_jsonl(promotion_payload_dir / payload["paths"]["questions"])
    baseline = read_jsonl(
        promotion_payload_dir / payload["paths"]["baseline_retrieval"]
    )
    if [str(row["id"]) for row in questions] != [
        str(row["id"]) for row in baseline
    ]:
        raise ValueError("Promotion canary question/R0 mismatch")
    states = prepare_states(
        baseline,
        Store(promotion_payload_dir / payload["paths"]["store"]),
        config,
        description="G3C canary replay prepare",
    )
    vector_cache = VectorCache(
        promotion_result_dir / "cache/embedding_vectors.npz"
    )
    score_cache = ScoreCache(
        promotion_result_dir / "cache/reranker_scores.json"
    )
    cache_contract = canonical_json_sha256({
        "config_sha256": config_fingerprint(config),
        "protocol_fingerprint": payload["protocol_fingerprint"],
    })
    embedding_revision = config["models"]["embedding"]["revision"]
    embedding_instruction = config["instructions"]["embedding_query"]
    embedding_batches = exact_embedding_batches(
        states,
        batch_size=int(config["runtime"]["embedding_batch_size"]),
    )
    selected_embedding = []
    expected_arrays = {}
    for kind in ("table", "query"):
        candidates = [batch for batch in embedding_batches if batch["kind"] == kind]
        for case_number, batch in enumerate(_select_batches(candidates)):
            expected = []
            cache_keys = []
            for text in batch["texts"]:
                key = vector_cache_key(
                    contract_fingerprint=cache_contract,
                    backend="qwen",
                    model_revision=embedding_revision,
                    kind=kind,
                    instruction=embedding_instruction if kind == "query" else "",
                    content=text,
                )
                vector = vector_cache.get(key)
                if vector is None:
                    raise ValueError(f"Promotion vector cache missing canary key {key}")
                cache_keys.append(key)
                expected.append(vector)
            array_name = f"{kind}_{case_number}"
            expected_arrays[array_name] = np.stack(expected)
            selected_embedding.append({
                "case_id": array_name,
                "kind": kind,
                "is_query": kind == "query",
                "instruction": embedding_instruction if kind == "query" else "",
                "texts": batch["texts"],
                "content_sha256": batch["content_sha256"],
                "cache_keys": cache_keys,
                "expected_array": array_name,
                "source_batch_index": batch["batch_index"],
            })

    query_key_by_state_leaf: dict[tuple[int, str], str] = {}
    passage_key_by_id: dict[str, str] = {}
    for state_index, state in enumerate(states):
        for leaf in state.leaves:
            text = state.query_by_leaf[leaf.leaf_id]
            query_key_by_state_leaf[(state_index, leaf.leaf_id)] = vector_cache_key(
                contract_fingerprint=cache_contract,
                backend="qwen",
                model_revision=embedding_revision,
                kind="query",
                instruction=embedding_instruction,
                content=text,
            )
        for passage in state.passage_by_key.values():
            passage_key_by_id[passage["passage_id"]] = vector_cache_key(
                contract_fingerprint=cache_contract,
                backend="qwen",
                model_revision=embedding_revision,
                kind="table",
                instruction="",
                content=passage["content"],
            )

    top_n = int(config["retrieval"]["dense_top_n_per_leaf"])
    for state_index, state in enumerate(states):
        query_vectors = {
            leaf.leaf_id: _required_vector(
                vector_cache, query_key_by_state_leaf[(state_index, leaf.leaf_id)]
            )
            for leaf in state.leaves
        }
        passage_vectors = {
            passage["passage_id"]: _required_vector(
                vector_cache, passage_key_by_id[passage["passage_id"]]
            )
            for passage in state.passage_by_key.values()
        }
        attach_dense_scores(state, query_vectors, passage_vectors, top_n)

    reranker_revision = config["models"]["reranker"]["revision"]
    reranker_instruction = config["instructions"]["reranker"]
    table_calls: list[dict] = []
    row_calls: list[dict] = []

    def cached_call(pairs: list[tuple[str, str]], *, kind: str) -> list[float]:
        values = []
        keys = []
        for query, document in pairs:
            key = score_cache_key(
                contract_fingerprint=cache_contract,
                backend="qwen",
                model_revision=reranker_revision,
                kind=kind,
                instruction=reranker_instruction,
                query=query,
                document=document,
            )
            value = score_cache.get(key)
            if value is None:
                raise ValueError(f"Promotion score cache missing canary key {key}")
            keys.append(key)
            values.append(float(value))
        (table_calls if kind == "table" else row_calls).append({
            "kind": kind,
            "pairs": pairs,
            "keys": keys,
            "values": values,
        })
        return values

    cap = int(config["retrieval"]["rerank_top_n_per_leaf"])
    rrf_k = int(config["retrieval"]["rrf_k"])
    for state in states:
        pools = rerank_pools(state, cap, rrf_k)
        pairs = []
        pair_keys = []
        for leaf in state.leaves:
            for key in pools[leaf.leaf_id]:
                pairs.append((
                    state.query_by_leaf[leaf.leaf_id],
                    state.passage_by_key[key]["content"],
                ))
                pair_keys.append((leaf.leaf_id, key))
        attach_reranker_scores(
            state, dict(zip(pair_keys, cached_call(pairs, kind="table")))
        )
    r3_rows = [r3_record(state, config) for state in states]
    store = Store(promotion_payload_dir / payload["paths"]["store"])
    for state, r3 in zip(states, r3_rows):
        selected = [table_key(candidate) for candidate in r3["candidates"]]
        attach_row_scores(state, store, selected, cached_call, config)
    replay_rows = [r4_record(state, config) for state in states]
    reference_rows = read_jsonl(promotion_result_dir / "r4_retrieval.jsonl")
    replay_equivalence = _semantic_replay_equivalence(
        replay_rows, reference_rows
    )
    if not replay_equivalence["passed"]:
        raise ValueError(
            "cached Promotion replay is not semantically R4-equivalent: "
            f"{replay_equivalence}"
        )

    reranker_cases = []
    for kind, calls in (("table", table_calls), ("row", row_calls)):
        batches = []
        batch_size = int(config["runtime"]["reranker_batch_size"])
        for call_index, call in enumerate(calls):
            for start in range(0, len(call["pairs"]), batch_size):
                batches.append({
                    "call_index": call_index,
                    "batch_start": start,
                    "kind": kind,
                    "pairs": call["pairs"][start:start + batch_size],
                    "keys": call["keys"][start:start + batch_size],
                    "values": call["values"][start:start + batch_size],
                })
        for case_number, batch in enumerate(_select_batches(batches)):
            reranker_cases.append({
                "case_id": f"{kind}_{case_number}",
                "kind": kind,
                "pairs": batch["pairs"],
                "cache_keys": batch["keys"],
                "expected_scores": batch["values"],
                "instruction": reranker_instruction,
                "max_length": int(config["runtime"][
                    "row_max_length" if kind == "row" else "reranker_max_length"
                ]),
                "source_call_index": batch["call_index"],
                "source_batch_start": batch["batch_start"],
            })

    _write_npz_atomic(vectors_output_path, expected_arrays)
    body = {
        "schema_version": CANARY_SCHEMA,
        "source": "frozen_g3c_one_shot_promotion_qwen_cache",
        "promotion_payload_fingerprint": payload["payload_fingerprint"],
        "promotion_r4_sha256": sha256_file(
            promotion_result_dir / "r4_retrieval.jsonl"
        ),
        "config_sha256": config_fingerprint(config),
        "protocol_fingerprint": payload["protocol_fingerprint"],
        "embedding_cases": selected_embedding,
        "reranker_cases": reranker_cases,
        "expected_vectors": {
            "path": vectors_output_path.name,
            "size": vectors_output_path.stat().st_size,
            "sha256": sha256_file(vectors_output_path),
        },
        "exact_cached_replay_passed": True,
        "cached_replay_equivalence": replay_equivalence,
        "gold_fields_included": False,
    }
    body["canary_fingerprint"] = canonical_json_sha256(body)
    write_json(manifest_output_path, body)
    return body


def run_embedding_canary(
    *,
    backend,
    canary_manifest_path: Path | str,
    expected_vectors_path: Path | str,
) -> dict:
    canary = load_numeric_canary(canary_manifest_path, expected_vectors_path)
    failures = []
    with np.load(expected_vectors_path, allow_pickle=False) as archive:
        for case in canary["embedding_cases"]:
            actual = backend.encode(
                case["texts"],
                is_query=bool(case["is_query"]),
                instruction=case["instruction"],
            )
            expected = archive[case["expected_array"]]
            if not np.array_equal(actual, expected):
                delta = float(np.max(np.abs(
                    actual.astype(np.float32) - expected.astype(np.float32)
                )))
                failures.append({"case_id": case["case_id"], "max_abs": delta})
    if failures:
        raise RuntimeError(f"embedding exact-canary mismatch: {failures}")
    return {"passed": True, "case_count": len(canary["embedding_cases"])}


def run_reranker_canary(
    *, backend, canary_manifest_path: Path | str,
    expected_vectors_path: Path | str,
) -> dict:
    canary = load_numeric_canary(canary_manifest_path, expected_vectors_path)
    failures = []
    for case in canary["reranker_cases"]:
        actual = backend.score(
            [(str(q), str(d)) for q, d in case["pairs"]],
            instruction=case["instruction"],
            max_length=int(case["max_length"]),
        )
        expected = [float(value) for value in case["expected_scores"]]
        if actual != expected:
            failures.append({
                "case_id": case["case_id"],
                "max_abs": max(
                    (abs(a - b) for a, b in zip(actual, expected)), default=0.0
                ),
            })
    if failures:
        raise RuntimeError(f"reranker exact-canary mismatch: {failures}")
    return {"passed": True, "case_count": len(canary["reranker_cases"])}


def load_numeric_canary(
    manifest_path: Path | str, vectors_path: Path | str,
) -> dict:
    canary = load_fingerprinted(
        manifest_path,
        schema=CANARY_SCHEMA,
        fingerprint_field="canary_fingerprint",
    )
    vectors_path = Path(vectors_path)
    record = canary["expected_vectors"]
    if (
        vectors_path.name != record["path"]
        or not vectors_path.is_file()
        or vectors_path.stat().st_size != int(record["size"])
        or sha256_file(vectors_path) != record["sha256"]
    ):
        raise ValueError("numeric canary vector artifact mismatch")
    return canary


def _required_vector(cache: VectorCache, key: str) -> np.ndarray:
    value = cache.get(key)
    if value is None:
        raise ValueError(f"required vector missing: {key}")
    return value


def _select_batches(batches: list[dict]) -> list[dict]:
    if not batches:
        raise ValueError("cannot select a canary from zero batches")
    positions = {0, len(batches) // 2, len(batches) - 1}
    partial = [
        index for index, batch in enumerate(batches)
        if len(batch.get("texts", batch.get("pairs", []))) == 1
    ]
    if partial:
        positions.add(partial[-1])
    return [batches[index] for index in sorted(positions)]


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.stem}.", suffix=".npz"
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temp_name, **arrays)
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _first_difference(left, right, path: str = "$") -> str:
    if type(left) is not type(right):
        return f"{path}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if set(left) != set(right):
            return f"{path}: keys {sorted(set(left) ^ set(right))}"
        for key in left:
            if left[key] != right[key]:
                return _first_difference(left[key], right[key], f"{path}.{key}")
        return f"{path}: unknown dict difference"
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for index, (a, b) in enumerate(zip(left, right)):
            if a != b:
                return _first_difference(a, b, f"{path}[{index}]")
        return f"{path}: unknown list difference"
    return f"{path}: {left!r} != {right!r}"


def _semantic_replay_equivalence(left: list[dict], right: list[dict]) -> dict:
    """Require exact output except bounded local BLAS dense diagnostics.

    Promotion was materialized with NumPy 2.0.2 while local replay may use a
    newer NumPy build. The cached FP16 vectors, candidate ranks, RRF output,
    reranker values and submission projection must remain exact. Only the
    non-ranking ``dense_score_max`` diagnostic may differ by at most 1e-6.
    """
    left_copy = deepcopy(left)
    right_copy = deepcopy(right)
    deltas = []
    for a_row, b_row in zip(left_copy, right_copy):
        for a_candidate, b_candidate in zip(
            a_row.get("candidates", []), b_row.get("candidates", [])
        ):
            a_value = a_candidate.get("g3c", {}).get("dense_score_max")
            b_value = b_candidate.get("g3c", {}).get("dense_score_max")
            if a_value is not None and b_value is not None:
                deltas.append(abs(float(a_value) - float(b_value)))
            elif a_value != b_value:
                return {
                    "passed": False,
                    "reason": "dense diagnostic nullability mismatch",
                }
            a_candidate.get("g3c", {})["dense_score_max"] = None
            b_candidate.get("g3c", {})["dense_score_max"] = None
    if left_copy != right_copy:
        return {
            "passed": False,
            "reason": _first_difference(left_copy, right_copy),
        }
    maximum = max(deltas, default=0.0)
    if maximum > 1e-6:
        return {
            "passed": False,
            "reason": "dense diagnostic exceeded 1e-6",
            "dense_score_max_abs_delta": maximum,
        }
    return {
        "passed": True,
        "rank_exact": True,
        "submission_projection_exact": True,
        "all_non_dense_diagnostic_fields_exact": True,
        "dense_score_max_abs_delta": maximum,
        "dense_score_tolerance": 1e-6,
        "tolerance_reason": "local NumPy BLAS differs from Kaggle NumPy 2.0.2",
    }
