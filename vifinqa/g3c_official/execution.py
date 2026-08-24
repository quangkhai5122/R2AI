"""Exact-batch embedding and whole-question reranking on two T4 workers."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ..extraction.build_store import Store
from ..g3c.cache import ScoreCache, score_cache_key, vector_cache_key
from ..g3c.common import (
    canonical_json_sha256,
    config_fingerprint,
    load_config,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from ..g3c.modeling import (
    build_embedding_backend,
    build_reranker_backend,
    runtime_fingerprint,
)
from ..g3c.retrieval import (
    attach_dense_scores,
    attach_reranker_scores,
    attach_row_scores,
    r3_record,
    r4_record,
    rerank_pools,
)
from ..g3c.serialize import table_key
from .canary import run_embedding_canary, run_reranker_canary
from .common import (
    OFFICIAL_EMBEDDING_SCHEMA,
    OFFICIAL_SHARD_SCHEMA,
)
from .payload import validate_official_payload
from .workload import (
    exact_embedding_batches,
    prepare_states_total,
    r0_unsupported_record,
    validate_workload_plan,
)

_VECTOR_CHUNK_SCHEMA = "g3c_official_vector_chunk_v1"
_RERANK_PAIR_SCHEMA = "g3c_official_rerank_pair_v1"


def run_embedding_orchestrator(
    *,
    payload_dir: Path | str,
    output_dir: Path | str,
    runner_path: Path | str,
    backend: str = "qwen",
) -> dict:
    payload_dir = Path(payload_dir).resolve()
    output_dir = Path(output_dir).resolve()
    runner_path = Path(runner_path).resolve()
    payload = validate_official_payload(payload_dir)
    execution = read_json(payload_dir / payload["paths"]["execution_config"])
    workload = read_json(payload_dir / payload["paths"]["workload"])
    validate_workload_plan(workload, execution)
    manifest_path = output_dir / "g3c_official_embedding_manifest.json"
    signature = canonical_json_sha256({
        "payload_fingerprint": payload["payload_fingerprint"],
        "workload_fingerprint": workload["workload_fingerprint"],
        "phase": "embedding",
        "backend": backend,
        "workers": 2,
    })
    if manifest_path.exists():
        existing = read_json(manifest_path)
        validate_embedding_result(
            payload_dir=payload_dir,
            result_dir=output_dir,
            expected_signature=signature,
            require_qwen=backend == "qwen",
        )
        return existing
    output_dir.mkdir(parents=True, exist_ok=True)
    if backend == "qwen":
        _require_two_matching_gpus()
    started = time.perf_counter()
    processes = []
    for worker_index in range(2):
        worker_dir = output_dir / f"worker_{worker_index}"
        command = [
            sys.executable,
            str(runner_path),
            "worker-embed",
            "--payload", str(payload_dir),
            "--out", str(worker_dir),
            "--worker-index", str(worker_index),
            "--backend", backend,
        ]
        environment = os.environ.copy()
        if backend == "qwen":
            environment["CUDA_VISIBLE_DEVICES"] = str(worker_index)
        processes.append(subprocess.Popen(
            command,
            env=environment,
            cwd=str(runner_path.parent),
        ))
    failures = []
    for worker_index, process in enumerate(processes):
        code = process.wait()
        if code:
            failures.append({"worker_index": worker_index, "exit_code": code})
    if failures:
        raise RuntimeError(f"embedding workers failed: {failures}")

    worker_records = []
    all_keys = set()
    for worker_index in range(2):
        worker_dir = output_dir / f"worker_{worker_index}"
        worker = _validate_embedding_worker(
            worker_dir,
            payload=payload,
            workload=workload,
            worker_index=worker_index,
            backend=backend,
        )
        overlap = all_keys & set(worker["vector_keys"])
        if overlap:
            raise ValueError(f"embedding workers overlap keys: {sorted(overlap)[:3]}")
        all_keys.update(worker["vector_keys"])
        worker_records.append({
            "worker_index": worker_index,
            "path": worker_dir.name,
            "manifest_sha256": sha256_file(
                worker_dir / "worker_manifest.json"
            ),
            "vector_count": worker["vector_count"],
            "batch_count": worker["batch_count"],
            "canary": worker["canary"],
        })
    expected_vectors = (
        int(workload["unique_table_passage_count"])
        + int(workload["unique_query_count"])
    )
    if len(all_keys) != expected_vectors:
        raise ValueError(
            f"embedding vector coverage {len(all_keys)} != {expected_vectors}"
        )
    result = {
        "schema_version": OFFICIAL_EMBEDDING_SCHEMA,
        "run_signature": signature,
        "backend": backend,
        "scientific_execution": backend == "qwen",
        "payload_fingerprint": payload["payload_fingerprint"],
        "official_protocol_fingerprint": payload[
            "official_protocol_fingerprint"
        ],
        "workload_fingerprint": workload["workload_fingerprint"],
        "question_count": 1012,
        "vector_count": len(all_keys),
        "table_vector_count": int(workload["unique_table_passage_count"]),
        "query_vector_count": int(workload["unique_query_count"]),
        "workers": worker_records,
        "cache_seeded_from_prior_qwen_runs": False,
        "exact_canary_passed_on_both_gpus": all(
            record["canary"].get("passed") is True
            for record in worker_records
        ),
        "total_seconds": round(time.perf_counter() - started, 6),
    }
    write_json(manifest_path, result)
    validate_embedding_result(
        payload_dir=payload_dir,
        result_dir=output_dir,
        expected_signature=signature,
        require_qwen=backend == "qwen",
    )
    return result


def run_embedding_worker(
    *,
    payload_dir: Path | str,
    output_dir: Path | str,
    worker_index: int,
    backend: str = "qwen",
) -> dict:
    payload_dir = Path(payload_dir).resolve()
    output_dir = Path(output_dir).resolve()
    payload = validate_official_payload(payload_dir)
    config = load_config(payload_dir / payload["paths"]["config"])
    execution = read_json(payload_dir / payload["paths"]["execution_config"])
    workload = read_json(payload_dir / payload["paths"]["workload"])
    validate_workload_plan(workload, execution)
    assignment = workload["embedding_workers"][worker_index]
    if int(assignment["worker_index"]) != worker_index:
        raise ValueError("embedding worker assignment mismatch")
    manifest_path = output_dir / "worker_manifest.json"
    worker_signature = canonical_json_sha256({
        "payload_fingerprint": payload["payload_fingerprint"],
        "workload_fingerprint": workload["workload_fingerprint"],
        "worker_index": worker_index,
        "batch_indices": assignment["batch_indices"],
        "backend": backend,
    })
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing.get("worker_signature") != worker_signature:
            raise ValueError("existing embedding worker signature mismatch")
        _validate_embedding_worker(
            output_dir,
            payload=payload,
            workload=workload,
            worker_index=worker_index,
            backend=backend,
        )
        return existing

    questions = read_jsonl(payload_dir / payload["paths"]["questions"])
    baseline = read_jsonl(
        payload_dir / payload["paths"]["baseline_retrieval"]
    )
    if len(questions) != len(baseline):
        raise ValueError("official question/R0 count mismatch")
    states, unsupported = prepare_states_total(
        baseline,
        Store(payload_dir / payload["paths"]["store"]),
        config,
        description=f"official embed worker {worker_index} prepare",
    )
    unsupported_ids = [
        str(states[index].baseline["id"]) for index in sorted(unsupported)
    ]
    if unsupported_ids != workload["unsupported_r0_passthrough_ids"]:
        raise ValueError("official unsupported-question partition drifted")
    supported_indices = [
        index for index in range(len(states)) if index not in unsupported
    ]
    batches = exact_embedding_batches(
        states,
        batch_size=int(config["runtime"]["embedding_batch_size"]),
        state_indices=supported_indices,
    )
    summary = [
        {key: value for key, value in batch.items() if key != "texts"}
        for batch in batches
    ]
    if canonical_json_sha256(summary) != workload["embedding_batches_sha256"]:
        raise ValueError("reconstructed frozen embedding batches drifted")
    selected = [
        batches[index] for index in assignment["batch_indices"]
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    cache_contract = _cache_contract(payload, config)
    revision = config["models"]["embedding"]["revision"]
    query_instruction = config["instructions"]["embedding_query"]
    checkpoint = int(execution["embedding_checkpoint_batches"])
    groups = [
        selected[start:start + checkpoint]
        for start in range(0, len(selected), checkpoint)
    ]
    completed_keys = set()
    chunk_records = []
    pending_groups = []
    for chunk_index, group in enumerate(groups):
        chunk_manifest_path = chunks_dir / f"chunk_{chunk_index:04d}.json"
        if chunk_manifest_path.exists():
            record = _validate_vector_chunk(
                chunks_dir, chunk_manifest_path, worker_index, group
            )
            overlap = completed_keys & set(record["vector_keys"])
            if overlap:
                raise ValueError("duplicate vector keys across existing chunks")
            completed_keys.update(record["vector_keys"])
            chunk_records.append(record)
        else:
            pending_groups.append((chunk_index, group))

    model = None
    canary_report = {"passed": backend != "qwen", "case_count": 0}
    started = time.perf_counter()
    try:
        if pending_groups or backend == "qwen":
            model = build_embedding_backend(
                backend, config["models"]["embedding"], config["runtime"]
            )
        if backend == "qwen":
            canary_report = run_embedding_canary(
                backend=model,
                canary_manifest_path=(
                    payload_dir / payload["paths"]["numeric_canary"]
                ),
                expected_vectors_path=(
                    payload_dir / payload["paths"]["numeric_canary_vectors"]
                ),
            )
        for chunk_index, group in _progress(
            pending_groups,
            f"official embed worker {worker_index} chunks",
        ):
            arrays = {}
            batch_indices = []
            for batch in group:
                values = model.encode(
                    batch["texts"],
                    is_query=batch["kind"] == "query",
                    instruction=(
                        query_instruction if batch["kind"] == "query" else ""
                    ),
                )
                if len(values) != len(batch["texts"]):
                    raise RuntimeError("embedding backend returned wrong count")
                for text, vector in zip(batch["texts"], values):
                    key = vector_cache_key(
                        contract_fingerprint=cache_contract,
                        backend=backend,
                        model_revision=revision,
                        kind=batch["kind"],
                        instruction=(
                            query_instruction
                            if batch["kind"] == "query" else ""
                        ),
                        content=text,
                    )
                    if key in arrays or key in completed_keys:
                        raise ValueError("duplicate vector key in exact workload")
                    arrays[key] = np.asarray(vector, dtype=np.float16)
                batch_indices.append(int(batch["batch_index"]))
            record = _write_vector_chunk(
                chunks_dir=chunks_dir,
                chunk_index=chunk_index,
                worker_index=worker_index,
                batch_indices=batch_indices,
                arrays=arrays,
            )
            completed_keys.update(record["vector_keys"])
            chunk_records.append(record)
    finally:
        if model is not None:
            model.close()
    chunk_records.sort(key=lambda item: int(item["chunk_index"]))
    if len(completed_keys) != int(assignment["vector_count"]):
        raise ValueError("embedding worker vector count incomplete")
    result = {
        "schema_version": "g3c_official_embedding_worker_v1",
        "worker_signature": worker_signature,
        "worker_index": worker_index,
        "backend": backend,
        "payload_fingerprint": payload["payload_fingerprint"],
        "workload_fingerprint": workload["workload_fingerprint"],
        "batch_count": len(assignment["batch_indices"]),
        "batch_indices": assignment["batch_indices"],
        "vector_count": len(completed_keys),
        "vector_keys": sorted(completed_keys),
        "chunks": [
            {
                "chunk_index": record["chunk_index"],
                "manifest": f"chunks/chunk_{int(record['chunk_index']):04d}.json",
                "manifest_sha256": sha256_file(
                    chunks_dir / f"chunk_{int(record['chunk_index']):04d}.json"
                ),
                "archive": f"chunks/{record['archive']['path']}",
                "archive_sha256": record["archive"]["sha256"],
                "vector_count": record["vector_count"],
            }
            for record in chunk_records
        ],
        "canary": canary_report,
        "runtime": runtime_fingerprint(backend),
        "total_seconds": round(time.perf_counter() - started, 6),
    }
    write_json(manifest_path, result)
    return result


def run_rerank_pair_orchestrator(
    *,
    payload_dir: Path | str,
    embedding_result_dir: Path | str,
    output_dir: Path | str,
    runner_path: Path | str,
    shard_indices: tuple[int, int],
    backend: str = "qwen",
) -> dict:
    payload_dir = Path(payload_dir).resolve()
    embedding_result_dir = Path(embedding_result_dir).resolve()
    output_dir = Path(output_dir).resolve()
    runner_path = Path(runner_path).resolve()
    if len(shard_indices) != 2 or len(set(shard_indices)) != 2:
        raise ValueError("a rerank pair must contain two distinct shards")
    payload = validate_official_payload(payload_dir)
    execution = read_json(payload_dir / payload["paths"]["execution_config"])
    workload = read_json(payload_dir / payload["paths"]["workload"])
    validate_workload_plan(workload, execution)
    embedding = validate_embedding_result(
        payload_dir=payload_dir,
        result_dir=embedding_result_dir,
        require_qwen=backend == "qwen",
    )
    if any(index < 0 or index >= len(workload["rerank_shards"]) for index in shard_indices):
        raise ValueError("rerank pair shard index out of range")
    manifest_path = output_dir / "g3c_official_rerank_pair_manifest.json"
    signature = canonical_json_sha256({
        "payload_fingerprint": payload["payload_fingerprint"],
        "embedding_run_signature": embedding["run_signature"],
        "workload_fingerprint": workload["workload_fingerprint"],
        "shard_indices": list(shard_indices),
        "backend": backend,
    })
    if manifest_path.exists():
        existing = read_json(manifest_path)
        validate_rerank_pair_result(
            payload_dir=payload_dir,
            embedding_result_dir=embedding_result_dir,
            result_dir=output_dir,
            expected_shards=shard_indices,
            expected_signature=signature,
            require_qwen=backend == "qwen",
        )
        return existing
    output_dir.mkdir(parents=True, exist_ok=True)
    if backend == "qwen":
        _require_two_matching_gpus()
    started = time.perf_counter()
    processes = []
    for local_worker, shard_index in enumerate(shard_indices):
        shard_dir = output_dir / f"shard_{shard_index}"
        command = [
            sys.executable,
            str(runner_path),
            "worker-rerank",
            "--payload", str(payload_dir),
            "--embedding-results", str(embedding_result_dir),
            "--out", str(shard_dir),
            "--shard-index", str(shard_index),
            "--backend", backend,
        ]
        environment = os.environ.copy()
        if backend == "qwen":
            environment["CUDA_VISIBLE_DEVICES"] = str(local_worker)
        processes.append((shard_index, subprocess.Popen(
            command,
            env=environment,
            cwd=str(runner_path.parent),
        )))
    failures = []
    for shard_index, process in processes:
        code = process.wait()
        if code:
            failures.append({"shard_index": shard_index, "exit_code": code})
    if failures:
        raise RuntimeError(f"rerank workers failed: {failures}")
    shard_records = []
    for shard_index in shard_indices:
        shard_dir = output_dir / f"shard_{shard_index}"
        shard = validate_rerank_shard(
            payload_dir=payload_dir,
            embedding_result_dir=embedding_result_dir,
            shard_dir=shard_dir,
            expected_shard_index=shard_index,
            require_qwen=backend == "qwen",
        )
        shard_records.append({
            "shard_index": shard_index,
            "path": shard_dir.name,
            "manifest_sha256": sha256_file(shard_dir / "shard_manifest.json"),
            "question_count": shard["question_count"],
            "run_signature": shard["run_signature"],
            "canary": shard["canary"],
        })
    result = {
        "schema_version": _RERANK_PAIR_SCHEMA,
        "run_signature": signature,
        "backend": backend,
        "scientific_execution": backend == "qwen",
        "payload_fingerprint": payload["payload_fingerprint"],
        "embedding_run_signature": embedding["run_signature"],
        "workload_fingerprint": workload["workload_fingerprint"],
        "shard_indices": list(shard_indices),
        "shards": shard_records,
        "exact_canary_passed_on_both_gpus": all(
            record["canary"].get("passed") is True
            for record in shard_records
        ),
        "total_seconds": round(time.perf_counter() - started, 6),
    }
    write_json(manifest_path, result)
    return result


def run_rerank_worker(
    *,
    payload_dir: Path | str,
    embedding_result_dir: Path | str,
    output_dir: Path | str,
    shard_index: int,
    backend: str = "qwen",
) -> dict:
    payload_dir = Path(payload_dir).resolve()
    embedding_result_dir = Path(embedding_result_dir).resolve()
    output_dir = Path(output_dir).resolve()
    payload = validate_official_payload(payload_dir)
    config = load_config(payload_dir / payload["paths"]["config"])
    execution = read_json(payload_dir / payload["paths"]["execution_config"])
    workload = read_json(payload_dir / payload["paths"]["workload"])
    validate_workload_plan(workload, execution)
    embedding = validate_embedding_result(
        payload_dir=payload_dir,
        result_dir=embedding_result_dir,
        require_qwen=backend == "qwen",
    )
    shard = workload["rerank_shards"][shard_index]
    if int(shard["shard_index"]) != shard_index:
        raise ValueError("rerank shard assignment mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "shard_manifest.json"
    signature = canonical_json_sha256({
        "payload_fingerprint": payload["payload_fingerprint"],
        "embedding_run_signature": embedding["run_signature"],
        "workload_fingerprint": workload["workload_fingerprint"],
        "shard_index": shard_index,
        "question_ids": shard["question_ids"],
        "backend": backend,
    })
    if manifest_path.exists():
        existing = read_json(manifest_path)
        _validate_rerank_shard(
            payload_dir=payload_dir,
            embedding_result_dir=embedding_result_dir,
            shard_dir=output_dir,
            expected_shard_index=shard_index,
            expected_signature=signature,
            require_qwen=backend == "qwen",
        )
        return existing

    all_baseline = read_jsonl(
        payload_dir / payload["paths"]["baseline_retrieval"]
    )
    selected_baseline = [
        all_baseline[index] for index in shard["state_indices"]
    ]
    if [str(row["id"]) for row in selected_baseline] != shard["question_ids"]:
        raise ValueError("rerank shard question order drift")
    store = Store(payload_dir / payload["paths"]["store"])
    states, unsupported_local = prepare_states_total(
        selected_baseline,
        store,
        config,
        description=f"official R4 shard {shard_index} prepare",
    )
    expected_unsupported_ids = set(workload["unsupported_r0_passthrough_ids"])
    actual_unsupported_ids = {
        str(states[index].baseline["id"]) for index in unsupported_local
    }
    if actual_unsupported_ids != (
        expected_unsupported_ids & set(shard["question_ids"])
    ):
        raise ValueError("official shard unsupported-question partition drifted")
    vectors = load_embedding_vectors(embedding_result_dir)
    cache_contract = _cache_contract(payload, config)
    embedding_revision = config["models"]["embedding"]["revision"]
    embedding_instruction = config["instructions"]["embedding_query"]
    dense_top_n = int(config["retrieval"]["dense_top_n_per_leaf"])
    for state in _progress(states, f"official R4 shard {shard_index} dense"):
        if str(state.baseline["id"]) in actual_unsupported_ids:
            continue
        query_vectors = {}
        for leaf in state.leaves:
            text = state.query_by_leaf[leaf.leaf_id]
            key = vector_cache_key(
                contract_fingerprint=cache_contract,
                backend=backend,
                model_revision=embedding_revision,
                kind="query",
                instruction=embedding_instruction,
                content=text,
            )
            query_vectors[leaf.leaf_id] = _required_vector(vectors, key)
        passage_vectors = {}
        for passage in state.passage_by_key.values():
            key = vector_cache_key(
                contract_fingerprint=cache_contract,
                backend=backend,
                model_revision=embedding_revision,
                kind="table",
                instruction="",
                content=passage["content"],
            )
            passage_vectors[passage["passage_id"]] = _required_vector(
                vectors, key
            )
        attach_dense_scores(
            state, query_vectors, passage_vectors, dense_top_n
        )
    del vectors

    score_cache = ScoreCache(output_dir / "reranker_scores.json")
    reranker_revision = config["models"]["reranker"]["revision"]
    reranker_instruction = config["instructions"]["reranker"]
    checkpoint_every = int(execution["reranker_checkpoint_questions"])
    reranker = build_reranker_backend(
        backend, config["models"]["reranker"], config["runtime"]
    )
    canary_report = {"passed": backend != "qwen", "case_count": 0}
    started = time.perf_counter()
    table_call_count = 0
    row_call_count = 0

    def cached_scores(
        pairs: list[tuple[str, str]], *, kind: str,
    ) -> list[float]:
        nonlocal table_call_count, row_call_count
        if kind == "table":
            table_call_count += 1
        else:
            row_call_count += 1
        max_length = int(config["runtime"][
            "row_max_length" if kind == "row" else "reranker_max_length"
        ])
        keys = [
            score_cache_key(
                contract_fingerprint=cache_contract,
                backend=backend,
                model_revision=reranker_revision,
                kind=kind,
                instruction=reranker_instruction,
                query=query,
                document=document,
            )
            for query, document in pairs
        ]
        missing = [
            index for index, key in enumerate(keys)
            if score_cache.get(key) is None
        ]
        if missing:
            values = reranker.score(
                [pairs[index] for index in missing],
                instruction=reranker_instruction,
                max_length=max_length,
            )
            if len(values) != len(missing):
                raise RuntimeError("reranker returned wrong score count")
            for index, value in zip(missing, values):
                score_cache.put(keys[index], value)
        return [float(score_cache.get(key)) for key in keys]

    try:
        if backend == "qwen":
            canary_report = run_reranker_canary(
                backend=reranker,
                canary_manifest_path=(
                    payload_dir / payload["paths"]["numeric_canary"]
                ),
                expected_vectors_path=(
                    payload_dir / payload["paths"]["numeric_canary_vectors"]
                ),
            )
        cap = int(config["retrieval"]["rerank_top_n_per_leaf"])
        rrf_k = int(config["retrieval"]["rrf_k"])
        for state_number, state in enumerate(
            _progress(states, f"official R4 shard {shard_index} table reranker"),
            1,
        ):
            if str(state.baseline["id"]) in actual_unsupported_ids:
                continue
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
                state, dict(zip(pair_keys, cached_scores(pairs, kind="table")))
            )
            if state_number % checkpoint_every == 0:
                score_cache.save()
        score_cache.save()
        r3_rows = {
            str(state.baseline["id"]): r3_record(state, config)
            for state in states
            if str(state.baseline["id"]) not in actual_unsupported_ids
        }
        for state_number, state in enumerate(
            _progress(
                states,
                f"official R4 shard {shard_index} row reranker",
                total=len(states),
            ),
            1,
        ):
            qid = str(state.baseline["id"])
            if qid in actual_unsupported_ids:
                continue
            r3 = r3_rows[qid]
            selected = [table_key(candidate) for candidate in r3["candidates"]]
            attach_row_scores(state, store, selected, cached_scores, config)
            if state_number % checkpoint_every == 0:
                score_cache.save()
        score_cache.save()
    finally:
        reranker.close()
    rows = [
        (
            r0_unsupported_record(state, unsupported_local[index])
            if index in unsupported_local
            else r4_record(state, config)
        )
        for index, state in enumerate(states)
    ]
    retrieval_path = output_dir / f"r4_shard_{shard_index}.jsonl"
    write_jsonl(retrieval_path, rows)
    hard_violations = sum(
        len(row["g3c"]["hard_constraint_violations"]) for row in rows
    )
    if hard_violations:
        raise ValueError("official R4 shard has hard-constraint violations")
    score_path = output_dir / "reranker_scores.json"
    result = {
        "schema_version": OFFICIAL_SHARD_SCHEMA,
        "run_signature": signature,
        "backend": backend,
        "scientific_execution": backend == "qwen",
        "payload_fingerprint": payload["payload_fingerprint"],
        "official_protocol_fingerprint": payload[
            "official_protocol_fingerprint"
        ],
        "embedding_run_signature": embedding["run_signature"],
        "workload_fingerprint": workload["workload_fingerprint"],
        "shard_index": shard_index,
        "question_ids": shard["question_ids"],
        "question_count": len(rows),
        "retrieval": {
            "path": retrieval_path.name,
            "size": retrieval_path.stat().st_size,
            "sha256": sha256_file(retrieval_path),
        },
        "score_cache": {
            "path": score_path.name,
            "size": score_path.stat().st_size,
            "sha256": sha256_file(score_path),
            "score_count": len(score_cache.values),
        },
        "hard_constraint_violation_count": 0,
        "canary": canary_report,
        "table_score_call_count": table_call_count,
        "row_score_call_count": row_call_count,
        "runtime": runtime_fingerprint(backend),
        "total_seconds": round(time.perf_counter() - started, 6),
    }
    write_json(manifest_path, result)
    return result


def validate_embedding_result(
    *,
    payload_dir: Path | str,
    result_dir: Path | str,
    expected_signature: str | None = None,
    require_qwen: bool = True,
) -> dict:
    payload_dir = Path(payload_dir)
    result_dir = Path(result_dir)
    payload = validate_official_payload(payload_dir)
    execution = read_json(payload_dir / payload["paths"]["execution_config"])
    workload = read_json(payload_dir / payload["paths"]["workload"])
    validate_workload_plan(workload, execution)
    manifest = read_json(result_dir / "g3c_official_embedding_manifest.json")
    if manifest.get("schema_version") != OFFICIAL_EMBEDDING_SCHEMA:
        raise ValueError("unknown official embedding result schema")
    if expected_signature and manifest.get("run_signature") != expected_signature:
        raise ValueError("official embedding run signature mismatch")
    if manifest.get("payload_fingerprint") != payload["payload_fingerprint"]:
        raise ValueError("official embedding/payload mismatch")
    if manifest.get("workload_fingerprint") != workload["workload_fingerprint"]:
        raise ValueError("official embedding/workload mismatch")
    if require_qwen and (
        manifest.get("backend") != "qwen"
        or manifest.get("scientific_execution") is not True
        or manifest.get("exact_canary_passed_on_both_gpus") is not True
    ):
        raise ValueError("official embedding is not exact-canary Qwen evidence")
    keys = set()
    expected_files = {"g3c_official_embedding_manifest.json"}
    for record in manifest["workers"]:
        worker_index = int(record["worker_index"])
        worker_dir = result_dir / record["path"]
        worker_manifest_path = worker_dir / "worker_manifest.json"
        expected_files.add(worker_manifest_path.relative_to(result_dir).as_posix())
        if sha256_file(worker_manifest_path) != record["manifest_sha256"]:
            raise ValueError("official embedding worker manifest hash mismatch")
        worker = _validate_embedding_worker(
            worker_dir,
            payload=payload,
            workload=workload,
            worker_index=worker_index,
            backend=manifest["backend"],
        )
        overlap = keys & set(worker["vector_keys"])
        if overlap:
            raise ValueError("official embedding worker key overlap")
        keys.update(worker["vector_keys"])
        for chunk in worker["chunks"]:
            expected_files.add(
                (worker_dir / chunk["manifest"])
                .relative_to(result_dir).as_posix()
            )
            expected_files.add(
                (worker_dir / chunk["archive"])
                .relative_to(result_dir).as_posix()
            )
    if len(keys) != int(manifest["vector_count"]):
        raise ValueError("official embedding manifest vector count mismatch")
    actual_files = {
        path.relative_to(result_dir).as_posix()
        for path in result_dir.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError(
            "official embedding file-set mismatch "
            f"extra={sorted(actual_files - expected_files)} "
            f"missing={sorted(expected_files - actual_files)}"
        )
    return manifest


def load_embedding_vectors(result_dir: Path | str) -> dict[str, np.ndarray]:
    result_dir = Path(result_dir)
    manifest = read_json(result_dir / "g3c_official_embedding_manifest.json")
    vectors = {}
    for worker in manifest["workers"]:
        worker_dir = result_dir / worker["path"]
        worker_manifest = read_json(worker_dir / "worker_manifest.json")
        for chunk in worker_manifest["chunks"]:
            archive_path = worker_dir / chunk["archive"]
            with np.load(archive_path, allow_pickle=False) as archive:
                for key in archive.files:
                    if key in vectors:
                        raise ValueError("duplicate key while loading embeddings")
                    vectors[key] = archive[key]
    if len(vectors) != int(manifest["vector_count"]):
        raise ValueError("loaded embedding vector count mismatch")
    return vectors


def validate_rerank_shard(
    *,
    payload_dir: Path | str,
    embedding_result_dir: Path | str,
    shard_dir: Path | str,
    expected_shard_index: int,
    require_qwen: bool = True,
) -> dict:
    return _validate_rerank_shard(
        payload_dir=Path(payload_dir),
        embedding_result_dir=Path(embedding_result_dir),
        shard_dir=Path(shard_dir),
        expected_shard_index=expected_shard_index,
        expected_signature=None,
        require_qwen=require_qwen,
    )


def validate_rerank_pair_result(
    *,
    payload_dir: Path | str,
    embedding_result_dir: Path | str,
    result_dir: Path | str,
    expected_shards: tuple[int, int],
    expected_signature: str | None = None,
    require_qwen: bool = True,
) -> dict:
    payload_dir = Path(payload_dir)
    embedding_result_dir = Path(embedding_result_dir)
    result_dir = Path(result_dir)
    payload = validate_official_payload(payload_dir)
    embedding = validate_embedding_result(
        payload_dir=payload_dir,
        result_dir=embedding_result_dir,
        require_qwen=require_qwen,
    )
    manifest_path = result_dir / "g3c_official_rerank_pair_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != _RERANK_PAIR_SCHEMA:
        raise ValueError("unknown official rerank pair schema")
    if expected_signature and manifest.get("run_signature") != expected_signature:
        raise ValueError("official rerank pair signature mismatch")
    if manifest.get("payload_fingerprint") != payload["payload_fingerprint"]:
        raise ValueError("official rerank pair/payload mismatch")
    if manifest.get("embedding_run_signature") != embedding["run_signature"]:
        raise ValueError("official rerank pair/embedding mismatch")
    if manifest.get("shard_indices") != list(expected_shards):
        raise ValueError("official rerank pair shard assignment mismatch")
    if require_qwen and (
        manifest.get("backend") != "qwen"
        or manifest.get("scientific_execution") is not True
        or manifest.get("exact_canary_passed_on_both_gpus") is not True
    ):
        raise ValueError("official rerank pair is not exact-canary Qwen output")
    expected_files = {"g3c_official_rerank_pair_manifest.json"}
    for record in manifest["shards"]:
        shard_index = int(record["shard_index"])
        shard_dir = result_dir / record["path"]
        shard_manifest_path = shard_dir / "shard_manifest.json"
        if sha256_file(shard_manifest_path) != record["manifest_sha256"]:
            raise ValueError("official rerank pair shard hash mismatch")
        shard = validate_rerank_shard(
            payload_dir=payload_dir,
            embedding_result_dir=embedding_result_dir,
            shard_dir=shard_dir,
            expected_shard_index=shard_index,
            require_qwen=require_qwen,
        )
        for path in shard_dir.rglob("*"):
            if path.is_file():
                expected_files.add(path.relative_to(result_dir).as_posix())
    actual_files = {
        path.relative_to(result_dir).as_posix()
        for path in result_dir.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("official rerank pair file-set mismatch")
    return manifest


def _validate_embedding_worker(
    worker_dir: Path,
    *,
    payload: dict,
    workload: dict,
    worker_index: int,
    backend: str,
) -> dict:
    manifest_path = worker_dir / "worker_manifest.json"
    worker = read_json(manifest_path)
    if worker.get("schema_version") != "g3c_official_embedding_worker_v1":
        raise ValueError("unknown embedding worker schema")
    if int(worker.get("worker_index", -1)) != worker_index:
        raise ValueError("embedding worker index mismatch")
    if worker.get("backend") != backend:
        raise ValueError("embedding worker backend mismatch")
    if worker.get("payload_fingerprint") != payload["payload_fingerprint"]:
        raise ValueError("embedding worker/payload mismatch")
    if worker.get("workload_fingerprint") != workload["workload_fingerprint"]:
        raise ValueError("embedding worker/workload mismatch")
    assignment = workload["embedding_workers"][worker_index]
    if worker.get("batch_indices") != assignment["batch_indices"]:
        raise ValueError("embedding worker batch assignment drift")
    keys = set()
    chunk_batch_indices = []
    for chunk in worker["chunks"]:
        chunk_manifest = worker_dir / chunk["manifest"]
        if sha256_file(chunk_manifest) != chunk["manifest_sha256"]:
            raise ValueError("embedding chunk manifest hash mismatch")
        record = read_json(chunk_manifest)
        group = [
            {"batch_index": index}
            for index in record["batch_indices"]
        ]
        _validate_vector_chunk(
            worker_dir / "chunks",
            chunk_manifest,
            worker_index,
            group,
        )
        chunk_batch_indices.extend(int(value) for value in record["batch_indices"])
        overlap = keys & set(record["vector_keys"])
        if overlap:
            raise ValueError("duplicate vector keys across worker chunks")
        keys.update(record["vector_keys"])
    if keys != set(worker["vector_keys"]):
        raise ValueError("embedding worker vector registry mismatch")
    if chunk_batch_indices != assignment["batch_indices"]:
        raise ValueError("embedding worker chunks do not cover its batch assignment")
    if len(keys) != int(worker["vector_count"]):
        raise ValueError("embedding worker vector count mismatch")
    return worker


def _validate_rerank_shard(
    *,
    payload_dir: Path,
    embedding_result_dir: Path,
    shard_dir: Path,
    expected_shard_index: int,
    expected_signature: str | None,
    require_qwen: bool,
) -> dict:
    payload = validate_official_payload(payload_dir)
    workload = read_json(payload_dir / payload["paths"]["workload"])
    embedding = validate_embedding_result(
        payload_dir=payload_dir,
        result_dir=embedding_result_dir,
        require_qwen=require_qwen,
    )
    shard = workload["rerank_shards"][expected_shard_index]
    manifest = read_json(shard_dir / "shard_manifest.json")
    if manifest.get("schema_version") != OFFICIAL_SHARD_SCHEMA:
        raise ValueError("unknown official rerank shard schema")
    if expected_signature and manifest.get("run_signature") != expected_signature:
        raise ValueError("official rerank shard signature mismatch")
    if int(manifest.get("shard_index", -1)) != expected_shard_index:
        raise ValueError("official rerank shard index mismatch")
    if manifest.get("payload_fingerprint") != payload["payload_fingerprint"]:
        raise ValueError("official rerank shard/payload mismatch")
    if manifest.get("embedding_run_signature") != embedding["run_signature"]:
        raise ValueError("official rerank shard/embedding mismatch")
    if manifest.get("question_ids") != shard["question_ids"]:
        raise ValueError("official rerank shard question assignment drift")
    if require_qwen and (
        manifest.get("backend") != "qwen"
        or manifest.get("scientific_execution") is not True
        or manifest.get("canary", {}).get("passed") is not True
    ):
        raise ValueError("official rerank shard is not exact-canary Qwen output")
    expected_files = {"shard_manifest.json"}
    for field in ("retrieval", "score_cache"):
        record = manifest[field]
        path = shard_dir / record["path"]
        expected_files.add(path.name)
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size"])
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"official rerank shard {field} hash mismatch")
    actual_files = {
        path.relative_to(shard_dir).as_posix()
        for path in shard_dir.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("official rerank shard file-set mismatch")
    rows = read_jsonl(shard_dir / manifest["retrieval"]["path"])
    if [str(row.get("id")) for row in rows] != shard["question_ids"]:
        raise ValueError("official rerank shard retrieval order mismatch")
    if any(
        row.get("g3c", {}).get("stage") != "R4"
        or row.get("g3c", {}).get("hard_constraint_violations")
        for row in rows
    ):
        raise ValueError("official rerank shard structural violation")
    return manifest


def _write_vector_chunk(
    *,
    chunks_dir: Path,
    chunk_index: int,
    worker_index: int,
    batch_indices: list[int],
    arrays: dict[str, np.ndarray],
) -> dict:
    archive_name = f"chunk_{chunk_index:04d}.npz"
    archive_path = chunks_dir / archive_name
    _write_npz_atomic(archive_path, arrays)
    body = {
        "schema_version": _VECTOR_CHUNK_SCHEMA,
        "worker_index": worker_index,
        "chunk_index": chunk_index,
        "batch_indices": batch_indices,
        "vector_count": len(arrays),
        "vector_keys": sorted(arrays),
        "archive": {
            "path": archive_name,
            "size": archive_path.stat().st_size,
            "sha256": sha256_file(archive_path),
        },
    }
    body["chunk_fingerprint"] = canonical_json_sha256(body)
    write_json(chunks_dir / f"chunk_{chunk_index:04d}.json", body)
    return body


def _validate_vector_chunk(
    chunks_dir: Path,
    manifest_path: Path,
    worker_index: int,
    expected_group: list[dict],
) -> dict:
    record = read_json(manifest_path)
    if record.get("schema_version") != _VECTOR_CHUNK_SCHEMA:
        raise ValueError("unknown official vector chunk schema")
    expected = canonical_json_sha256({
        key: value for key, value in record.items()
        if key != "chunk_fingerprint"
    })
    if record.get("chunk_fingerprint") != expected:
        raise ValueError("official vector chunk fingerprint mismatch")
    if int(record.get("worker_index", -1)) != worker_index:
        raise ValueError("official vector chunk worker mismatch")
    if record.get("batch_indices") != [
        int(batch["batch_index"]) for batch in expected_group
    ]:
        raise ValueError("official vector chunk batch assignment mismatch")
    archive = record["archive"]
    path = chunks_dir / archive["path"]
    if (
        not path.is_file()
        or path.stat().st_size != int(archive["size"])
        or sha256_file(path) != archive["sha256"]
    ):
        raise ValueError("official vector chunk archive hash mismatch")
    with np.load(path, allow_pickle=False) as values:
        if sorted(values.files) != record["vector_keys"]:
            raise ValueError("official vector chunk key registry mismatch")
        if any(not np.isfinite(values[key]).all() for key in values.files):
            raise ValueError("official vector chunk has non-finite values")
    return record


def _cache_contract(payload: dict, config: dict) -> str:
    return canonical_json_sha256({
        "config_sha256": config_fingerprint(config),
        "protocol_fingerprint": payload["g3c_protocol_fingerprint"],
    })


def _required_vector(
    vectors: dict[str, np.ndarray], key: str,
) -> np.ndarray:
    try:
        return vectors[key]
    except KeyError as error:
        raise ValueError(f"official embedding vector missing: {key}") from error


def _require_two_matching_gpus() -> None:
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("exact official runner requires exactly two CUDA GPUs")
    signatures = [
        (
            torch.cuda.get_device_name(index),
            tuple(torch.cuda.get_device_capability(index)),
            int(torch.cuda.get_device_properties(index).total_memory),
        )
        for index in range(2)
    ]
    if signatures[0] != signatures[1]:
        raise RuntimeError(f"exact workers require matching GPUs: {signatures}")


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


def _progress(iterable, description: str, total: int | None = None):
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=description, total=total)
    except ImportError:
        return iterable
