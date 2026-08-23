"""End-to-end GPU retrieval runner with no evaluation-label dependency."""
from __future__ import annotations

import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ..extraction.build_store import Store
from .cache import (
    ScoreCache,
    VectorCache,
    score_cache_key,
    vector_cache_key,
)
from .common import (
    GPU_RESULT_SCHEMA,
    STAGES,
    canonical_json_sha256,
    config_fingerprint,
    load_config,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from .modeling import (
    build_embedding_backend,
    build_reranker_backend,
    runtime_fingerprint,
)
from .payload import validate_gpu_payload
from .retrieval import (
    QuestionState,
    attach_dense_scores,
    attach_reranker_scores,
    attach_row_scores,
    prepare_state,
    r0_record,
    r0l_record,
    r1_record,
    r2_record,
    r3_record,
    r4_record,
    rerank_pools,
)
from .serialize import table_key


def run_gpu_pipeline(
    payload_dir: Path | str,
    output_dir: Path | str,
    *,
    backend: str = "qwen",
    limit: int = 0,
) -> dict:
    payload_dir = Path(payload_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "g3c_gpu_result_manifest.json"

    payload = _validate_payload_for_runner(payload_dir)
    config_path = payload_dir / payload["paths"]["config"]
    config = load_config(config_path)
    config_sha256 = config_fingerprint(config)
    cache_contract = canonical_json_sha256({
        "config_sha256": config_sha256,
        "protocol_fingerprint": payload["protocol_fingerprint"],
    })
    questions_path = payload_dir / payload["paths"]["questions"]
    baseline_path = payload_dir / payload["paths"]["baseline_retrieval"]
    store_dir = payload_dir / payload["paths"]["store"]
    questions = read_jsonl(questions_path)
    baseline_rows = read_jsonl(baseline_path)
    if limit:
        questions = questions[:limit]
        baseline_rows = baseline_rows[:limit]
    _validate_question_boundary(questions, baseline_rows)

    mode = payload["mode"]
    selected_stage = payload.get("selected_stage")
    if mode == "dev":
        stages_to_write = list(STAGES)
    else:
        if selected_stage not in STAGES[1:]:
            raise ValueError("promotion payload must bind one non-R0 stage")
        stages_to_write = ["R0", selected_stage]
    if backend == "fake" and not limit:
        raise ValueError("fake backend requires --limit and is smoke-only")

    run_signature = canonical_json_sha256({
        "payload_fingerprint": payload["payload_fingerprint"],
        "config_sha256": config_sha256,
        "backend": backend,
        "mode": mode,
        "selected_stage": selected_stage,
        "limit": int(limit),
    })
    if manifest_path.exists():
        existing = read_json(manifest_path)
        _validate_completed_result(
            existing,
            output_dir,
            expected_run_signature=run_signature,
        )
        return existing
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    stage_ready_seconds: dict[str, float] = {}
    started = time.perf_counter()

    store = Store(store_dir)
    baseline_by_id = {str(row["id"]): row for row in baseline_rows}
    states: list[QuestionState] = []
    prepare_start = time.perf_counter()
    for question in _progress(questions, "G3C prepare leaf/table universe"):
        state = prepare_state(baseline_by_id[str(question["id"])], store, config)
        if not state.leaves:
            raise ValueError(f"question {question['id']} produced zero leaves")
        if any(not leaf.report_ids for leaf in state.leaves):
            missing = [
                leaf.to_dict() for leaf in state.leaves if not leaf.report_ids
            ]
            raise ValueError(
                f"question {question['id']} has no exact report for leaves: "
                f"{missing}"
            )
        states.append(state)
    timings["prepare_seconds"] = time.perf_counter() - prepare_start

    stage_rows: dict[str, list[dict]] = {
        "R0": [r0_record(state) for state in states],
        "R0L": [r0l_record(state, config) for state in states],
    }
    ready = time.perf_counter() - started
    stage_ready_seconds["R0"] = ready
    stage_ready_seconds["R0L"] = ready

    target_index = max(STAGES.index(stage) for stage in stages_to_write)
    vector_cache = VectorCache(cache_dir / "embedding_vectors.npz")
    embedding_revision = config["models"]["embedding"]["revision"]
    embedding_instruction = config["instructions"]["embedding_query"]
    vector_cache_keys: dict[tuple[str, str], str] = {}

    if target_index >= STAGES.index("R1"):
        embedding_start = time.perf_counter()
        embedding = build_embedding_backend(
            backend, config["models"]["embedding"], config["runtime"]
        )
        try:
            passage_texts: dict[str, str] = {}
            query_texts: dict[str, str] = {}
            for state_index, state in enumerate(states):
                for passage in state.passage_by_key.values():
                    passage_texts[passage["passage_id"]] = passage["content"]
                for leaf in state.leaves:
                    local_id = f"{state_index}:{leaf.leaf_id}"
                    query_texts[local_id] = state.query_by_leaf[leaf.leaf_id]

            _encode_missing(
                embedding, vector_cache, passage_texts,
                backend=backend,
                revision=embedding_revision,
                kind="table",
                instruction="",
                is_query=False,
                contract_fingerprint=cache_contract,
                cache_keys=vector_cache_keys,
            )
            _encode_missing(
                embedding, vector_cache, query_texts,
                backend=backend,
                revision=embedding_revision,
                kind="query",
                instruction=embedding_instruction,
                is_query=True,
                contract_fingerprint=cache_contract,
                cache_keys=vector_cache_keys,
            )
            vector_cache.save()
        finally:
            embedding.close()
        timings["embedding_seconds"] = time.perf_counter() - embedding_start

        dense_start = time.perf_counter()
        dense_top_n = int(
            config["retrieval"]["dense_top_n_per_leaf"]
        )
        for state_index, state in enumerate(
            _progress(states, "G3C dense rankings")
        ):
            query_vectors = {
                leaf.leaf_id: vector_cache.get(vector_cache_keys[(
                    "query", f"{state_index}:{leaf.leaf_id}"
                )])
                for leaf in state.leaves
            }
            passage_vectors = {
                passage["passage_id"]: vector_cache.get(vector_cache_keys[(
                    "table", passage["passage_id"]
                )])
                for passage in state.passage_by_key.values()
            }
            if any(value is None for value in query_vectors.values()):
                raise RuntimeError("query embedding cache is incomplete")
            if any(value is None for value in passage_vectors.values()):
                raise RuntimeError("table embedding cache is incomplete")
            attach_dense_scores(
                state, query_vectors, passage_vectors, dense_top_n
            )
        timings["dense_fusion_seconds"] = time.perf_counter() - dense_start
        stage_rows["R1"] = [r1_record(state, config) for state in states]
        stage_ready_seconds["R1"] = (
            time.perf_counter() - started
        )

    score_cache = ScoreCache(cache_dir / "reranker_scores.json")
    reranker_revision = config["models"]["reranker"]["revision"]
    reranker_instruction = config["instructions"]["reranker"]
    if target_index >= STAGES.index("R2"):
        rerank_start = time.perf_counter()
        reranker = build_reranker_backend(
            backend, config["models"]["reranker"], config["runtime"]
        )
        checkpoint_every = int(config["retrieval"]["checkpoint_every"])

        def cached_scores(
            pairs: list[tuple[str, str]], *, kind: str
        ) -> list[float]:
            max_length = int(config["runtime"][
                "row_max_length"
                if kind == "row" else "reranker_max_length"
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
            missing_indexes = [
                index for index, key in enumerate(keys)
                if score_cache.get(key) is None
            ]
            if missing_indexes:
                missing_pairs = [pairs[index] for index in missing_indexes]
                values = reranker.score(
                    missing_pairs,
                    instruction=reranker_instruction,
                    max_length=max_length,
                )
                if len(values) != len(missing_indexes):
                    raise RuntimeError("reranker returned wrong score count")
                for index, value in zip(missing_indexes, values):
                    score_cache.put(keys[index], value)
            return [float(score_cache.get(key)) for key in keys]

        try:
            cap = int(config["retrieval"]["rerank_top_n_per_leaf"])
            rrf_k = int(config["retrieval"]["rrf_k"])
            for state_index, state in enumerate(
                _progress(states, "G3C table reranker"), 1
            ):
                pools = rerank_pools(state, cap, rrf_k)
                pairs = []
                pair_keys = []
                for leaf in state.leaves:
                    leaf_id = leaf.leaf_id
                    for key in pools[leaf_id]:
                        pairs.append((
                            state.query_by_leaf[leaf_id],
                            state.passage_by_key[key]["content"],
                        ))
                        pair_keys.append((leaf_id, key))
                values = cached_scores(pairs, kind="table")
                attach_reranker_scores(
                    state, dict(zip(pair_keys, values))
                )
                if state_index % checkpoint_every == 0:
                    score_cache.save()
            score_cache.save()
            stage_rows["R2"] = [
                r2_record(state, config) for state in states
            ]
            stage_rows["R3"] = [
                r3_record(state, config) for state in states
            ]
            ready = time.perf_counter() - started
            stage_ready_seconds["R2"] = ready
            stage_ready_seconds["R3"] = ready

            if target_index >= STAGES.index("R4"):
                for state_index, (state, r3) in enumerate(
                    _progress(
                        zip(states, stage_rows["R3"]),
                        "G3C bounded row reranker",
                        total=len(states),
                    ), 1
                ):
                    selected = [
                        table_key(candidate)
                        for candidate in r3["candidates"]
                    ]
                    attach_row_scores(
                        state, store, selected, cached_scores, config
                    )
                    if state_index % checkpoint_every == 0:
                        score_cache.save()
                score_cache.save()
                stage_rows["R4"] = [
                    r4_record(state, config) for state in states
                ]
                stage_ready_seconds["R4"] = (
                    time.perf_counter() - started
                )
        finally:
            reranker.close()
        timings["reranker_seconds"] = time.perf_counter() - rerank_start

    write_start = time.perf_counter()
    stage_artifacts = {}
    for stage in stages_to_write:
        if stage not in stage_rows:
            raise RuntimeError(f"stage {stage} was not computed")
        path = output_dir / f"{stage.lower()}_retrieval.jsonl"
        if stage == "R0" and not limit:
            shutil.copyfile(baseline_path, path)
        else:
            write_jsonl(path, stage_rows[stage])
        stage_artifacts[stage] = _stage_artifact(path, stage_rows[stage])
    timings["write_seconds"] = time.perf_counter() - write_start
    timings["total_seconds"] = time.perf_counter() - started

    cache_artifacts = {}
    for path in sorted(cache_dir.glob("*")):
        if path.is_file():
            cache_artifacts[path.name] = {
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    runtime = runtime_fingerprint(backend)
    runtime["timings"] = {
        key: round(value, 6) for key, value in timings.items()
    }
    runtime["stage_ready_seconds"] = {
        key: round(value, 6)
        for key, value in stage_ready_seconds.items()
    }
    result = {
        "schema_version": GPU_RESULT_SCHEMA,
        "run_signature": run_signature,
        "mode": mode,
        "backend": backend,
        "scientific_evidence_valid": backend == "qwen" and limit == 0,
        "smoke_limit": int(limit),
        "payload_fingerprint": payload["payload_fingerprint"],
        "payload_manifest_sha256": sha256_file(
            payload_dir / "g3c_gpu_payload_manifest.json"
        ),
        "config_sha256": config_sha256,
        "g3_evaluation_freeze_sha256": (
            config["g3_evaluation_freeze_sha256"]
        ),
        "protocol_fingerprint": payload["protocol_fingerprint"],
        "selected_stage": selected_stage,
        "stages_written": stages_to_write,
        "question_count": len(states),
        "model_revisions": {
            name: {
                "model_id": value["model_id"],
                "revision": value["revision"],
                "tokenizer_revision": value["tokenizer_revision"],
            }
            for name, value in config["models"].items()
        },
        "instructions_sha256": canonical_json_sha256(
            config["instructions"]
        ),
        "stage_artifacts": stage_artifacts,
        "cache_artifacts": cache_artifacts,
        "runtime": runtime,
    }
    write_json(manifest_path, result)
    return result


def _encode_missing(
    backend_object,
    cache: VectorCache,
    texts: dict[str, str],
    *,
    backend: str,
    revision: str,
    kind: str,
    instruction: str,
    is_query: bool,
    contract_fingerprint: str,
    cache_keys: dict[tuple[str, str], str],
) -> None:
    missing_ids = []
    missing_texts = []
    for identity, text in sorted(texts.items()):
        key = vector_cache_key(
            contract_fingerprint=contract_fingerprint,
            backend=backend,
            model_revision=revision,
            kind=kind,
            instruction=instruction,
            content=text,
        )
        cache_keys[(kind, identity)] = key
        if cache.get(key) is None:
            missing_ids.append(identity)
            missing_texts.append(text)
    if not missing_ids:
        return
    vectors = backend_object.encode(
        missing_texts, is_query=is_query, instruction=instruction
    )
    if len(vectors) != len(missing_ids):
        raise RuntimeError("embedding backend returned wrong vector count")
    for identity, vector in zip(missing_ids, vectors):
        cache.put(cache_keys[(kind, identity)], vector)


def _stage_artifact(path: Path, rows: list[dict]) -> dict:
    counts = [len(row.get("candidates", [])) for row in rows]
    source_counts: Counter[str] = Counter()
    hard_violations = 0
    for row in rows:
        hard_violations += len(
            row.get("g3c", {}).get("hard_constraint_violations", [])
        )
        for candidate in row.get("candidates", []):
            source_counts.update(
                candidate.get("g3c", {}).get("sources", [])
            )
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
        "record_count": len(rows),
        "candidate_count_min": min(counts) if counts else 0,
        "candidate_count_max": max(counts) if counts else 0,
        "candidate_count_mean": (
            round(sum(counts) / len(counts), 6) if counts else 0.0
        ),
        "hard_constraint_violation_count": hard_violations,
        "source_counts": dict(sorted(source_counts.items())),
    }


def _validate_question_boundary(
    questions: list[dict], baseline_rows: list[dict]
) -> None:
    if len(questions) != len(baseline_rows):
        raise ValueError("question and R0 retrieval counts differ")
    question_ids = [str(row.get("id")) for row in questions]
    baseline_ids = [str(row.get("id")) for row in baseline_rows]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("duplicate question IDs")
    if question_ids != baseline_ids:
        raise ValueError("question/R0 order or ID mismatch")
    for question, baseline in zip(questions, baseline_rows):
        if set(question) != {"id", "question"}:
            raise ValueError(
                "GPU question rows must contain only id and question"
            )
        if question["question"] != baseline.get("question"):
            raise ValueError(f"question text mismatch for {question['id']}")


def _validate_payload_for_runner(payload_dir: Path) -> dict:
    return validate_gpu_payload(payload_dir)


def _validate_completed_result(
    manifest: dict,
    output_dir: Path,
    *,
    expected_run_signature: str,
) -> None:
    if manifest.get("schema_version") != GPU_RESULT_SCHEMA:
        raise ValueError("existing output has unknown result schema")
    if manifest.get("run_signature") != expected_run_signature:
        raise ValueError(
            "existing output run signature does not match this request"
        )
    expected_files = {"g3c_gpu_result_manifest.json"}
    for artifact in manifest.get("stage_artifacts", {}).values():
        path = output_dir / artifact["path"]
        expected_files.add(path.relative_to(output_dir).as_posix())
        if (
            not path.is_file()
            or path.stat().st_size != int(artifact["size"])
            or sha256_file(path) != artifact["sha256"]
        ):
            raise ValueError("existing completed result failed hash validation")
    for name, artifact in manifest.get("cache_artifacts", {}).items():
        path = output_dir / "cache" / name
        expected_files.add(path.relative_to(output_dir).as_posix())
        if (
            not path.is_file()
            or path.stat().st_size != int(artifact["size"])
            or sha256_file(path) != artifact["sha256"]
        ):
            raise ValueError("existing completed cache failed hash validation")
    actual_files = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("existing completed result has a stale file set")


def _progress(iterable, description: str, total: int | None = None):
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=description, total=total)
    except ImportError:
        return iterable
