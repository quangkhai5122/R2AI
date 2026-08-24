"""Build and validate the label-free official 1,012-question R4 payload."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

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
from ..g3c.freeze import load_candidate_freeze
from ..g3c.protocol import load_protocol_freeze
from .canary import load_numeric_canary
from .closeout import load_pb_closeout
from .common import (
    OFFICIAL_PAYLOAD_SCHEMA,
    load_execution_config,
)
from .protocol import load_official_protocol, validate_official_protocol
from .workload import build_workload_plan, validate_workload_plan

_UPLOAD_SIDECARS = {"dataset-metadata.json"}
_FORBIDDEN_PATH_PARTS = {"gold", "corpus", "oracle", "review", "evaluation"}


def build_official_payload(
    *,
    repo_root: Path | str,
    output_dir: Path | str,
    questions_path: Path | str,
    baseline_retrieval_path: Path | str,
    store_dir: Path | str,
    config_path: Path | str,
    execution_config_path: Path | str,
    source_protocol_path: Path | str,
    official_protocol_path: Path | str,
    candidate_path: Path | str,
    closeout_path: Path | str,
    canary_manifest_path: Path | str,
    canary_vectors_path: Path | str,
    promotion_result_dir: Path | str,
    kaggle_dataset_id: str,
    source_git_head: str,
    source_git_dirty: bool,
    source_git_status_sha256: str,
) -> dict:
    repo_root = Path(repo_root).resolve()
    output_dir = Path(output_dir).resolve()
    questions_path = Path(questions_path).resolve()
    baseline_retrieval_path = Path(baseline_retrieval_path).resolve()
    store_dir = Path(store_dir).resolve()
    config_path = Path(config_path).resolve()
    execution_config_path = Path(execution_config_path).resolve()
    source_protocol_path = Path(source_protocol_path).resolve()
    official_protocol_path = Path(official_protocol_path).resolve()
    candidate_path = Path(candidate_path).resolve()
    closeout_path = Path(closeout_path).resolve()
    canary_manifest_path = Path(canary_manifest_path).resolve()
    canary_vectors_path = Path(canary_vectors_path).resolve()
    promotion_result_dir = Path(promotion_result_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty official payload: {output_dir}"
        )
    if "/" not in kaggle_dataset_id:
        raise ValueError("Kaggle dataset id must be owner/slug")

    config = load_config(config_path)
    execution = load_execution_config(execution_config_path)
    source_protocol = load_protocol_freeze(source_protocol_path)
    official_protocol = validate_official_protocol(
        repo_root=repo_root,
        freeze_path=official_protocol_path,
        verify_worktree=True,
    )
    candidate = load_candidate_freeze(candidate_path)
    closeout = load_pb_closeout(closeout_path)
    canary = load_numeric_canary(canary_manifest_path, canary_vectors_path)
    if execution["source_config_sha256"] != config_fingerprint(config):
        raise ValueError("official execution/config mismatch")
    if execution["candidate_fingerprint"] != candidate["candidate_fingerprint"]:
        raise ValueError("official execution/candidate mismatch")
    if closeout["candidate_fingerprint"] != candidate["candidate_fingerprint"]:
        raise ValueError("official closeout/candidate mismatch")
    if official_protocol["candidate_fingerprint"] != candidate[
        "candidate_fingerprint"
    ]:
        raise ValueError("official protocol/candidate mismatch")
    if official_protocol["source_g3c_protocol_fingerprint"] != source_protocol[
        "protocol_fingerprint"
    ]:
        raise ValueError("official/source protocol mismatch")
    if canary["canary_fingerprint"] != official_protocol[
        "numeric_canary_fingerprint"
    ]:
        raise ValueError("official protocol/numeric canary mismatch")

    questions_source = read_jsonl(questions_path)
    baseline_rows = read_jsonl(baseline_retrieval_path)
    questions = [
        {"id": row["id"], "question": row["question"]}
        for row in questions_source
    ]
    if any(set(row) != {"id", "question"} for row in questions_source):
        raise ValueError("official source questions expose fields beyond id/question")
    if len(questions) != 1012 or len(baseline_rows) != 1012:
        raise ValueError("official payload requires exactly 1,012 records")
    if [str(row["id"]) for row in questions] != [
        str(row["id"]) for row in baseline_rows
    ]:
        raise ValueError("official questions/R0 ID order mismatch")
    if any(
        question["question"] != baseline.get("question")
        for question, baseline in zip(questions, baseline_rows)
    ):
        raise ValueError("official questions/R0 text mismatch")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "config": "config/g3c_config.json",
        "execution_config": "config/g3c_official_execution.json",
        "questions": "questions/questions.jsonl",
        "baseline_retrieval": "baseline/r0_retrieval.jsonl",
        "store": "store",
        "source_protocol": "freeze/g3c_dev_protocol_freeze.json",
        "official_protocol": "freeze/g3c_official_protocol_freeze.json",
        "candidate_freeze": "freeze/g3c_candidate_freeze.json",
        "pb_closeout": "freeze/g3c_pb_r4_closeout.json",
        "numeric_canary": "canary/exact_numeric_canary.json",
        "numeric_canary_vectors": "canary/exact_numeric_canary_vectors.npz",
        "workload": "workload/g3c_official_workload.json",
        "runner": "code/kaggle_g3c_official.py",
        "requirements": "code/requirements-g3c.txt",
        "dataset_metadata": "dataset-metadata.json",
    }
    write_json(output_dir / paths["dataset_metadata"], {
        "title": kaggle_dataset_id.split("/", 1)[1],
        "id": kaggle_dataset_id,
        "licenses": [{"name": "CC-BY-NC-4.0"}],
    })
    _copy_file(config_path, output_dir / paths["config"])
    _copy_file(execution_config_path, output_dir / paths["execution_config"])
    write_jsonl(output_dir / paths["questions"], questions)
    _copy_file(baseline_retrieval_path, output_dir / paths["baseline_retrieval"])
    _copy_tree_files(store_dir, output_dir / paths["store"])
    _copy_file(source_protocol_path, output_dir / paths["source_protocol"])
    _copy_file(official_protocol_path, output_dir / paths["official_protocol"])
    _copy_file(candidate_path, output_dir / paths["candidate_freeze"])
    _copy_file(closeout_path, output_dir / paths["pb_closeout"])
    _copy_file(canary_manifest_path, output_dir / paths["numeric_canary"])
    _copy_file(canary_vectors_path, output_dir / paths["numeric_canary_vectors"])
    _copy_python_package(repo_root / "vifinqa", output_dir / "code/vifinqa")
    _copy_file(
        repo_root / "kaggle/kaggle_g3c_official.py",
        output_dir / paths["runner"],
    )
    _copy_file(
        repo_root / "kaggle/requirements-g3c.txt",
        output_dir / paths["requirements"],
    )

    promotion_manifest = read_json(
        promotion_result_dir / "g3c_gpu_result_manifest.json"
    )
    with np.load(
        promotion_result_dir / "cache/embedding_vectors.npz",
        allow_pickle=False,
    ) as archive:
        promotion_vector_count = len(archive.files)
    promotion_scores = read_json(
        promotion_result_dir / "cache/reranker_scores.json"
    )["scores"]
    workload, _states = build_workload_plan(
        questions=questions,
        baseline_rows=baseline_rows,
        store_dir=store_dir,
        config=config,
        execution=execution,
        output_path=output_dir / paths["workload"],
        promotion_runtime=promotion_manifest["runtime"],
        promotion_vector_count=promotion_vector_count,
        promotion_score_count=len(promotion_scores),
    )
    files = _manifest_files(output_dir, excluded_paths=_UPLOAD_SIDECARS)
    sidecars = [
        _file_record(output_dir, relative)
        for relative in sorted(_UPLOAD_SIDECARS)
    ]
    _assert_no_forbidden_paths([*files, *sidecars])
    body = {
        "schema_version": OFFICIAL_PAYLOAD_SCHEMA,
        "mode": "official_engineering_audit",
        "selected_stage": "R4",
        "question_count": 1012,
        "question_ids_sha256": canonical_json_sha256([
            str(row["id"]) for row in questions
        ]),
        "question_texts_sha256": canonical_json_sha256([
            row["question"] for row in questions
        ]),
        "questions_file_sha256": sha256_file(questions_path),
        "baseline_retrieval_sha256": sha256_file(baseline_retrieval_path),
        "config_sha256": config_fingerprint(config),
        "execution_config_sha256": sha256_file(execution_config_path),
        "g3c_protocol_fingerprint": source_protocol["protocol_fingerprint"],
        "official_protocol_fingerprint": official_protocol[
            "official_protocol_fingerprint"
        ],
        "candidate_fingerprint": candidate["candidate_fingerprint"],
        "pb_closeout_fingerprint": closeout["closeout_fingerprint"],
        "numeric_canary_fingerprint": canary["canary_fingerprint"],
        "workload_fingerprint": workload["workload_fingerprint"],
        "source_git_head": str(source_git_head),
        "source_git_dirty": bool(source_git_dirty),
        "source_git_status_sha256": str(source_git_status_sha256),
        "kaggle_dataset_id": kaggle_dataset_id,
        "paths": paths,
        "label_boundary": {
            "question_fields": ["id", "question"],
            "official_public_questions_included": True,
            "gold_included": False,
            "public_scores_included": False,
            "g3b_corpus_included": False,
            "manual_reviews_included": False,
            "selection_role": "none_post_freeze_engineering_only",
        },
        "equivalence_boundary": {
            "prior_qwen_vector_cache_included": False,
            "prior_qwen_score_cache_included": False,
            "embedding_batch_size_changed": False,
            "reranker_batch_size_changed": False,
            "model_or_tokenizer_changed": False,
            "precision_or_attention_changed": False,
            "approximate_search_enabled": False,
            "supported_questions_use_exact_frozen_r4": True,
            "missing_exact_report_action": "r0_passthrough_only",
            "fallback_report_search_enabled": False,
        },
        "upload_sidecars": sidecars,
        "files": files,
    }
    body["payload_fingerprint"] = canonical_json_sha256(body)
    write_json(output_dir / "g3c_official_payload_manifest.json", body)
    validate_official_payload(output_dir)
    return body


def validate_official_payload(payload_dir: Path | str) -> dict:
    payload_dir = Path(payload_dir)
    manifest_path = payload_dir / "g3c_official_payload_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != OFFICIAL_PAYLOAD_SCHEMA:
        raise ValueError("unknown official G3C payload schema")
    expected = canonical_json_sha256({
        key: value for key, value in manifest.items()
        if key != "payload_fingerprint"
    })
    if manifest.get("payload_fingerprint") != expected:
        raise ValueError("official G3C payload fingerprint mismatch")
    _validate_file_set(payload_dir, manifest)
    _assert_no_forbidden_paths([
        *manifest.get("files", []), *manifest.get("upload_sidecars", [])
    ])
    paths = manifest["paths"]
    config = load_config(payload_dir / paths["config"])
    execution = load_execution_config(payload_dir / paths["execution_config"])
    source_protocol = load_protocol_freeze(payload_dir / paths["source_protocol"])
    official_protocol = load_official_protocol(
        payload_dir / paths["official_protocol"]
    )
    candidate = load_candidate_freeze(payload_dir / paths["candidate_freeze"])
    closeout = load_pb_closeout(payload_dir / paths["pb_closeout"])
    canary = load_numeric_canary(
        payload_dir / paths["numeric_canary"],
        payload_dir / paths["numeric_canary_vectors"],
    )
    workload = read_json(payload_dir / paths["workload"])
    validate_workload_plan(workload, execution)
    if manifest.get("question_count") != 1012:
        raise ValueError("official payload question count drift")
    if manifest.get("config_sha256") != config_fingerprint(config):
        raise ValueError("official payload/config mismatch")
    if manifest.get("execution_config_sha256") != sha256_file(
        payload_dir / paths["execution_config"]
    ):
        raise ValueError("official payload/execution config mismatch")
    bindings = {
        "g3c_protocol_fingerprint": source_protocol["protocol_fingerprint"],
        "official_protocol_fingerprint": official_protocol[
            "official_protocol_fingerprint"
        ],
        "candidate_fingerprint": candidate["candidate_fingerprint"],
        "pb_closeout_fingerprint": closeout["closeout_fingerprint"],
        "numeric_canary_fingerprint": canary["canary_fingerprint"],
        "workload_fingerprint": workload["workload_fingerprint"],
    }
    for field, value in bindings.items():
        if manifest.get(field) != value:
            raise ValueError(f"official payload binding mismatch: {field}")
    questions = read_jsonl(payload_dir / paths["questions"])
    baseline = read_jsonl(payload_dir / paths["baseline_retrieval"])
    if len(questions) != 1012 or len(baseline) != 1012:
        raise ValueError("official payload row count mismatch")
    if any(set(row) != {"id", "question"} for row in questions):
        raise ValueError("official payload question field boundary violated")
    if [str(row["id"]) for row in questions] != [
        str(row["id"]) for row in baseline
    ]:
        raise ValueError("official payload question/R0 order mismatch")
    if any(
        question["question"] != row.get("question")
        for question, row in zip(questions, baseline)
    ):
        raise ValueError("official payload question/R0 text mismatch")
    if manifest["label_boundary"] != {
        "question_fields": ["id", "question"],
        "official_public_questions_included": True,
        "gold_included": False,
        "public_scores_included": False,
        "g3b_corpus_included": False,
        "manual_reviews_included": False,
        "selection_role": "none_post_freeze_engineering_only",
    }:
        raise ValueError("official payload label boundary drift")
    return manifest


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree_files(source: Path, target: Path) -> None:
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def _copy_python_package(source: Path, target: Path) -> None:
    allowed_packages = {
        "extraction", "finance", "g3c", "g3c_official",
        "retrieval", "router", "utils",
    }
    allowed_root_files = {"__init__.py", "config.py"}
    for path in sorted(source.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(source)
        if len(relative.parts) == 1:
            if relative.name not in allowed_root_files:
                continue
        elif relative.parts[0] not in allowed_packages:
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def _file_record(root: Path, relative: str) -> dict:
    path = root / relative
    return {
        "path": relative,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _manifest_files(
    root: Path, *, excluded_paths: set[str] | None = None,
) -> list[dict]:
    excluded_paths = excluded_paths or set()
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if (
            relative == "g3c_official_payload_manifest.json"
            or relative in excluded_paths
        ):
            continue
        rows.append(_file_record(root, relative))
    return rows


def _validate_file_set(payload_dir: Path, manifest: dict) -> None:
    core = {str(record["path"]): record for record in manifest["files"]}
    sidecars = {
        str(record["path"]): record for record in manifest["upload_sidecars"]
    }
    if set(sidecars) != {manifest["paths"]["dataset_metadata"]}:
        raise ValueError("official payload upload-sidecar contract mismatch")
    if set(core) & set(sidecars):
        raise ValueError("official payload sidecar also listed as core")
    expected = {"g3c_official_payload_manifest.json", *core}
    actual_all = {
        path.relative_to(payload_dir).as_posix()
        for path in payload_dir.rglob("*") if path.is_file()
    }
    actual = actual_all - set(sidecars)
    if actual != expected:
        raise ValueError(
            "official payload file-set mismatch "
            f"extra={sorted(actual - expected)} missing={sorted(expected - actual)}"
        )
    for relative, record in core.items():
        _validate_record(payload_dir, relative, record, required=True)
    for relative, record in sidecars.items():
        _validate_record(payload_dir, relative, record, required=False)


def _validate_record(
    root: Path, relative: str, record: dict, *, required: bool,
) -> None:
    path = root / relative
    if not path.is_file():
        if required:
            raise ValueError(f"official payload file missing: {relative}")
        return
    if (
        path.stat().st_size != int(record["size"])
        or sha256_file(path) != record["sha256"]
    ):
        raise ValueError(f"official payload file hash mismatch: {relative}")


def _assert_no_forbidden_paths(records: list[dict]) -> None:
    bad = []
    for record in records:
        parts = {
            part.lower().replace("-", "_")
            for part in Path(str(record["path"])).parts
        }
        if any(
            any(token in part.split("_") for token in _FORBIDDEN_PATH_PARTS)
            for part in parts
        ):
            bad.append(record["path"])
    if bad:
        raise ValueError(f"official payload contains forbidden paths: {bad}")
