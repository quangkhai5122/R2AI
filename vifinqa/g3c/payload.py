"""Build and validate label-free local-to-Kaggle G3C payloads."""
from __future__ import annotations

import shutil
from pathlib import Path

from .common import (
    GPU_PAYLOAD_SCHEMA,
    canonical_json_sha256,
    config_fingerprint,
    load_config,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from .freeze import load_candidate_freeze
from .protocol import (
    load_protocol_freeze,
    validate_protocol_freeze,
)

_FORBIDDEN_PAYLOAD_PARTS = {
    "gold", "corpus", "oracle", "review", "evaluation", "public",
}
_UPLOAD_SIDECAR_PATHS = {"dataset-metadata.json"}


def build_gpu_payload(
    *,
    repo_root: Path | str,
    output_dir: Path | str,
    mode: str,
    config_path: Path | str,
    questions_path: Path | str,
    baseline_retrieval_path: Path | str,
    store_dir: Path | str,
    source_git_head: str,
    source_git_dirty: bool,
    source_git_status_sha256: str,
    kaggle_dataset_id: str,
    protocol_freeze_path: Path | str,
    candidate_freeze_path: Path | str | None = None,
) -> dict:
    repo_root = Path(repo_root).resolve()
    output_dir = Path(output_dir).resolve()
    config_path = Path(config_path).resolve()
    questions_path = Path(questions_path).resolve()
    baseline_retrieval_path = Path(baseline_retrieval_path).resolve()
    store_dir = Path(store_dir).resolve()
    protocol_freeze_path = Path(protocol_freeze_path).resolve()
    if mode not in {"dev", "promotion"}:
        raise ValueError("mode must be dev or promotion")
    protocol_freeze = validate_protocol_freeze(
        repo_root=repo_root,
        config_path=config_path,
        freeze_path=protocol_freeze_path,
        verify_worktree=True,
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty payload directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    questions_source = read_jsonl(questions_path)
    baseline_rows = read_jsonl(baseline_retrieval_path)
    _validate_source_split(mode, questions_source)
    questions = [
        {"id": row["id"], "question": row["question"]}
        for row in questions_source
    ]
    _validate_question_match(questions, baseline_rows)

    selected_stage = None
    candidate_freeze = None
    if mode == "promotion":
        if candidate_freeze_path is None:
            raise ValueError("promotion payload requires a G3C candidate freeze")
        candidate_freeze = load_candidate_freeze(candidate_freeze_path)
        if candidate_freeze["config_sha256"] != config_fingerprint(config):
            raise ValueError("candidate freeze/config mismatch")
        if candidate_freeze.get("protocol_fingerprint") != (
            protocol_freeze["protocol_fingerprint"]
        ):
            raise ValueError("candidate freeze/protocol mismatch")
        selected_stage = candidate_freeze["selected_stage"]
        if candidate_freeze.get("gate_passed") is not True:
            raise ValueError("promotion candidate did not pass the dev gate")
    elif candidate_freeze_path is not None:
        raise ValueError("dev payload must not include a candidate freeze")

    paths = {
        "config": "config/g3c_config.json",
        "questions": "questions/questions.jsonl",
        "baseline_retrieval": "baseline/r0_retrieval.jsonl",
        "store": "store",
        "runner": "code/kaggle_g3c_qwen_retrieval.py",
        "dataset_metadata": "dataset-metadata.json",
        "protocol_freeze": "freeze/g3c_dev_protocol_freeze.json",
    }
    if "/" not in kaggle_dataset_id:
        raise ValueError("Kaggle dataset id must be owner/slug")
    write_json(output_dir / paths["dataset_metadata"], {
        "title": kaggle_dataset_id.split("/", 1)[1],
        "id": kaggle_dataset_id,
        "licenses": [{"name": "CC-BY-NC-4.0"}],
    })
    _copy_file(config_path, output_dir / paths["config"])
    write_jsonl(output_dir / paths["questions"], questions)
    _copy_file(
        baseline_retrieval_path, output_dir / paths["baseline_retrieval"]
    )
    _copy_tree_files(store_dir, output_dir / paths["store"])
    _copy_python_package(
        repo_root / "vifinqa", output_dir / "code" / "vifinqa"
    )
    _copy_file(
        repo_root / "kaggle" / "kaggle_g3c_qwen_retrieval.py",
        output_dir / paths["runner"],
    )
    requirements = repo_root / "kaggle" / "requirements-g3c.txt"
    _copy_file(requirements, output_dir / "code" / requirements.name)
    paths["requirements"] = "code/requirements-g3c.txt"
    _copy_file(
        protocol_freeze_path,
        output_dir / paths["protocol_freeze"],
    )

    if candidate_freeze is not None:
        freeze_target = output_dir / "freeze" / "g3c_candidate_freeze.json"
        _copy_file(Path(candidate_freeze_path), freeze_target)
        paths["candidate_freeze"] = freeze_target.relative_to(
            output_dir
        ).as_posix()

    files = _manifest_files(
        output_dir, excluded_paths=_UPLOAD_SIDECAR_PATHS,
    )
    upload_sidecars = [
        _file_record(output_dir, relative)
        for relative in sorted(_UPLOAD_SIDECAR_PATHS)
    ]
    _assert_no_forbidden_paths(files)
    body = {
        "schema_version": GPU_PAYLOAD_SCHEMA,
        "mode": mode,
        "selected_stage": selected_stage,
        "question_count": len(questions),
        "question_ids_sha256": canonical_json_sha256(
            [str(row["id"]) for row in questions]
        ),
        "question_texts_sha256": canonical_json_sha256(
            [row["question"] for row in questions]
        ),
        "config_sha256": config_fingerprint(config),
        "g3_evaluation_freeze_sha256": (
            config["g3_evaluation_freeze_sha256"]
        ),
        "source_git_head": str(source_git_head),
        "source_git_dirty": bool(source_git_dirty),
        "source_git_status_sha256": str(source_git_status_sha256),
        "kaggle_dataset_id": kaggle_dataset_id,
        "protocol_fingerprint": protocol_freeze[
            "protocol_fingerprint"
        ],
        "protocol_freeze_sha256": sha256_file(protocol_freeze_path),
        "candidate_freeze_sha256": (
            sha256_file(candidate_freeze_path)
            if candidate_freeze_path is not None else None
        ),
        "paths": paths,
        "label_boundary": {
            "question_fields": ["id", "question"],
            "gold_included": False,
            "g3b_corpus_included": False,
            "evaluation_reports_included": False,
            "official_public_questions_included": False,
        },
        "upload_sidecars": upload_sidecars,
        "files": files,
    }
    body["payload_fingerprint"] = canonical_json_sha256(body)
    write_json(output_dir / "g3c_gpu_payload_manifest.json", body)
    validate_gpu_payload(output_dir)
    return body


def validate_gpu_payload(payload_dir: Path | str) -> dict:
    from .common import read_json

    payload_dir = Path(payload_dir)
    manifest_path = payload_dir / "g3c_gpu_payload_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != GPU_PAYLOAD_SCHEMA:
        raise ValueError("unknown G3C payload schema")
    expected = canonical_json_sha256({
        key: value for key, value in manifest.items()
        if key != "payload_fingerprint"
    })
    if manifest.get("payload_fingerprint") != expected:
        raise ValueError("G3C payload fingerprint mismatch")
    _validate_payload_file_set(payload_dir, manifest)
    _assert_no_forbidden_paths(
        [*manifest["files"], *manifest["upload_sidecars"]]
    )
    questions = read_jsonl(payload_dir / manifest["paths"]["questions"])
    if any(set(row) != {"id", "question"} for row in questions):
        raise ValueError("payload questions expose fields beyond id/question")
    if len(questions) != int(manifest["question_count"]):
        raise ValueError("payload question count mismatch")
    config = load_config(payload_dir / manifest["paths"]["config"])
    protocol_path = payload_dir / manifest["paths"]["protocol_freeze"]
    protocol = load_protocol_freeze(protocol_path)
    if protocol.get("config_sha256") != config_fingerprint(config):
        raise ValueError("payload protocol/config mismatch")
    if manifest.get("protocol_fingerprint") != (
        protocol["protocol_fingerprint"]
    ):
        raise ValueError("payload protocol fingerprint mismatch")
    if manifest.get("protocol_freeze_sha256") != sha256_file(protocol_path):
        raise ValueError("payload protocol file hash mismatch")
    metadata_path = payload_dir / manifest["paths"]["dataset_metadata"]
    if metadata_path.is_file():
        metadata = read_json(metadata_path)
        if metadata.get("id") != manifest.get("kaggle_dataset_id"):
            raise ValueError("Kaggle dataset metadata/manifest id mismatch")
    if manifest["label_boundary"] != {
        "question_fields": ["id", "question"],
        "gold_included": False,
        "g3b_corpus_included": False,
        "evaluation_reports_included": False,
        "official_public_questions_included": False,
    }:
        raise ValueError("payload label-boundary declaration drift")
    if manifest["mode"] == "dev":
        if manifest.get("selected_stage") is not None:
            raise ValueError("dev payload unexpectedly binds a selected stage")
        if "candidate_freeze" in manifest["paths"]:
            raise ValueError("dev payload unexpectedly contains a freeze")
    elif manifest["mode"] == "promotion":
        if manifest.get("selected_stage") not in {"R0L", "R1", "R2", "R3", "R4"}:
            raise ValueError("promotion payload has no valid selected stage")
        freeze = load_candidate_freeze(
            payload_dir / manifest["paths"]["candidate_freeze"]
        )
        if freeze["selected_stage"] != manifest["selected_stage"]:
            raise ValueError("promotion payload/freeze stage mismatch")
    else:
        raise ValueError("unknown payload mode")
    return manifest


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_python_package(source: Path, target: Path) -> None:
    allowed_packages = {
        "extraction",
        "finance",
        "g3c",
        "retrieval",
        "router",
        "utils",
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


def _copy_tree_files(source: Path, target: Path) -> None:
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def _manifest_files(
    root: Path, *, excluded_paths: set[str] | None = None,
) -> list[dict]:
    excluded_paths = excluded_paths or set()
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if (
            relative == "g3c_gpu_payload_manifest.json"
            or relative in excluded_paths
        ):
            continue
        rows.append(_file_record(root, relative))
    return rows


def _file_record(root: Path, relative: str) -> dict:
    path = root / relative
    return {
        "path": relative,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validate_payload_file_set(payload_dir: Path, manifest: dict) -> None:
    core_paths = _record_paths(manifest.get("files"), "core payload")
    sidecar_rows = manifest.get("upload_sidecars")
    sidecar_paths = _record_paths(sidecar_rows, "upload sidecar")
    expected_sidecars = {manifest["paths"]["dataset_metadata"]}
    if sidecar_paths != expected_sidecars:
        raise ValueError(
            "payload upload-sidecar contract mismatch "
            f"expected={sorted(expected_sidecars)} actual={sorted(sidecar_paths)}"
        )
    if sidecar_paths & core_paths:
        raise ValueError("upload sidecar cannot also be a core payload file")

    expected_core_paths = {
        "g3c_gpu_payload_manifest.json", *core_paths,
    }
    actual_paths = {
        path.relative_to(payload_dir).as_posix()
        for path in payload_dir.rglob("*") if path.is_file()
    }
    actual_core_paths = actual_paths - sidecar_paths
    if expected_core_paths != actual_core_paths:
        raise ValueError(
            "payload file-set mismatch "
            f"extra={sorted(actual_core_paths - expected_core_paths)} "
            f"missing={sorted(expected_core_paths - actual_core_paths)}"
        )

    for row in manifest["files"]:
        _validate_file_record(payload_dir, row, required=True)
    for row in sidecar_rows:
        _validate_file_record(payload_dir, row, required=False)


def _record_paths(rows: object, label: str) -> set[str]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} records must be a list")
    paths = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError(f"invalid {label} record")
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe {label} path: {row['path']}")
        paths.append(relative.as_posix())
    if len(paths) != len(set(paths)):
        raise ValueError(f"duplicate {label} paths")
    return set(paths)


def _validate_file_record(
    payload_dir: Path, row: dict, *, required: bool,
) -> None:
    path = payload_dir / row["path"]
    if not path.is_file():
        if required:
            raise ValueError(f"payload file missing: {row['path']}")
        return
    if (
        path.stat().st_size != int(row["size"])
        or sha256_file(path) != row["sha256"]
    ):
        raise ValueError(f"payload hash mismatch: {row['path']}")


def _assert_no_forbidden_paths(files: list[dict]) -> None:
    violations = []
    for row in files:
        parts = {
            part.lower().replace("-", "_")
            for part in Path(row["path"]).parts
        }
        tokens = set()
        for part in parts:
            tokens.update(part.replace(".", "_").split("_"))
        if tokens & _FORBIDDEN_PAYLOAD_PARTS:
            violations.append(row["path"])
    if violations:
        raise ValueError(f"forbidden evaluation-label paths in payload: {violations}")


def _validate_source_split(mode: str, rows: list[dict]) -> None:
    expected = (
        {"primary_tune"} if mode == "dev"
        else {"primary_locked", "hard"}
    )
    actual = {str(row.get("split")) for row in rows}
    if not actual or not actual <= expected:
        raise ValueError(
            f"{mode} question source split mismatch: {sorted(actual)}"
        )


def _validate_question_match(
    questions: list[dict], baseline_rows: list[dict]
) -> None:
    if len(questions) != len(baseline_rows):
        raise ValueError("question/R0 retrieval count mismatch")
    for question, baseline in zip(questions, baseline_rows):
        if str(question["id"]) != str(baseline.get("id")):
            raise ValueError("question/R0 ID or order mismatch")
        if question["question"] != baseline.get("question"):
            raise ValueError(f"question mismatch for {question['id']}")
