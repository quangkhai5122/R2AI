"""Deterministic candidate and evaluation-contract freeze manifests."""
from __future__ import annotations

from pathlib import Path

from .builder import (
    EXT_MANIFEST_NAME,
    MANIFEST_NAME,
    REVIEW_LEDGER_NAME,
    validate_corpus,
)
from .common import (
    canonical_sha256,
    file_sha256,
    load_json,
    tree_rows,
    tree_sha256,
    write_json,
)

CANDIDATE_FREEZE_SCHEMA = "g3b_candidate_freeze_v1"
EVALUATION_FREEZE_SCHEMA = "g3_evaluation_freeze_v1"
DEFAULT_CONTRACT_FILES = (
    "configs/g3b_evaluation_v1.json",
    "vifinqa/g3b/__init__.py",
    "vifinqa/g3b/builder.py",
    "vifinqa/g3b/common.py",
    "vifinqa/g3b/evaluate.py",
    "vifinqa/g3b/freeze.py",
    "vifinqa/g3b/generate.py",
    "vifinqa/g3b/source.py",
    "vifinqa/g3b/views.py",
    "scripts/69_g3b_build.py",
    "scripts/70_g3b_review.py",
    "scripts/71_g3b_freeze_candidate.py",
    "scripts/72_g3b_evaluate.py",
    "scripts/73_freeze_g3_evaluation.py",
    "scripts/74_g3b_build_submission.py",
)


def _results_path(submission: Path | str | None) -> Path | None:
    if submission is None:
        return None
    path = Path(submission)
    return path / "results.json" if path.is_dir() else path


def _seal(payload: dict) -> dict:
    sealed = dict(payload)
    sealed["fingerprint_sha256"] = canonical_sha256(sealed)
    return sealed


def create_candidate_freeze(
    candidate_name: str,
    corpus_dir: Path | str,
    config_path: Path | str,
    *,
    submission: Path | str | None = None,
    typed_predictions: Path | str | None = None,
    output_path: Path | str | None = None,
) -> dict:
    """Bind a promotion candidate to exact predictions and evaluation inputs."""
    candidate_name = str(candidate_name).strip()
    if not candidate_name:
        raise ValueError("candidate name must not be blank")
    submission_path = _results_path(submission)
    typed_path = (
        Path(typed_predictions) if typed_predictions is not None else None
    )
    if submission_path is None and typed_path is None:
        raise ValueError(
            "candidate freeze requires a submission or typed predictions"
        )
    for path in (submission_path, typed_path):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    corpus_dir = Path(corpus_dir)
    config_path = Path(config_path)
    payload = _seal({
        "schema_version": CANDIDATE_FREEZE_SCHEMA,
        "candidate_name": candidate_name,
        "policy_mode": "promotion",
        "corpus_manifest_sha256": file_sha256(
            corpus_dir / MANIFEST_NAME
        ),
        "config_sha256": file_sha256(config_path),
        "submission_results_sha256": (
            file_sha256(submission_path) if submission_path else None
        ),
        "typed_predictions_sha256": (
            file_sha256(typed_path) if typed_path else None
        ),
    })
    if output_path is not None:
        write_json(output_path, payload)
    return payload


def _contract_hashes(repo_root: Path) -> dict:
    output = {}
    for relative in DEFAULT_CONTRACT_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        output[relative] = file_sha256(path)
    return output


def create_evaluation_freeze(
    *,
    g3a_v1_dir: Path | str = "data/g3a_v1",
    extension_dir: Path | str = "data/g3a_extension_v1",
    corpus_dir: Path | str = "data/g3b_v1",
    config_path: Path | str = "configs/g3b_evaluation_v1.json",
    repo_root: Path | str = ".",
    output_path: Path | str | None = None,
) -> dict:
    """Seal the complete G3A/G3B evaluation contract before G3C."""
    g3a_v1_dir = Path(g3a_v1_dir)
    extension_dir = Path(extension_dir)
    corpus_dir = Path(corpus_dir)
    config_path = Path(config_path)
    repo_root = Path(repo_root)
    validation = validate_corpus(
        extension_dir,
        corpus_dir,
        config_path,
        g3a_v1_dir=g3a_v1_dir,
    )
    g3a_manifest_path = g3a_v1_dir / "g3a_manifest.json"
    extension_manifest_path = extension_dir / EXT_MANIFEST_NAME
    corpus_manifest_path = corpus_dir / MANIFEST_NAME
    review_path = corpus_dir / REVIEW_LEDGER_NAME
    g3a_manifest = load_json(g3a_manifest_path)
    extension_manifest = load_json(extension_manifest_path)
    corpus_manifest = load_json(corpus_manifest_path)
    payload = _seal({
        "schema_version": EVALUATION_FREEZE_SCHEMA,
        "purpose": (
            "freeze evaluation contract before G3C retrieval ablations"
        ),
        "policy": {
            "dev": "primary_tune only",
            "promotion": (
                "primary_locked plus hard; candidate freeze required"
            ),
            "ood_views": (
                "overlapping diagnostics, not independent replications"
            ),
            "public_questions": "exact id/text exclusion only",
        },
        "g3a_v1": {
            "tree_sha256": tree_sha256(g3a_v1_dir),
            "tree_files": tree_rows(g3a_v1_dir),
            "manifest_sha256": file_sha256(g3a_manifest_path),
            "fingerprint_sha256": g3a_manifest[
                "bundle_fingerprint_sha256"
            ],
        },
        "g3a_extension": {
            "manifest_sha256": file_sha256(extension_manifest_path),
            "fingerprint_sha256": extension_manifest[
                "fingerprint_sha256"
            ],
        },
        "g3b": {
            "manifest_sha256": file_sha256(corpus_manifest_path),
            "fingerprint_sha256": corpus_manifest[
                "fingerprint_sha256"
            ],
            "review_ledger_sha256": file_sha256(review_path),
            "view_hashes": corpus_manifest["view_hashes"],
            "questions": validation["questions"],
            "reviews_pending": validation["reviews_pending"],
        },
        "contract_files": _contract_hashes(repo_root),
    })
    if output_path is not None:
        write_json(output_path, payload)
    return payload


def validate_evaluation_freeze(
    freeze_path: Path | str,
    **kwargs,
) -> dict:
    """Recompute the deterministic freeze and reject any drift."""
    freeze_path = Path(freeze_path)
    observed = load_json(freeze_path)
    expected = create_evaluation_freeze(**kwargs)
    if observed != expected:
        raise ValueError("G3 evaluation freeze does not match current files")
    return {
        "valid": True,
        "fingerprint_sha256": observed["fingerprint_sha256"],
        "g3a_v1_tree_sha256": observed["g3a_v1"]["tree_sha256"],
        "g3b_fingerprint_sha256": observed["g3b"][
            "fingerprint_sha256"
        ],
    }
