from pathlib import Path

import pytest

from vifinqa.g3b.evaluate import _verify_freeze
from vifinqa.g3b.freeze import (
    create_candidate_freeze,
    create_evaluation_freeze,
    validate_evaluation_freeze,
)


ROOT = Path(__file__).resolve().parents[1]


def test_candidate_freeze_binds_exact_typed_predictions(tmp_path):
    typed = ROOT / "data/g3b_v1/g3b_oracle_predictions.jsonl"
    output = tmp_path / "candidate.json"
    freeze = create_candidate_freeze(
        "oracle-control",
        ROOT / "data/g3b_v1",
        ROOT / "configs/g3b_evaluation_v1.json",
        typed_predictions=typed,
        output_path=output,
    )
    observed = _verify_freeze(
        output,
        ROOT / "data/g3b_v1",
        ROOT / "configs/g3b_evaluation_v1.json",
        None,
        typed,
    )
    assert observed == freeze


def test_candidate_freeze_requires_a_prediction_artifact():
    with pytest.raises(ValueError, match="requires"):
        create_candidate_freeze(
            "empty",
            ROOT / "data/g3b_v1",
            ROOT / "configs/g3b_evaluation_v1.json",
        )


def test_final_evaluation_freeze_round_trip(tmp_path):
    output = tmp_path / "evaluation-freeze.json"
    freeze = create_evaluation_freeze(
        g3a_v1_dir=ROOT / "data/g3a_v1",
        extension_dir=ROOT / "data/g3a_extension_v1",
        corpus_dir=ROOT / "data/g3b_v1",
        config_path=ROOT / "configs/g3b_evaluation_v1.json",
        repo_root=ROOT,
        output_path=output,
    )
    validated = validate_evaluation_freeze(
        output,
        g3a_v1_dir=ROOT / "data/g3a_v1",
        extension_dir=ROOT / "data/g3a_extension_v1",
        corpus_dir=ROOT / "data/g3b_v1",
        config_path=ROOT / "configs/g3b_evaluation_v1.json",
        repo_root=ROOT,
    )
    assert validated["valid"]
    assert validated["fingerprint_sha256"] == freeze[
        "fingerprint_sha256"
    ]
