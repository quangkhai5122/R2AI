from pathlib import Path

import pytest

from vifinqa.g3b.evaluate import _retrieval, _summarize, evaluate_g3b


ROOT = Path(__file__).resolve().parents[1]


def _metric_row(*, typed: bool, correct: bool) -> dict:
    row = {
        f"{prefix}_{metric}": 1.0
        for prefix in ("docs", "tables")
        for metric in ("precision", "recall", "f2", "mrr5")
    }
    row.update({
        "answer_correct": correct,
        "execution_correct": correct,
        "execution_ran": correct,
        "full_plan_coverage": correct,
        "typed_present": typed,
        "operator_correct": correct,
        "operand_role_correct": correct,
        "output_type_correct": correct,
        "ast_match": correct,
        "typed_execution_correct": correct,
        "typed_execution_ran": correct,
        "leaf_recall_at_k": 1.0 if correct else 0.0,
    })
    return row


def test_retrieval_metrics_are_competition_shaped():
    metric = _retrieval(["x", "gold", "z"], ["gold"])
    assert metric["precision"] == 1 / 3
    assert metric["recall"] == 1.0
    assert metric["f2"] == 5 / 7
    assert metric["mrr5"] == 0.5


def test_missing_typed_output_scores_zero_on_full_denominator():
    metrics = _summarize([
        _metric_row(typed=True, correct=True),
        _metric_row(typed=False, correct=False),
    ])
    assert metrics["typed_output_coverage"] == 0.5
    assert metrics["operator_accuracy"] == 0.5
    assert metrics["canonical_ast_match_accuracy"] == 0.5
    assert metrics["operator_accuracy_given_typed"] == 1.0


def test_oracle_dev_scores_reasoning_and_marks_retrieval_bypassed():
    report = evaluate_g3b(
        ROOT / "data/g3b_v1",
        ROOT / "data/g3a_extension_v1",
        ROOT / "configs/g3b_evaluation_v1.json",
        policy_mode="dev",
        evidence_mode="oracle_evidence",
        typed_predictions=(
            ROOT / "data/g3b_v1/g3b_oracle_predictions.jsonl"
        ),
    )
    assert report["integrity"]["passed"]
    assert report["metrics"]["typed_output_coverage"] == 1.0
    assert report["metrics"]["canonical_ast_match_accuracy"] == 1.0
    assert report["metrics"]["typed_execution_accuracy"] == 1.0
    assert "do not interpret" in report["retrieval_interpretation"]
    ood = report["breakdown"]["ood_views"]
    assert "not independent" in ood["interpretation"]
    assert set(ood["views"]) >= {
        "loto", "loyo", "loro", "lomo", "composition",
        "scope_period_stress",
    }


def test_promotion_requires_candidate_freeze():
    with pytest.raises(ValueError, match="candidate-freeze"):
        evaluate_g3b(
            ROOT / "data/g3b_v1",
            ROOT / "data/g3a_extension_v1",
            ROOT / "configs/g3b_evaluation_v1.json",
            policy_mode="promotion",
            evidence_mode="oracle_evidence",
            typed_predictions=(
                ROOT / "data/g3b_v1/g3b_oracle_predictions.jsonl"
            ),
        )
