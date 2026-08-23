import json
from pathlib import Path

import pytest

from vifinqa.g3a.builder import (
    GOLD_NAME,
    MANIFEST_NAME,
    QUESTIONS_NAME,
    G3AValidationError,
    validate_bundle,
)
from vifinqa.g3a.common import canonical_sha256, file_sha256, write_json, write_jsonl
from vifinqa.g3a.evaluate import compare_reports, evaluate_submission


def _make_bundle(root: Path) -> Path:
    bundle = root / "bundle"
    question = {
        "id": 3300001,
        "question": "Synthetic source question?",
        "split": "hard",
        "set": "hard",
        "stratum": "lookup|single_cell",
        "operator": "lookup",
        "difficulty": "single_cell",
    }
    gold = {
        **question,
        "answer": 42.0,
        "output_type": "money_million_vnd",
        "tolerance": 0.01,
        "relevant_docs": ["DOC"],
        "relevant_tables": ["DOC|10"],
        "review": {"status": "approved"},
    }
    write_jsonl(bundle / QUESTIONS_NAME, [question])
    write_jsonl(bundle / GOLD_NAME, [gold])
    manifest = {
        "schema_version": "g3a_bundle_v1",
        "files": {
            QUESTIONS_NAME: {"sha256": file_sha256(bundle / QUESTIONS_NAME)},
            GOLD_NAME: {"sha256": file_sha256(bundle / GOLD_NAME)},
        },
    }
    manifest["bundle_fingerprint_sha256"] = canonical_sha256(manifest)
    write_json(bundle / MANIFEST_NAME, manifest)
    return bundle


def _make_config(root: Path) -> Path:
    path = root / "config.json"
    write_json(path, {
        "weight_scenarios": {
            "balanced": {
                "docs_f2_macro": 1,
                "tables_f2_macro": 1,
                "answer_accuracy": 1,
            },
        },
        "promotion_gate": {
            "metric_paths": [
                "metrics.docs_f2_macro",
                "metrics.tables_f2_macro",
                "metrics.answer_accuracy",
                "metrics.execution_accuracy",
            ],
            "max_regression": {
                "metrics.docs_f2_macro": 0,
                "metrics.tables_f2_macro": 0,
                "metrics.answer_accuracy": 0,
                "metrics.execution_accuracy": 0,
            },
            "min_material_gain": {
                "metrics.docs_f2_macro": 0.01,
                "metrics.tables_f2_macro": 0.01,
                "metrics.answer_accuracy": 0.01,
                "metrics.execution_accuracy": 0.01,
            },
            "hard_answer_max_regression": 0,
        },
    })
    return path


def _make_submission(root: Path) -> Path:
    submission = root / "submission"
    (submission / "data").mkdir(parents=True)
    (submission / "data" / "table.csv").write_text(
        "value\n42.0\n", encoding="utf-8"
    )
    write_json(submission / "results.json", [{
        "id": 3300001,
        "question": "Synthetic source question?",
        "answer": 42.0,
        "relevant_docs": ["DOC"],
        "relevant_tables": ["DOC|10"],
        "evidence": [{"variable": "df1", "csv_path": "data/table.csv"}],
        "pandas_query": "float(df1['value'].iloc[0])",
    }])
    return submission


def test_evaluator_scores_all_competition_tasks(tmp_path):
    report = evaluate_submission(
        _make_submission(tmp_path),
        _make_bundle(tmp_path),
        _make_config(tmp_path),
    )
    assert report["integrity"]["passed"]
    assert report["metrics"]["docs_f2_macro"] == 1.0
    assert report["metrics"]["docs_mrr5"] == 1.0
    assert report["metrics"]["tables_f2_macro"] == 1.0
    assert report["metrics"]["tables_mrr5"] == 1.0
    assert report["metrics"]["answer_accuracy"] == 1.0
    assert report["metrics"]["execution_accuracy"] == 1.0


def test_retrieval_metric_is_macro_f2_and_mrr5():
    from vifinqa.g3a.evaluate import _retrieval_metrics

    metric = _retrieval_metrics(["x", "gold", "z"], ["gold"])
    assert metric["precision"] == 1 / 3
    assert metric["recall"] == 1.0
    assert metric["f2"] == 5 / 7
    assert metric["mrr5"] == 0.5


def test_compare_blocks_unknown_weight_regression(tmp_path):
    config = _make_config(tmp_path)
    base = {
        "integrity": {"passed": True},
        "metrics": {
            "docs_f2_macro": 0.8,
            "tables_f2_macro": 0.6,
            "answer_accuracy": 0.5,
            "execution_accuracy": 0.5,
        },
        "weight_scenarios": {"balanced": {"score": 0.633333}},
        "breakdown": {"set": {"hard": {"answer_accuracy": 0.5}}},
    }
    candidate = json.loads(json.dumps(base))
    candidate["metrics"]["answer_accuracy"] = 0.6
    candidate["metrics"]["docs_f2_macro"] = 0.7
    candidate["weight_scenarios"]["balanced"]["score"] = 0.633333
    base_path, candidate_path = tmp_path / "base.json", tmp_path / "candidate.json"
    write_json(base_path, base)
    write_json(candidate_path, candidate)
    report = compare_reports(base_path, candidate_path, config)
    assert report["decision"] == "block"
    assert "metric_regression_guard_failed" in report["blockers"]


def test_bundle_validation_rejects_manifest_tamper(tmp_path):
    bundle = _make_bundle(tmp_path)
    manifest_path = bundle / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tampered_after_seal"] = True
    write_json(manifest_path, manifest)
    with pytest.raises(G3AValidationError, match="fingerprint"):
        validate_bundle(bundle)