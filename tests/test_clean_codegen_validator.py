import importlib.util
import json
from pathlib import Path

import pytest


def _load_validator():
    path = Path(__file__).resolve().parents[1] / "kaggle" / "validate_clean_codegen.py"
    spec = importlib.util.spec_from_file_location("clean_codegen_validator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )


def _result(qid, question, signature="sig"):
    return {
        "id": qid,
        "question": question,
        "answer": 1.0,
        "pandas_query": "1.0",
        "used_vars": [],
        "status": "ok",
        "source": "llm_select_v2",
        "run_signature": signature,
        "llm_attempt_status": "completed",
        "selection_trace": {
            "schema_version": 1,
            "outcome": "accepted",
            "rejection_counts": {},
        },
    }


def test_validator_accepts_complete_exact_signature_checkpoint(tmp_path):
    validator = _load_validator()
    retrieval = tmp_path / "retrieval.jsonl"
    codegen = tmp_path / "results.jsonl"
    _write_jsonl(retrieval, [{"id": 1, "question": "q1"}, {"id": 2, "question": "q2"}])
    _write_jsonl(codegen, [_result(1, "q1"), _result(2, "q2")])
    report = validator.validate_codegen(
        retrieval, codegen, require_complete_llm=True,
    )
    assert report["validated_records"] == 2
    assert report["llm_completed"] == 2
    assert report["llm_accepted"] == 2


def test_validator_rejects_partial_or_mixed_checkpoint(tmp_path):
    validator = _load_validator()
    retrieval = tmp_path / "retrieval.jsonl"
    codegen = tmp_path / "results.jsonl"
    _write_jsonl(retrieval, [{"id": 1, "question": "q1"}, {"id": 2, "question": "q2"}])
    partial = _result(1, "q1")
    partial.pop("llm_attempt_status")
    partial.pop("selection_trace")
    _write_jsonl(codegen, [partial, _result(2, "q2", signature="other")])
    with pytest.raises(SystemExit, match="incomplete"):
        validator.validate_codegen(
            retrieval, codegen, require_complete_llm=True,
        )
