import importlib.util
import json
from pathlib import Path

import pytest


def _load_validator():
    path = Path(__file__).resolve().parents[1] / "kaggle" / "validate_clean_codegen_v2.py"
    spec = importlib.util.spec_from_file_location("clean_codegen_validator_v2", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )


def _result(trace_schema=2, trace_mode="select_v2"):
    return {
        "id": 1,
        "question": "q1",
        "answer": 1.0,
        "pandas_query": "1.0",
        "used_vars": [],
        "status": "ok",
        "source": "llm_select_v2",
        "run_signature": "sig",
        "llm_attempt_status": "completed",
        "selection_trace": {
            "schema_version": trace_schema,
            "mode": trace_mode,
            "outcome": "accepted",
            "attempts": [{"accepted": True}],
            "rejection_counts": {},
        },
    }


def test_validator_accepts_real_selection_v2_trace_contract(tmp_path):
    validator = _load_validator()
    retrieval = tmp_path / "retrieval.jsonl"
    codegen = tmp_path / "results.jsonl"
    _write_jsonl(retrieval, [{"id": 1, "question": "q1"}])
    _write_jsonl(codegen, [_result()])
    report = validator.validate_codegen(
        retrieval, codegen, require_complete_llm=True,
    )
    assert report["validator_profile"] == "clean-codegen-select-v2-v2"
    assert report["llm_completed"] == 1
    assert report["llm_accepted"] == 1


def test_validator_rejects_selection_v1_trace_schema(tmp_path):
    validator = _load_validator()
    retrieval = tmp_path / "retrieval.jsonl"
    codegen = tmp_path / "results.jsonl"
    _write_jsonl(retrieval, [{"id": 1, "question": "q1"}])
    _write_jsonl(codegen, [_result(trace_schema=1, trace_mode=None)])
    with pytest.raises(SystemExit, match="trace schema must be 2"):
        validator.validate_codegen(
            retrieval, codegen, require_complete_llm=True,
        )
