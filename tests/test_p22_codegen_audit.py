import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_p22_codegen", ROOT / "scripts" / "48_audit_p22_codegen.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _files(tmp_path, *, attempted=True, mask_ids=(1,)):
    retrieval = tmp_path / "retrieval.jsonl"
    codegen = tmp_path / "codegen.jsonl"
    mask = tmp_path / "mask.json"
    _write_jsonl(retrieval, [
        {"id": 1, "question": "q1"}, {"id": 2, "question": "q2"},
    ])
    first = {
        "id": 1, "question": "q1", "answer": 3.0, "run_signature": "sig",
        "source": "llm_select_v2", "status": "ok",
        "pandas_query": "float(df1.loc[(df1['row'] == 1), 'value'].iloc[0])",
        "used_vars": [{"var": "df1"}],
    }
    if attempted:
        first.update({
            "llm_attempt_status": "completed",
            "selection_trace": {
                "schema_version": 2, "mode": "select_v2", "outcome": "accepted",
                "rejection_counts": {}, "shortlist": {"rescue_mode": "none"},
            },
        })
    _write_jsonl(codegen, [
        first,
        {"id": 2, "question": "q2", "answer": 2.0, "run_signature": "sig",
         "source": "rule", "status": "ok", "pandas_query": "2.0", "used_vars": []},
    ])
    mask.write_text(json.dumps({"count": len(mask_ids), "ids": list(mask_ids)}),
                    encoding="utf-8")
    return codegen, retrieval, mask


def test_complete_masked_v2_output_passes(tmp_path):
    codegen, retrieval, mask = _files(tmp_path)
    report = MODULE.audit(codegen, retrieval, mask)
    assert report["counts"] == {
        "rows": 2, "target": 1, "attempted": 1, "pending": 0,
        "accepted": 1, "rejected": 0,
        "inherited_attempted": 0,
    }


def test_incomplete_requires_explicit_diagnostic_mode(tmp_path):
    codegen, retrieval, mask = _files(tmp_path, attempted=False)
    with pytest.raises(ValueError, match="incomplete"):
        MODULE.audit(codegen, retrieval, mask)
    report = MODULE.audit(codegen, retrieval, mask, allow_incomplete=True)
    assert report["pending_ids"] == [1]


def test_attempt_outside_mask_fails_closed(tmp_path):
    codegen, retrieval, mask = _files(tmp_path, mask_ids=(2,))
    with pytest.raises(ValueError, match="outside mask"):
        MODULE.audit(codegen, retrieval, mask, allow_incomplete=True)


def test_explicit_upstream_mask_allows_inherited_attempts(tmp_path):
    codegen, retrieval, target = _files(tmp_path, mask_ids=(2,))
    rows = [
        json.loads(line)
        for line in codegen.read_text(encoding="utf-8").splitlines()
    ]
    rows[1].update({
        "source": "llm_select_v2",
        "status": "ok",
        "pandas_query": "float(df2.loc[(df2['row'] == 1), 'value'].iloc[0])",
        "used_vars": [{"var": "df2"}],
        "llm_attempt_status": "completed",
        "selection_trace": {
            "schema_version": 2,
            "mode": "select_v2",
            "outcome": "accepted",
            "rejection_counts": {},
            "shortlist": {"rescue_mode": "atomic_partial_schema"},
        },
    })
    _write_jsonl(codegen, rows)
    upstream = tmp_path / "upstream.json"
    upstream.write_text(json.dumps({"count": 1, "ids": [1]}), encoding="utf-8")

    report = MODULE.audit(
        codegen, retrieval, target,
        allowed_attempt_mask_paths=(upstream,),
    )
