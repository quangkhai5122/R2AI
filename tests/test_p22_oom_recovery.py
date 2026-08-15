from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "make_p22_oom_tail_mask",
    ROOT / "scripts" / "49_make_p22_oom_tail_mask.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )


def _fixture(tmp_path: Path, *, completed=(1, 2), parent=(1, 2, 3)):
    checkpoint = tmp_path / "checkpoint.jsonl"
    retrieval = tmp_path / "retrieval.jsonl"
    parent_mask = tmp_path / "parent.json"
    retrieval_rows = [
        {"id": qid, "question": f"q{qid}",
         "route": {"output_type": "number", "plan": {"op": "lookup"}}}
        for qid in (1, 2, 3, 4)
    ]
    checkpoint_rows = []
    for qid in (1, 2, 3, 4):
        row = {
            "id": qid, "question": f"q{qid}", "run_signature": "sig-a",
            "answer": 0.0, "source": "none", "status": "failed",
            "pandas_query": "0.0", "used_vars": [],
        }
        if qid in completed:
            row.update({
                "llm_attempt_status": "completed",
                "selection_trace": {
                    "schema_version": 2,
                    "mode": "select_v2",
                    "outcome": "rejected",
                },
            })
        checkpoint_rows.append(row)
    _write_jsonl(checkpoint, checkpoint_rows)
    _write_jsonl(retrieval, retrieval_rows)
    parent_mask.write_text(
        json.dumps({"name": "B", "count": len(parent), "ids": list(parent)}),
        encoding="utf-8",
    )
    return checkpoint, parent_mask, retrieval


def test_tail_is_parent_minus_durable_completed_attempts(tmp_path):
    checkpoint, parent, retrieval = _fixture(tmp_path)
    mask = MODULE.build_tail_mask(
        checkpoint, parent, retrieval, expect_pending=1,
    )
    assert mask["ids"] == [3]
    assert mask["checkpoint"]["completed_ids"] == [1, 2]
    assert mask["checkpoint"]["run_signature"] == "sig-a"
    assert mask["recovery_contract"]["new_output_required"] is True


def test_completed_attempt_outside_parent_fails_closed(tmp_path):
    checkpoint, parent, retrieval = _fixture(
        tmp_path, completed=(1, 2, 4), parent=(1, 2, 3),
    )
    with pytest.raises(ValueError, match="outside parent mask"):
        MODULE.build_tail_mask(checkpoint, parent, retrieval)


def test_partial_marker_trace_contract_fails_closed(tmp_path):
    checkpoint, parent, retrieval = _fixture(tmp_path, completed=(1,))
    rows = MODULE.read_jsonl(checkpoint)
    rows[1]["llm_attempt_status"] = "completed"
    _write_jsonl(checkpoint, rows)
    with pytest.raises(ValueError, match="incomplete Selection-v2"):
        MODULE.build_tail_mask(checkpoint, parent, retrieval)


def test_frozen_tail_write_is_idempotent_and_refuses_drift(tmp_path):
    out = tmp_path / "tail.json"
    first = {"schema_version": "v1", "ids": [3]}
    MODULE._write_idempotent(out, first)
    MODULE._write_idempotent(out, first)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        MODULE._write_idempotent(out, {"schema_version": "v1", "ids": [4]})
