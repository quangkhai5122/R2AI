from __future__ import annotations

import hashlib
import json

import pytest

from vifinqa.codegen.selection_v2_replay import _saved_samples, _scope_attempted, _signature


def _row(raw: str, *, truncated: bool = False) -> dict:
    return {"selection_trace": {
        "samples_received": 1,
        "attempts": [{
            "index": 1,
            "raw_response": raw,
            "raw_truncated": truncated,
            "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        }],
    }}


def test_saved_samples_requires_complete_hashed_raw_response():
    raw = json.dumps({"schema_version": 2})
    assert _saved_samples(_row(raw), 7) == [raw]
    with pytest.raises(ValueError, match="truncated"):
        _saved_samples(_row(raw, truncated=True), 7)
    bad = _row(raw)
    bad["selection_trace"]["attempts"][0]["raw_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        _saved_samples(bad, 7)


def test_saved_samples_requires_contiguous_attempt_indices():
    row = _row("a")
    row["selection_trace"]["attempts"][0]["index"] = 2
    with pytest.raises(ValueError, match="not contiguous"):
        _saved_samples(row, 8)


def test_replay_signature_fingerprints_rescue_contract():
    kwargs = dict(
        input_hashes={"checkpoint": "a"}, semantic_sha="b", store_sha="c",
        checkpoint_signature="d", control_signature="e", k=0, top_n=24,
        rescue_no_candidates=False, rescue_table_k=20, rescue_min_score=28.0,
    )
    base = _signature(**kwargs)
    assert base != _signature(**{**kwargs, "rescue_no_candidates": True})
    assert base != _signature(**{**kwargs, "rescue_table_k": 21})
    assert base != _signature(**{**kwargs, "rescue_min_score": 27.0})
    assert base != _signature(**{**kwargs, "allow_checkpoint_superset": True})


def test_checkpoint_superset_is_fail_closed_by_default():
    with pytest.raises(ValueError, match="outside mask"):
        _scope_attempted({1, 2, 3}, {2, 3}, allow_checkpoint_superset=False)


def test_checkpoint_superset_opt_in_scopes_replay_and_audits_ignored_ids():
    attempted, ignored = _scope_attempted(
        {5, 1, 3, 2}, {2, 3, 4}, allow_checkpoint_superset=True,
    )
    assert attempted == {2, 3}
    assert ignored == [1, 5]


