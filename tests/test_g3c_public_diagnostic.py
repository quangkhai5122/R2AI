from __future__ import annotations

import json
import zipfile
from copy import deepcopy

import pytest

from vifinqa.g3c_official.public_diagnostic import (
    audit_retrieval_only_rows,
    audit_zip_delta,
)


def _row(qid: int) -> dict:
    return {
        "id": qid,
        "question": f"q{qid}",
        "answer": 1.0,
        "relevant_docs": ["R0"],
        "relevant_tables": ["R0|10"],
        "evidence": [{"variable": "df1", "csv_path": "data/t.csv"}],
        "pandas_query": "df1['value'].iloc[0]",
    }


def test_public_diagnostic_allows_only_retrieval_field_changes():
    baseline = [_row(1), _row(2)]
    candidate = deepcopy(baseline)
    candidate[0]["relevant_docs"] = ["R4"]
    candidate[0]["relevant_tables"] = ["R4|20"]
    audit = audit_retrieval_only_rows(
        baseline, candidate, expected_count=2
    )
    assert audit["non_retrieval_fields_exact"] is True
    assert audit["table_set_changed_count"] == 1
    assert audit["doc_set_changed_count"] == 1
    assert audit["changed_field_counts"] == {
        "relevant_docs": 1,
        "relevant_tables": 1,
    }


def test_public_diagnostic_rejects_answer_drift():
    baseline = [_row(1)]
    candidate = deepcopy(baseline)
    candidate[0]["answer"] = 2.0
    candidate[0]["relevant_docs"] = ["R4"]
    candidate[0]["relevant_tables"] = ["R4|20"]
    with pytest.raises(ValueError, match="non-retrieval field drift"):
        audit_retrieval_only_rows(baseline, candidate, expected_count=1)


def test_public_diagnostic_zip_diff_is_results_only(tmp_path):
    baseline_zip = tmp_path / "baseline.zip"
    candidate_zip = tmp_path / "candidate.zip"
    candidate_results = tmp_path / "results.json"
    baseline_rows = [_row(1)]
    candidate_rows = deepcopy(baseline_rows)
    candidate_rows[0]["relevant_docs"] = ["R4"]
    candidate_results.write_text(
        json.dumps(candidate_rows), encoding="utf-8"
    )
    for path, rows in (
        (baseline_zip, baseline_rows),
        (candidate_zip, candidate_rows),
    ):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("results.json", json.dumps(rows))
            archive.writestr("data/t.csv", "value\n1\n")
    audit = audit_zip_delta(
        baseline_zip, candidate_zip, candidate_results
    )
    assert audit["differing_members_vs_b1"] == ["results.json"]
    assert audit["data_members_byte_exact_vs_b1"] is True
