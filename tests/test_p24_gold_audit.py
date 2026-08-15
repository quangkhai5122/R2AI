from vifinqa.devset.p24_gold_audit import build_tune_gold_audit


def test_audit_counts_ops_and_flags():
    question = {"id": 1}
    spec = {"id": 1, "expression": "E1 / E2"}
    evidence = [
        {"report_id": "R", "table_pos": 1, "row": 1, "col": 1},
        {"report_id": "R", "table_pos": 1, "row": 1, "col": 2},
    ]
    gold = [{
        "id": 1, "stratum": "ratio|multi_2_4", "evidence": evidence,
        "output": {"type": "ratio", "value": 125.0, "unit": "times"},
        "ast": {"kind": "op", "op": "divide", "args": [
            {"kind": "evidence", "evidence_id": "E1"},
            {"kind": "evidence", "evidence_id": "E2"},
        ]},
        "annotator_notes": "fixture",
    }]
    audit = build_tune_gold_audit(gold, [question], [spec])
    assert audit["count"] == 1
    assert audit["operation_counts"] == {"divide": 1}
    assert audit["provenance"]["evidence_unique_exact_cells"] == 2
    assert audit["review_flags"][0]["reasons"] == ["ratio_magnitude_gt_100"]


def test_audit_rejects_incomplete_id_universe():
    try:
        build_tune_gold_audit([], [{"id": 1}], [])
    except ValueError as exc:
        assert "exactly cover" in str(exc)
    else:
        raise AssertionError("expected incomplete universe to fail")
