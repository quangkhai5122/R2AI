"""Tests for conservative replay of saved P2.1 Selection attempts."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from vifinqa.codegen.selection_replay import (
    replay_selection_artifact,
    replay_selection_record,
)


class _Candidate:
    def __init__(self, *, var="df1", row=1, col=1, value=5.0,
                 fact_year=None, score=80.0):
        self.var = var
        self.row = row
        self.col = col
        self.value = value
        self.unit_scale = 1.0
        self.fact_year = fact_year
        self.col_name = str(fact_year or 2024)
        self.label = "Metric"
        self.score = score


class _Bundle:
    def __init__(self, candidates, output_type="number"):
        self._candidates = candidates
        self.route = {
            "unit_scale": 1.0,
            "output_type": output_type,
            "years": [c.fact_year for c in candidates if c.fact_year],
        }
        self.dfs = {
            c.var: pd.DataFrame([{
                "row": c.row, "col": c.col, "value": c.value,
            }]) for c in candidates
        }

    def shortlist(self, _encoder=None, top_n=12):
        return self._candidates[:top_n]

    def used_vars(self, query):
        return [{"var": name, "report_id": f"AAA_{name}", "table_pos": i}
                for i, name in enumerate(sorted(self.dfs), 1) if name in query]


def _none_record(op="lookup", operands=None, reason="answer_mismatch"):
    operands = [1] if operands is None else operands
    return {
        "id": 1,
        "question": "Q",
        "answer": 0.0,
        "pandas_query": "0.0",
        "used_vars": [],
        "status": "failed",
        "source": "none",
        "run_signature": "original",
        "selection_trace": {
            "outcome": "rejected",
            "attempts": [{
                "index": 1,
                "reason_code": reason,
                "selection": {"op": op, "operands": operands},
            }],
        },
    }


class ReplayRecordTests(unittest.TestCase):
    def test_default_policy_replays_structural_none(self):
        original = _none_record()
        before = dict(original)
        row, outcome, meta = replay_selection_record(
            original, _Bundle([_Candidate(value=7.0)]),
        )
        self.assertEqual(outcome, "replayed")
        self.assertEqual(row["answer"], 7.0)
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["source"], "llm_select_p21r")
        self.assertIn("['row'] == 1", row["pandas_query"])
        self.assertEqual(meta["original_reason_code"], "answer_mismatch")
        self.assertEqual(original, before)  # input record is never mutated

    def test_default_policy_never_overwrites_successful_rule(self):
        original = _none_record()
        original.update({
            "answer": 9.0,
            "pandas_query": "float(df1.loc[df1['row'] == 1, 'value'].iloc[0])",
            "status": "ok",
            "source": "rule",
        })
        row, outcome, meta = replay_selection_record(
            original, _Bundle([_Candidate(value=7.0)]),
        )
        self.assertEqual(outcome, "not_eligible")
        self.assertIsNone(meta)
        self.assertEqual(row["answer"], 9.0)
        self.assertEqual(row["source"], "rule")

    def test_saved_ranking_selection_can_be_replayed_as_year(self):
        original = _none_record("ranking_max", [1, 2],
                                "semantic_validation_failed")
        candidates = [
            _Candidate(var="df1", row=1, value=3.0, fact_year=2020),
            _Candidate(var="df2", row=2, value=8.0, fact_year=2024),
        ]
        row, outcome, meta = replay_selection_record(
            original, _Bundle(candidates, output_type="year"),
        )
        self.assertEqual(outcome, "replayed")
        self.assertEqual(row["answer"], 2024.0)
        self.assertEqual(meta["selection"]["op"], "ranking_max")

    def test_saved_count_selection_returns_number_of_picks(self):
        original = _none_record("count", [1, 2], "invalid_selection")
        candidates = [
            _Candidate(var="df1", row=1),
            _Candidate(var="df2", row=2),
        ]
        row, outcome, _meta = replay_selection_record(
            original, _Bundle(candidates, output_type="count"),
        )
        self.assertEqual(outcome, "replayed")
        self.assertEqual(row["answer"], 2.0)
        self.assertEqual(len(row["used_vars"]), 2)

    def test_saved_count_rejects_two_indices_for_same_stable_cell(self):
        original = _none_record("count", [1, 2], "invalid_selection")
        candidates = [
            _Candidate(var="df1", row=4, col=2),
            _Candidate(var="df1", row=4, col=2),
        ]
        row, outcome, meta = replay_selection_record(
            original, _Bundle(candidates, output_type="count"),
        )
        self.assertEqual(row["source"], "none")
        self.assertIsNone(meta)
        self.assertIn("duplicate stable cells", outcome)

    def test_saved_year_rejects_same_cell_with_different_fact_years(self):
        original = _none_record("ranking_max", [1, 2], "invalid_selection")
        candidates = [
            _Candidate(var="df1", row=4, col=2, fact_year=2020),
            _Candidate(var="df1", row=4, col=2, fact_year=2024),
        ]
        row, outcome, meta = replay_selection_record(
            original, _Bundle(candidates, output_type="year"),
        )
        self.assertEqual(row["source"], "none")
        self.assertIsNone(meta)
        self.assertIn("duplicate stable cells", outcome)

    def test_extra_fixed_arity_operands_remain_unresolved(self):
        original = _none_record("lookup", [1, 2], "invalid_selection")
        row, outcome, _meta = replay_selection_record(
            original, _Bundle([_Candidate(), _Candidate(var="df2")]),
        )
        self.assertEqual(row["source"], "none")
        self.assertIn("needs exactly 1", outcome)


class ReplayArtifactTests(unittest.TestCase):
    @staticmethod
    def _write_jsonl(path, rows):
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_output_type_filter_is_counted_and_changes_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = root / "store"
            (store / "tables").mkdir(parents=True)
            (store / "reports.parquet").write_bytes(b"reports-v1")
            (store / "tables" / "AAA.parquet").write_bytes(b"tables-v1")

            number = _none_record("lookup", [1])
            number.update({"id": 1, "question": "number question"})
            year = _none_record("ranking_max", [1, 2])
            year.update({"id": 2, "question": "year question"})
            retrieval = [
                {
                    "id": 1,
                    "question": "number question",
                    "route": {"output_type": "number"},
                    "candidates": [{"ticker": "AAA", "report_id": "AAA_R1"}],
                },
                {
                    "id": 2,
                    "question": "year question",
                    "route": {"output_type": "year"},
                    "candidates": [{"ticker": "AAA", "report_id": "AAA_R1"}],
                },
            ]
            retrieval_path = root / "retrieval.jsonl"
            codegen_path = root / "codegen.jsonl"
            self._write_jsonl(retrieval_path, retrieval)
            self._write_jsonl(codegen_path, [number, year])

            def bundle_factory(rec, _store, **_kwargs):
                if rec["id"] == 2:
                    candidates = [
                        _Candidate(var="df1", value=3.0, fact_year=2020),
                        _Candidate(var="df2", row=2, value=8.0, fact_year=2024),
                    ]
                else:
                    candidates = [_Candidate(value=7.0)]
                return _Bundle(candidates, rec["route"]["output_type"])

            with patch("vifinqa.codegen.selection_replay.Store"), patch(
                "vifinqa.codegen.selection_replay.QuestionBundle",
                side_effect=bundle_factory,
            ):
                year_audit = replay_selection_artifact(
                    retrieval_path, codegen_path, store,
                    root / "year.jsonl", root / "year.audit.json",
                    output_types="year",
                )
            with patch("vifinqa.codegen.selection_replay.Store"), patch(
                "vifinqa.codegen.selection_replay.QuestionBundle",
                side_effect=bundle_factory,
            ):
                all_audit = replay_selection_artifact(
                    retrieval_path, codegen_path, store,
                    root / "all.jsonl", root / "all.audit.json",
                )

            self.assertEqual(year_audit["parameters"]["output_types"], ["year"])
            self.assertEqual(year_audit["counts"]["replayed"], 1)
            self.assertEqual(year_audit["counts"]["skipped_by_output_type"], 1)
            self.assertEqual(year_audit["skipped_by_output_type_ids"], [1])
            self.assertEqual(year_audit["counts"]["unresolved"], 0)
            self.assertEqual(all_audit["parameters"]["output_types"], "all")
            self.assertEqual(all_audit["counts"]["replayed"], 2)
            self.assertNotEqual(
                year_audit["output"]["run_signature"],
                all_audit["output"]["run_signature"],
            )
            self.assertIn("retrieval/shortlist.py", {
                item["name"] for item in year_audit["semantic_inputs"]["files"]
            })
            self.assertEqual(
                {item["path"] for item in year_audit["store"]["files"]},
                {"reports.parquet", "tables/AAA.parquet"},
            )

            (store / "tables" / "AAA.parquet").write_bytes(b"tables-v2")
            with patch("vifinqa.codegen.selection_replay.Store"), patch(
                "vifinqa.codegen.selection_replay.QuestionBundle",
                side_effect=bundle_factory,
            ):
                changed_store_audit = replay_selection_artifact(
                    retrieval_path, codegen_path, store,
                    root / "changed-store.jsonl",
                    root / "changed-store.audit.json",
                )
            self.assertNotEqual(
                all_audit["store"]["manifest_sha256"],
                changed_store_audit["store"]["manifest_sha256"],
            )
            self.assertNotEqual(
                all_audit["output"]["run_signature"],
                changed_store_audit["output"]["run_signature"],
            )


if __name__ == "__main__":
    unittest.main()
