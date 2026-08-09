import json
import tempfile
import unittest
from pathlib import Path

from vifinqa.codegen.hybrid import is_structural_none, merge_codegen_hybrid
from vifinqa.utils.io import read_jsonl


def _row(qid, signature, *, status="ok", source="llm_select", answer=2.0,
         query="float(df1['value'].iloc[0])", question=None):
    return {
        "id": qid,
        "question": question or f"question {qid}",
        "answer": answer,
        "pandas_query": query,
        "used_vars": ([{"var": "df1", "report_id": "AAA_report", "table_pos": 1}]
                      if "df1" in query else []),
        "status": status,
        "source": source,
        "run_signature": signature,
        "detail": f"from {signature}",
    }


def _none(qid, signature):
    return _row(qid, signature, status="failed", source="none", answer=0.0,
                query="0.0")


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


class HybridCodegenTests(unittest.TestCase):
    def test_fallback_only_replaces_structural_none(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            primary_path, fallback_path = root / "p.jsonl", root / "f.jsonl"
            out_path = root / "hybrid.jsonl"
            primary = [
                _row(1, "primary", source="rule", answer=10.0),
                _none(2, "primary"),
                _row(3, "primary", source="rule", answer=0.0),
                _none(4, "primary"),
            ]
            fallback = [
                _row(1, "fallback", answer=99.0),
                _row(2, "fallback", answer=22.0),
                _row(3, "fallback", answer=33.0),
                _none(4, "fallback"),
            ]
            _write(primary_path, primary)
            _write(fallback_path, fallback)

            audit = merge_codegen_hybrid(primary_path, fallback_path, out_path)
            rows = {row["id"]: row for row in read_jsonl(out_path)}

            self.assertEqual(rows[1]["answer"], 10.0)
            self.assertEqual(rows[1]["source"], "rule")
            self.assertEqual(rows[2]["answer"], 22.0)
            self.assertEqual(rows[2]["hybrid_provenance"]["selected_from"], "fallback")
            self.assertEqual(rows[3]["answer"], 0.0)
            self.assertEqual(rows[3]["hybrid_provenance"]["selected_from"], "primary")
            self.assertEqual(rows[4]["status"], "failed")
            self.assertEqual(audit["counts"], {
                "total": 4, "kept_primary": 2, "used_fallback": 1, "unresolved": 1,
            })
            self.assertEqual(audit["selected_fallback_ids"], [2])
            self.assertEqual(audit["unresolved_ids"], [4])

    def test_successful_computed_zero_is_not_structural_none(self):
        computed_zero = _row(1, "sig", answer=0.0)
        self.assertFalse(is_structural_none(computed_zero))
        self.assertTrue(is_structural_none(_none(1, "sig")))

    def test_preserves_selected_query_and_evidence_and_records_signatures(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p, f, out = root / "p.jsonl", root / "f.jsonl", root / "out.jsonl"
            fallback = _row(1, "sig-7b", answer=7.0)
            fallback["used_vars"][0]["extra"] = "preserve-me"
            _write(p, [_none(1, "sig-14b")])
            _write(f, [fallback])
            audit = merge_codegen_hybrid(p, f, out)
            row = read_jsonl(out)[0]

            self.assertEqual(row["pandas_query"], fallback["pandas_query"])
            self.assertEqual(row["used_vars"], fallback["used_vars"])
            self.assertEqual(row["detail"], "from sig-7b")
            self.assertEqual(
                row["hybrid_provenance"]["selected_run_signature"], "sig-7b"
            )
            self.assertEqual(row["run_signature"], audit["hybrid_run_signature"])
            self.assertNotEqual(row["run_signature"], "sig-7b")

    def test_rejects_id_question_and_signature_mismatches(self):
        cases = (
            ([_none(1, "p")], [_row(2, "f")], "id sets differ"),
            ([_none(1, "p")], [_row(1, "f", question="different")],
             "questions differ"),
            ([_none(1, "p"), _none(2, "p2")],
             [_row(1, "f"), _row(2, "f")], "multiple run_signatures"),
        )
        for primary, fallback, error in cases:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                p, f = root / "p.jsonl", root / "f.jsonl"
                _write(p, primary)
                _write(f, fallback)
                with self.assertRaisesRegex(ValueError, error):
                    merge_codegen_hybrid(p, f, root / "out.jsonl")

    def test_expected_signature_guard(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p, f = root / "p.jsonl", root / "f.jsonl"
            _write(p, [_none(1, "actual-primary")])
            _write(f, [_row(1, "actual-fallback")])
            with self.assertRaisesRegex(ValueError, "signature mismatch"):
                merge_codegen_hybrid(
                    p, f, root / "out.jsonl",
                    expected_primary_signature="wrong-primary",
                )


if __name__ == "__main__":
    unittest.main()
