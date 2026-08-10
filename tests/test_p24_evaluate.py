import copy
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from vifinqa.devset.evaluate import evaluate_codegen, fill_gold_hashes
from vifinqa.devset.p24 import (
    LOCKED_QUESTIONS,
    LOCKED_TEMPLATE,
    TUNE_QUESTIONS,
    TUNE_TEMPLATE,
    P24ValidationError,
    build_bundle,
    canonical_sha256,
    seal_locked_gold,
)
from vifinqa.utils.io import read_jsonl, write_jsonl


REPORT_ID = "ABC_financial_statements_2024_consolidated"
QUERY = (
    "round(float(df1.loc[(df1['row'] == 1) & "
    "(df1['col'] == 2), 'value'].iloc[0]), 2)"
)
USED_VARS = [{"var": "df1", "report_id": REPORT_ID, "table_pos": 7}]
EVIDENCE = [{
    "evidence_id": "E1",
    "variable": "df1",
    "report_id": REPORT_ID,
    "table_pos": 7,
    "row": 1,
    "col": 2,
    "label": "Doanh thu",
    "code": "1.0",
    "col_name": "2024.0",
    "value": 2.5,
    "unit_scale": 1.0,
}]
AST = {
    "kind": "op",
    "op": "lookup",
    "args": [{"kind": "evidence", "evidence_id": "E1"}],
}


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    questions = root / "questions.jsonl"
    retrieval = root / "retrieval.jsonl"
    qrows, rrows = [], []
    for qid in range(1, 7):
        question = f"Doanh thu kiểm thử {qid}?"
        qrows.append({"id": qid, "question": question})
        rrows.append({
            "id": qid,
            "question": question,
            "route": {
                "tickers": ["ABC"],
                "years": [2024],
                "output_type": "number",
                "unit_name": "triệu đồng",
                "unit_scale": 1_000_000.0,
                "plan": {
                    "op": "lookup",
                    "facts": [{
                        "ticker": "ABC", "year": 2024,
                        "doc_type": "consolidated", "metric": "doanh thu",
                        "role": "value",
                    }],
                    "n_entities": 1,
                    "n_periods": 1,
                },
            },
            "candidates": [],
        })
    write_jsonl(questions, qrows)
    write_jsonl(retrieval, rrows)

    store = root / "store"
    (store / "tables").mkdir(parents=True)
    (store / "cells").mkdir()
    pd.DataFrame([{
        "report_id": REPORT_ID,
        "ticker": "ABC",
        "year": 2024,
        "doc_type": "consolidated",
        "n_tables": 1,
    }]).to_parquet(store / "reports.parquet", index=False)
    pd.DataFrame([{
        "report_id": REPORT_ID,
        "ticker": "ABC",
        "year": 2024,
        "doc_type": "consolidated",
        "table_pos": 7,
        "line_no": 70,
        "grid_json": json.dumps([
            ["", "Mã", "2024"], ["Doanh thu", "01", "2.5"]
        ]),
        "unit_scale": 1.0,
        "unit_source": "header",
    }]).to_parquet(store / "tables" / "ABC.parquet", index=False)
    return questions, retrieval, store


def _complete_gold(template_rows: list[dict], split: str) -> list[dict]:
    rows = []
    for template in template_rows:
        row = copy.deepcopy(template)
        row["split"] = split
        row["label_status"] = "verified"
        row["evidence"] = copy.deepcopy(EVIDENCE)
        row["output"].update({
            "type": "number", "value": 2.5, "unit": "triệu đồng",
            "scale": 1_000_000.0, "round_decimals": 2,
        })
        row["ast"] = copy.deepcopy(AST)
        row["replay"].update({
            "pandas_query": QUERY,
            "used_vars": copy.deepcopy(USED_VARS),
            "expected_answer": 2.5,
            "tolerance": 0.01,
            "status": "verified",
            "evidence_sha256": canonical_sha256(row["evidence"]),
            "ast_sha256": canonical_sha256(row["ast"]),
        })
        rows.append(row)
    return rows


def _codegen_rows(questions_path: Path) -> list[dict]:
    rows = []
    for q in read_jsonl(questions_path):
        rows.append({
            "id": q["id"],
            "question": q["question"],
            "answer": 2.5,
            "pandas_query": QUERY,
            "used_vars": copy.deepcopy(USED_VARS),
            "status": "ok",
            "source": "llm_select",
            "run_signature": "fixture-run-signature",
        })
    return rows


class P24EvaluationTests(unittest.TestCase):
    def _bundle(self, root: Path):
        questions, retrieval, store = _write_fixture(root)
        bundle = root / "bundle"
        build_bundle(
            questions, retrieval, bundle, tune_size=4, locked_size=2,
            expected_source_count=6,
        )
        return questions, store, bundle

    def test_fill_hashes_writes_new_file_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _questions, store, bundle = self._bundle(root)
            draft = read_jsonl(bundle / TUNE_TEMPLATE)
            draft[0]["label_status"] = "draft"
            draft[0]["evidence"] = copy.deepcopy(EVIDENCE)
            draft[0]["ast"] = copy.deepcopy(AST)
            input_path = root / "draft.jsonl"
            output_path = root / "hashed.jsonl"
            write_jsonl(input_path, draft)

            summary = fill_gold_hashes(
                input_path, output_path, bundle, "tune", store_dir=store,
                verify_bundle=False,
            )
            self.assertEqual(summary["hashes_filled"], 1)
            self.assertEqual(summary["blank_preserved"], 3)
            self.assertEqual(read_jsonl(input_path)[0]["replay"]["ast_sha256"], "")
            filled = read_jsonl(output_path)
            self.assertEqual(filled[0]["replay"]["ast_sha256"], canonical_sha256(AST))
            with self.assertRaisesRegex(P24ValidationError, "refusing to overwrite"):
                fill_gold_hashes(
                    input_path, output_path, bundle, "tune", store_dir=store,
                    verify_bundle=False,
                )

    def test_evaluator_reports_answer_execution_coverage_and_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            questions, store, bundle = self._bundle(root)
            tune_gold = root / "tune_gold.jsonl"
            write_jsonl(
                tune_gold, _complete_gold(read_jsonl(bundle / TUNE_TEMPLATE), "tune")
            )
            tune_ids = [row["id"] for row in read_jsonl(bundle / TUNE_QUESTIONS)]
            codegen = _codegen_rows(questions)
            by_id = {row["id"]: row for row in codegen}
            by_id[tune_ids[0]]["answer"] = 3.0  # answer wrong, query correct
            by_id[tune_ids[1]]["pandas_query"] = f"({QUERY}) + 0.5"  # exec wrong
            by_id[tune_ids[2]].update({
                "answer": 0.0,
                "pandas_query": "0.0",
                "used_vars": [],
                "status": "failed",
                "source": "none",
            })
            codegen_path = root / "codegen.jsonl"
            write_jsonl(codegen_path, codegen)
            report_path = root / "report.json"

            report = evaluate_codegen(
                codegen_path, tune_gold, bundle, "tune", store_dir=store,
                output_path=report_path, verify_bundle=False,
            )
            self.assertEqual(report["metrics"]["count"], 4)
            self.assertEqual(report["metrics"]["answer_accuracy"], 0.5)
            self.assertEqual(report["metrics"]["execution_accuracy"], 0.5)
            self.assertEqual(report["metrics"]["query_executable_rate"], 1.0)
            self.assertEqual(report["metrics"]["coverage"], 0.75)
            self.assertEqual(
                report["provenance"]["run_signatures"], ["fixture-run-signature"]
            )
            self.assertEqual(len(report["provenance"]["codegen_sha256"]), 64)
            self.assertTrue(report["population_weighted"]["complete_population_coverage"])
            self.assertIn("llm_select", report["breakdown"]["source"])
            self.assertIn("none", report["breakdown"]["source"])
            self.assertTrue(report_path.exists())
            with self.assertRaisesRegex(P24ValidationError, "refusing to overwrite"):
                evaluate_codegen(
                    codegen_path, tune_gold, bundle, "tune", store_dir=store,
                    output_path=report_path, verify_bundle=False,
                )

    def test_evaluator_requires_complete_single_signature_codegen(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            questions, store, bundle = self._bundle(root)
            gold = root / "gold.jsonl"
            write_jsonl(gold, _complete_gold(read_jsonl(bundle / TUNE_TEMPLATE), "tune"))
            codegen = _codegen_rows(questions)
            codegen.pop()
            incomplete = root / "incomplete.jsonl"
            write_jsonl(incomplete, codegen)
            with self.assertRaisesRegex(P24ValidationError, "not complete"):
                evaluate_codegen(
                    incomplete, gold, bundle, "tune", store_dir=store,
                    verify_bundle=False,
                )

            codegen = _codegen_rows(questions)
            codegen[0]["run_signature"] = "another-signature"
            mixed = root / "mixed.jsonl"
            write_jsonl(mixed, codegen)
            with self.assertRaisesRegex(P24ValidationError, "multiple run signatures"):
                evaluate_codegen(
                    mixed, gold, bundle, "tune", store_dir=store,
                    verify_bundle=False,
                )

            codegen = _codegen_rows(questions)
            codegen[0]["run_signature"] = ""
            missing_signature = root / "missing_signature.jsonl"
            write_jsonl(missing_signature, codegen)
            with self.assertRaisesRegex(P24ValidationError, "non-empty run_signature"):
                evaluate_codegen(
                    missing_signature, gold, bundle, "tune", store_dir=store,
                    verify_bundle=False,
                )

    def test_locked_evaluation_requires_and_records_seal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            questions, store, bundle = self._bundle(root)
            locked_gold = root / "locked_gold.jsonl"
            write_jsonl(
                locked_gold,
                _complete_gold(read_jsonl(bundle / LOCKED_TEMPLATE), "locked"),
            )
            codegen_path = root / "codegen.jsonl"
            write_jsonl(codegen_path, _codegen_rows(questions))
            with self.assertRaisesRegex(P24ValidationError, "requires --seal"):
                evaluate_codegen(
                    codegen_path, locked_gold, bundle, "locked", store_dir=store,
                    verify_bundle=False,
                )
            seal = root / "locked.seal.json"
            seal_locked_gold(
                locked_gold, bundle, seal, store_dir=store, verify_bundle=False
            )
            report = evaluate_codegen(
                codegen_path, locked_gold, bundle, "locked", store_dir=store,
                seal_path=seal, verify_bundle=False,
            )
            self.assertEqual(report["metrics"]["answer_accuracy"], 1.0)
            self.assertEqual(len(report["provenance"]["locked_seal_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
