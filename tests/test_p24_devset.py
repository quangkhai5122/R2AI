import copy
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from vifinqa.devset.p24 import (
    LOCKED_QUESTIONS,
    LOCKED_TEMPLATE,
    MANIFEST_NAME,
    TUNE_QUESTIONS,
    TUNE_TEMPLATE,
    P24ValidationError,
    build_bundle,
    canonical_sha256,
    check_tune_input,
    seal_locked_gold,
    validate_bundle,
    validate_gold_records,
    verify_locked_seal,
)
from vifinqa.utils.io import read_jsonl, write_jsonl


def _write_source(root: Path, count: int = 60) -> tuple[Path, Path]:
    questions_path = root / "questions.jsonl"
    retrieval_path = root / "retrieval.jsonl"
    ops = ["lookup", "difference", "ranking", "average", "growth_pct"]
    output_types = {
        "lookup": "number",
        "difference": "number",
        "ranking": "year",
        "average": "number",
        "growth_pct": "percent",
    }
    questions, retrieval = [], []
    for qid in range(1, count + 1):
        op = ops[(qid - 1) % len(ops)]
        n_facts = 1 if qid % 3 == 0 else (3 if qid % 3 == 1 else 6)
        question = f"Câu hỏi kiểm thử {qid} cho {op}?"
        questions.append({"id": qid, "question": question})
        facts = [
            {
                "ticker": f"T{i}", "year": 2024 - (i % 2),
                "doc_type": "consolidated", "metric": "metric", "role": "value",
            }
            for i in range(n_facts)
        ]
        retrieval.append({
            "id": qid,
            "question": question,
            "route": {
                "tickers": sorted({f["ticker"] for f in facts}),
                "years": sorted({f["year"] for f in facts}),
                "output_type": output_types[op],
                "unit_name": "%" if op == "growth_pct" else "triệu đồng",
                "unit_scale": 1.0 if op == "growth_pct" else 1_000_000.0,
                "plan": {
                    "op": op,
                    "facts": facts,
                    "n_entities": len({f["ticker"] for f in facts}),
                    "n_periods": len({f["year"] for f in facts}),
                },
            },
            "candidates": [],
        })
    write_jsonl(questions_path, questions)
    write_jsonl(retrieval_path, retrieval)
    return questions_path, retrieval_path


def _question_row(qid: int = 1, split: str = "tune") -> dict:
    question = f"Câu hỏi gold {qid}?"
    import hashlib

    return {
        "schema_version": "p24_question_v1",
        "split": split,
        "id": qid,
        "question": question,
        "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        "stratum": "lookup|single",
        "route_summary": {
            "op": "lookup",
            "n_facts": 1,
            "n_entities": 1,
            "n_periods": 1,
            "output_type": "number",
        },
    }


def _verified_record(question_row: dict) -> dict:
    evidence = [{
        "evidence_id": "E1",
        "variable": "df1",
        "report_id": "ABC_financial_statements_2024_consolidated",
        "table_pos": 7,
        "row": 0,
        "col": 1,
        "label": "Doanh thu",
        "code": "01",
        "col_name": "2024",
        "value": 2.5,
        "unit_scale": 1.0,
    }]
    ast_node = {
        "kind": "op",
        "op": "lookup",
        "args": [{"kind": "evidence", "evidence_id": "E1"}],
    }
    return {
        "schema_version": "p24_gold_v1",
        "split": question_row["split"],
        "id": question_row["id"],
        "question": question_row["question"],
        "question_sha256": question_row["question_sha256"],
        "stratum": question_row["stratum"],
        "label_status": "verified",
        "evidence": evidence,
        "output": {
            "type": "number",
            "value": 2.5,
            "unit": "triệu đồng",
            "scale": 1_000_000.0,
            "round_decimals": 2,
        },
        "ast": ast_node,
        "replay": {
            "pandas_query": (
                "round(float(df1.loc[(df1['row'] == 0) & "
                "(df1['col'] == 1), 'value'].iloc[0]), 2)"
            ),
            "used_vars": [{
                "var": "df1",
                "report_id": "ABC_financial_statements_2024_consolidated",
                "table_pos": 7,
            }],
            "expected_answer": 2.5,
            "tolerance": 0.01,
            "status": "verified",
            "evidence_sha256": canonical_sha256(evidence),
            "ast_sha256": canonical_sha256(ast_node),
        },
        "annotator_notes": "Double-checked against source table.",
    }


def _table_loader(_report_id: str, _table_pos: int) -> pd.DataFrame:
    return pd.DataFrame([{
        "row": 0,
        "label": "Doanh thu",
        "code": "01",
        "col": 1,
        "col_name": "2024",
        "value": 2.5,
        "unit_scale": 1.0,
    }])


class P24DevsetTests(unittest.TestCase):
    def test_build_is_deterministic_stratified_and_disjoint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            questions, retrieval = _write_source(root)
            first, second = root / "first", root / "second"
            one = build_bundle(
                questions, retrieval, first, seed=2404, tune_size=10,
                locked_size=5, expected_source_count=60,
            )
            two = build_bundle(
                questions, retrieval, second, seed=2404, tune_size=10,
                locked_size=5, expected_source_count=60,
            )
            self.assertEqual(one, two)
            for name in (TUNE_QUESTIONS, LOCKED_QUESTIONS, TUNE_TEMPLATE, LOCKED_TEMPLATE):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            tune = read_jsonl(first / TUNE_QUESTIONS)
            locked = read_jsonl(first / LOCKED_QUESTIONS)
            self.assertEqual(len(tune), 10)
            self.assertEqual(len(locked), 5)
            self.assertFalse({r["id"] for r in tune} & {r["id"] for r in locked})
            self.assertGreater(len({r["stratum"] for r in tune + locked}), 5)
            validate_bundle(
                first, questions_path=questions, retrieval_path=retrieval,
                expected_source_count=60, expected_tune_size=10,
                expected_locked_size=5,
            )

    def test_bundle_hash_guard_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            questions, retrieval = _write_source(root, 30)
            bundle = root / "bundle"
            build_bundle(
                questions, retrieval, bundle, tune_size=6, locked_size=4,
                expected_source_count=30,
            )
            with open(bundle / TUNE_QUESTIONS, "a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(P24ValidationError, "hash mismatch"):
                validate_bundle(
                    bundle, questions_path=questions, retrieval_path=retrieval,
                    expected_source_count=30, expected_tune_size=6,
                    expected_locked_size=4,
                )

    def test_build_refuses_to_overwrite_frozen_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            questions, retrieval = _write_source(root, 30)
            bundle = root / "bundle"
            build_bundle(
                questions, retrieval, bundle, tune_size=6, locked_size=4,
                expected_source_count=30,
            )
            with self.assertRaisesRegex(
                P24ValidationError, "refusing to overwrite existing P2.4 bundle"
            ):
                build_bundle(
                    questions, retrieval, bundle, tune_size=6, locked_size=4,
                    expected_source_count=30,
                )

    def test_source_hash_guard_rejects_drift(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            questions, retrieval = _write_source(root, 30)
            bundle = root / "bundle"
            build_bundle(
                questions, retrieval, bundle, tune_size=6, locked_size=4,
                expected_source_count=30,
            )
            with open(questions, "a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(P24ValidationError, "source questions hash"):
                validate_bundle(
                    bundle, questions_path=questions, retrieval_path=retrieval,
                    expected_source_count=30, expected_tune_size=6,
                    expected_locked_size=4,
                )

    def test_verified_gold_checks_exact_evidence_ast_and_replay(self):
        question = _question_row()
        record = _verified_record(question)
        summary = validate_gold_records(
            [record], [question], "tune", table_loader=_table_loader,
            require_complete=True,
        )
        self.assertEqual(summary["count"], 1)

        bad_evidence = copy.deepcopy(record)
        bad_evidence["evidence"][0]["col_name"] = "2023"
        bad_evidence["replay"]["evidence_sha256"] = canonical_sha256(
            bad_evidence["evidence"]
        )
        with self.assertRaisesRegex(P24ValidationError, "exact evidence col_name"):
            validate_gold_records(
                [bad_evidence], [question], "tune", table_loader=_table_loader
            )

        bad_ast = copy.deepcopy(record)
        bad_ast["ast"]["args"] = []
        bad_ast["replay"]["ast_sha256"] = canonical_sha256(bad_ast["ast"])
        with self.assertRaisesRegex(P24ValidationError, "exactly 1 args"):
            validate_gold_records(
                [bad_ast], [question], "tune", table_loader=_table_loader
            )

        bad_replay = copy.deepcopy(record)
        bad_replay["replay"]["expected_answer"] = 3.0
        with self.assertRaisesRegex(P24ValidationError, "match output.value"):
            validate_gold_records(
                [bad_replay], [question], "tune", table_loader=_table_loader
            )

    def test_template_mode_allows_blanks_but_strict_mode_does_not(self):
        question = _question_row()
        record = _verified_record(question)
        record["label_status"] = "unlabeled"
        record["evidence"] = []
        record["output"]["value"] = None
        record["ast"] = None
        record["replay"].update({
            "pandas_query": "",
            "used_vars": [],
            "expected_answer": None,
            "status": "unverified",
            "evidence_sha256": "",
            "ast_sha256": "",
        })
        validate_gold_records(
            [record], [question], "tune", require_complete=False
        )
        with self.assertRaisesRegex(P24ValidationError, "label_status='verified'"):
            validate_gold_records(
                [record], [question], "tune", table_loader=_table_loader,
                require_complete=True,
            )

    def test_tune_input_rejects_locked_ids(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            questions, retrieval = _write_source(root, 30)
            bundle = root / "bundle"
            build_bundle(
                questions, retrieval, bundle, tune_size=6, locked_size=4,
                expected_source_count=30,
            )
            tune_input = root / "tune.jsonl"
            write_jsonl(tune_input, [read_jsonl(bundle / TUNE_QUESTIONS)[0]])
            self.assertEqual(
                check_tune_input(tune_input, bundle, verify_bundle=False)["locked_overlap"],
                0,
            )
            write_jsonl(tune_input, [read_jsonl(bundle / LOCKED_QUESTIONS)[0]])
            with self.assertRaisesRegex(P24ValidationError, "locked-set leakage"):
                check_tune_input(tune_input, bundle, verify_bundle=False)

    def test_locked_seal_detects_raw_file_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundle = root / "bundle"
            bundle.mkdir()
            question = _question_row(split="locked")
            record = _verified_record(question)
            write_jsonl(bundle / LOCKED_QUESTIONS, [question])
            # The seal also binds these files; minimal valid placeholders suffice here.
            (bundle / MANIFEST_NAME).write_text("{}\n", encoding="utf-8")
            gold = bundle / "p24_locked_gold.jsonl"
            write_jsonl(gold, [record])
            seal = bundle / "seal.json"
            seal_locked_gold(
                gold, bundle, seal, table_loader=_table_loader,
                verify_bundle=False,
            )
            with self.assertRaisesRegex(
                P24ValidationError, "refusing to overwrite existing P2.4 artifact"
            ):
                seal_locked_gold(
                    gold, bundle, seal, table_loader=_table_loader,
                    verify_bundle=False,
                )
            verify_locked_seal(
                gold, bundle, seal, table_loader=_table_loader,
                verify_bundle=False,
            )
            with open(gold, "a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(P24ValidationError, "locked seal mismatch"):
                verify_locked_seal(
                    gold, bundle, seal, table_loader=_table_loader,
                    verify_bundle=False,
                )

    def test_json_schema_is_valid_json(self):
        schema = Path(__file__).resolve().parents[1] / "schemas" / "p24_gold.schema.json"
        obj = json.loads(schema.read_text(encoding="utf-8"))
        self.assertEqual(obj["properties"]["schema_version"]["const"], "p24_gold_v1")


if __name__ == "__main__":
    unittest.main()
