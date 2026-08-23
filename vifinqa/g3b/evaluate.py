"""G3B oracle-evidence and end-to-end diagnostic evaluator."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ..codegen.executor import run_code
from ..codegen.selection_v2 import compile_program
from .builder import (
    CORPUS_NAME,
    MANIFEST_NAME,
    validate_corpus,
)
from .common import (
    canonical_sha256,
    file_sha256,
    read_jsonl,
    write_json,
)

EVALUATOR_VERSION = "g3b_dual_mode_vector_v1"


def _ratio(total: float, count: int) -> float:
    return round(float(total) / count, 6) if count else 0.0


def _retrieval(predicted: list[str], gold: list[str]) -> dict:
    predicted = [str(value) for value in predicted]
    gold_set = {str(value) for value in gold}
    true_positive = len(set(predicted) & gold_set)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold_set) if gold_set else 0.0
    f2 = (
        5 * precision * recall / (4 * precision + recall)
        if precision + recall
        else 0.0
    )
    reciprocal_rank = 0.0
    for rank, value in enumerate(predicted[:5], 1):
        if value in gold_set:
            reciprocal_rank = 1 / rank
            break
    return {
        "precision": precision,
        "recall": recall,
        "f2": f2,
        "mrr5": reciprocal_rank,
    }


def _load_submission(path: Path) -> tuple[list[dict], Path, Path]:
    results_path = path / "results.json" if path.is_dir() else path
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("submission results must be a JSON list")
    return rows, results_path.parent, results_path


def _replay_submission(row: dict, root: Path) -> dict:
    query = str(row.get("pandas_query") or "").strip()
    if not query:
        return {"ok": False, "value": None, "error": "empty query"}
    frames = {}
    try:
        for evidence in row.get("evidence", []):
            variable = str(evidence["variable"])
            if variable in frames:
                raise ValueError(f"duplicate variable {variable}")
            frames[variable] = pd.read_csv(
                root / str(evidence["csv_path"])
            )
        result = run_code(query, frames, timeout=10)
    except Exception as exc:
        return {
            "ok": False,
            "value": None,
            "error": f"{type(exc).__name__}: {exc}"[:500],
        }
    return {
        "ok": result.get("status") == "ok",
        "value": result.get("value"),
        "error": str(result.get("error") or "")[:500],
    }


def _prediction_map(rows: list[dict], label: str) -> tuple[dict, list[int]]:
    output = {}
    duplicates = []
    for row in rows:
        question_id = int(row["id"])
        if question_id in output:
            duplicates.append(question_id)
        output[question_id] = row
    if duplicates:
        raise ValueError(
            f"{label} duplicate ids: {sorted(set(duplicates))[:10]}"
        )
    return output, duplicates


def _expand_program(program: dict, fact_ids: list[str]) -> dict:
    facts = dict(program.get("facts") or {})
    bindings = dict(program.get("bindings") or {})
    resolving = set()

    def expand_name(name: str) -> object:
        if name in resolving:
            raise ValueError("typed program has a binding cycle")
        resolving.add(name)
        try:
            if name in facts:
                spec = facts[name]
                index = int(spec["ref"]) - 1
                if not 0 <= index < len(fact_ids):
                    raise ValueError("typed fact ref outside candidate_fact_ids")
                return {
                    "fact_id": str(fact_ids[index]),
                    "as": str(spec.get("as", "auto")),
                    "role": str(spec.get("role", "value")),
                }
            if name in bindings:
                return expand_node(bindings[name])
            raise ValueError(f"unknown typed binding {name}")
        finally:
            resolving.remove(name)

    def expand_node(node: dict) -> object:
        if "var" in node:
            return expand_name(str(node["var"]))
        if "year" in node:
            return {"year": int(node["year"])}
        if "literal" in node:
            return {
                "literal": float(node["literal"]),
                "type": str(node.get("type", "number")),
            }
        op = str(node.get("op") or "")
        if op in {"argmax_project", "argmin_project"}:
            return {
                "op": op,
                "items": [
                    {
                        "score": expand_node(item["score"]),
                        "result": expand_node(item["result"]),
                        **(
                            {"when": expand_node(item["when"])}
                            if "when" in item
                            else {}
                        ),
                    }
                    for item in node.get("items", [])
                ],
            }
        return {
            "op": op,
            "args": [
                expand_node(child)
                for child in node.get("args", [])
            ],
            **(
                {"periods": int(node["periods"])}
                if "periods" in node
                else {}
            ),
        }

    return {
        "schema_version": int(program.get("schema_version", 0)),
        "output_type": str(program.get("output_type") or ""),
        "root": expand_node(program["root"]),
    }


def _root_operator(program: dict | None) -> str:
    if not isinstance(program, dict):
        return ""
    return str((program.get("root") or {}).get("op") or "")


def _roles(program: dict, fact_ids: list[str]) -> list[tuple[str, str]]:
    output = []
    for spec in (program.get("facts") or {}).values():
        index = int(spec["ref"]) - 1
        if not 0 <= index < len(fact_ids):
            raise ValueError("typed fact ref outside candidate_fact_ids")
        output.append((
            str(fact_ids[index]),
            str(spec.get("role", "value")),
        ))
    return sorted(output)


def _candidate_objects(evidence_rows: list[dict]) -> list[SimpleNamespace]:
    table_variables = {}
    output = []
    for index, evidence in enumerate(evidence_rows, 1):
        key = (
            str(evidence["report_id"]),
            int(evidence["table_pos"]),
        )
        table_variables.setdefault(key, f"df{len(table_variables) + 1}")
        output.append(SimpleNamespace(
            var=table_variables[key],
            row=int(evidence["row"]),
            col=int(evidence["col"]),
            label=str(evidence["label"]),
            code=str(evidence.get("row_code") or ""),
            col_name=str(evidence["col_name"]),
            value=float(evidence["value"]),
            unit_scale=float(evidence["unit_scale"]),
            score=100.0,
            rescue=False,
            fact_year=int(evidence["period_year"]),
            report_year=int(evidence["report_year"]),
            fact_slot=f"F{index}",
            fact_role="value",
            fact_metric=str(evidence["metric_key"]),
            ticker=str(evidence["ticker"]),
            report_id=str(evidence["report_id"]),
            table_pos=int(evidence["table_pos"]),
            metric_grounded=True,
        ))
    return output


def _execute_typed(
    program: dict,
    evidence_rows: list[dict],
    gold: dict,
) -> dict:
    candidates = _candidate_objects(evidence_rows)
    route = {
        "output_type": gold["output_type"],
        "unit_scale": gold["unit_scale"],
        "years": gold["years"],
        "plan": {
            "op": (
                "ranking"
                if gold["family"].startswith("ranking_")
                else gold["family"]
            )
        },
    }
    try:
        compiled = compile_program(
            program,
            candidates,
            route,
            gold["question"],
            atomic_facts=gold["atomic_facts"],
        )
        frames: dict[str, list[dict]] = defaultdict(list)
        for evidence, candidate in zip(evidence_rows, candidates):
            frames[candidate.var].append({
                "row": int(evidence["row"]),
                "col": int(evidence["col"]),
                "value": float(evidence["value"]),
                "unit_scale": float(evidence["unit_scale"]),
                "label": str(evidence["label"]),
                "col_name": str(evidence["col_name"]),
            })
        result = run_code(
            compiled.query,
            {
                variable: pd.DataFrame(rows)
                for variable, rows in frames.items()
            },
        )
    except Exception as exc:
        return {
            "ok": False,
            "value": None,
            "error": f"{type(exc).__name__}: {exc}"[:500],
        }
    return {
        "ok": result.get("status") == "ok",
        "value": result.get("value"),
        "error": str(result.get("error") or "")[:500],
    }


def _verify_freeze(
    freeze_path: Path,
    corpus_dir: Path,
    config_path: Path,
    submission_results: Path | None,
    typed_path: Path | None,
) -> dict:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != "g3b_candidate_freeze_v1":
        raise ValueError("invalid candidate freeze schema")
    expected = {
        "corpus_manifest_sha256": file_sha256(
            corpus_dir / MANIFEST_NAME
        ),
        "config_sha256": file_sha256(config_path),
        "submission_results_sha256": (
            file_sha256(submission_results)
            if submission_results
            else None
        ),
        "typed_predictions_sha256": (
            file_sha256(typed_path) if typed_path else None
        ),
    }
    if any(freeze.get(key) != value for key, value in expected.items()):
        raise ValueError("candidate freeze hash mismatch")
    fingerprint = freeze.get("fingerprint_sha256")
    payload = dict(freeze)
    payload.pop("fingerprint_sha256", None)
    if canonical_sha256(payload) != fingerprint:
        raise ValueError("candidate freeze fingerprint mismatch")
    return freeze


def _summarize(rows: list[dict]) -> dict:
    count = len(rows)
    if not count:
        return {"count": 0}
    output: dict[str, float | int | None] = {"count": count}
    for prefix in ("docs", "tables"):
        for metric in ("precision", "recall", "f2", "mrr5"):
            key = (
                f"{prefix}_{metric}_macro"
                if metric != "mrr5"
                else f"{prefix}_mrr5"
            )
            output[key] = _ratio(
                sum(float(row[f"{prefix}_{metric}"]) for row in rows),
                count,
            )
    rate_names = {
        "answer_correct": "answer_accuracy",
        "execution_correct": "execution_accuracy",
        "execution_ran": "execution_run_rate",
        "full_plan_coverage": "full_plan_coverage_rate",
        "typed_present": "typed_output_coverage",
        "operator_correct": "operator_accuracy",
        "operand_role_correct": "operand_role_accuracy",
        "output_type_correct": "output_type_accuracy",
        "ast_match": "canonical_ast_match_accuracy",
        "typed_execution_correct": "typed_execution_accuracy",
        "typed_execution_ran": "typed_execution_run_rate",
    }
    for name, rate_name in rate_names.items():
        total = sum(bool(row[name]) for row in rows)
        output[name] = total
        output[rate_name] = _ratio(total, count)
    output["leaf_recall_at_k"] = _ratio(
        sum(float(row["leaf_recall_at_k"]) for row in rows),
        count,
    )
    typed_count = int(output["typed_present"])
    output["operator_accuracy_given_typed"] = (
        _ratio(sum(bool(row["operator_correct"]) for row in rows), typed_count)
        if typed_count
        else None
    )
    output["ast_match_given_typed"] = (
        _ratio(sum(bool(row["ast_match"]) for row in rows), typed_count)
        if typed_count
        else None
    )
    return output


def _breakdown(rows: list[dict], key: str) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {
        name: _summarize(grouped[name])
        for name in sorted(grouped)
    }



def _ood_view_breakdown(
    records: list[dict],
    views_dir: Path,
) -> dict:
    by_id = {int(row["id"]): row for row in records}
    views = {}
    for path in sorted(views_dir.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        folds = document.get("folds")
        if isinstance(folds, dict):
            fold_reports = {}
            unique_eval_ids = set()
            for fold_name, fold in sorted(folds.items()):
                eval_ids = {
                    int(value) for value in fold.get("eval_ids", [])
                }
                unique_eval_ids.update(eval_ids)
                fold_reports[str(fold_name)] = _summarize([
                    by_id[question_id]
                    for question_id in sorted(eval_ids)
                    if question_id in by_id
                ])
            aggregate = _summarize([
                by_id[question_id]
                for question_id in sorted(unique_eval_ids)
                if question_id in by_id
            ])
        else:
            eval_ids = {
                int(value) for value in document.get("eval_ids", [])
            }
            fold_reports = None
            aggregate = _summarize([
                by_id[question_id]
                for question_id in sorted(eval_ids)
                if question_id in by_id
            ])
        views[path.stem] = {
            "kind": document["kind"],
            "overlap_policy": document.get("overlap_policy"),
            "aggregate_unique_eval": aggregate,
            "folds": fold_reports,
        }
    return {
        "interpretation": (
            "overlapping diagnostic views; not independent replications"
        ),
        "views": views,
    }


def evaluate_g3b(
    corpus_dir: Path | str,
    extension_dir: Path | str,
    config_path: Path | str,
    *,
    policy_mode: str,
    evidence_mode: str,
    submission: Path | str | None = None,
    typed_predictions: Path | str | None = None,
    candidate_freeze: Path | str | None = None,
    output_path: Path | str | None = None,
) -> dict:
    if policy_mode not in {"dev", "promotion"}:
        raise ValueError("policy_mode must be dev or promotion")
    if evidence_mode not in {"oracle_evidence", "end_to_end"}:
        raise ValueError(
            "evidence_mode must be oracle_evidence or end_to_end"
        )
    corpus_dir = Path(corpus_dir)
    extension_dir = Path(extension_dir)
    config_path = Path(config_path)
    validate_corpus(extension_dir, corpus_dir, config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    all_gold = read_jsonl(corpus_dir / CORPUS_NAME)
    wanted_splits = (
        {"primary_tune"}
        if policy_mode == "dev"
        else {"primary_locked", "hard"}
    )
    gold_rows = [
        row for row in all_gold if row["split"] in wanted_splits
    ]
    gold_ids = {int(row["id"]) for row in gold_rows}
    corpus_ids = {int(row["id"]) for row in all_gold}

    submission_rows = []
    submission_root = Path(".")
    submission_results_path = None
    if submission is not None:
        (
            submission_rows,
            submission_root,
            submission_results_path,
        ) = _load_submission(Path(submission))
    if evidence_mode == "end_to_end" and submission is None:
        raise ValueError("end_to_end evaluation requires --submission")

    typed_path = (
        Path(typed_predictions)
        if typed_predictions is not None
        else None
    )
    typed_rows = read_jsonl(typed_path) if typed_path else []
    submission_map, _ = _prediction_map(
        submission_rows, "submission"
    )
    typed_map, _ = _prediction_map(typed_rows, "typed predictions")

    freeze = None
    if policy_mode == "promotion":
        if candidate_freeze is None:
            raise ValueError(
                "promotion mode requires --candidate-freeze"
            )
        freeze = _verify_freeze(
            Path(candidate_freeze),
            corpus_dir,
            config_path,
            submission_results_path,
            typed_path,
        )

    prediction_ids = (
        set(submission_map)
        if evidence_mode == "end_to_end"
        else set(typed_map)
    )
    # A frozen candidate may contain predictions for the full corpus while a
    # policy evaluates only its allowed split. In-corpus rows outside that
    # policy are neither leakage nor extra predictions.
    missing_ids = sorted(gold_ids - prediction_ids)
    extra_ids = sorted(prediction_ids - corpus_ids)

    question_mismatch_ids = []
    nonfinite_answer_ids = []
    duplicate_docs_ids = []
    duplicate_tables_ids = []
    records = []
    for gold in gold_rows:
        question_id = int(gold["id"])
        submission_row = submission_map.get(question_id, {})
        typed_row = typed_map.get(question_id, {})
        if evidence_mode == "oracle_evidence":
            predicted_docs = list(gold["relevant_docs"])
            predicted_tables = list(gold["relevant_tables"])
        else:
            predicted_docs = list(
                submission_row.get("relevant_docs") or []
            )
            predicted_tables = list(
                submission_row.get("relevant_tables") or []
            )
        prediction_row = (
            typed_row
            if evidence_mode == "oracle_evidence"
            else submission_row
        )
        if (
            prediction_row
            and str(prediction_row.get("question") or "")
            != str(gold["question"])
        ):
            question_mismatch_ids.append(question_id)
        if len(predicted_docs) != len(set(predicted_docs)):
            duplicate_docs_ids.append(question_id)
        if len(predicted_tables) != len(set(predicted_tables)):
            duplicate_tables_ids.append(question_id)
        docs = _retrieval(predicted_docs, gold["relevant_docs"])
        tables = _retrieval(
            predicted_tables, gold["relevant_tables"]
        )

        leaf_tables = [
            str(leaf["table_ref"]) for leaf in gold["leaf_specs"]
        ]
        top_tables = set(
            map(str, predicted_tables[:int(config["leaf_k"])])
        )
        leaf_hits = sum(
            table_ref in top_tables for table_ref in leaf_tables
        )
        leaf_recall = (
            leaf_hits / len(leaf_tables) if leaf_tables else 0.0
        )
        full_plan = bool(
            leaf_tables
            and all(table_ref in top_tables for table_ref in leaf_tables)
        )

        typed_program = typed_row.get("program")
        fact_ids = list(typed_row.get("candidate_fact_ids") or [])
        typed_present = isinstance(typed_program, dict) and bool(fact_ids)
        operator_correct = False
        roles_correct = False
        output_type_correct = False
        ast_match = False
        typed_execution = {
            "ok": False,
            "value": None,
            "error": "typed output missing",
        }
        if typed_present:
            try:
                gold_expanded = _expand_program(
                    gold["typed_program"], gold["fact_ids"]
                )
                predicted_expanded = _expand_program(
                    typed_program, fact_ids
                )
                operator_correct = (
                    _root_operator(typed_program)
                    == _root_operator(gold["typed_program"])
                )
                roles_correct = (
                    _roles(typed_program, fact_ids)
                    == _roles(
                        gold["typed_program"], gold["fact_ids"]
                    )
                )
                output_type_correct = (
                    typed_program.get("output_type")
                    == gold["output_type"]
                )
                ast_match = (
                    canonical_sha256(predicted_expanded)
                    == canonical_sha256(gold_expanded)
                )
                if evidence_mode == "oracle_evidence":
                    evidence_rows = gold["evidence"]
                else:
                    evidence_rows = list(
                        typed_row.get("candidate_evidence") or []
                    )
                if evidence_rows:
                    typed_execution = _execute_typed(
                        typed_program, evidence_rows, gold
                    )
            except Exception as exc:
                typed_execution = {
                    "ok": False,
                    "value": None,
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                }

        if evidence_mode == "oracle_evidence":
            answer_value = typed_execution["value"]
            finite = (
                answer_value is not None
                and math.isfinite(float(answer_value))
            )
            replay = typed_execution
        else:
            try:
                answer_value = float(submission_row.get("answer"))
                finite = math.isfinite(answer_value)
            except (TypeError, ValueError):
                answer_value = None
                finite = False
            replay = (
                _replay_submission(submission_row, submission_root)
                if submission_row
                else {
                    "ok": False,
                    "value": None,
                    "error": "missing submission row",
                }
            )
        if not finite:
            nonfinite_answer_ids.append(question_id)

        tolerance = float(gold["tolerance"])
        answer_correct = bool(
            finite
            and math.isclose(
                float(answer_value),
                float(gold["answer"]),
                rel_tol=0.0,
                abs_tol=tolerance,
            )
        )
        execution_correct = bool(
            replay["ok"]
            and replay["value"] is not None
            and math.isclose(
                float(replay["value"]),
                float(gold["answer"]),
                rel_tol=0.0,
                abs_tol=tolerance,
            )
        )
        typed_execution_correct = bool(
            typed_execution["ok"]
            and typed_execution["value"] is not None
            and math.isclose(
                float(typed_execution["value"]),
                float(gold["answer"]),
                rel_tol=0.0,
                abs_tol=tolerance,
            )
        )
        records.append({
            "id": question_id,
            "split": gold["split"],
            "set": gold["set"],
            "family": gold["family"],
            "operator": gold["operator"],
            "metric_family": gold["metric_family"],
            "output_type": gold["output_type"],
            **{f"docs_{key}": value for key, value in docs.items()},
            **{f"tables_{key}": value for key, value in tables.items()},
            "leaf_recall_at_k": leaf_recall,
            "full_plan_coverage": full_plan,
            "answer_correct": answer_correct,
            "execution_correct": execution_correct,
            "execution_ran": bool(replay["ok"]),
            "typed_present": typed_present,
            "operator_correct": operator_correct,
            "operand_role_correct": roles_correct,
            "output_type_correct": output_type_correct,
            "ast_match": ast_match,
            "typed_execution_ran": bool(typed_execution["ok"]),
            "typed_execution_correct": typed_execution_correct,
            "typed_error": typed_execution["error"],
        })

    integrity_inputs = (
        missing_ids,
        extra_ids,
        question_mismatch_ids,
        nonfinite_answer_ids,
        duplicate_docs_ids,
        duplicate_tables_ids,
    )
    integrity = {
        "passed": not any(integrity_inputs),
        "gold_count": len(gold_rows),
        "prediction_count": len(gold_ids & prediction_ids),
        "provided_prediction_count": len(prediction_ids),
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        "question_mismatch_ids": question_mismatch_ids,
        "nonfinite_answer_ids": nonfinite_answer_ids,
        "duplicate_docs_ids": duplicate_docs_ids,
        "duplicate_tables_ids": duplicate_tables_ids,
    }
    report = {
        "schema_version": "g3b_evaluation_report_v1",
        "evaluator_version": EVALUATOR_VERSION,
        "policy_mode": policy_mode,
        "evidence_mode": evidence_mode,
        "retrieval_interpretation": (
            "bypassed with gold evidence; do not interpret retrieval metrics"
            if evidence_mode == "oracle_evidence"
            else "measured from submitted relevant_docs/relevant_tables"
        ),
        "provenance": {
            "corpus_manifest_sha256": file_sha256(
                corpus_dir / MANIFEST_NAME
            ),
            "config_sha256": file_sha256(config_path),
            "submission_results_sha256": (
                file_sha256(submission_results_path)
                if submission_results_path
                else None
            ),
            "typed_predictions_sha256": (
                file_sha256(typed_path) if typed_path else None
            ),
            "candidate_freeze_sha256": (
                file_sha256(candidate_freeze)
                if candidate_freeze
                else None
            ),
            "candidate_name": (
                freeze.get("candidate_name") if freeze else None
            ),
        },
        "metric_definitions": {
            "leaf_recall_at_k": (
                "mean fraction of gold leaf table references present in "
                f"the top {int(config['leaf_k'])} submitted tables"
            ),
            "full_plan_coverage": (
                "all gold leaf table references occur in the top-K tables"
            ),
            "typed_denominator": (
                "all evaluated records; missing typed output scores zero; "
                "given_typed metrics are secondary diagnostics"
            ),
            "ast_match": (
                "Selection-v2 facts/bindings/root expanded to stable fact IDs"
            ),
        },
        "integrity": integrity,
        "metrics": _summarize(records),
        "breakdown": {
            "split": _breakdown(records, "split"),
            "family": _breakdown(records, "family"),
            "output_type": _breakdown(records, "output_type"),
            "metric_family": _breakdown(records, "metric_family"),
            "ood_views": _ood_view_breakdown(records, corpus_dir / "views"),
        },
        "records": records,
    }
    if output_path is not None:
        write_json(output_path, report)
    return report
