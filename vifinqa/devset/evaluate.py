"""Practical P2.4 gold hashing and codegen evaluation helpers."""
from __future__ import annotations

import copy
import json
import math
from collections import defaultdict
from pathlib import Path

from ..codegen.executor import check_safe, run_code
from ..codegen.semantic import all_dataframe_refs
from ..utils.io import read_json, read_jsonl
from .p24 import (
    LOCKED_QUESTIONS,
    MANIFEST_NAME,
    TUNE_QUESTIONS,
    USED_VAR_KEYS,
    P24ValidationError,
    StoreTableLoader,
    _exact_keys,
    _read_unique_jsonl,
    _validate_ast_node,
    _validate_evidence,
    canonical_sha256,
    file_sha256,
    question_sha256,
    validate_bundle,
    validate_gold_file,
    verify_locked_seal,
)


EVALUATOR_VERSION = "p24_codegen_eval_v1"
REPORT_SCHEMA_VERSION = "p24_eval_report_v1"


def _fail(message: str) -> None:
    raise P24ValidationError(message)


def _write_text_new(path: Path | str, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except FileExistsError as exc:
        raise P24ValidationError(f"refusing to overwrite existing output: {path}") from exc


def _write_jsonl_new(path: Path | str, rows: list[dict]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
    )
    _write_text_new(path, text)


def _write_json_new(path: Path | str, value: dict) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _write_text_new(path, text)


def fill_gold_hashes(
    input_path: Path | str,
    output_path: Path | str,
    bundle_dir: Path | str,
    split: str,
    *,
    store_dir: Path | str,
    verify_bundle: bool = True,
) -> dict:
    """Fill evidence/AST hashes into a new file without changing the draft.

    Fully blank records are preserved.  Any record with evidence or AST must
    have both and must resolve to exact cells before hashes are written.
    """
    if Path(output_path).exists():
        _fail(f"refusing to overwrite existing output: {output_path}")
    validate_gold_file(
        input_path, bundle_dir, split, require_complete=False,
        verify_bundle=verify_bundle,
    )
    rows = _read_unique_jsonl(input_path, f"{split} gold draft")
    loader = StoreTableLoader(store_dir)
    output_rows: list[dict] = []
    filled = 0
    blank = 0
    for index, original in enumerate(rows, 1):
        row = copy.deepcopy(original)
        evidence = row.get("evidence")
        ast_node = row.get("ast")
        has_evidence = isinstance(evidence, list) and bool(evidence)
        has_ast = isinstance(ast_node, dict)
        if not has_evidence and ast_node is None:
            row["replay"]["evidence_sha256"] = ""
            row["replay"]["ast_sha256"] = ""
            blank += 1
            output_rows.append(row)
            continue
        if not has_evidence or not has_ast:
            _fail(
                f"{split} gold line {index}: hash fill needs both non-empty "
                "evidence and an AST"
            )
        evidence_ids, _var_tables, _dfs = _validate_evidence(
            evidence, f"{split} gold line {index}.evidence", loader, True
        )
        if ast_node.get("kind") != "op":
            _fail(f"{split} gold line {index}.ast: root must be an operation node")
        ast_refs = _validate_ast_node(
            ast_node, evidence_ids, f"{split} gold line {index}.ast"
        )
        if ast_refs != evidence_ids:
            _fail(
                f"{split} gold line {index}.ast: evidence refs must exactly "
                "equal labeled evidence"
            )
        row["replay"]["evidence_sha256"] = canonical_sha256(evidence)
        row["replay"]["ast_sha256"] = canonical_sha256(ast_node)
        filled += 1
        output_rows.append(row)
    if not filled:
        _fail("no labeled records found; no hash-filled output was created")
    _write_jsonl_new(output_path, output_rows)
    return {
        "split": split,
        "records": len(output_rows),
        "hashes_filled": filled,
        "blank_preserved": blank,
        "input_sha256": file_sha256(input_path),
        "output_sha256": file_sha256(output_path),
    }


def _resolve_source_questions(bundle_dir: Path, manifest: dict) -> Path:
    raw = manifest.get("source", {}).get("questions_path")
    if not raw:
        _fail("manifest has no source questions path")
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            expected = manifest["source"].get("questions_sha256")
            if file_sha256(candidate) != expected:
                _fail("source questions hash differs from manifest")
            return candidate
    _fail(f"source questions file is unavailable: {raw}")
    raise AssertionError("unreachable")


def _validate_complete_codegen(
    codegen_path: Path | str, source_questions_path: Path, source_count: int
) -> tuple[list[dict], list[str]]:
    rows = _read_unique_jsonl(codegen_path, "codegen")
    questions = _read_unique_jsonl(source_questions_path, "source questions")
    if len(questions) != source_count:
        _fail(
            f"source questions count {len(questions)} differs from manifest {source_count}"
        )
    source_map = {int(row["id"]): row["question"] for row in questions}
    codegen_map = {int(row["id"]): row for row in rows}
    if set(codegen_map) != set(source_map):
        missing = sorted(set(source_map) - set(codegen_map))[:10]
        extra = sorted(set(codegen_map) - set(source_map))[:10]
        _fail(f"codegen is not complete; missing={missing}, extra={extra}")
    required = {"id", "answer", "pandas_query", "used_vars", "status", "source"}
    signatures: set[str] = set()
    for qid in sorted(codegen_map):
        row = codegen_map[qid]
        missing = sorted(required - set(row))
        if missing:
            _fail(f"codegen id {qid}: missing fields {missing}")
        if "question" in row and row["question"] != source_map[qid]:
            _fail(f"codegen id {qid}: question differs from source")
        try:
            answer = float(row["answer"])
        except (TypeError, ValueError) as exc:
            raise P24ValidationError(f"codegen id {qid}: answer is not numeric") from exc
        if not math.isfinite(answer):
            _fail(f"codegen id {qid}: answer is not finite")
        if not isinstance(row["pandas_query"], str):
            _fail(f"codegen id {qid}: pandas_query must be a string")
        if not isinstance(row["used_vars"], list):
            _fail(f"codegen id {qid}: used_vars must be a list")
        signature = str(row.get("run_signature") or "").strip()
        if not signature:
            _fail(f"codegen id {qid}: missing non-empty run_signature")
        signatures.add(signature)
    if len(signatures) > 1:
        _fail(f"codegen mixes multiple run signatures: {sorted(signatures)}")
    return [codegen_map[qid] for qid in sorted(codegen_map)], sorted(signatures)


def _prediction_replay(row: dict, loader: StoreTableLoader) -> dict:
    query = row["pandas_query"].strip()
    if not query:
        return {"ok": False, "value": None, "error": "empty pandas_query"}
    try:
        compile(query, "<p24-codegen>", "eval")
    except SyntaxError as exc:
        return {"ok": False, "value": None, "error": f"not eval expression: {exc.msg}"}
    banned = check_safe(query)
    if banned:
        return {"ok": False, "value": None, "error": f"unsafe token: {banned}"}

    bindings: dict[str, tuple[str, int]] = {}
    try:
        for index, item in enumerate(row["used_vars"], 1):
            _exact_keys(item, USED_VAR_KEYS, f"codegen id {row['id']} used_vars[{index}]")
            var = str(item["var"])
            if var in bindings:
                return {"ok": False, "value": None, "error": f"duplicate variable {var}"}
            bindings[var] = (str(item["report_id"]), int(item["table_pos"]))
        refs = all_dataframe_refs(query)
    except (P24ValidationError, SyntaxError, TypeError, ValueError) as exc:
        return {"ok": False, "value": None, "error": str(exc)[:500]}
    if refs != set(bindings):
        return {
            "ok": False,
            "value": None,
            "error": (
                f"dataframe refs {sorted(refs)} != used_vars {sorted(bindings)}"
            ),
        }
    try:
        dfs = {
            var: loader(report_id, table_pos)
            for var, (report_id, table_pos) in bindings.items()
        }
        result = run_code(query, dfs, timeout=10)
    except Exception as exc:  # evidence/load failures are per-query evaluation failures
        return {
            "ok": False,
            "value": None,
            "error": f"{type(exc).__name__}: {exc}"[:500],
        }
    return {
        "ok": result.get("status") == "ok",
        "value": result.get("value"),
        "error": result.get("error") or "",
    }


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _metric_summary(items: list[dict]) -> dict:
    count = len(items)
    answer_correct = sum(bool(item["answer_correct"]) for item in items)
    execution_correct = sum(bool(item["execution_correct"]) for item in items)
    executable = sum(bool(item["query_executable"]) for item in items)
    covered = sum(bool(item["covered"]) for item in items)
    return {
        "count": count,
        "answer_correct": answer_correct,
        "answer_accuracy": _ratio(answer_correct, count),
        "execution_correct": execution_correct,
        "execution_accuracy": _ratio(execution_correct, count),
        "query_executable": executable,
        "query_executable_rate": _ratio(executable, count),
        "covered": covered,
        "coverage": _ratio(covered, count),
    }


def _group_metrics(items: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        groups[str(item[key])].append(item)
    return {name: _metric_summary(groups[name]) for name in sorted(groups)}


def _population_weighted(items: list[dict], population: dict[str, int]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        groups[item["stratum"]].append(item)
    represented = sorted(groups)
    missing = sorted(set(population) - set(groups))
    represented_population = sum(int(population[name]) for name in represented)
    population_total = sum(int(value) for value in population.values())
    metrics = {
        "answer_accuracy": "answer_correct",
        "execution_accuracy": "execution_correct",
        "query_executable_rate": "query_executable",
        "coverage": "covered",
    }
    weighted: dict[str, float] = {}
    for output_name, item_key in metrics.items():
        value = 0.0
        for stratum in represented:
            stratum_items = groups[stratum]
            stratum_rate = sum(bool(item[item_key]) for item in stratum_items) / len(stratum_items)
            value += int(population[stratum]) / represented_population * stratum_rate
        weighted[output_name] = round(value, 6) if represented_population else 0.0
    return {
        "method": "post-stratified by manifest population; conditional on represented strata",
        "population_count": population_total,
        "represented_population_count": represented_population,
        "represented_population_mass": _ratio(represented_population, population_total),
        "complete_population_coverage": not missing,
        "missing_strata": missing,
        **weighted,
    }


def evaluate_codegen(
    codegen_path: Path | str,
    gold_path: Path | str,
    bundle_dir: Path | str,
    split: str,
    *,
    store_dir: Path | str,
    output_path: Path | str | None = None,
    seal_path: Path | str | None = None,
    verify_bundle: bool = True,
) -> dict:
    """Replay a complete codegen artifact and evaluate it on validated gold."""
    if split not in {"tune", "locked"}:
        _fail("split must be 'tune' or 'locked'")
    if output_path is not None and Path(output_path).exists():
        _fail(f"refusing to overwrite existing output: {output_path}")
    bundle_dir = Path(bundle_dir)
    if split == "locked":
        if seal_path is None:
            _fail("locked evaluation requires --seal")
        verify_locked_seal(
            gold_path, bundle_dir, seal_path, store_dir=store_dir,
            verify_bundle=verify_bundle,
        )
    else:
        validate_gold_file(
            gold_path, bundle_dir, split, store_dir=store_dir,
            require_complete=True, verify_bundle=verify_bundle,
        )

    manifest = read_json(bundle_dir / MANIFEST_NAME)
    source_questions_path = _resolve_source_questions(bundle_dir, manifest)
    codegen_rows, run_signatures = _validate_complete_codegen(
        codegen_path, source_questions_path, int(manifest["source"]["count"])
    )
    codegen_map = {int(row["id"]): row for row in codegen_rows}
    gold_rows = _read_unique_jsonl(gold_path, f"{split} gold")
    question_name = TUNE_QUESTIONS if split == "tune" else LOCKED_QUESTIONS
    split_questions = _read_unique_jsonl(bundle_dir / question_name, f"{split} questions")
    qmap = {int(row["id"]): row for row in split_questions}
    loader = StoreTableLoader(store_dir)

    diagnostics: list[dict] = []
    for gold in sorted(gold_rows, key=lambda row: int(row["id"])):
        qid = int(gold["id"])
        prediction = codegen_map[qid]
        tolerance = float(gold["replay"]["tolerance"])
        gold_value = float(gold["output"]["value"])
        predicted_value = float(prediction["answer"])
        replay = _prediction_replay(prediction, loader)
        execution_value = replay["value"]
        query_executable = bool(replay["ok"])
        answer_correct = math.isclose(
            predicted_value, gold_value, rel_tol=0.0, abs_tol=tolerance
        )
        execution_correct = bool(
            query_executable
            and execution_value is not None
            and math.isclose(
                float(execution_value), gold_value,
                rel_tol=0.0, abs_tol=tolerance,
            )
        )
        source = str(prediction.get("source") or "unknown")
        covered = bool(
            prediction.get("status") == "ok"
            and source != "none"
            and prediction.get("used_vars")
        )
        diagnostics.append({
            "id": qid,
            "stratum": qmap[qid]["stratum"],
            "output_type": gold["output"]["type"],
            "source": source,
            "status": str(prediction.get("status") or ""),
            "gold_answer": gold_value,
            "predicted_answer": predicted_value,
            "execution_value": execution_value,
            "answer_correct": answer_correct,
            "query_executable": query_executable,
            "execution_correct": execution_correct,
            "covered": covered,
            "execution_error": str(replay.get("error") or "")[:500],
        })

    population = {
        str(name): int(count)
        for name, count in manifest["allocation"]["population_strata"].items()
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "split": split,
        "provenance": {
            "bundle_fingerprint_sha256": manifest["bundle_fingerprint_sha256"],
            "manifest_sha256": file_sha256(bundle_dir / MANIFEST_NAME),
            "gold_sha256": file_sha256(gold_path),
            "codegen_sha256": file_sha256(codegen_path),
            "codegen_canonical_sha256": canonical_sha256(codegen_rows),
            "run_signatures": run_signatures,
            "run_signature_set_sha256": canonical_sha256(run_signatures),
            "locked_seal_sha256": (
                file_sha256(seal_path) if split == "locked" and seal_path else None
            ),
        },
        "metric_definitions": {
            "answer_accuracy": (
                "predicted answer equals gold output.value within gold replay tolerance"
            ),
            "execution_accuracy": (
                "pandas_query replays and its value equals gold within tolerance"
            ),
            "query_executable_rate": (
                "pandas_query is a safe eval expression and replays to a finite scalar"
            ),
            "coverage": (
                "status is ok, source is not none, and at least one evidence table is bound"
            ),
        },
        "metrics": _metric_summary(diagnostics),
        "population_weighted": _population_weighted(diagnostics, population),
        "breakdown": {
            "stratum": _group_metrics(diagnostics, "stratum"),
            "output_type": _group_metrics(diagnostics, "output_type"),
            "source": _group_metrics(diagnostics, "source"),
        },
        "records": diagnostics,
    }
    if output_path is not None:
        _write_json_new(output_path, report)
    return report
