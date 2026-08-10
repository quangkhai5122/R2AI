"""P2.4: deterministic, leakage-resistant human-gold dev-set scaffolding.

This module is deliberately separate from retrieval and code generation.  It
samples the official question pool, writes immutable question/template files,
and validates completed labels down to their exact source cells and replayed
pandas expression.
"""
from __future__ import annotations

import ast as py_ast
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from ..codegen.executor import check_safe, run_code
from ..codegen.semantic import all_dataframe_refs
from ..extraction.build_store import Store
from ..retrieval.serialize import df_roundtrip, tidy_csv_text
from ..utils.io import read_json, read_jsonl


SAMPLER_VERSION = "p24_stratified_v1"
QUESTION_SCHEMA_VERSION = "p24_question_v1"
GOLD_SCHEMA_VERSION = "p24_gold_v1"
MANIFEST_SCHEMA_VERSION = "p24_split_manifest_v1"
SEAL_SCHEMA_VERSION = "p24_locked_seal_v1"

DEFAULT_SEED = 2404
DEFAULT_TUNE_SIZE = 100
DEFAULT_LOCKED_SIZE = 50
DEFAULT_EXPECTED_SOURCE_COUNT = 1012

TUNE_QUESTIONS = "p24_tune_questions.jsonl"
LOCKED_QUESTIONS = "p24_locked_questions.jsonl"
TUNE_TEMPLATE = "p24_tune_gold.template.jsonl"
LOCKED_TEMPLATE = "p24_locked_gold.template.jsonl"
MANIFEST_NAME = "p24_manifest.json"
LOCKED_SEAL_NAME = "p24_locked_gold.seal.json"


def _write_jsonl_new(path: Path | str, records: Iterable[dict]) -> None:
    """Write a JSONL artifact once; immutable P2.4 files are never overwritten."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except FileExistsError:
        _fail(f"refusing to overwrite existing P2.4 artifact: {target}")


def _write_json_new(path: Path | str, obj: object) -> None:
    """Write a JSON artifact once; immutable P2.4 files are never overwritten."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8") as handle:
            json.dump(obj, handle, ensure_ascii=False, indent=2)
    except FileExistsError:
        _fail(f"refusing to overwrite existing P2.4 artifact: {target}")

ALLOWED_OUTPUT_TYPES = {
    "number", "percent", "percentage_point", "ratio", "count", "year"
}
ALLOWED_AST_OPS = {
    "lookup", "sum", "average", "difference", "growth_pct", "ratio",
    "margin", "ratio_times", "ranking", "ranking_max", "ranking_min",
    "argmax", "argmin", "count", "percentage_point", "cagr",
    "hypothetical", "add", "subtract", "multiply", "divide", "abs",
    "negate",
}
EXACT_ARITY = {
    "lookup": 1,
    "difference": 2,
    "growth_pct": 2,
    "ratio": 2,
    "margin": 2,
    "ratio_times": 2,
    "percentage_point": 2,
    "add": 2,
    "subtract": 2,
    "multiply": 2,
    "divide": 2,
    "abs": 1,
    "negate": 1,
}
MIN_ARITY = {
    "sum": 1,
    "average": 1,
    "count": 1,
    "ranking": 2,
    "ranking_max": 2,
    "ranking_min": 2,
    "argmax": 2,
    "argmin": 2,
    "hypothetical": 1,
}

QUESTION_KEYS = {
    "schema_version", "split", "id", "question", "question_sha256",
    "stratum", "route_summary",
}
ROUTE_SUMMARY_KEYS = {
    "op", "n_facts", "n_entities", "n_periods", "output_type",
}
GOLD_KEYS = {
    "schema_version", "split", "id", "question", "question_sha256",
    "stratum", "label_status", "evidence", "output", "ast", "replay",
    "annotator_notes",
}
EVIDENCE_KEYS = {
    "evidence_id", "variable", "report_id", "table_pos", "row", "col",
    "label", "code", "col_name", "value", "unit_scale",
}
OUTPUT_KEYS = {"type", "value", "unit", "scale", "round_decimals"}
REPLAY_KEYS = {
    "pandas_query", "used_vars", "expected_answer", "tolerance", "status",
    "evidence_sha256", "ast_sha256",
}
USED_VAR_KEYS = {"var", "report_id", "table_pos"}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_EVIDENCE_ID = re.compile(r"^E[1-9][0-9]*$")
_DF_VAR = re.compile(r"^df[1-9][0-9]*$")


class P24ValidationError(ValueError):
    """Raised when a P2.4 split, label, replay, or seal is invalid."""


def _canonical_json(value) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def question_sha256(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def _fail(message: str) -> None:
    raise P24ValidationError(message)


def _exact_keys(value: dict, expected: set[str], where: str) -> None:
    if not isinstance(value, dict):
        _fail(f"{where}: expected object")
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        _fail(f"{where}: fields mismatch; missing={missing}, extra={extra}")


def _read_unique_jsonl(path: Path | str, label: str) -> list[dict]:
    rows = read_jsonl(path)
    seen: set[int] = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or "id" not in row:
            _fail(f"{label} line {index}: missing id")
        qid = int(row["id"])
        if qid in seen:
            _fail(f"{label}: duplicate id {qid}")
        seen.add(qid)
    return rows


def _complexity_bucket(n_facts: int) -> str:
    if n_facts <= 1:
        return "single"
    if n_facts <= 4:
        return "multi_2_4"
    return "multi_5_plus"


def _load_source_rows(
    questions_path: Path | str,
    retrieval_path: Path | str,
    expected_source_count: int | None,
) -> list[dict]:
    questions = _read_unique_jsonl(questions_path, "questions")
    retrieval = _read_unique_jsonl(retrieval_path, "retrieval")
    if expected_source_count is not None and len(questions) != expected_source_count:
        _fail(
            f"questions: expected exactly {expected_source_count} rows, "
            f"found {len(questions)}"
        )
    q_by_id = {int(row["id"]): row for row in questions}
    r_by_id = {int(row["id"]): row for row in retrieval}
    if set(q_by_id) != set(r_by_id):
        missing = sorted(set(q_by_id) - set(r_by_id))[:10]
        extra = sorted(set(r_by_id) - set(q_by_id))[:10]
        _fail(f"retrieval ids differ from questions; missing={missing}, extra={extra}")

    source_rows: list[dict] = []
    seen_question_hashes: set[str] = set()
    for qid in sorted(q_by_id):
        question = q_by_id[qid].get("question")
        if not isinstance(question, str) or not question.strip():
            _fail(f"question {qid}: empty question text")
        retrieval_question = r_by_id[qid].get("question")
        if retrieval_question != question:
            _fail(f"question {qid}: retrieval question text differs from source")
        qhash = question_sha256(question)
        if qhash in seen_question_hashes:
            _fail(f"question {qid}: duplicate question text")
        seen_question_hashes.add(qhash)

        route = r_by_id[qid].get("route") or {}
        plan = route.get("plan") or {}
        op = str(plan.get("op") or "lookup")
        facts = plan.get("facts") or []
        n_facts = len(facts)
        output_type = str(route.get("output_type") or "number")
        if output_type not in ALLOWED_OUTPUT_TYPES:
            _fail(f"question {qid}: unsupported output_type {output_type!r}")
        summary = {
            "op": op,
            "n_facts": n_facts,
            "n_entities": int(plan.get("n_entities") or len(route.get("tickers") or [])),
            "n_periods": int(plan.get("n_periods") or len(route.get("years") or [])),
            "output_type": output_type,
        }
        source_rows.append({
            "id": qid,
            "question": question,
            "question_sha256": qhash,
            "stratum": f"{op}|{_complexity_bucket(n_facts)}",
            "route_summary": summary,
            "unit": str(route.get("unit_name") or ""),
            "scale": float(route.get("unit_scale") or 1.0),
        })
    return source_rows


def _allocate_strata(
    counts: dict[str, int], total: int, ensure_each: bool = True
) -> dict[str, int]:
    """Allocate an exact total with deterministic, capacity-aware quotas.

    One slot is reserved per non-empty stratum when capacity permits; remaining
    slots use largest-remainder allocation over the remaining population.
    """
    counts = {str(k): int(v) for k, v in counts.items() if int(v) > 0}
    population = sum(counts.values())
    if total < 0 or total > population:
        _fail(f"cannot allocate {total} rows from population {population}")
    if total == 0:
        return {}

    base = {key: 0 for key in counts}
    if ensure_each and total >= len(counts):
        for key in base:
            base[key] = 1
    reserved = sum(base.values())
    remaining = total - reserved
    capacity = {key: counts[key] - base[key] for key in counts}
    cap_total = sum(capacity.values())
    if remaining and not cap_total:
        _fail("stratified allocation has no remaining capacity")

    ideals = {
        key: (remaining * capacity[key] / cap_total if cap_total else 0.0)
        for key in counts
    }
    allocations = {
        key: base[key] + min(capacity[key], int(math.floor(ideals[key])))
        for key in counts
    }
    left = total - sum(allocations.values())
    while left:
        candidates = [key for key in counts if allocations[key] < counts[key]]
        if not candidates:
            _fail("stratified allocation exhausted capacity")
        candidates.sort(
            key=lambda key: (
                -(ideals[key] - math.floor(ideals[key])), key
            )
        )
        for key in candidates:
            if not left:
                break
            allocations[key] += 1
            left -= 1
    return {
        key: allocations[key]
        for key in sorted(allocations)
        if allocations[key] > 0
    }


def _rank_key(seed: int, phase: str, row: dict) -> str:
    payload = (
        f"{SAMPLER_VERSION}|{seed}|{phase}|{row['stratum']}|"
        f"{row['id']}|{row['question_sha256']}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _select_by_quota(
    rows: Iterable[dict], quotas: dict[str, int], seed: int, phase: str
) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["stratum"]].append(row)
    selected: list[dict] = []
    for stratum, quota in quotas.items():
        pool = sorted(
            grouped.get(stratum, []),
            key=lambda row: (_rank_key(seed, phase, row), int(row["id"])),
        )
        if len(pool) < quota:
            _fail(f"stratum {stratum}: quota {quota} exceeds pool {len(pool)}")
        selected.extend(pool[:quota])
    return sorted(selected, key=lambda row: int(row["id"]))


def _question_record(row: dict, split: str) -> dict:
    return {
        "schema_version": QUESTION_SCHEMA_VERSION,
        "split": split,
        "id": int(row["id"]),
        "question": row["question"],
        "question_sha256": row["question_sha256"],
        "stratum": row["stratum"],
        "route_summary": dict(row["route_summary"]),
    }


def _gold_template(row: dict, split: str) -> dict:
    return {
        "schema_version": GOLD_SCHEMA_VERSION,
        "split": split,
        "id": int(row["id"]),
        "question": row["question"],
        "question_sha256": row["question_sha256"],
        "stratum": row["stratum"],
        "label_status": "unlabeled",
        "evidence": [],
        "output": {
            "type": row["route_summary"]["output_type"],
            "value": None,
            "unit": row["unit"],
            "scale": float(row["scale"]),
            "round_decimals": 2,
        },
        "ast": None,
        "replay": {
            "pandas_query": "",
            "used_vars": [],
            "expected_answer": None,
            "tolerance": 0.01,
            "status": "unverified",
            "evidence_sha256": "",
            "ast_sha256": "",
        },
        "annotator_notes": "",
    }


def _split_metadata(rows: list[dict]) -> dict:
    strata: dict[str, int] = defaultdict(int)
    for row in rows:
        strata[row["stratum"]] += 1
    ids = [int(row["id"]) for row in rows]
    qhashes = [row["question_sha256"] for row in rows]
    return {
        "count": len(rows),
        "ids": ids,
        "ids_sha256": canonical_sha256(ids),
        "question_hashes_sha256": canonical_sha256(qhashes),
        "strata": dict(sorted(strata.items())),
    }


def build_bundle(
    questions_path: Path | str,
    retrieval_path: Path | str,
    out_dir: Path | str,
    *,
    seed: int = DEFAULT_SEED,
    tune_size: int = DEFAULT_TUNE_SIZE,
    locked_size: int = DEFAULT_LOCKED_SIZE,
    expected_source_count: int | None = DEFAULT_EXPECTED_SOURCE_COUNT,
) -> dict:
    """Build deterministic tune/locked question files and gold templates."""
    out_dir = Path(out_dir)
    immutable_paths = [
        out_dir / name
        for name in (
            TUNE_QUESTIONS,
            LOCKED_QUESTIONS,
            TUNE_TEMPLATE,
            LOCKED_TEMPLATE,
            MANIFEST_NAME,
        )
    ]
    existing = [path for path in immutable_paths if path.exists()]
    if existing:
        names = ", ".join(path.name for path in existing)
        _fail(
            "refusing to overwrite existing P2.4 bundle files: "
            f"{names}; validate the frozen bundle or choose a new --out-dir"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    source_rows = _load_source_rows(
        questions_path, retrieval_path, expected_source_count
    )
    total_size = int(tune_size) + int(locked_size)
    if tune_size <= 0 or locked_size <= 0:
        _fail("tune_size and locked_size must both be positive")

    population_counts: dict[str, int] = defaultdict(int)
    for row in source_rows:
        population_counts[row["stratum"]] += 1
    sample_quotas = _allocate_strata(population_counts, total_size, ensure_each=True)
    selected = _select_by_quota(source_rows, sample_quotas, seed, "sample")

    selected_counts: dict[str, int] = defaultdict(int)
    for row in selected:
        selected_counts[row["stratum"]] += 1
    locked_quotas = _allocate_strata(
        selected_counts, int(locked_size), ensure_each=True
    )
    locked_rows = _select_by_quota(selected, locked_quotas, seed, "locked")
    locked_ids = {int(row["id"]) for row in locked_rows}
    tune_rows = [row for row in selected if int(row["id"]) not in locked_ids]
    if len(tune_rows) != tune_size or len(locked_rows) != locked_size:
        _fail(
            f"split size mismatch: tune={len(tune_rows)}, locked={len(locked_rows)}"
        )

    tune_questions = [_question_record(row, "tune") for row in tune_rows]
    locked_questions = [_question_record(row, "locked") for row in locked_rows]
    tune_templates = [_gold_template(row, "tune") for row in tune_rows]
    locked_templates = [_gold_template(row, "locked") for row in locked_rows]

    file_rows = {
        TUNE_QUESTIONS: tune_questions,
        LOCKED_QUESTIONS: locked_questions,
        TUNE_TEMPLATE: tune_templates,
        LOCKED_TEMPLATE: locked_templates,
    }
    for name, rows in file_rows.items():
        _write_jsonl_new(out_dir / name, rows)

    tune_meta = _split_metadata(tune_questions)
    locked_meta = _split_metadata(locked_questions)
    overlap_ids = sorted(set(tune_meta["ids"]) & set(locked_meta["ids"]))
    overlap_qhash = sorted(
        {row["question_sha256"] for row in tune_questions}
        & {row["question_sha256"] for row in locked_questions}
    )
    if overlap_ids or overlap_qhash:
        _fail("sampler produced tune/locked leakage")

    source_fingerprint_rows = [
        {
            "id": row["id"],
            "question_sha256": row["question_sha256"],
            "stratum": row["stratum"],
            "route_summary": row["route_summary"],
        }
        for row in source_rows
    ]
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "sampler_version": SAMPLER_VERSION,
        "seed": int(seed),
        "source": {
            "count": len(source_rows),
            "questions_path": Path(questions_path).as_posix(),
            "questions_sha256": file_sha256(questions_path),
            "retrieval_path": Path(retrieval_path).as_posix(),
            "retrieval_sha256": file_sha256(retrieval_path),
            "fingerprint_sha256": canonical_sha256(source_fingerprint_rows),
        },
        "splits": {"tune": tune_meta, "locked": locked_meta},
        "allocation": {
            "population_strata": dict(sorted(population_counts.items())),
            "sample_quotas": sample_quotas,
            "locked_quotas": locked_quotas,
        },
        "files": {
            name: {
                "count": len(rows),
                "sha256": file_sha256(out_dir / name),
            }
            for name, rows in file_rows.items()
        },
        "leakage_guard": {
            "id_overlap_sha256": canonical_sha256(overlap_ids),
            "question_overlap_sha256": canonical_sha256(overlap_qhash),
            "selected_ids_sha256": canonical_sha256(
                sorted(tune_meta["ids"] + locked_meta["ids"])
            ),
        },
    }
    manifest["bundle_fingerprint_sha256"] = canonical_sha256({
        "source": manifest["source"]["fingerprint_sha256"],
        "tune": tune_meta["ids_sha256"],
        "locked": locked_meta["ids_sha256"],
        "files": manifest["files"],
    })
    _write_json_new(out_dir / MANIFEST_NAME, manifest)
    validate_bundle(
        out_dir,
        questions_path=questions_path,
        retrieval_path=retrieval_path,
        expected_source_count=expected_source_count,
        expected_tune_size=tune_size,
        expected_locked_size=locked_size,
    )
    return manifest


def _validate_question_records(rows: list[dict], split: str) -> None:
    seen_ids: set[int] = set()
    seen_hashes: set[str] = set()
    for index, row in enumerate(rows, 1):
        where = f"{split} questions line {index}"
        _exact_keys(row, QUESTION_KEYS, where)
        _exact_keys(row["route_summary"], ROUTE_SUMMARY_KEYS, f"{where}.route_summary")
        if row["schema_version"] != QUESTION_SCHEMA_VERSION:
            _fail(f"{where}: invalid schema_version")
        if row["split"] != split:
            _fail(f"{where}: expected split {split!r}")
        qid = int(row["id"])
        if qid in seen_ids:
            _fail(f"{where}: duplicate id {qid}")
        seen_ids.add(qid)
        question = row["question"]
        if not isinstance(question, str) or not question.strip():
            _fail(f"{where}: empty question")
        qhash = row["question_sha256"]
        if not _HEX64.fullmatch(str(qhash)) or qhash != question_sha256(question):
            _fail(f"{where}: question_sha256 mismatch")
        if qhash in seen_hashes:
            _fail(f"{where}: duplicate question text")
        seen_hashes.add(qhash)
        if row["route_summary"]["output_type"] not in ALLOWED_OUTPUT_TYPES:
            _fail(f"{where}: unsupported output_type")


def _validate_template_identity(
    gold_rows: list[dict], question_rows: list[dict], split: str
) -> None:
    if len(gold_rows) != len(question_rows):
        _fail(
            f"{split} template count {len(gold_rows)} != question count "
            f"{len(question_rows)}"
        )
    qmap = {int(row["id"]): row for row in question_rows}
    seen: set[int] = set()
    for index, row in enumerate(gold_rows, 1):
        where = f"{split} template line {index}"
        _exact_keys(row, GOLD_KEYS, where)
        qid = int(row["id"])
        if qid in seen:
            _fail(f"{where}: duplicate id {qid}")
        seen.add(qid)
        question = qmap.get(qid)
        if question is None:
            _fail(f"{where}: id {qid} not in {split} questions")
        for key in ("split", "question", "question_sha256", "stratum"):
            if row[key] != question[key]:
                _fail(f"{where}: identity field {key!r} differs from split")
        if row["schema_version"] != GOLD_SCHEMA_VERSION:
            _fail(f"{where}: invalid schema_version")
        if row["label_status"] not in {"unlabeled", "draft", "verified"}:
            _fail(f"{where}: invalid label_status")
        _exact_keys(row["output"], OUTPUT_KEYS, f"{where}.output")
        _exact_keys(row["replay"], REPLAY_KEYS, f"{where}.replay")
    if seen != set(qmap):
        _fail(f"{split} template ids differ from question ids")


def validate_bundle(
    bundle_dir: Path | str,
    *,
    questions_path: Path | str | None = None,
    retrieval_path: Path | str | None = None,
    expected_source_count: int | None = DEFAULT_EXPECTED_SOURCE_COUNT,
    expected_tune_size: int | None = DEFAULT_TUNE_SIZE,
    expected_locked_size: int | None = DEFAULT_LOCKED_SIZE,
) -> dict:
    """Validate file hashes, source hashes, exact split sizes, and no leakage."""
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / MANIFEST_NAME
    if not manifest_path.exists():
        _fail(f"missing manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        _fail("manifest schema_version mismatch")
    if manifest.get("sampler_version") != SAMPLER_VERSION:
        _fail("manifest sampler_version mismatch")

    required_files = {
        TUNE_QUESTIONS, LOCKED_QUESTIONS, TUNE_TEMPLATE, LOCKED_TEMPLATE
    }
    if set(manifest.get("files") or {}) != required_files:
        _fail("manifest file registry is incomplete or has extra files")
    for name in sorted(required_files):
        path = bundle_dir / name
        if not path.exists():
            _fail(f"missing bundle file: {name}")
        expected_hash = manifest["files"][name].get("sha256")
        if file_sha256(path) != expected_hash:
            _fail(f"bundle file hash mismatch: {name}")

    tune = _read_unique_jsonl(bundle_dir / TUNE_QUESTIONS, "tune questions")
    locked = _read_unique_jsonl(bundle_dir / LOCKED_QUESTIONS, "locked questions")
    _validate_question_records(tune, "tune")
    _validate_question_records(locked, "locked")
    tune_gold = _read_unique_jsonl(bundle_dir / TUNE_TEMPLATE, "tune template")
    locked_gold = _read_unique_jsonl(bundle_dir / LOCKED_TEMPLATE, "locked template")
    _validate_template_identity(tune_gold, tune, "tune")
    _validate_template_identity(locked_gold, locked, "locked")
    if expected_tune_size is not None and len(tune) != expected_tune_size:
        _fail(f"tune split must contain exactly {expected_tune_size} rows")
    if expected_locked_size is not None and len(locked) != expected_locked_size:
        _fail(f"locked split must contain exactly {expected_locked_size} rows")

    for split, rows in (("tune", tune), ("locked", locked)):
        metadata = _split_metadata(rows)
        recorded = manifest["splits"].get(split) or {}
        if metadata != recorded:
            _fail(f"manifest {split} split metadata mismatch")
        if manifest["files"][
            TUNE_QUESTIONS if split == "tune" else LOCKED_QUESTIONS
        ]["count"] != len(rows):
            _fail(f"manifest {split} file count mismatch")
    if manifest["files"][TUNE_TEMPLATE]["count"] != len(tune_gold):
        _fail("manifest tune template count mismatch")
    if manifest["files"][LOCKED_TEMPLATE]["count"] != len(locked_gold):
        _fail("manifest locked template count mismatch")

    tune_ids = {int(row["id"]) for row in tune}
    locked_ids = {int(row["id"]) for row in locked}
    id_overlap = sorted(tune_ids & locked_ids)
    q_overlap = sorted(
        {row["question_sha256"] for row in tune}
        & {row["question_sha256"] for row in locked}
    )
    if id_overlap or q_overlap:
        _fail(f"tune/locked leakage: ids={id_overlap[:10]}, questions={q_overlap[:3]}")
    leakage = manifest.get("leakage_guard") or {}
    if leakage.get("id_overlap_sha256") != canonical_sha256(id_overlap):
        _fail("manifest id-overlap guard mismatch")
    if leakage.get("question_overlap_sha256") != canonical_sha256(q_overlap):
        _fail("manifest question-overlap guard mismatch")
    selected_hash = canonical_sha256(sorted(tune_ids | locked_ids))
    if leakage.get("selected_ids_sha256") != selected_hash:
        _fail("manifest selected-id guard mismatch")

    actual_sample_strata: dict[str, int] = defaultdict(int)
    for row in tune + locked:
        actual_sample_strata[row["stratum"]] += 1
    allocation = manifest.get("allocation") or {}
    if allocation.get("sample_quotas") != dict(sorted(actual_sample_strata.items())):
        _fail("manifest sample quotas differ from selected rows")
    if allocation.get("locked_quotas") != _split_metadata(locked)["strata"]:
        _fail("manifest locked quotas differ from locked rows")

    source_info = manifest.get("source") or {}
    if expected_source_count is not None and source_info.get("count") != expected_source_count:
        _fail("manifest source count mismatch")
    if questions_path is not None:
        if file_sha256(questions_path) != source_info.get("questions_sha256"):
            _fail("source questions hash mismatch")
    if retrieval_path is not None:
        if file_sha256(retrieval_path) != source_info.get("retrieval_sha256"):
            _fail("source retrieval hash mismatch")
    if questions_path is not None and retrieval_path is not None:
        source_rows = _load_source_rows(
            questions_path, retrieval_path, expected_source_count
        )
        fingerprint = canonical_sha256([
            {
                "id": row["id"],
                "question_sha256": row["question_sha256"],
                "stratum": row["stratum"],
                "route_summary": row["route_summary"],
            }
            for row in source_rows
        ])
        if fingerprint != source_info.get("fingerprint_sha256"):
            _fail("source semantic fingerprint mismatch")
        population_strata: dict[str, int] = defaultdict(int)
        for row in source_rows:
            population_strata[row["stratum"]] += 1
        if allocation.get("population_strata") != dict(sorted(population_strata.items())):
            _fail("manifest population strata differ from source")
        source_map = {int(row["id"]): row for row in source_rows}
        for row in tune + locked:
            source = source_map.get(int(row["id"]))
            if source is None or row["question"] != source["question"]:
                _fail(f"sampled question {row['id']} differs from source")
            if row["stratum"] != source["stratum"]:
                _fail(f"sampled question {row['id']} stratum differs from source")

    expected_bundle_fingerprint = canonical_sha256({
        "source": source_info.get("fingerprint_sha256"),
        "tune": manifest["splits"]["tune"].get("ids_sha256"),
        "locked": manifest["splits"]["locked"].get("ids_sha256"),
        "files": manifest["files"],
    })
    if manifest.get("bundle_fingerprint_sha256") != expected_bundle_fingerprint:
        _fail("manifest bundle fingerprint mismatch")

    return {
        "source_count": int(source_info.get("count", 0)),
        "tune_count": len(tune),
        "locked_count": len(locked),
        "bundle_fingerprint_sha256": manifest.get("bundle_fingerprint_sha256"),
    }


class StoreTableLoader:
    """Resolve an exact report/table pair to the grader-style tidy DataFrame."""

    def __init__(self, store_dir: Path | str):
        self.store = Store(Path(store_dir))
        self.cache: dict[tuple[str, int], pd.DataFrame] = {}

    def __call__(self, report_id: str, table_pos: int) -> pd.DataFrame:
        key = (str(report_id), int(table_pos))
        if key not in self.cache:
            ticker = key[0].split("_")[0]
            tables = self.store.tables_of(ticker, [key[0]])
            hit = tables[
                (tables.report_id == key[0]) & (tables.table_pos == key[1])
            ]
            if len(hit) != 1:
                _fail(
                    f"store evidence table {key[0]}|{key[1]}: expected one row, "
                    f"found {len(hit)}"
                )
            self.cache[key] = df_roundtrip(tidy_csv_text(hit.iloc[0].to_dict()))
        return self.cache[key].copy()


def _finite_number(value, where: str) -> float:
    if isinstance(value, bool):
        _fail(f"{where}: boolean is not a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise P24ValidationError(f"{where}: expected finite number") from exc
    if not math.isfinite(number):
        _fail(f"{where}: expected finite number")
    return number


def _same_text(actual, expected) -> bool:
    if pd.isna(actual):
        actual = ""
    if expected is None:
        expected = ""
    return str(actual) == str(expected)


def _validate_evidence(
    evidence: list[dict],
    where: str,
    table_loader: Callable[[str, int], pd.DataFrame] | None,
    require_complete: bool,
) -> tuple[set[str], dict[str, tuple[str, int]], dict[str, pd.DataFrame]]:
    if not isinstance(evidence, list):
        _fail(f"{where}: evidence must be a list")
    if require_complete and not evidence:
        _fail(f"{where}: verified label needs at least one evidence cell")
    evidence_ids: set[str] = set()
    var_tables: dict[str, tuple[str, int]] = {}
    dfs: dict[str, pd.DataFrame] = {}
    for index, cell in enumerate(evidence, 1):
        cell_where = f"{where}[{index}]"
        _exact_keys(cell, EVIDENCE_KEYS, cell_where)
        eid = str(cell["evidence_id"])
        if not _EVIDENCE_ID.fullmatch(eid) or eid in evidence_ids:
            _fail(f"{cell_where}: evidence_id must be a unique E<number>")
        evidence_ids.add(eid)
        var = str(cell["variable"])
        if not _DF_VAR.fullmatch(var):
            _fail(f"{cell_where}: invalid dataframe variable {var!r}")
        report_id = str(cell["report_id"])
        table_pos = int(cell["table_pos"])
        table_key = (report_id, table_pos)
        if var in var_tables and var_tables[var] != table_key:
            _fail(f"{cell_where}: variable {var} maps to multiple tables")
        var_tables[var] = table_key
        row_no, col_no = int(cell["row"]), int(cell["col"])
        _finite_number(cell["value"], f"{cell_where}.value")
        _finite_number(cell["unit_scale"], f"{cell_where}.unit_scale")
        for key in ("label", "code", "col_name"):
            if not isinstance(cell[key], str):
                _fail(f"{cell_where}.{key}: expected string")

        if table_loader is None:
            if require_complete:
                _fail(f"{cell_where}: table loader is required for exact evidence")
            continue
        df = dfs.get(var)
        if df is None:
            df = table_loader(report_id, table_pos)
            dfs[var] = df
        required_columns = {
            "row", "label", "code", "col", "col_name", "value", "unit_scale"
        }
        if not required_columns.issubset(df.columns):
            _fail(f"{cell_where}: evidence table lacks tidy columns")
        hit = df[(df["row"] == row_no) & (df["col"] == col_no)]
        if len(hit) != 1:
            _fail(
                f"{cell_where}: exact cell row={row_no}, col={col_no} "
                f"matched {len(hit)} rows"
            )
        actual = hit.iloc[0]
        for key in ("label", "code", "col_name"):
            if not _same_text(actual[key], cell[key]):
                _fail(f"{cell_where}: exact evidence {key} mismatch")
        for key in ("value", "unit_scale"):
            if not math.isclose(
                float(actual[key]), float(cell[key]), rel_tol=0.0, abs_tol=1e-9
            ):
                _fail(f"{cell_where}: exact evidence {key} mismatch")
    return evidence_ids, var_tables, dfs


def _validate_ast_node(
    node: dict,
    evidence_ids: set[str],
    where: str,
    *,
    depth: int = 0,
    node_budget: list[int] | None = None,
) -> set[str]:
    if node_budget is None:
        node_budget = [0]
    node_budget[0] += 1
    if depth > 24 or node_budget[0] > 256:
        _fail(f"{where}: AST is too deep or too large")
    if not isinstance(node, dict):
        _fail(f"{where}: AST node must be an object")
    kind = node.get("kind")
    if kind == "evidence":
        _exact_keys(node, {"kind", "evidence_id"}, where)
        eid = str(node["evidence_id"])
        if eid not in evidence_ids:
            _fail(f"{where}: unknown evidence_id {eid!r}")
        return {eid}
    if kind == "literal":
        _exact_keys(node, {"kind", "value"}, where)
        _finite_number(node["value"], f"{where}.value")
        return set()
    if kind != "op":
        _fail(f"{where}: kind must be evidence, literal, or op")
    _exact_keys(node, {"kind", "op", "args"}, where)
    op = str(node["op"])
    if op not in ALLOWED_AST_OPS:
        _fail(f"{where}: unsupported AST op {op!r}")
    args = node["args"]
    if not isinstance(args, list):
        _fail(f"{where}.args: expected list")
    exact = EXACT_ARITY.get(op)
    minimum = MIN_ARITY.get(op)
    if exact is not None and len(args) != exact:
        _fail(f"{where}: op {op} needs exactly {exact} args")
    if minimum is not None and len(args) < minimum:
        _fail(f"{where}: op {op} needs at least {minimum} args")
    if op == "cagr" and len(args) not in {2, 3}:
        _fail(f"{where}: op cagr needs 2 facts and optional horizon literal")
    refs: set[str] = set()
    for index, arg in enumerate(args, 1):
        refs |= _validate_ast_node(
            arg, evidence_ids, f"{where}.args[{index}]",
            depth=depth + 1, node_budget=node_budget,
        )
    return refs


def _validate_output(output: dict, where: str, require_complete: bool) -> float | None:
    _exact_keys(output, OUTPUT_KEYS, where)
    output_type = output["type"]
    if output_type not in ALLOWED_OUTPUT_TYPES:
        _fail(f"{where}.type: unsupported output type {output_type!r}")
    if not isinstance(output["unit"], str):
        _fail(f"{where}.unit: expected string")
    scale = _finite_number(output["scale"], f"{where}.scale")
    if scale <= 0:
        _fail(f"{where}.scale: must be positive")
    decimals = output["round_decimals"]
    if not isinstance(decimals, int) or isinstance(decimals, bool) or not 0 <= decimals <= 8:
        _fail(f"{where}.round_decimals: expected integer 0..8")
    if output["value"] is None and not require_complete:
        return None
    value = _finite_number(output["value"], f"{where}.value")
    if output_type in {"count", "year"} and not float(value).is_integer():
        _fail(f"{where}.value: {output_type} output must be an integer")
    return value


def _validate_replay(
    replay: dict,
    output_value: float,
    evidence: list[dict],
    ast_node: dict,
    var_tables: dict[str, tuple[str, int]],
    dfs: dict[str, pd.DataFrame],
    table_loader: Callable[[str, int], pd.DataFrame] | None,
    where: str,
) -> None:
    _exact_keys(replay, REPLAY_KEYS, where)
    if replay["status"] != "verified":
        _fail(f"{where}.status: verified label requires status='verified'")
    query = replay["pandas_query"]
    if not isinstance(query, str) or not query.strip():
        _fail(f"{where}.pandas_query: empty expression")
    try:
        py_ast.parse(query, mode="eval")
    except SyntaxError as exc:
        raise P24ValidationError(
            f"{where}.pandas_query: must be one eval expression: {exc.msg}"
        ) from exc
    banned = check_safe(query)
    if banned:
        _fail(f"{where}.pandas_query: unsafe token {banned!r}")

    used_vars = replay["used_vars"]
    if not isinstance(used_vars, list) or not used_vars:
        _fail(f"{where}.used_vars: expected non-empty list")
    replay_vars: dict[str, tuple[str, int]] = {}
    for index, item in enumerate(used_vars, 1):
        item_where = f"{where}.used_vars[{index}]"
        _exact_keys(item, USED_VAR_KEYS, item_where)
        var = str(item["var"])
        if not _DF_VAR.fullmatch(var) or var in replay_vars:
            _fail(f"{item_where}: duplicate or invalid variable")
        replay_vars[var] = (str(item["report_id"]), int(item["table_pos"]))
    if replay_vars != var_tables:
        _fail(f"{where}.used_vars: must exactly match evidence table bindings")
    try:
        refs = all_dataframe_refs(query)
    except SyntaxError as exc:
        raise P24ValidationError(f"{where}.pandas_query: invalid syntax") from exc
    if refs != set(replay_vars):
        _fail(
            f"{where}.pandas_query: dataframe refs {sorted(refs)} do not exactly "
            f"match used_vars {sorted(replay_vars)}"
        )

    expected = _finite_number(replay["expected_answer"], f"{where}.expected_answer")
    if not math.isclose(expected, output_value, rel_tol=0.0, abs_tol=1e-12):
        _fail(f"{where}.expected_answer: must exactly match output.value")
    tolerance = _finite_number(replay["tolerance"], f"{where}.tolerance")
    if tolerance <= 0 or tolerance > 1.0:
        _fail(f"{where}.tolerance: expected 0 < tolerance <= 1")
    if replay["evidence_sha256"] != canonical_sha256(evidence):
        _fail(f"{where}.evidence_sha256 mismatch")
    if replay["ast_sha256"] != canonical_sha256(ast_node):
        _fail(f"{where}.ast_sha256 mismatch")
    if table_loader is None:
        _fail(f"{where}: table loader is required for replay")
    for var, (report_id, table_pos) in replay_vars.items():
        if var not in dfs:
            dfs[var] = table_loader(report_id, table_pos)
    result = run_code(query, dfs, timeout=10)
    if result.get("status") != "ok":
        _fail(f"{where}: replay failed: {result.get('error')}")
    if not math.isclose(
        float(result["value"]), expected, rel_tol=0.0, abs_tol=tolerance
    ):
        _fail(
            f"{where}: replay value {result['value']} != expected {expected} "
            f"within {tolerance}"
        )


def validate_gold_records(
    records: list[dict],
    question_rows: list[dict],
    split: str,
    *,
    table_loader: Callable[[str, int], pd.DataFrame] | None = None,
    require_complete: bool = True,
) -> dict:
    """Validate exact identities and, for complete gold, cell/AST/replay truth."""
    _validate_question_records(question_rows, split)
    if len(records) != len(question_rows):
        _fail(
            f"{split} gold count {len(records)} != question count {len(question_rows)}"
        )
    qmap = {int(row["id"]): row for row in question_rows}
    seen: set[int] = set()
    for index, record in enumerate(records, 1):
        where = f"{split} gold line {index}"
        _exact_keys(record, GOLD_KEYS, where)
        qid = int(record["id"])
        if qid in seen:
            _fail(f"{where}: duplicate id {qid}")
        seen.add(qid)
        source = qmap.get(qid)
        if source is None:
            _fail(f"{where}: id {qid} is not in {split} split")
        if record["schema_version"] != GOLD_SCHEMA_VERSION:
            _fail(f"{where}: schema_version mismatch")
        for key in ("split", "question", "question_sha256", "stratum"):
            if record[key] != source[key]:
                _fail(f"{where}: identity field {key!r} differs from split")
        if not isinstance(record["annotator_notes"], str):
            _fail(f"{where}.annotator_notes: expected string")
        allowed_status = {"unlabeled", "draft", "verified"}
        if record["label_status"] not in allowed_status:
            _fail(f"{where}.label_status: invalid status")
        if require_complete and record["label_status"] != "verified":
            _fail(f"{where}: complete gold requires label_status='verified'")

        output_value = _validate_output(
            record["output"], f"{where}.output", require_complete
        )
        _exact_keys(record["replay"], REPLAY_KEYS, f"{where}.replay")
        if not require_complete:
            continue
        evidence_ids, var_tables, dfs = _validate_evidence(
            record["evidence"], f"{where}.evidence", table_loader, True
        )
        ast_node = record["ast"]
        if not isinstance(ast_node, dict) or ast_node.get("kind") != "op":
            _fail(f"{where}.ast: root must be an operation node")
        ast_refs = _validate_ast_node(ast_node, evidence_ids, f"{where}.ast")
        if ast_refs != evidence_ids:
            _fail(
                f"{where}.ast: evidence refs must exactly equal labeled evidence; "
                f"refs={sorted(ast_refs)}, evidence={sorted(evidence_ids)}"
            )
        assert output_value is not None
        _validate_replay(
            record["replay"], output_value, record["evidence"], ast_node,
            var_tables, dfs, table_loader, f"{where}.replay",
        )
    if seen != set(qmap):
        _fail(f"{split} gold ids differ from split question ids")
    return {
        "split": split,
        "count": len(records),
        "complete": bool(require_complete),
        "records_sha256": canonical_sha256(records),
    }


def validate_gold_file(
    gold_path: Path | str,
    bundle_dir: Path | str,
    split: str,
    *,
    store_dir: Path | str | None = None,
    table_loader: Callable[[str, int], pd.DataFrame] | None = None,
    require_complete: bool = True,
    verify_bundle: bool = True,
) -> dict:
    if split not in {"tune", "locked"}:
        _fail("split must be 'tune' or 'locked'")
    bundle_dir = Path(bundle_dir)
    if verify_bundle:
        validate_bundle(bundle_dir)
    question_name = TUNE_QUESTIONS if split == "tune" else LOCKED_QUESTIONS
    question_rows = _read_unique_jsonl(bundle_dir / question_name, f"{split} questions")
    records = _read_unique_jsonl(gold_path, f"{split} gold")
    if table_loader is None and store_dir is not None:
        table_loader = StoreTableLoader(store_dir)
    return validate_gold_records(
        records, question_rows, split,
        table_loader=table_loader, require_complete=require_complete,
    )


def check_tune_input(
    input_path: Path | str, bundle_dir: Path | str, *, verify_bundle: bool = True
) -> dict:
    """Reject locked or unknown records before using a file for tuning."""
    bundle_dir = Path(bundle_dir)
    if verify_bundle:
        validate_bundle(bundle_dir)
    tune = _read_unique_jsonl(bundle_dir / TUNE_QUESTIONS, "tune questions")
    locked = _read_unique_jsonl(bundle_dir / LOCKED_QUESTIONS, "locked questions")
    incoming = _read_unique_jsonl(input_path, "tune input")
    tune_map = {int(row["id"]): row for row in tune}
    locked_ids = {int(row["id"]) for row in locked}
    locked_hashes = {row["question_sha256"] for row in locked}
    leaked_ids: list[int] = []
    leaked_questions: list[int] = []
    unknown: list[int] = []
    for row in incoming:
        qid = int(row["id"])
        question = row.get("question")
        qhash = question_sha256(question) if isinstance(question, str) else None
        if qid in locked_ids:
            leaked_ids.append(qid)
        if qhash is not None and qhash in locked_hashes:
            leaked_questions.append(qid)
        if qid not in tune_map:
            unknown.append(qid)
        elif question is not None and question != tune_map[qid]["question"]:
            _fail(f"tune input id {qid}: question text differs from tune split")
    if leaked_ids or leaked_questions:
        _fail(
            f"locked-set leakage in tune input: ids={sorted(set(leaked_ids))[:10]}, "
            f"question_matches={sorted(set(leaked_questions))[:10]}"
        )
    if unknown:
        _fail(f"tune input contains ids outside tune split: {sorted(set(unknown))[:10]}")
    return {"count": len(incoming), "locked_overlap": 0, "unknown": 0}


def seal_locked_gold(
    gold_path: Path | str,
    bundle_dir: Path | str,
    seal_path: Path | str,
    *,
    store_dir: Path | str | None = None,
    table_loader: Callable[[str, int], pd.DataFrame] | None = None,
    verify_bundle: bool = True,
) -> dict:
    """Validate complete locked gold and write a deterministic tamper seal."""
    summary = validate_gold_file(
        gold_path, bundle_dir, "locked", store_dir=store_dir,
        table_loader=table_loader, require_complete=True,
        verify_bundle=verify_bundle,
    )
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / MANIFEST_NAME
    locked_questions_path = bundle_dir / LOCKED_QUESTIONS
    rows = _read_unique_jsonl(gold_path, "locked gold")
    seal = {
        "schema_version": SEAL_SCHEMA_VERSION,
        "manifest_sha256": file_sha256(manifest_path),
        "locked_questions_sha256": file_sha256(locked_questions_path),
        "locked_gold_sha256": file_sha256(gold_path),
        "locked_gold_canonical_sha256": canonical_sha256(rows),
        "locked_ids_sha256": canonical_sha256(sorted(int(row["id"]) for row in rows)),
        "record_count": summary["count"],
    }
    _write_json_new(seal_path, seal)
    return seal


def verify_locked_seal(
    gold_path: Path | str,
    bundle_dir: Path | str,
    seal_path: Path | str,
    *,
    store_dir: Path | str | None = None,
    table_loader: Callable[[str, int], pd.DataFrame] | None = None,
    verify_bundle: bool = True,
) -> dict:
    seal = read_json(seal_path)
    if seal.get("schema_version") != SEAL_SCHEMA_VERSION:
        _fail("locked seal schema_version mismatch")
    bundle_dir = Path(bundle_dir)
    checks = {
        "manifest_sha256": file_sha256(bundle_dir / MANIFEST_NAME),
        "locked_questions_sha256": file_sha256(bundle_dir / LOCKED_QUESTIONS),
        "locked_gold_sha256": file_sha256(gold_path),
    }
    for key, actual in checks.items():
        if seal.get(key) != actual:
            _fail(f"locked seal mismatch: {key}")
    rows = _read_unique_jsonl(gold_path, "locked gold")
    if seal.get("locked_gold_canonical_sha256") != canonical_sha256(rows):
        _fail("locked seal mismatch: canonical gold hash")
    if seal.get("locked_ids_sha256") != canonical_sha256(
        sorted(int(row["id"]) for row in rows)
    ):
        _fail("locked seal mismatch: id hash")
    if int(seal.get("record_count", -1)) != len(rows):
        _fail("locked seal mismatch: record count")
    return validate_gold_file(
        gold_path, bundle_dir, "locked", store_dir=store_dir,
        table_loader=table_loader, require_complete=True,
        verify_bundle=verify_bundle,
    )
