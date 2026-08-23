"""Build and validate the immutable G3A extension plus G3B corpus."""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ..codegen.executor import run_code
from ..codegen.selection_v2 import compile_program
from ..finance.operators import operator_registry_fingerprint
from .common import (
    canonical_sha256,
    file_sha256,
    normalize_question,
    read_jsonl,
    tree_sha256,
    write_json,
    write_jsonl,
)
from .generate import build_candidates
from .source import Fact, load_source_facts
from .views import build_views, validate_views

SCHEMA_VERSION = "g3b_corpus_v1"
EXTENSION_SCHEMA = "g3a_extension_v1"
BUILDER_VERSION = "g3b_source_program_first_v1"
MANIFEST_NAME = "g3b_manifest.json"
CORPUS_NAME = "g3b_corpus.jsonl"
QUESTIONS_NAME = "g3b_questions.jsonl"
DEV_QUESTIONS_NAME = "g3b_dev_questions.jsonl"
PROMOTION_QUESTIONS_NAME = "g3b_promotion_questions.jsonl"
ORACLE_NAME = "g3b_oracle_predictions.jsonl"
REVIEW_QUEUE_NAME = "g3b_review_queue.jsonl"
REVIEW_LEDGER_NAME = "g3b_reviews.jsonl"
EXT_MANIFEST_NAME = "g3a_extension_manifest.json"
EXT_GOLD_NAME = "g3a_extension_gold.jsonl"
EXT_QUESTIONS_NAME = "g3a_extension_questions.jsonl"


class G3BValidationError(ValueError):
    """Raised when corpus, review, or leakage invariants fail."""


def _choose_splits(candidates: list[dict], config: dict) -> dict:
    randomizer = random.Random(int(config["seed"]))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in candidates:
        grouped[item["family"]].append(item)
    for values in grouped.values():
        randomizer.shuffle(values)

    selected: dict[str, list[dict]] = defaultdict(list)
    used_facts: set[str] = set()
    for split in ("hard", "primary_locked", "primary_tune"):
        for family, quota in config["allocation"][split].items():
            taken = 0
            for item in grouped.get(family, []):
                if used_facts.intersection(item["fact_ids"]):
                    continue
                selected[split].append(item)
                used_facts.update(item["fact_ids"])
                taken += 1
                if taken == int(quota):
                    break
            if taken != int(quota):
                raise G3BValidationError(
                    f"cannot fill {split}/{family}: "
                    f"requested={quota}, found={taken}, "
                    f"pool={len(grouped.get(family, []))}"
                )
    return selected


def _candidate_objects(
    facts: list[Fact],
) -> tuple[list[SimpleNamespace], list[dict]]:
    table_variables = {}
    used_variables = []
    candidates = []
    for index, fact in enumerate(facts, 1):
        key = (fact.report_id, fact.table_pos)
        if key not in table_variables:
            variable = f"df{len(table_variables) + 1}"
            table_variables[key] = variable
            used_variables.append({
                "var": variable,
                "report_id": fact.report_id,
                "table_pos": fact.table_pos,
            })
        candidates.append(SimpleNamespace(
            var=table_variables[key],
            row=fact.row,
            col=fact.col,
            label=fact.label,
            code=fact.row_code,
            col_name=fact.col_name,
            value=fact.value,
            unit_scale=fact.unit_scale,
            score=100.0,
            rescue=False,
            fact_year=fact.period_year,
            report_year=fact.report_year,
            fact_slot=f"F{index}",
            fact_role="value",
            fact_metric=fact.metric_key,
            ticker=fact.ticker,
            report_id=fact.report_id,
            table_pos=fact.table_pos,
            metric_grounded=True,
        ))
    return candidates, used_variables


def _atomic_facts(facts: list[Fact], typed_program: dict) -> list[dict]:
    by_reference = {
        int(spec["ref"]): spec
        for spec in typed_program["facts"].values()
    }
    return [
        {
            "ticker": fact.ticker,
            "year": fact.period_year,
            "metric": fact.metric_key,
            "role": by_reference[index]["role"],
            "family": "g3b_source_fact",
        }
        for index, fact in enumerate(facts, 1)
    ]


def _compile_oracle(item: dict) -> tuple[list[dict], str, float]:
    candidates, used_variables = _candidate_objects(item["facts"])
    route = {
        "output_type": item["output_type"],
        "unit_scale": item["unit_scale"],
        "years": item["years"],
        "plan": {
            "op": (
                "ranking"
                if item["family"].startswith("ranking_")
                else item["family"]
            )
        },
    }
    compiled = compile_program(
        item["program"],
        candidates,
        route,
        item["question"],
        atomic_facts=_atomic_facts(item["facts"], item["program"]),
    )
    rows_by_variable: dict[str, list[dict]] = defaultdict(list)
    for fact, candidate in zip(item["facts"], candidates):
        rows_by_variable[candidate.var].append({
            "row": fact.row,
            "col": fact.col,
            "value": fact.value,
            "unit_scale": fact.unit_scale,
            "label": fact.label,
            "col_name": fact.col_name,
        })
    result = run_code(
        compiled.query,
        {
            variable: pd.DataFrame(rows)
            for variable, rows in rows_by_variable.items()
        },
    )
    if result.get("status") != "ok":
        raise G3BValidationError(
            f"gold program failed: {item['family']} {result}"
        )
    value = float(result["value"])
    if not math.isclose(
        value,
        float(item["answer"]),
        rel_tol=0.0,
        abs_tol=float(item["tolerance"]),
    ):
        raise G3BValidationError(
            f"gold mismatch for {item['family']}: "
            f"compiled={value} expected={item['answer']}"
        )
    return used_variables, compiled.query, value


def review_subject(record: dict) -> str:
    return canonical_sha256({
        key: record[key]
        for key in (
            "id",
            "question",
            "family",
            "answer",
            "output_type",
            "tolerance",
            "relevant_docs",
            "relevant_tables",
            "evidence",
            "leaf_specs",
            "typed_program",
        )
    })


def _load_reviews(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    output = {}
    for row in read_jsonl(path):
        subject = str(row.get("subject_sha256") or "")
        if not subject or subject in output:
            raise G3BValidationError(
                "review ledger has blank or duplicate subject"
            )
        output[subject] = row
    return output


def _question_record(
    question_id: int,
    split: str,
    item: dict,
) -> dict:
    return {
        "id": question_id,
        "question": item["question"],
        "split": split,
        "set": "hard" if split == "hard" else "primary",
        "family": item["family"],
        "operator": item["operator"],
        "metric_family": item["metric_family"],
        "output_type": item["output_type"],
        "tree_shape": item["tree_shape"],
        "stress_tags": item["stress_tags"],
    }


def _leaf_specs(item: dict) -> list[dict]:
    by_reference = {
        int(spec["ref"]): (name, spec)
        for name, spec in item["program"]["facts"].items()
    }
    output = []
    for index, fact in enumerate(item["facts"], 1):
        name, spec = by_reference[index]
        output.append({
            "slot": f"F{index}",
            "name": name,
            "role": spec["role"],
            "as": spec["as"],
            "fact_id": fact.fact_id,
            "ticker": fact.ticker,
            "year": fact.period_year,
            "report_year": fact.report_year,
            "scope": fact.doc_type,
            "metric_key": fact.metric_key,
            "report_id": fact.report_id,
            "table_ref": f"{fact.report_id}|{fact.table_line}",
            "row": fact.row,
            "col": fact.col,
        })
    return output


def _cross_split_facts(records: list[dict]) -> dict:
    fact_sets = {
        split: {
            fact_id
            for row in records
            if row["split"] == split
            for fact_id in row["fact_ids"]
        }
        for split in ("primary_tune", "primary_locked", "hard")
    }
    output = {}
    names = list(fact_sets)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            output[f"{left}__{right}"] = sorted(
                fact_sets[left] & fact_sets[right]
            )
    return output


def _manifest(
    *,
    schema_version: str,
    source: dict,
    counts: dict,
    leakage: dict,
    root: Path,
    files: list[str],
    extra: dict | None = None,
) -> dict:
    output = {
        "schema_version": schema_version,
        "builder_version": BUILDER_VERSION,
        "source": source,
        "counts": counts,
        "leakage_guard": leakage,
        **(extra or {}),
        "files": {
            name: {"sha256": file_sha256(root / name)}
            for name in files
        },
    }
    output["fingerprint_sha256"] = canonical_sha256(output)
    return output


def build_corpus(
    store_dir: Path | str,
    public_questions_path: Path | str,
    config_path: Path | str,
    extension_dir: Path | str,
    corpus_dir: Path | str,
    *,
    review_path: Path | str | None = None,
    g3a_v1_dir: Path | str = "data/g3a_v1",
) -> dict:
    store_dir = Path(store_dir)
    public_questions_path = Path(public_questions_path)
    config_path = Path(config_path)
    extension_dir = Path(extension_dir)
    corpus_dir = Path(corpus_dir)
    g3a_v1_dir = Path(g3a_v1_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    before = tree_sha256(g3a_v1_dir)
    if before != config["g3a_v1_tree_sha256"]:
        raise G3BValidationError(
            f"G3A v1 immutable tree mismatch: {before}"
        )

    candidates = build_candidates(load_source_facts(store_dir))
    splits = _choose_splits(candidates, config)
    review_file = Path(review_path) if review_path else None
    reviews = _load_reviews(review_file)
    public_rows = read_jsonl(public_questions_path)
    public_text = {
        normalize_question(row["question"]) for row in public_rows
    }
    public_ids = {int(row["id"]) for row in public_rows}
    id_bases = {
        "primary_tune": 3_400_000,
        "primary_locked": 3_500_000,
        "hard": 3_600_000,
    }

    staged = []
    for split in ("primary_tune", "primary_locked", "hard"):
        ordered = sorted(
            splits[split],
            key=lambda item: (
                item["family"],
                item["program_key"],
            ),
        )
        for offset, item in enumerate(ordered, 1):
            staged.append((
                _question_record(
                    id_bases[split] + offset,
                    split,
                    item,
                ),
                item,
            ))

    required_families = set(
        config["review"]["required_families"]
    )
    optional_ids = [
        question["id"]
        for question, item in staged
        if question["split"] != "hard"
        and item["family"] not in required_families
    ]
    randomizer = random.Random(int(config["seed"]) + 91)
    audit_ids = set(randomizer.sample(
        optional_ids,
        min(
            int(config["review"]["primary_random_audit"]),
            len(optional_ids),
        ),
    ))

    records = []
    questions = []
    oracle_rows = []
    review_queue = []
    for question, item in staged:
        evidence = [asdict(fact) for fact in item["facts"]]
        leaves = _leaf_specs(item)
        record = {
            **question,
            "answer": item["answer"],
            "tolerance": item["tolerance"],
            "unit_scale": item["unit_scale"],
            "relevant_docs": item["relevant_docs"],
            "relevant_tables": item["relevant_tables"],
            "evidence": evidence,
            "atomic_facts": _atomic_facts(
                item["facts"], item["program"]
            ),
            "leaf_specs": leaves,
            "typed_program": item["program"],
            "fact_ids": item["fact_ids"],
            "fact_group": item["fact_group"],
            "program_key": item["program_key"],
            "tickers": item["tickers"],
            "years": item["years"],
            "report_ids": item["report_ids"],
            "scopes": item["scopes"],
            "tree_shape_value": item["tree_shape_value"],
            "primitive_ops": item["primitive_ops"],
        }
        review_required = (
            question["split"] == "hard"
            or item["family"] in required_families
            or question["id"] in audit_ids
        )
        subject = review_subject(record)
        review = reviews.get(subject)
        approved = bool(
            review and review.get("status") == "approved"
        )
        record["review"] = {
            "required": review_required,
            "status": (
                "approved"
                if review_required and approved
                else "pending"
                if review_required
                else "not_required"
            ),
            "subject_sha256": subject if review_required else None,
            "reviewer": (
                review.get("reviewer") if approved else None
            ),
            "review_method": (
                review.get("method") if approved else None
            ),
        }
        if review_required:
            review_queue.append({
                "id": question["id"],
                "split": question["split"],
                "family": item["family"],
                "reason": (
                    "hard"
                    if question["split"] == "hard"
                    else "family_required"
                    if item["family"] in required_families
                    else "primary_random_audit"
                ),
                "subject_sha256": subject,
                "question": question["question"],
                "answer": item["answer"],
                "evidence": evidence,
                "leaf_specs": leaves,
                "typed_program": item["program"],
            })
        used_variables, query, value = _compile_oracle(item)
        oracle_rows.append({
            "id": question["id"],
            "question": question["question"],
            "program": item["program"],
            "candidate_fact_ids": item["fact_ids"],
            "used_vars": used_variables,
            "pandas_query": query,
            "answer": value,
            "source": "g3b_gold_typed_oracle",
        })
        records.append(record)
        questions.append(question)

    text_overlap = [
        int(row["id"])
        for row in questions
        if normalize_question(row["question"]) in public_text
    ]
    id_overlap = sorted(
        {int(row["id"]) for row in questions} & public_ids
    )
    if text_overlap or id_overlap:
        raise G3BValidationError(
            f"public leakage: text={text_overlap}, ids={id_overlap}"
        )
    cross_split = _cross_split_facts(records)
    if any(cross_split.values()):
        raise G3BValidationError(
            f"cross-split fact leakage: {cross_split}"
        )

    extension_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "views").mkdir(
        parents=True, exist_ok=True
    )
    write_jsonl(
        extension_dir / EXT_QUESTIONS_NAME,
        questions,
    )
    write_jsonl(extension_dir / EXT_GOLD_NAME, records)
    write_jsonl(corpus_dir / QUESTIONS_NAME, questions)
    write_jsonl(
        corpus_dir / DEV_QUESTIONS_NAME,
        [
            row
            for row in questions
            if row["split"] == "primary_tune"
        ],
    )
    write_jsonl(
        corpus_dir / PROMOTION_QUESTIONS_NAME,
        [
            row
            for row in questions
            if row["split"] in {"primary_locked", "hard"}
        ],
    )
    write_jsonl(corpus_dir / CORPUS_NAME, records)
    write_jsonl(corpus_dir / ORACLE_NAME, oracle_rows)
    write_jsonl(
        corpus_dir / REVIEW_QUEUE_NAME,
        review_queue,
    )
    ledger_path = corpus_dir / REVIEW_LEDGER_NAME
    if (
        review_file
        and review_file.exists()
        and review_file.resolve() != ledger_path.resolve()
    ):
        write_jsonl(ledger_path, read_jsonl(review_file))

    views = build_views(records, config)
    for name, document in views.items():
        write_json(
            corpus_dir / "views" / f"{name}.json",
            document,
        )

    source = {
        "g3a_v1_tree_sha256": before,
        "store_reports_sha256": file_sha256(
            store_dir / "reports.parquet"
        ),
        "public_questions_sha256": file_sha256(
            public_questions_path
        ),
        "public_questions_usage": config[
            "public_question_use"
        ],
        "config_sha256": file_sha256(config_path),
    }
    leakage = {
        "public_exact_text_overlap": text_overlap,
        "public_id_overlap": id_overlap,
        "cross_split_fact_overlap": cross_split,
    }
    counts = {
        "questions": len(records),
        "by_split": dict(Counter(
            row["split"] for row in records
        )),
        "by_family": dict(Counter(
            row["family"] for row in records
        )),
        "by_output_type": dict(Counter(
            row["output_type"] for row in records
        )),
    }
    extension_manifest = _manifest(
        schema_version=EXTENSION_SCHEMA,
        source=source,
        counts=counts,
        leakage=leakage,
        root=extension_dir,
        files=[EXT_QUESTIONS_NAME, EXT_GOLD_NAME],
    )
    write_json(
        extension_dir / EXT_MANIFEST_NAME,
        extension_manifest,
    )

    corpus_files = [
        QUESTIONS_NAME,
        DEV_QUESTIONS_NAME,
        PROMOTION_QUESTIONS_NAME,
        CORPUS_NAME,
        ORACLE_NAME,
        REVIEW_QUEUE_NAME,
        *(
            f"views/{name}.json"
            for name in views
        ),
    ]
    if ledger_path.exists():
        corpus_files.append(REVIEW_LEDGER_NAME)
    corpus_source = {
        **source,
        "g3a_extension_manifest_sha256": file_sha256(
            extension_dir / EXT_MANIFEST_NAME
        ),
        "review_ledger_sha256": (
            file_sha256(ledger_path)
            if ledger_path.exists()
            else None
        ),
    }
    corpus_counts = {
        **counts,
        "reviews_required": sum(
            row["review"]["required"] for row in records
        ),
        "reviews_approved": sum(
            row["review"]["status"] == "approved"
            for row in records
        ),
        "reviews_pending": sum(
            row["review"]["status"] == "pending"
            for row in records
        ),
    }
    corpus_manifest = _manifest(
        schema_version=SCHEMA_VERSION,
        source=corpus_source,
        counts=corpus_counts,
        leakage={
            **leakage,
            "g3a_v1_unchanged_after_build": (
                tree_sha256(g3a_v1_dir) == before
            ),
        },
        root=corpus_dir,
        files=corpus_files,
        extra={
            "semantic_contract": {
                "selection_v2_schema_version": 2,
                "operator_registry_sha256": (
                    operator_registry_fingerprint()
                ),
                "ir_policy": (
                    "reuse facts/bindings/root; no third IR"
                ),
            },
            "view_hashes": {
                name: file_sha256(
                    corpus_dir / "views" / f"{name}.json"
                )
                for name in views
            },
        },
    )
    write_json(
        corpus_dir / MANIFEST_NAME,
        corpus_manifest,
    )
    if tree_sha256(g3a_v1_dir) != before:
        raise G3BValidationError(
            "G3A v1 changed during G3B build"
        )
    return {
        "g3a_extension": extension_manifest,
        "g3b": corpus_manifest,
        "candidate_pool": dict(Counter(
            item["family"] for item in candidates
        )),
    }


def _validate_manifest(
    root: Path,
    name: str,
    schema: str,
) -> dict:
    manifest = json.loads(
        (root / name).read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != schema:
        raise G3BValidationError(
            f"unexpected schema: {root / name}"
        )
    fingerprint = manifest.get("fingerprint_sha256")
    payload = dict(manifest)
    payload.pop("fingerprint_sha256", None)
    if (
        not fingerprint
        or canonical_sha256(payload) != fingerprint
    ):
        raise G3BValidationError(
            f"manifest fingerprint mismatch: {root / name}"
        )
    for relative, expected in manifest["files"].items():
        path = root / relative
        if (
            not path.exists()
            or file_sha256(path) != expected["sha256"]
        ):
            raise G3BValidationError(
                f"file hash mismatch: {path}"
            )
    return manifest


def validate_corpus(
    extension_dir: Path | str,
    corpus_dir: Path | str,
    config_path: Path | str,
    *,
    require_reviews: bool = True,
    g3a_v1_dir: Path | str = "data/g3a_v1",
) -> dict:
    extension_dir = Path(extension_dir)
    corpus_dir = Path(corpus_dir)
    config = json.loads(
        Path(config_path).read_text(encoding="utf-8")
    )
    g3a_hash = tree_sha256(Path(g3a_v1_dir))
    if g3a_hash != config["g3a_v1_tree_sha256"]:
        raise G3BValidationError(
            "G3A v1 immutable tree mismatch"
        )
    extension = _validate_manifest(
        extension_dir,
        EXT_MANIFEST_NAME,
        EXTENSION_SCHEMA,
    )
    manifest = _validate_manifest(
        corpus_dir,
        MANIFEST_NAME,
        SCHEMA_VERSION,
    )
    records = read_jsonl(corpus_dir / CORPUS_NAME)
    if (
        not records
        or len(records) != manifest["counts"]["questions"]
    ):
        raise G3BValidationError(
            "empty corpus or count mismatch"
        )
    pending = [
        int(row["id"])
        for row in records
        if row["review"]["required"]
        and row["review"]["status"] != "approved"
    ]
    if require_reviews and pending:
        raise G3BValidationError(
            f"required reviews pending: {pending[:10]}"
        )
    if any(
        manifest["leakage_guard"][
            "cross_split_fact_overlap"
        ].values()
    ):
        raise G3BValidationError(
            "cross-split fact leakage"
        )
    if (
        manifest["source"][
            "g3a_extension_manifest_sha256"
        ]
        != file_sha256(
            extension_dir / EXT_MANIFEST_NAME
        )
    ):
        raise G3BValidationError(
            "extension/G3B provenance mismatch"
        )
    view_documents = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(
            (corpus_dir / "views").glob("*.json")
        )
    ]
    try:
        validate_views(view_documents)
    except ValueError as exc:
        raise G3BValidationError(str(exc)) from exc
    return {
        "valid": True,
        "g3a_v1_tree_sha256": g3a_hash,
        "g3a_extension_fingerprint_sha256": (
            extension["fingerprint_sha256"]
        ),
        "g3b_fingerprint_sha256": (
            manifest["fingerprint_sha256"]
        ),
        "questions": len(records),
        "reviews_pending": len(pending),
    }
