"""Build the G3A same-corpus/new-question bundle.

The generator is program-first and reads only source tables. Public questions
are used solely as an exclusion set; their wording never seeds a template.
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from ..extraction.build_store import Store
from ..utils.viet_text import fuzz_token_set
from .common import (
    canonical_sha256,
    file_sha256,
    normalize_question,
    read_jsonl,
    write_json,
    write_jsonl,
)

SCHEMA_VERSION = "g3a_bundle_v1"
BUILDER_VERSION = "g3a_program_first_v1"
MANIFEST_NAME = "g3a_manifest.json"
QUESTIONS_NAME = "g3a_questions.jsonl"
GOLD_NAME = "g3a_gold.jsonl"
ORACLE_NAME = "g3a_oracle_codegen.jsonl"
REVIEW_QUEUE_NAME = "g3a_hard_review_queue.jsonl"

METRICS = {
    "10": ("doanh thu thu\u1ea7n", "Doanh thu thu\u1ea7n v\u1ec1 b\u00e1n h\u00e0ng v\u00e0 cung c\u1ea5p d\u1ecbch v\u1ee5"),
    "11": ("gi\u00e1 v\u1ed1n h\u00e0ng b\u00e1n", "Gi\u00e1 v\u1ed1n h\u00e0ng b\u00e1n"),
    "20": ("l\u1ee3i nhu\u1eadn g\u1ed9p", "L\u1ee3i nhu\u1eadn g\u1ed9p v\u1ec1 b\u00e1n h\u00e0ng v\u00e0 cung c\u1ea5p d\u1ecbch v\u1ee5"),
    "50": ("l\u1ee3i nhu\u1eadn tr\u01b0\u1edbc thu\u1ebf", "T\u1ed5ng l\u1ee3i nhu\u1eadn k\u1ebf to\u00e1n tr\u01b0\u1edbc thu\u1ebf"),
    "60": ("l\u1ee3i nhu\u1eadn sau thu\u1ebf", "L\u1ee3i nhu\u1eadn sau thu\u1ebf thu nh\u1eadp doanh nghi\u1ec7p"),
    "270": ("t\u1ed5ng t\u00e0i s\u1ea3n", "T\u1ed4NG C\u1ed8NG T\u00c0I S\u1ea2N"),
    "300": ("n\u1ee3 ph\u1ea3i tr\u1ea3", "N\u1ee2 PH\u1ea2I TR\u1ea2"),
    "400": ("v\u1ed1n ch\u1ee7 s\u1edf h\u1eefu", "V\u1ed0N CH\u1ee6 S\u1ede H\u1eeeU"),
}


class G3AValidationError(ValueError):
    """Raised when a G3A bundle violates a provenance or leakage guard."""


@dataclass(frozen=True)
class Fact:
    fact_id: str
    ticker: str
    year: int
    doc_type: str
    report_id: str
    table_pos: int
    table_line: int
    row: int
    col: int
    row_code: str
    label: str
    col_name: str
    value: float
    unit_scale: float
    base_value: float


def _row_code(value: object) -> str:
    raw = str(value or "").strip()
    return raw[:-2] if raw.endswith(".0") else raw


def _progress(values: Iterable[str]) -> Iterable[str]:
    try:
        from tqdm import tqdm
        return tqdm(values, desc="G3A source facts")
    except ImportError:
        return values


def _load_facts(store_dir: Path) -> list[Fact]:
    store = Store(store_dir, cache_size=4)
    tickers = sorted({str(value) for value in store.reports.ticker})
    facts: list[Fact] = []
    for ticker in _progress(tickers):
        cells = store.cells_of(ticker)
        tables = store.tables_of(ticker)
        if not len(cells) or not len(tables):
            continue
        lines = {
            (str(row.report_id), int(row.table_pos)): int(row.line_no)
            for row in tables.itertuples()
        }
        best: dict[tuple[str, str], tuple[float, Fact]] = {}
        for cell in cells.itertuples():
            code = _row_code(cell.row_code)
            if code not in METRICS or not bool(cell.unit_known):
                continue
            year = int(cell.year)
            if not re.search(rf"(?<!\d){year}(?!\d)", str(cell.col_name)):
                continue
            value, scale = float(cell.value), float(cell.unit_scale)
            if (
                not math.isfinite(value)
                or not math.isfinite(scale)
                or scale <= 0
                or abs(value * scale) < 1.0
            ):
                continue
            score = float(fuzz_token_set(str(cell.label), METRICS[code][1]))
            if score < 85.0:
                continue
            key = (str(cell.report_id), code)
            table_key = (str(cell.report_id), int(cell.table_pos))
            if table_key not in lines:
                continue
            fact = Fact(
                fact_id=(
                    f"{cell.report_id}|{int(cell.table_pos)}|{int(cell.row)}|"
                    f"{int(cell.col)}|{code}"
                ),
                ticker=str(cell.ticker),
                year=year,
                doc_type=str(cell.doc_type),
                report_id=str(cell.report_id),
                table_pos=int(cell.table_pos),
                table_line=lines[table_key],
                row=int(cell.row),
                col=int(cell.col),
                row_code=code,
                label=str(cell.label),
                col_name=str(cell.col_name),
                value=value,
                unit_scale=scale,
                base_value=value * scale,
            )
            rank = score - 0.001 * fact.table_pos
            if key not in best or rank > best[key][0]:
                best[key] = (rank, fact)
        facts.extend(item[1] for item in best.values())
    return sorted(facts, key=lambda fact: fact.fact_id)


def _scope(doc_type: str) -> str:
    return {
        "consolidated": "h\u1ee3p nh\u1ea5t",
        "separate": "ri\u00eang c\u1ee7a c\u00f4ng ty m\u1eb9",
    }.get(doc_type, doc_type)


def _candidate(
    operator: str,
    question: str,
    facts: list[Fact],
    answer: float,
    output_type: str,
    template_family: str,
) -> dict:
    fact_ids = sorted(fact.fact_id for fact in facts)
    documents = list(dict.fromkeys(fact.report_id for fact in facts))
    tables = list(dict.fromkeys(
        f"{fact.report_id}|{fact.table_line}" for fact in facts
    ))
    return {
        "operator": operator,
        "question": question,
        "facts": facts,
        "fact_ids": fact_ids,
        "fact_group": canonical_sha256(fact_ids),
        "program_key": canonical_sha256({"operator": operator, "facts": fact_ids}),
        "answer": round(float(answer), 2),
        "output_type": output_type,
        "tolerance": 0.01,
        "relevant_docs": documents,
        "relevant_tables": tables,
        "template_family": template_family,
        "difficulty": (
            "single_cell" if len(facts) == 1
            else "multi_document" if len(documents) > 1
            else "multi_leaf"
        ),
    }


def _build_candidates(facts: list[Fact]) -> list[dict]:
    candidates: list[dict] = []
    by_report: dict[str, dict[str, Fact]] = defaultdict(dict)
    by_series: dict[tuple[str, str, str], list[Fact]] = defaultdict(list)
    for fact in facts:
        by_report[fact.report_id][fact.row_code] = fact
        by_series[(fact.ticker, fact.doc_type, fact.row_code)].append(fact)
        metric = METRICS[fact.row_code][0]
        candidates.append(_candidate(
            "lookup",
            f"Theo b\u00e1o c\u00e1o t\u00e0i ch\u00ednh {_scope(fact.doc_type)} n\u0103m {fact.year} "
            f"c\u1ee7a {fact.ticker}, {metric} l\u00e0 bao nhi\u00eau tri\u1ec7u \u0111\u1ed3ng?",
            [fact],
            fact.base_value / 1e6,
            "money_million_vnd",
            "source_lookup_v1",
        ))

    for values in by_report.values():
        revenue, profit = values.get("10"), values.get("60")
        if revenue and profit and abs(revenue.base_value) > 1.0:
            candidates.append(_candidate(
                "net_margin",
                f"Trong n\u0103m {profit.year}, bi\u00ean l\u1ee3i nhu\u1eadn sau thu\u1ebf tr\u00ean doanh "
                f"thu thu\u1ea7n c\u1ee7a {profit.ticker} theo b\u00e1o c\u00e1o "
                f"{_scope(profit.doc_type)} l\u00e0 bao nhi\u00eau ph\u1ea7n tr\u0103m?",
                [profit, revenue],
                profit.base_value / revenue.base_value * 100.0,
                "percent",
                "source_ratio_v1",
            ))
        liabilities, assets = values.get("300"), values.get("270")
        if liabilities and assets and abs(assets.base_value) > 1.0:
            candidates.append(_candidate(
                "debt_to_assets",
                f"N\u1ee3 ph\u1ea3i tr\u1ea3 chi\u1ebfm bao nhi\u00eau ph\u1ea7n tr\u0103m t\u1ed5ng t\u00e0i s\u1ea3n c\u1ee7a "
                f"{assets.ticker} trong b\u00e1o c\u00e1o {_scope(assets.doc_type)} "
                f"n\u0103m {assets.year}?",
                [liabilities, assets],
                liabilities.base_value / assets.base_value * 100.0,
                "percent",
                "source_ratio_v1",
            ))

    for (_ticker, _doc_type, code), series in sorted(by_series.items()):
        ordered = sorted(series, key=lambda fact: (fact.year, fact.report_id))
        for previous, current in zip(ordered, ordered[1:]):
            if current.year - previous.year != 1:
                continue
            common = (
                f"{METRICS[code][0]} c\u1ee7a {current.ticker} theo b\u00e1o c\u00e1o "
                f"{_scope(current.doc_type)}"
            )
            candidates.append(_candidate(
                "difference",
                f"Ch\u00eanh l\u1ec7ch {common} n\u0103m {current.year} so v\u1edbi n\u0103m "
                f"{previous.year} l\u00e0 bao nhi\u00eau tri\u1ec7u \u0111\u1ed3ng?",
                [current, previous],
                (current.base_value - previous.base_value) / 1e6,
                "money_million_vnd",
                "source_change_v1",
            ))
            candidates.append(_candidate(
                "average",
                f"Gi\u00e1 tr\u1ecb b\u00ecnh qu\u00e2n {common} trong hai n\u0103m {previous.year} v\u00e0 "
                f"{current.year} l\u00e0 bao nhi\u00eau tri\u1ec7u \u0111\u1ed3ng?",
                [previous, current],
                (previous.base_value + current.base_value) / 2.0 / 1e6,
                "money_million_vnd",
                "source_average_v1",
            ))
            if abs(previous.base_value) > 1.0:
                candidates.append(_candidate(
                    "growth_pct",
                    f"T\u1ed1c \u0111\u1ed9 t\u0103ng ho\u1eb7c gi\u1ea3m {common} t\u1eeb n\u0103m {previous.year} "
                    f"\u0111\u1ebfn n\u0103m {current.year} l\u00e0 bao nhi\u00eau ph\u1ea7n tr\u0103m?",
                    [current, previous],
                    (
                        (current.base_value - previous.base_value)
                        / abs(previous.base_value)
                        * 100.0
                    ),
                    "percent",
                    "source_growth_v1",
                ))
    return candidates


def _choose_splits(candidates: list[dict], config: dict) -> dict[str, list[dict]]:
    rng = random.Random(int(config["seed"]))
    by_operator: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        by_operator[candidate["operator"]].append(candidate)
    for values in by_operator.values():
        rng.shuffle(values)

    selected: dict[str, list[dict]] = defaultdict(list)
    used_facts: set[str] = set()
    for split in ("hard", "primary_locked", "primary_tune"):
        for operator, quota in config["allocation"][split].items():
            taken = 0
            for candidate in by_operator.get(operator, []):
                if used_facts.intersection(candidate["fact_ids"]):
                    continue
                selected[split].append(candidate)
                used_facts.update(candidate["fact_ids"])
                taken += 1
                if taken == int(quota):
                    break
            if taken != int(quota):
                raise G3AValidationError(
                    f"cannot fill {split}/{operator}: "
                    f"requested={quota}, found={taken}"
                )
    return selected


def _oracle_query(candidate: dict) -> tuple[list[dict], str]:
    table_vars: dict[tuple[str, int], str] = {}
    used_vars: list[dict] = []
    expressions: list[str] = []
    for fact in candidate["facts"]:
        key = (fact.report_id, fact.table_pos)
        if key not in table_vars:
            variable = f"df{len(table_vars) + 1}"
            table_vars[key] = variable
            used_vars.append({
                "var": variable,
                "report_id": fact.report_id,
                "table_pos": fact.table_pos,
            })
        variable = table_vars[key]
        expressions.append(
            f"(float({variable}.loc[({variable}['row']=={fact.row}) & "
            f"({variable}['col']=={fact.col}), 'value'].iloc[0]) * "
            f"float({variable}.loc[({variable}['row']=={fact.row}) & "
            f"({variable}['col']=={fact.col}), 'unit_scale'].iloc[0]))"
        )
    left = expressions[0]
    right = expressions[1] if len(expressions) > 1 else ""
    operator = candidate["operator"]
    if operator == "lookup":
        query = f"({left}) / 1000000.0"
    elif operator == "difference":
        query = f"(({left}) - ({right})) / 1000000.0"
    elif operator == "average":
        query = f"(({left}) + ({right})) / 2.0 / 1000000.0"
    elif operator == "growth_pct":
        query = f"((({left}) - ({right})) / abs({right})) * 100.0"
    elif operator in {"net_margin", "debt_to_assets"}:
        query = f"(({left}) / ({right})) * 100.0"
    else:
        raise G3AValidationError(f"unsupported operator: {operator}")
    return used_vars, query


def _review_subject(gold: dict) -> str:
    return canonical_sha256({
        key: gold[key]
        for key in (
            "id",
            "question",
            "operator",
            "answer",
            "output_type",
            "tolerance",
            "relevant_docs",
            "relevant_tables",
            "evidence",
            "program",
        )
    })


def _load_reviews(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    reviews: dict[str, dict] = {}
    for row in read_jsonl(path):
        subject = str(row.get("subject_sha256") or "")
        if not subject or subject in reviews:
            raise G3AValidationError(
                "hard review ledger has blank/duplicate subject hash"
            )
        reviews[subject] = row
    return reviews


def build_bundle(
    store_dir: Path | str,
    public_questions_path: Path | str,
    config_path: Path | str,
    out_dir: Path | str,
    *,
    review_path: Path | str | None = None,
) -> dict:
    store_dir = Path(store_dir)
    public_questions_path = Path(public_questions_path)
    config_path = Path(config_path)
    out_dir = Path(out_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    splits = _choose_splits(
        _build_candidates(_load_facts(store_dir)),
        config,
    )
    review_file = Path(review_path) if review_path else None
    reviews = _load_reviews(review_file)
    public_rows = read_jsonl(public_questions_path)
    public_text = {normalize_question(row["question"]) for row in public_rows}
    public_ids = {int(row["id"]) for row in public_rows}

    questions: list[dict] = []
    gold_rows: list[dict] = []
    oracle_rows: list[dict] = []
    review_queue: list[dict] = []
    id_bases = {
        "primary_tune": 3_100_000,
        "primary_locked": 3_200_000,
        "hard": 3_300_000,
    }
    for split in ("primary_tune", "primary_locked", "hard"):
        ordered = sorted(
            splits[split],
            key=lambda row: (row["operator"], row["program_key"]),
        )
        for offset, candidate in enumerate(ordered, 1):
            qid = id_bases[split] + offset
            question = {
                "id": qid,
                "question": candidate["question"],
                "split": split,
                "set": "hard" if split == "hard" else "primary",
                "stratum": (
                    f"{candidate['operator']}|{candidate['difficulty']}"
                ),
                "operator": candidate["operator"],
                "difficulty": candidate["difficulty"],
                "fact_group": candidate["fact_group"],
                "program_key": candidate["program_key"],
                "template_family": candidate["template_family"],
            }
            evidence = [asdict(fact) for fact in candidate["facts"]]
            gold = {
                **question,
                "answer": candidate["answer"],
                "output_type": candidate["output_type"],
                "tolerance": candidate["tolerance"],
                "relevant_docs": candidate["relevant_docs"],
                "relevant_tables": candidate["relevant_tables"],
                "evidence": evidence,
                "program": {
                    "operator": candidate["operator"],
                    "fact_ids": candidate["fact_ids"],
                },
            }
            if split == "hard":
                subject = _review_subject(gold)
                review = reviews.get(subject)
                approved = bool(
                    review and review.get("decision") == "approve"
                )
                gold["review"] = {
                    "status": "approved" if approved else "pending",
                    "subject_sha256": subject,
                    "reviewer": str(review.get("reviewer", "")) if review else "",
                    "reviewed_at": (
                        str(review.get("reviewed_at", "")) if review else ""
                    ),
                    "checks": (
                        list(review.get("checks", [])) if review else []
                    ),
                    "notes": str(review.get("notes", "")) if review else "",
                }
                review_queue.append({
                    "id": qid,
                    "subject_sha256": subject,
                    "question": question["question"],
                    "operator": question["operator"],
                    "answer": gold["answer"],
                    "output_type": gold["output_type"],
                    "relevant_docs": gold["relevant_docs"],
                    "relevant_tables": gold["relevant_tables"],
                    "evidence": evidence,
                    "decision": "pending",
                })
            else:
                gold["review"] = {"status": "not_required"}
            used_vars, pandas_query = _oracle_query(candidate)
            oracle_rows.append({
                "id": qid,
                "question": question["question"],
                "answer": gold["answer"],
                "pandas_query": pandas_query,
                "used_vars": used_vars,
                "status": "ok",
                "source": "g3a_oracle",
                "run_signature": "g3a-oracle-v1",
            })
            questions.append(question)
            gold_rows.append(gold)

    text_overlap = [
        row["id"]
        for row in questions
        if normalize_question(row["question"]) in public_text
    ]
    id_overlap = sorted(
        {int(row["id"]) for row in questions} & public_ids
    )
    if text_overlap or id_overlap:
        raise G3AValidationError(
            f"public leakage guard failed: "
            f"text={text_overlap}, ids={id_overlap}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / QUESTIONS_NAME, questions)
    write_jsonl(out_dir / GOLD_NAME, gold_rows)
    write_jsonl(out_dir / ORACLE_NAME, oracle_rows)
    write_jsonl(out_dir / REVIEW_QUEUE_NAME, review_queue)
    for split in ("primary_tune", "primary_locked", "hard"):
        write_jsonl(
            out_dir / f"g3a_{split}_questions.jsonl",
            [row for row in questions if row["split"] == split],
        )

    fact_sets = {
        split: sorted({
            fact_id
            for row in gold_rows
            if row["split"] == split
            for fact_id in row["program"]["fact_ids"]
        })
        for split in ("primary_tune", "primary_locked", "hard")
    }
    overlaps = {}
    names = list(fact_sets)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlaps[f"{left}__{right}"] = sorted(
                set(fact_sets[left]) & set(fact_sets[right])
            )
    if any(overlaps.values()):
        raise G3AValidationError(f"cross-split fact leakage: {overlaps}")

    all_facts = {
        fact["fact_id"]
        for row in gold_rows
        for fact in row["evidence"]
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "seed": int(config["seed"]),
        "source": {
            "store_dir": str(store_dir),
            "reports_sha256": file_sha256(
                store_dir / "reports.parquet"
            ),
            "selected_evidence_sha256": canonical_sha256([
                row["evidence"] for row in gold_rows
            ]),
            "public_questions_path": str(public_questions_path),
            "public_questions_sha256": file_sha256(
                public_questions_path
            ),
            "public_questions_usage": (
                "exact-id/text exclusion only; never template input"
            ),
            "config_path": str(config_path),
            "config_sha256": file_sha256(config_path),
            "hard_reviews_path": (
                str(review_file) if review_file else None
            ),
            "hard_reviews_sha256": (
                file_sha256(review_file) if review_file else None
            ),
        },
        "counts": {
            "selected_facts": len(all_facts),
            "questions": len(questions),
            "by_split": dict(Counter(
                row["split"] for row in questions
            )),
            "by_operator": dict(Counter(
                row["operator"] for row in questions
            )),
            "hard_approved": sum(
                row["review"]["status"] == "approved"
                for row in gold_rows
                if row["split"] == "hard"
            ),
            "hard_pending": sum(
                row["review"]["status"] != "approved"
                for row in gold_rows
                if row["split"] == "hard"
            ),
        },
        "leakage_guard": {
            "public_exact_text_overlap": text_overlap,
            "public_id_overlap": id_overlap,
            "cross_split_fact_overlap": overlaps,
            "fact_sets_sha256": {
                split: canonical_sha256(values)
                for split, values in fact_sets.items()
            },
        },
        "metric_contract": {
            "retrieval": [
                "docs_precision_macro",
                "docs_recall_macro",
                "docs_f2_macro",
                "docs_mrr5",
                "tables_precision_macro",
                "tables_recall_macro",
                "tables_f2_macro",
                "tables_mrr5",
            ],
            "submission": [
                "answer_accuracy",
                "execution_accuracy",
            ],
            "private_weight_policy": (
                "unknown; retain vector plus configured stress scenarios"
            ),
        },
        "files": {},
    }
    file_names = [
        QUESTIONS_NAME,
        GOLD_NAME,
        ORACLE_NAME,
        REVIEW_QUEUE_NAME,
        "g3a_primary_tune_questions.jsonl",
        "g3a_primary_locked_questions.jsonl",
        "g3a_hard_questions.jsonl",
    ]
    if (
        review_file is not None
        and review_file.parent.resolve() == out_dir.resolve()
    ):
        file_names.append(review_file.name)
    for name in file_names:
        manifest["files"][name] = {
            "sha256": file_sha256(out_dir / name)
        }
    manifest["bundle_fingerprint_sha256"] = canonical_sha256(
        manifest
    )
    write_json(out_dir / MANIFEST_NAME, manifest)
    return manifest


def validate_bundle(
    bundle_dir: Path | str,
    *,
    require_hard_approved: bool = True,
) -> dict:
    bundle_dir = Path(bundle_dir)
    manifest = json.loads(
        (bundle_dir / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise G3AValidationError("unexpected G3A manifest schema")
    expected_fingerprint = manifest.get(
        "bundle_fingerprint_sha256"
    )
    fingerprint_payload = dict(manifest)
    fingerprint_payload.pop("bundle_fingerprint_sha256", None)
    if (
        not expected_fingerprint
        or canonical_sha256(fingerprint_payload)
        != expected_fingerprint
    ):
        raise G3AValidationError(
            "manifest fingerprint mismatch"
        )
    for name, expected in manifest["files"].items():
        path = bundle_dir / name
        if (
            not path.exists()
            or file_sha256(path) != expected["sha256"]
        ):
            raise G3AValidationError(
                f"bundle file missing or hash mismatch: {name}"
            )
    questions = read_jsonl(bundle_dir / QUESTIONS_NAME)
    gold = read_jsonl(bundle_dir / GOLD_NAME)
    if len(questions) != len(gold) or not questions:
        raise G3AValidationError(
            "questions/gold count mismatch or empty bundle"
        )
    qmap = {int(row["id"]): row for row in questions}
    gmap = {int(row["id"]): row for row in gold}
    if len(qmap) != len(questions) or set(qmap) != set(gmap):
        raise G3AValidationError(
            "duplicate or mismatched question/gold ids"
        )
    for qid, row in qmap.items():
        if row["question"] != gmap[qid]["question"]:
            raise G3AValidationError(
                f"question text mismatch for id {qid}"
            )
    hard = [row for row in gold if row["split"] == "hard"]
    pending = [
        row["id"]
        for row in hard
        if row["review"]["status"] != "approved"
    ]
    if require_hard_approved and pending:
        raise G3AValidationError(
            f"hard set is not fully approved; "
            f"pending ids={pending[:10]}"
        )
    return {
        "bundle_fingerprint_sha256": manifest[
            "bundle_fingerprint_sha256"
        ],
        "questions": len(questions),
        "hard": len(hard),
        "hard_pending": len(pending),
        "valid": True,
    }
