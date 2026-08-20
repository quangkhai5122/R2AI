"""Automatic P2.4 silver benchmark from adjacent-report duplicate facts.

The benchmark is self-supervised: a target-year cell in report Y is accepted as
silver only when the same signed, unit-normalized fact appears in report Y+1's
prior-period column.  Gold construction uses exact normalized labels (and row
codes when available), while evaluation invokes the fuzzy v5.3 resolver without
access to the expected value.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from ..codegen.semantic_repair import (
    _StoreView,
    _full_header,
    _sha256,
    _write_json_exclusive,
    _write_jsonl_exclusive,
    _store_manifest,
    classify_column_role,
)
from ..codegen.single_cell_consensus import resolve_single_fact_candidates
from ..extraction.unit_policy import resolve_stored_table_unit
from ..utils.io import read_jsonl
from ..utils.viet_text import norm


POLICY_VERSION = "p24_auto_silver_adjacent_report_v1"
SCHEMA_VERSION = "p24_auto_silver_fact_v1"
MANIFEST_VERSION = "p24_auto_silver_manifest_v1"
EVALUATOR_VERSION = "p24_auto_silver_eval_v1"
DEFAULT_SEED = 2453
DEFAULT_MAX_PER_REPORT_PAIR = 12
DEFAULT_MAX_TICKERS_PER_SPLIT = 8

_FILES = {
    "train": "p24_silver_train.jsonl",
    "tune": "p24_silver_tune.jsonl",
    "locked": "p24_silver_locked.jsonl",
}
_GENERIC = {
    "cong", "tong", "tong cong", "gia tri", "chi tieu", "nam nay",
    "nam truoc", "so cuoi nam", "so dau nam", "ma so", "thuyet minh",
}


@dataclass(frozen=True)
class SilverCell:
    ticker: str
    report_id: str
    report_year: int
    doc_type: str
    table_pos: int
    row: int
    col: int
    label: str
    label_norm: str
    row_code: str
    col_name: str
    raw_value: float
    absolute_value: float
    role: dict
    unit: dict

    @property
    def stable_cell(self) -> tuple[str, int, int, int]:
        return self.report_id, self.table_pos, self.row, self.col


def build_auto_silver_bundle(
    store_dir: Path,
    out_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    max_per_report_pair: int = DEFAULT_MAX_PER_REPORT_PAIR,
    max_tickers_per_split: int = DEFAULT_MAX_TICKERS_PER_SPLIT,
) -> dict:
    """Build immutable train/tune/locked automatic silver splits."""
    store_dir, out_dir = Path(store_dir), Path(out_dir)
    if max_per_report_pair <= 0:
        raise ValueError("max_per_report_pair must be positive")
    if max_tickers_per_split <= 0:
        raise ValueError("max_tickers_per_split must be positive")
    paths = {split: out_dir / name for split, name in _FILES.items()}
    manifest_path = out_dir / "p24_silver_manifest.json"
    for path in [*paths.values(), manifest_path]:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite silver artifact: {path}")

    view = _StoreView(store_dir)
    records: dict[str, list[dict]] = {split: [] for split in _FILES}
    counts = Counter()
    tickers_used: set[str] = set()

    report_groups: dict[tuple[str, str], dict[int, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    cell_cache: dict[tuple[str, int], list[SilverCell]] = {}

    def report_cells(ticker: str, report_id: str, target_year: int) -> list[SilverCell]:
        key = (report_id, int(target_year))
        if key not in cell_cache:
            cell_cache[key] = _report_cells(
                view, ticker, report_id, target_year=target_year,
            )
        return cell_cache[key]

    for report in view.store.reports.itertuples():
        report_groups[(str(report.ticker), str(report.doc_type))][
            int(report.year)
        ].append(str(report.report_id))

    tickers_by_split: dict[str, list[str]] = defaultdict(list)
    for ticker in sorted({key[0] for key in report_groups}):
        tickers_by_split[_ticker_split(ticker, seed)].append(ticker)
    selected_tickers = {
        ticker
        for split, tickers in tickers_by_split.items()
        for ticker in sorted(
            tickers,
            key=lambda value: hashlib.sha256(
                f"select|{seed}|{split}|{value}".encode("utf-8")
            ).hexdigest(),
        )[:max_tickers_per_split]
    }

    for (ticker, doc_type), by_year in sorted(report_groups.items()):
        if ticker not in selected_tickers:
            continue
        split = _ticker_split(ticker, seed)
        for year in sorted(by_year):
            if year + 1 not in by_year:
                continue
            pair_records: list[dict] = []
            for target_report_id in sorted(by_year[year]):
                target_cells = report_cells(ticker, target_report_id, year)
                if not target_cells:
                    continue
                target_by_label: dict[str, list[SilverCell]] = defaultdict(list)
                for cell in target_cells:
                    target_by_label[cell.label_norm].append(cell)
                for support_report_id in sorted(by_year[year + 1]):
                    support_cells = report_cells(ticker, support_report_id, year)
                    support_by_label: dict[str, list[SilverCell]] = defaultdict(list)
                    for cell in support_cells:
                        support_by_label[cell.label_norm].append(cell)
                    for label_norm in sorted(set(target_by_label) & set(support_by_label)):
                        targets = target_by_label[label_norm]
                        if len({cell.stable_cell for cell in targets}) != 1:
                            counts["target_label_not_unique"] += 1
                            continue
                        target = targets[0]
                        supports = [
                            cell for cell in support_by_label[label_norm]
                            if _same_signed_value(
                                cell.absolute_value, target.absolute_value,
                            ) and _codes_compatible(target.row_code, cell.row_code)
                        ]
                        support_map = {cell.stable_cell: cell for cell in supports}
                        supports = list(support_map.values())
                        if not supports:
                            counts["no_adjacent_signed_support"] += 1
                            continue
                        if target.absolute_value == 0.0:
                            counts["zero_value_excluded"] += 1
                            continue
                        output_scale = _requested_scale(
                            ticker, doc_type, year, label_norm, seed,
                        )
                        expected_answer = round(
                            target.absolute_value / output_scale, 2,
                        )
                        if not math.isfinite(expected_answer):
                            continue
                        silver_id = hashlib.sha256(
                            f"{ticker}|{doc_type}|{year}|{label_norm}|{seed}".encode("utf-8")
                        ).hexdigest()[:20]
                        pair_records.append({
                            "schema_version": SCHEMA_VERSION,
                            "policy": POLICY_VERSION,
                            "silver_id": silver_id,
                            "split": split,
                            "ticker": ticker,
                            "doc_type": doc_type,
                            "target_year": year,
                            "metric": target.label,
                            "metric_norm": label_norm,
                            "output_type": "number",
                            "output_scale": output_scale,
                            "expected_answer": expected_answer,
                            "target": asdict(target),
                            "supports": [asdict(cell) for cell in sorted(
                                supports, key=lambda value: value.stable_cell,
                            )],
                        })
            pair_records = _deterministic_cap(
                pair_records, max_per_report_pair,
                salt=f"{ticker}|{doc_type}|{year}|{seed}",
            )
            if pair_records:
                tickers_used.add(ticker)
                records[split].extend(pair_records)
                counts["report_pairs_with_records"] += 1

    for split in records:
        records[split].sort(key=lambda row: row["silver_id"])
        if not records[split]:
            raise ValueError(f"automatic silver split is empty: {split}")
        _write_jsonl_exclusive(paths[split], records[split])

    file_meta = {
        split: {
            "path": str(paths[split]),
            "sha256": _sha256(paths[split]),
            "count": len(records[split]),
            "tickers": sorted({row["ticker"] for row in records[split]}),
        }
        for split in records
    }
    store_manifest = _store_manifest(store_dir, sorted(tickers_used))
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "policy": POLICY_VERSION,
        "seed": int(seed),
        "max_per_report_pair": int(max_per_report_pair),
        "split_policy": "ticker_hash_70_15_15",
        "leakage_guard": "a ticker belongs to exactly one split",
        "counts": {**dict(sorted(counts.items())),
                   "total": sum(len(value) for value in records.values())},
        "files": file_meta,
        "store": {"path": str(store_dir), "manifest": store_manifest},
        "max_tickers_per_split": int(max_tickers_per_split),
    }
    manifest["bundle_fingerprint_sha256"] = _canonical_sha(manifest)
    _write_json_exclusive(manifest_path, manifest)
    return manifest


def evaluate_auto_silver(
    split_path: Path,
    store_dir: Path,
    out_path: Path,
    *,
    expected_split_sha256: str = "",
) -> dict:
    """Evaluate the v5.3 resolver without exposing expected values to it."""
    split_path, out_path = Path(split_path), Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite silver evaluation: {out_path}")
    actual_sha = _sha256(split_path)
    if expected_split_sha256 and actual_sha != expected_split_sha256.lower():
        raise ValueError(
            f"silver split SHA mismatch: expected={expected_split_sha256}, actual={actual_sha}"
        )
    rows = read_jsonl(split_path)
    if not rows:
        raise ValueError("silver split is empty")
    view = _StoreView(Path(store_dir))
    results = []
    counts = Counter()
    for row in rows:
        target = row["target"]
        route = {
            "doc_type": row["doc_type"],
            "output_type": "number",
            "unit_scale": float(row["output_scale"]),
            "metric_norm": row["metric_norm"],
            "report_ids": [target["report_id"]],
        }
        fact = {
            "ticker": row["ticker"], "year": int(row["target_year"]),
            "doc_type": row["doc_type"], "metric": row["metric"],
            "role": "value",
        }
        candidates, reason, detail = resolve_single_fact_candidates(
            view=view, route=route, fact=fact,
        )
        predicted = candidates[0] if candidates else None
        accepted = predicted is not None
        cell_correct = bool(
            predicted and predicted.stable_cell == (
                str(target["report_id"]), int(target["table_pos"]),
                int(target["row"]), int(target["col"]),
            )
        )
        answer_correct = bool(
            predicted and math.isclose(
                predicted.answer, float(row["expected_answer"]),
                rel_tol=0.0, abs_tol=0.005,
            )
        )
        counts["rows"] += 1
        counts["accepted"] += int(accepted)
        counts["cell_correct"] += int(cell_correct)
        counts["answer_correct"] += int(answer_correct)
        if accepted and answer_correct:
            counts["accepted_answer_correct"] += 1
        results.append({
            "silver_id": row["silver_id"], "ticker": row["ticker"],
            "accepted": accepted, "reason": reason,
            "cell_correct": cell_correct, "answer_correct": answer_correct,
            "expected_answer": row["expected_answer"],
            "predicted_answer": predicted.answer if predicted else None,
            "expected_cell": {
                key: target[key] for key in ("report_id", "table_pos", "row", "col")
            },
            "predicted_cell": ({
                "report_id": predicted.report_id, "table_pos": predicted.table_pos,
                "row": predicted.row, "col": predicted.col,
            } if predicted else None),
            "detail": detail,
        })

    n, accepted = counts["rows"], counts["accepted"]
    report = {
        "schema_version": "p24_auto_silver_eval_report_v1",
        "evaluator_version": EVALUATOR_VERSION,
        "policy": POLICY_VERSION,
        "input": {"path": str(split_path), "sha256": actual_sha},
        "metrics": {
            "count": n,
            "coverage": accepted / n,
            "cell_accuracy": counts["cell_correct"] / n,
            "answer_accuracy": counts["answer_correct"] / n,
            "accepted_answer_precision": (
                counts["accepted_answer_correct"] / accepted if accepted else 0.0
            ),
        },
        "counts": dict(sorted(counts.items())),
        "records": results,
    }
    _write_json_exclusive(out_path, report)
    return report


def _report_cells(
    view: _StoreView, ticker: str, report_id: str, *, target_year: int,
) -> list[SilverCell]:
    report = view.report(report_id)
    if report is None:
        return []
    cells = view.store.cells_of(ticker, [report_id])
    output: list[SilverCell] = []
    table_cache = {}
    unit_cache = {}
    role_cache = {}
    for cell in cells.itertuples():
        table_pos = int(cell.table_pos)
        table = table_cache.get(table_pos)
        if table_pos not in table_cache:
            table = view.table(ticker, report_id, table_pos)
            table_cache[table_pos] = table
        if table is None:
            continue
        if table_pos not in unit_cache:
            unit_cache[table_pos] = resolve_stored_table_unit(
                table.unit_scale, table.unit_source, str(table.context or ""),
            )
        unit = unit_cache[table_pos]
        role_key = (table_pos, int(cell.col))
        if role_key not in role_cache:
            header = _full_header(table, int(cell.col), str(cell.col_name or ""))
            role_cache[role_key] = classify_column_role(
                header, int(report.year), target_year, positional_current=False,
            )
        role = role_cache[role_key]
        if role.role != "target_value" or role.confidence < 3:
            continue
        label = str(cell.label or "").strip()
        label_norm = norm(label)
        if label_norm in _GENERIC or len(label_norm) < 5 \
                or len(label_norm.split()) < 2:
            continue
        raw_value = float(cell.value)
        absolute = raw_value * unit.effective_scale
        if not math.isfinite(absolute):
            continue
        output.append(SilverCell(
            ticker=ticker, report_id=report_id,
            report_year=int(report.year), doc_type=str(report.doc_type),
            table_pos=table_pos, row=int(cell.row), col=int(cell.col),
            label=label, label_norm=label_norm,
            row_code=str(getattr(cell, "row_code", "") or ""),
            col_name=str(cell.col_name or ""), raw_value=raw_value,
            absolute_value=absolute, role=asdict(role),
            unit={
                "stored_scale": unit.stored_scale,
                "effective_scale": unit.effective_scale,
                "stored_source": unit.stored_source,
                "effective_source": unit.effective_source,
                "resolution": unit.reason,
            },
        ))
    return output


def _ticker_split(ticker: str, seed: int) -> str:
    value = int(hashlib.sha256(f"{seed}|{ticker}".encode("utf-8")).hexdigest()[:8], 16) % 100
    if value < 70:
        return "train"
    if value < 85:
        return "tune"
    return "locked"


def _requested_scale(
    ticker: str, doc_type: str, year: int, metric: str, seed: int,
) -> float:
    options = (1.0, 1_000_000.0, 1_000_000_000.0)
    value = int(hashlib.sha256(
        f"scale|{seed}|{ticker}|{doc_type}|{year}|{metric}".encode("utf-8")
    ).hexdigest()[:8], 16)
    return options[value % len(options)]


def _deterministic_cap(rows: list[dict], limit: int, *, salt: str) -> list[dict]:
    ranked = sorted(rows, key=lambda row: hashlib.sha256(
        f"{salt}|{row['silver_id']}".encode("utf-8")
    ).hexdigest())
    return ranked[:limit]


def _same_signed_value(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=max(1.0, abs(b) * 1e-9))


def _codes_compatible(a: str, b: str) -> bool:
    left, right = str(a or "").strip(), str(b or "").strip()
    return not (left and right) or left == right


def _canonical_sha(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
