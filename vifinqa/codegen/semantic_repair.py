"""Deterministic column/period/unit repairs with independent silver evidence.

This module is intentionally narrower than the general rule/Selection pipelines.
It repairs an already-successful lookup only when the saved query is a single-cell
lookup, the selected column is a note/code column or carries a proven unit conflict,
and another table/report independently confirms the replacement value.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from ..extraction.build_store import Store
from ..extraction.unit_policy import resolve_stored_table_unit
from ..utils.io import read_jsonl
from ..utils.viet_text import label_metric_score, norm
from .semantic import all_dataframe_refs


POLICY_VERSION = "v52a_column_period_unit_silver_v1"
SOURCE_NAME = "deterministic_v52a"

_NOTE_OR_CODE_EXACT = {
    "thuyet minh", "ghi chu", "note", "notes", "ma so", "stt", "code",
}
_NOTE_OR_CODE_PREFIX = ("thuyet minh ", "ghi chu ", "note ")
_CURRENT_CUES = re.compile(
    r"\b(?:nam nay|ky nay|so cuoi nam|so du cuoi|cuoi ky|current year|"
    r"ending balance|closing balance)\b"
)
_PRIOR_CUES = re.compile(
    r"\b(?:nam truoc|ky truoc|so dau nam|so du dau|dau ky|prior year|"
    r"previous year|beginning balance|opening balance)\b"
)
_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_SIMPLE_LOOKUP_RE = re.compile(
    r"^\s*round\(float\((?P<var>df\d+)\.loc\["
    r"(?P=var)\['label'\]\.str\.contains\("
    r"'(?P<label>(?:\\'|[^'])*)',\s*case=False,\s*regex=False,\s*na=False\)"
    r"\s*&\s*\((?P=var)\['col'\]\s*==\s*(?P<col>\d+)\),\s*'value'\]"
    r"\.iloc\[0\]\)\s*\*\s*(?P<input_scale>[-+0-9.eE]+)\s*/\s*"
    r"(?P<output_scale>[-+0-9.eE]+)\s*,\s*2\)\s*$"
)


@dataclass(frozen=True)
class ColumnRole:
    role: str
    reason: str
    header: str
    mapped_year: int | None = None
    confidence: int = 0


@dataclass(frozen=True)
class ParsedLookup:
    var: str
    label_needle: str
    selected_col: int
    query_input_scale: float
    query_output_scale: float


@dataclass
class RepairProposal:
    qid: int
    answer: float
    query: str
    used_vars: list[dict]
    provenance: dict


class _StoreView:
    def __init__(self, store_dir: Path):
        self.store_dir = Path(store_dir)
        self.store = Store(self.store_dir)
        self.report_by_id = {
            str(row.report_id): row for row in self.store.reports.itertuples()
        }
        self._table_maps: dict[str, dict[tuple[str, int], object]] = {}

    def report(self, report_id: str):
        return self.report_by_id.get(str(report_id))

    def table_map(self, ticker: str) -> dict[tuple[str, int], object]:
        ticker = str(ticker)
        if ticker not in self._table_maps:
            self._table_maps[ticker] = {
                (str(row.report_id), int(row.table_pos)): row
                for row in self.store.tables_of(ticker).itertuples()
            }
        return self._table_maps[ticker]

    def table(self, ticker: str, report_id: str, table_pos: int):
        return self.table_map(ticker).get((str(report_id), int(table_pos)))


def classify_column_role(
    header: str,
    report_year: int,
    target_year: int,
    *,
    positional_current: bool = False,
) -> ColumnRole:
    """Classify a numeric column relative to the requested target year."""
    raw = str(header or "")
    text = norm(raw)
    if text in _NOTE_OR_CODE_EXACT or text.startswith(_NOTE_OR_CODE_PREFIX):
        role = "code" if text in {"ma so", "stt", "code"} else "note"
        return ColumnRole(role, "non_value_column", raw)

    mapped_year = _date_mapped_year(text)
    reason = "explicit_date"
    confidence = 4
    if mapped_year is None:
        years = {int(value) for value in _YEAR_RE.findall(text)}
        if len(years) == 1:
            mapped_year = next(iter(years))
            reason = "explicit_year"
            confidence = 4
        elif len(years) > 1:
            return ColumnRole("ambiguous", "multiple_explicit_years", raw)

    if mapped_year is None and _CURRENT_CUES.search(text):
        mapped_year = int(report_year)
        reason = "current_period_header"
        confidence = 3
    elif mapped_year is None and _PRIOR_CUES.search(text):
        mapped_year = int(report_year) - 1
        reason = "prior_period_header"
        confidence = 3

    if mapped_year is not None:
        role = "target_value" if mapped_year == int(target_year) else "other_period"
        return ColumnRole(role, reason, raw, mapped_year, confidence)
    if positional_current and int(report_year) == int(target_year):
        return ColumnRole(
            "target_value", "first_numeric_value_positional", raw,
            int(target_year), 1,
        )
    return ColumnRole("unknown", "no_period_evidence", raw)


def parse_simple_lookup(query: str) -> ParsedLookup | None:
    """Parse only the historical one-cell lookup template; fail closed otherwise."""
    text = str(query or "")
    try:
        ast.parse(text, mode="eval")
    except SyntaxError:
        return None
    match = _SIMPLE_LOOKUP_RE.fullmatch(text)
    if not match:
        return None
    try:
        input_scale = float(match.group("input_scale"))
        output_scale = float(match.group("output_scale"))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and value > 0
               for value in (input_scale, output_scale)):
        return None
    return ParsedLookup(
        var=match.group("var"),
        label_needle=match.group("label").replace("\\'", "'"),
        selected_col=int(match.group("col")),
        query_input_scale=input_scale,
        query_output_scale=output_scale,
    )


def discover_semantic_repairs(
    primary_rows: list[dict],
    retrieval_rows: list[dict],
    store_dir: Path,
) -> tuple[list[RepairProposal], dict]:
    """Discover strict v5.2a repairs without mutating or writing an artifact."""
    primary = _unique_rows(primary_rows, "primary")
    retrieval = _unique_rows(retrieval_rows, "retrieval")
    _validate_universe(primary, retrieval)
    view = _StoreView(Path(store_dir))
    proposals: list[RepairProposal] = []
    counts = Counter()
    rejected: list[dict] = []

    for qid in sorted(primary):
        row = primary[qid]
        route = retrieval[qid].get("route") or {}
        counts["rows"] += 1
        if str(row.get("status", "")).lower() != "ok":
            counts["out_of_scope_not_ok"] += 1
            continue
        counts["status_ok"] += 1
        plan = route.get("plan") or {}
        facts = plan.get("facts") or []
        if plan.get("op") != "lookup" or route.get("output_type") != "number" \
                or len(facts) != 1:
            counts["out_of_scope_not_lookup_number"] += 1
            continue
        counts["lookup_number"] += 1
        proposal, reason, detail = _proposal_for_row(
            qid, row, route, facts[0], view,
        )
        if proposal is None:
            counts[reason] += 1
            if detail.get("repair_triggered"):
                rejected.append({"id": qid, "reason": reason, **detail})
            continue
        counts["accepted"] += 1
        proposals.append(proposal)

    return proposals, {
        "counts": dict(sorted(counts.items())),
        "rejected_triggered": rejected,
    }


def build_semantic_repair_overlay(
    primary_path: Path,
    retrieval_path: Path,
    store_dir: Path,
    out_path: Path,
    audit_path: Path | None = None,
    *,
    expected_selected_ids: set[int] | None = None,
    expected_primary_signature: str = "",
    expected_primary_sha256: str = "",
    expected_retrieval_sha256: str = "",
) -> dict:
    """Build a complete overlay and an audit, never overwriting existing files."""
    primary_path = Path(primary_path)
    retrieval_path = Path(retrieval_path)
    store_dir = Path(store_dir)
    out_path = Path(out_path)
    audit_path = Path(audit_path) if audit_path else out_path.with_suffix(".audit.json")
    for path in (out_path, audit_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    primary_sha = _sha256(primary_path)
    retrieval_sha = _sha256(retrieval_path)
    _check_hash(primary_sha, expected_primary_sha256, "primary")
    _check_hash(retrieval_sha, expected_retrieval_sha256, "retrieval")
    primary_rows = read_jsonl(primary_path)
    retrieval_rows = read_jsonl(retrieval_path)
    primary = _unique_rows(primary_rows, "primary")
    primary_signature = _one_signature(primary, "primary")
    if expected_primary_signature and primary_signature != expected_primary_signature:
        raise ValueError(
            "primary run_signature mismatch: "
            f"expected={expected_primary_signature}, actual={primary_signature}"
        )

    proposals, discovery = discover_semantic_repairs(
        primary_rows, retrieval_rows, store_dir,
    )
    selected_ids = [proposal.qid for proposal in proposals]
    if expected_selected_ids is not None and set(selected_ids) != {
            int(value) for value in expected_selected_ids}:
        raise ValueError(
            "selected id guard mismatch: "
            f"expected={sorted(expected_selected_ids)}, actual={selected_ids}"
        )

    selected_tickers = sorted({
        str(item["ticker"])
        for proposal in proposals
        for item in proposal.provenance.get("selected_cells", [])
    })
    semantic_manifest = _semantic_manifest()
    store_manifest = _store_manifest(store_dir, selected_tickers)
    repair_digest = _json_sha([
        {
            "id": proposal.qid,
            "answer": proposal.answer,
            "query": proposal.query,
            "provenance": proposal.provenance,
        }
        for proposal in proposals
    ])
    run_signature = _json_sha({
        "policy": POLICY_VERSION,
        "primary_signature": primary_signature,
        "primary_sha256": primary_sha,
        "retrieval_sha256": retrieval_sha,
        "semantic_manifest": semantic_manifest,
        "store_manifest": store_manifest,
        "repair_digest": repair_digest,
        "selected_ids": selected_ids,
    })

    proposal_by_id = {proposal.qid: proposal for proposal in proposals}
    output: list[dict] = []
    for qid in sorted(primary):
        original = primary[qid]
        row = dict(original)
        row["run_signature"] = run_signature
        proposal = proposal_by_id.get(qid)
        if proposal is not None:
            row.update({
                "answer": proposal.answer,
                "pandas_query": proposal.query,
                "used_vars": proposal.used_vars,
                "source": SOURCE_NAME,
                "status": "ok",
                "detail": f"deterministic repair {POLICY_VERSION}",
                "semantic_repair_provenance": proposal.provenance,
            })
            _validate_repaired_record(row, qid)
        output.append(row)

    _write_jsonl_exclusive(out_path, output)
    output_sha = _sha256(out_path)
    audit = {
        "schema_version": "v52a_semantic_repair_audit_v1",
        "policy": POLICY_VERSION,
        "inputs": {
            "primary": {
                "path": str(primary_path), "sha256": primary_sha,
                "run_signature": primary_signature,
            },
            "retrieval": {"path": str(retrieval_path), "sha256": retrieval_sha},
            "store_dir": str(store_dir),
            "semantic_manifest": semantic_manifest,
            "store_manifest": store_manifest,
        },
        "run_signature": run_signature,
        "repair_digest": repair_digest,
        "discovery": discovery,
        "counts": {
            "rows": len(output),
            "selected": len(proposals),
            "unchanged_semantic_rows": len(output) - len(proposals),
        },
        "selected_ids": selected_ids,
        "selected": [
            {
                "id": proposal.qid,
                "answer": proposal.answer,
                "query": proposal.query,
                "trigger_reasons": proposal.provenance["trigger_reasons"],
                "silver_support_count": proposal.provenance["silver_support_count"],
            }
            for proposal in proposals
        ],
        "output": {"path": str(out_path), "sha256": output_sha, "rows": len(output)},
    }
    _write_json_exclusive(audit_path, audit)
    return audit


def _proposal_for_row(qid: int, row: dict, route: dict, fact: dict,
                      view: _StoreView) -> tuple[RepairProposal | None, str, dict]:
    parsed = parse_simple_lookup(row.get("pandas_query", ""))
    if parsed is None:
        return None, "not_simple_lookup", {}
    try:
        refs = all_dataframe_refs(str(row.get("pandas_query") or ""))
    except SyntaxError:
        return None, "invalid_primary_query", {}
    if refs != {parsed.var}:
        return None, "not_single_dataframe_lookup", {}
    bindings = [item for item in row.get("used_vars") or []
                if isinstance(item, dict) and str(item.get("var")) == parsed.var]
    if len(bindings) != 1:
        return None, "ambiguous_evidence_binding", {}
    binding = dict(bindings[0])
    report_id = str(binding.get("report_id") or "")
    try:
        table_pos = int(binding.get("table_pos"))
    except (TypeError, ValueError):
        return None, "invalid_evidence_binding", {}
    report = view.report(report_id)
    if report is None:
        return None, "report_not_found", {}
    ticker = str(report.ticker)
    table = view.table(ticker, report_id, table_pos)
    if table is None:
        return None, "table_not_found", {}
    if str(fact.get("ticker") or ticker) != ticker \
            or str(fact.get("doc_type") or report.doc_type) != str(report.doc_type):
        return None, "fact_report_mismatch", {}
    try:
        target_year = int(fact.get("year"))
    except (TypeError, ValueError):
        return None, "invalid_fact_year", {}
    if target_year != int(report.year):
        return None, "target_not_report_year", {}

    cells = view.store.cells_of(ticker, [report_id])
    table_cells = cells[cells.table_pos == table_pos]
    selected = table_cells[
        table_cells.label.str.contains(
            parsed.label_needle, case=False, regex=False, na=False,
        ) & (table_cells.col == parsed.selected_col)
    ]
    if len(selected) != 1:
        return None, "selected_cell_not_unique", {}
    selected_cell = selected.iloc[0]
    same_row = table_cells[table_cells.row == int(selected_cell.row)].copy()
    if same_row.empty:
        return None, "selected_row_missing", {}

    value_cols = []
    role_by_col: dict[int, ColumnRole] = {}
    for cell in same_row.itertuples():
        header = _full_header(table, int(cell.col), str(cell.col_name))
        base_role = classify_column_role(header, int(report.year), target_year)
        role_by_col[int(cell.col)] = base_role
        if base_role.role not in {"note", "code"}:
            value_cols.append(int(cell.col))
    first_value_col = min(value_cols) if value_cols else None
    for cell in same_row.itertuples():
        col = int(cell.col)
        header = _full_header(table, col, str(cell.col_name))
        role_by_col[col] = classify_column_role(
            header, int(report.year), target_year,
            positional_current=(col == first_value_col),
        )

    selected_role = role_by_col.get(parsed.selected_col)
    unit = resolve_stored_table_unit(
        table.unit_scale, table.unit_source, str(table.context or ""),
    )
    triggers = []
    if selected_role and selected_role.role in {"note", "code"}:
        triggers.append("selected_non_value_column")
    if unit.changed:
        triggers.append("terminal_vnd_unit_override")
    if not triggers:
        return None, "no_repair_trigger", {}
    trigger_detail = {
        "repair_triggered": True,
        "trigger_reasons": triggers,
        "selected_role": asdict(selected_role) if selected_role else None,
    }

    candidates = []
    for cell in same_row.itertuples():
        role = role_by_col[int(cell.col)]
        if role.role == "target_value":
            candidates.append((role.confidence, int(cell.col), cell, role))
    if not candidates:
        return None, "target_column_missing", trigger_detail
    best_conf = max(item[0] for item in candidates)
    candidates = [item for item in candidates if item[0] == best_conf]
    if len(candidates) != 1:
        return None, "target_column_ambiguous", {
            **trigger_detail, "candidate_cols": sorted(item[1] for item in candidates),
        }
    _confidence, _col, target, target_role = candidates[0]
    metric = str(fact.get("metric") or route.get("metric_norm") or "")
    metric_score = label_metric_score(str(target.label), metric)
    if metric_score < 55.0:
        return None, "target_metric_mismatch", {
            **trigger_detail, "metric_score": round(metric_score, 2),
        }

    absolute_value = float(target.value) * unit.effective_scale
    try:
        output_scale = float(route.get("unit_scale") or parsed.query_output_scale)
    except (TypeError, ValueError):
        return None, "invalid_output_scale", trigger_detail
    if not math.isfinite(absolute_value) or not math.isfinite(output_scale) \
            or output_scale <= 0:
        return None, "invalid_repaired_value", trigger_detail
    answer = round(absolute_value / output_scale, 2)
    if not math.isfinite(answer):
        return None, "invalid_repaired_value", trigger_detail
    try:
        original_answer = float(row.get("answer"))
    except (TypeError, ValueError):
        return None, "invalid_original_answer", trigger_detail
    if math.isclose(answer, original_answer, rel_tol=0.0, abs_tol=1e-12):
        return None, "repair_does_not_change_answer", trigger_detail

    supports = _silver_supports(
        view=view,
        ticker=ticker,
        report=report,
        table=table,
        target=target,
        target_year=target_year,
        metric=metric,
        absolute_value=absolute_value,
    )
    if not supports:
        return None, "silver_no_signed_support", {
            **trigger_detail,
            "candidate_answer": answer,
            "candidate_cell": _cell_summary(target, target_role),
        }

    query = (
        f"round(float({parsed.var}.loc[({parsed.var}['row'] == {int(target.row)}) "
        f"& ({parsed.var}['col'] == {int(target.col)}), 'value'].iloc[0]) * "
        f"{_number_literal(unit.effective_scale)} / {_number_literal(output_scale)}, 2)"
    )
    context = str(table.context or "")
    provenance = {
        "schema_version": 1,
        "policy": POLICY_VERSION,
        "trigger_reasons": triggers,
        "original": {
            "source": row.get("source"),
            "status": row.get("status"),
            "answer": original_answer,
            "query": row.get("pandas_query"),
            "selected_col": parsed.selected_col,
            "selected_role": asdict(selected_role) if selected_role else None,
            "query_input_scale": parsed.query_input_scale,
            "query_output_scale": parsed.query_output_scale,
        },
        "selected_cells": [{
            "ticker": ticker,
            "report_id": report_id,
            "report_year": int(report.year),
            "doc_type": str(report.doc_type),
            "table_pos": table_pos,
            **_cell_summary(target, target_role),
            "metric_score": round(float(metric_score), 2),
            "unit": {
                "stored_scale": unit.stored_scale,
                "effective_scale": unit.effective_scale,
                "stored_source": unit.stored_source,
                "effective_source": unit.effective_source,
                "resolution": unit.reason,
                "terminal_bare_vnd": unit.terminal_bare_vnd,
                "context_sha256": _text_sha(context),
            },
            "absolute_value": absolute_value,
            "output_scale": output_scale,
            "output_unit": route.get("unit_name"),
        }],
        "silver_support_count": len(supports),
        "silver_supports": supports,
    }
    return RepairProposal(
        qid=qid,
        answer=answer,
        query=query,
        used_vars=[binding],
        provenance=provenance,
    ), "accepted", trigger_detail


def _silver_supports(*, view: _StoreView, ticker: str, report, table, target,
                     target_year: int, metric: str,
                     absolute_value: float) -> list[dict]:
    report_ids = [str(report.report_id)]
    report_ids.extend(view.store.find_reports(
        ticker, target_year + 1, str(report.doc_type), allow_fallback=False,
    ))
    report_ids = list(dict.fromkeys(report_ids))
    cells = view.store.cells_of(ticker, report_ids)
    table_map = view.table_map(ticker)
    supports: list[dict] = []
    seen_tables = set()
    tolerance = max(1.0, abs(absolute_value) * 1e-9)

    for cell in cells.itertuples():
        key = (str(cell.report_id), int(cell.table_pos))
        if key == (str(report.report_id), int(table.table_pos)):
            continue
        other_table = table_map.get(key)
        other_report = view.report(str(cell.report_id))
        if other_table is None or other_report is None:
            continue
        other_unit = resolve_stored_table_unit(
            other_table.unit_scale, other_table.unit_source,
            str(other_table.context or ""),
        )
        other_absolute = float(cell.value) * other_unit.effective_scale
        if not math.isclose(
            other_absolute, absolute_value, rel_tol=1e-9, abs_tol=tolerance,
        ):
            continue

        same_row = cells[
            (cells.report_id == str(cell.report_id))
            & (cells.table_pos == int(cell.table_pos))
            & (cells.row == int(cell.row))
        ]
        value_cols = []
        for row_cell in same_row.itertuples():
            header = _full_header(
                other_table, int(row_cell.col), str(row_cell.col_name),
            )
            base = classify_column_role(
                header, int(other_report.year), target_year,
            )
            if base.role not in {"note", "code"}:
                value_cols.append(int(row_cell.col))
        first_value_col = min(value_cols) if value_cols else None
        header = _full_header(other_table, int(cell.col), str(cell.col_name))
        role = classify_column_role(
            header, int(other_report.year), target_year,
            positional_current=(int(cell.col) == first_value_col),
        )
        if role.role != "target_value":
            continue

        context = str(other_table.context or "")
        metric_score = max(
            label_metric_score(str(cell.label), metric),
            label_metric_score(f"{context} {cell.label}", metric),
        )
        target_code = str(getattr(target, "row_code", "") or "").strip()
        support_code = str(getattr(cell, "row_code", "") or "").strip()
        same_code = bool(target_code and support_code and target_code == support_code)
        if metric_score < 65.0 and not same_code:
            continue
        if key in seen_tables:
            continue
        seen_tables.add(key)
        relation = (
            "same_report_other_table"
            if str(cell.report_id) == str(report.report_id)
            else "next_report_prior_period"
        )
        supports.append({
            "relation": relation,
            "ticker": ticker,
            "report_id": str(cell.report_id),
            "report_year": int(other_report.year),
            "doc_type": str(other_report.doc_type),
            "table_pos": int(cell.table_pos),
            **_cell_summary(cell, role),
            "metric_score": round(float(metric_score), 2),
            "same_row_code": same_code,
            "absolute_value": other_absolute,
            "unit": {
                "stored_scale": other_unit.stored_scale,
                "effective_scale": other_unit.effective_scale,
                "stored_source": other_unit.stored_source,
                "effective_source": other_unit.effective_source,
                "resolution": other_unit.reason,
                "terminal_bare_vnd": other_unit.terminal_bare_vnd,
                "context_sha256": _text_sha(context),
            },
        })
    supports.sort(key=lambda item: (
        item["relation"], item["report_id"], item["table_pos"],
        item["row"], item["col"],
    ))
    return supports


def _date_mapped_year(text: str) -> int | None:
    ending = re.search(
        r"(?:31\s*/\s*12\s*/\s*|31\s+thang\s+12\s+nam\s+)((?:19|20)\d{2})",
        text,
    )
    if ending:
        return int(ending.group(1))
    beginning = re.search(
        r"(?:1\s*/\s*1\s*/\s*|01\s*/\s*01\s*/\s*|"
        r"1\s+thang\s+1\s+nam\s+)((?:19|20)\d{2})",
        text,
    )
    if beginning:
        return int(beginning.group(1)) - 1
    return None


def _full_header(table, col: int, fallback: str = "") -> str:
    try:
        grid = json.loads(str(table.grid_json))
        if isinstance(grid, list) and grid and isinstance(grid[0], list) \
                and 0 <= int(col) < len(grid[0]):
            value = str(grid[0][int(col)] or "").strip()
            if value:
                return value
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return str(fallback or "")


def _cell_summary(cell, role: ColumnRole) -> dict:
    return {
        "row": int(cell.row),
        "col": int(cell.col),
        "label": str(cell.label),
        "row_code": str(getattr(cell, "row_code", "") or ""),
        "col_name": str(cell.col_name),
        "raw_value": float(cell.value),
        "column_role": asdict(role),
    }


def _validate_repaired_record(row: dict, qid: int) -> None:
    answer = float(row.get("answer"))
    if not math.isfinite(answer):
        raise ValueError(f"id={qid}: repaired answer is non-finite")
    code = str(row.get("pandas_query") or "")
    try:
        refs = all_dataframe_refs(code)
    except SyntaxError as exc:
        raise ValueError(f"id={qid}: repaired query has invalid syntax") from exc
    used = {str(item.get("var")) for item in row.get("used_vars") or []
            if isinstance(item, dict)}
    if not refs or refs - used:
        raise ValueError(
            f"id={qid}: repaired query/evidence mismatch refs={sorted(refs)} "
            f"used={sorted(used)}"
        )


def _unique_rows(rows: list[dict], label: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for row in rows:
        if "id" not in row:
            raise ValueError(f"{label} row has no id")
        qid = int(row["id"])
        if qid in out:
            raise ValueError(f"{label} contains duplicate id={qid}")
        out[qid] = row
    if not out:
        raise ValueError(f"{label} is empty")
    return out


def _validate_universe(primary: dict[int, dict], retrieval: dict[int, dict]) -> None:
    if set(primary) != set(retrieval):
        raise ValueError("primary/retrieval id universes differ")
    mismatches = []
    for qid in sorted(primary):
        a = unicodedata.normalize("NFC", str(primary[qid].get("question") or "")).strip()
        b = unicodedata.normalize("NFC", str(retrieval[qid].get("question") or "")).strip()
        if not b:
            b = unicodedata.normalize(
                "NFC", str((retrieval[qid].get("route") or {}).get("question") or ""),
            ).strip()
        if not a or a != b:
            mismatches.append(qid)
    if mismatches:
        raise ValueError(f"primary/retrieval questions differ: ids={mismatches[:10]}")


def _one_signature(rows: dict[int, dict], label: str) -> str:
    signatures = {str(row.get("run_signature") or "").strip()
                  for row in rows.values()}
    if "" in signatures or len(signatures) != 1:
        raise ValueError(f"{label} must contain exactly one non-empty run_signature")
    return next(iter(signatures))


def _semantic_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = [
        Path(__file__).resolve(),
        root / "vifinqa" / "extraction" / "unit_policy.py",
        root / "vifinqa" / "utils" / "viet_text.py",
        root / "vifinqa" / "codegen" / "semantic.py",
    ]
    return {str(path.relative_to(root)).replace("\\", "/"): _sha256(path)
            for path in paths}


def _store_manifest(store_dir: Path, tickers: list[str]) -> dict[str, str]:
    store_dir = Path(store_dir)
    paths = [store_dir / "reports.parquet"]
    for ticker in tickers:
        paths.extend([
            store_dir / "tables" / f"{ticker}.parquet",
            store_dir / "cells" / f"{ticker}.parquet",
        ])
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"store manifest files missing: {missing}")
    return {str(path.relative_to(store_dir)).replace("\\", "/"): _sha256(path)
            for path in paths}


def _check_hash(actual: str, expected: str, label: str) -> None:
    if expected and actual.lower() != expected.strip().lower():
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected}, actual={actual}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _json_sha(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _number_literal(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return format(number, ".17g")


def _write_jsonl_exclusive(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
