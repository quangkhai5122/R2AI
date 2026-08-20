"""High-precision single-cell consensus overlays for v5.3.

V5.3a repairs successful one-fact lookups whose current answer disagrees with
a uniquely supported target cell.  V5.3b uses the same resolver for the frozen
structural-none lookup universe.  Both modes fail closed unless a target cell
is confirmed by an independent table or the following report's prior-period
column.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from ..extraction.unit_policy import resolve_stored_table_unit
from ..utils.io import read_jsonl
from ..utils.viet_text import label_metric_score, norm
from .hybrid import is_structural_none
from .semantic import all_dataframe_refs
from .semantic_repair import (
    SOURCE_NAME as V52A_SOURCE,
    ColumnRole,
    _StoreView,
    _check_hash,
    _full_header,
    _json_sha,
    _number_literal,
    _one_signature,
    _sha256,
    _silver_supports,
    _store_manifest,
    _unique_rows,
    _validate_repaired_record,
    _validate_universe,
    _write_json_exclusive,
    _write_jsonl_exclusive,
    classify_column_role,
    parse_simple_lookup,
)
from .units import cell_is_already_percent, percent_from_cell


POLICY_REPAIR = "v53a_single_cell_consensus_v1"
POLICY_RESCUE = "v53b_structural_lookup_consensus_v1"
SOURCE_REPAIR = "deterministic_v53a"
SOURCE_RESCUE = "deterministic_v53b"

_ALLOWED_OUTPUTS = {"number", "percent"}
_GENERIC_LABELS = {
    "cong",
    "tong cong",
    "tong",
    "so cuoi nam",
    "so dau nam",
    "nam nay",
    "nam truoc",
    "gia tri",
    "chi tieu",
    "ma so",
    "thuyet minh",
}


@dataclass
class CellCandidate:
    ticker: str
    report_id: str
    report_year: int
    doc_type: str
    table_pos: int
    row: int
    col: int
    label: str
    row_code: str
    col_name: str
    raw_value: float
    absolute_value: float
    answer: float
    metric_score: float
    role: dict
    input_scale: float
    output_scale: float
    unit: dict
    supports: list[dict]
    evidence_mode: str

    @property
    def stable_cell(self) -> tuple[str, int, int, int]:
        return (self.report_id, self.table_pos, self.row, self.col)


@dataclass
class ConsensusProposal:
    qid: int
    answer: float
    query: str
    used_vars: list[dict]
    provenance: dict


def discover_single_cell_consensus(
    primary_rows: list[dict],
    retrieval_rows: list[dict],
    store_dir: Path,
    *,
    mode: str,
) -> tuple[list[ConsensusProposal], dict]:
    """Discover v5.3a or v5.3b proposals without writing output."""
    if mode not in {"repair", "rescue"}:
        raise ValueError("mode must be repair or rescue")
    primary = _unique_rows(primary_rows, "primary")
    retrieval = _unique_rows(retrieval_rows, "retrieval")
    _validate_universe(primary, retrieval)
    view = _StoreView(Path(store_dir))
    # Questions are processed by id rather than ticker; retain all 100 ticker
    # shards so preflight does not repeatedly reload the same parquet files.
    view.store._cache_size = max(int(view.store._cache_size), 256)
    proposals: list[ConsensusProposal] = []
    counts = Counter()
    rejected: list[dict] = []
    target_ids: list[int] = []

    for qid in sorted(primary):
        row = primary[qid]
        route = retrieval[qid].get("route") or {}
        plan = route.get("plan") or {}
        counts["rows"] += 1

        if mode == "repair":
            if str(row.get("status") or "").lower() != "ok":
                counts["out_of_scope_not_ok"] += 1
                continue
            if str(row.get("source") or "") == V52A_SOURCE:
                counts["frozen_v52a_repair"] += 1
                continue
        else:
            if not is_structural_none(row):
                counts["out_of_scope_not_structural_none"] += 1
                continue

        if str(plan.get("op") or "") != "lookup":
            counts["out_of_scope_not_lookup"] += 1
            continue
        target_ids.append(qid)
        counts["target_lookup"] += 1
        facts = plan.get("facts") or []
        if len(facts) != 1:
            counts["not_single_fact"] += 1
            rejected.append(
                {"id": qid, "reason": "not_single_fact", "n_facts": len(facts)}
            )
            continue
        output_type = str(route.get("output_type") or "number")
        if output_type not in _ALLOWED_OUTPUTS:
            counts["unsupported_output_type"] += 1
            rejected.append(
                {
                    "id": qid,
                    "reason": "unsupported_output_type",
                    "output_type": output_type,
                }
            )
            continue
        if mode == "repair":
            suspect, suspect_reasons = _repair_suspect(row, output_type, view)
            if not suspect:
                counts["primary_not_suspect"] += 1
                continue
        else:
            suspect_reasons = ["structural_none_lookup"]

        proposal, reason, detail = _proposal_for_fact(
            qid,
            row,
            route,
            dict(facts[0]),
            view,
            mode=mode,
            trigger_reasons=suspect_reasons,
        )
        if proposal is None:
            counts[reason] += 1
            rejected.append({"id": qid, "reason": reason, **detail})
            continue
        counts["accepted"] += 1
        proposals.append(proposal)

    return proposals, {
        "mode": mode,
        "counts": dict(sorted(counts.items())),
        "target_ids": target_ids,
        "selected_ids": [proposal.qid for proposal in proposals],
        "rejected": rejected,
    }


def resolve_single_fact_candidates(
    *,
    view: _StoreView,
    route: dict,
    fact: dict,
    allow_direct_exact: bool = False,
) -> tuple[list[CellCandidate], str, dict]:
    """Resolve one routed fact to a unique signed-value consensus cluster."""
    ticker = str(fact.get("ticker") or "").upper()
    doc_type = str(fact.get("doc_type") or route.get("doc_type") or "")
    metric = str(fact.get("metric") or route.get("metric_norm") or "").strip()
    try:
        target_year = int(fact.get("year"))
    except (TypeError, ValueError):
        return [], "invalid_fact_year", {}
    if not ticker or not doc_type or not metric:
        return [], "incomplete_fact", {}

    report_ids = view.store.find_reports(
        ticker,
        target_year,
        doc_type,
        allow_fallback=False,
    )
    routed = {str(value) for value in route.get("report_ids") or []}
    if routed:
        exact_routed = [report_id for report_id in report_ids if report_id in routed]
        if exact_routed:
            report_ids = exact_routed
    if not report_ids:
        return (
            [],
            "target_report_missing",
            {"ticker": ticker, "year": target_year, "doc_type": doc_type},
        )

    candidates: list[CellCandidate] = []
    rejection_counts = Counter()
    for report_id in report_ids:
        report = view.report(report_id)
        if report is None:
            rejection_counts["report_not_found"] += 1
            continue
        cells = view.store.cells_of(ticker, [report_id])
        if cells.empty:
            rejection_counts["report_has_no_cells"] += 1
            continue
        for (table_pos, row_no), same_row in cells.groupby(
            ["table_pos", "row"], sort=False
        ):
            table = view.table(ticker, report_id, int(table_pos))
            if table is None:
                continue
            roles = _roles_for_row(table, report, target_year, same_row)
            unit = resolve_stored_table_unit(
                table.unit_scale,
                table.unit_source,
                str(table.context or ""),
            )
            context = str(table.context or "")
            for cell in same_row.itertuples():
                role = roles.get(int(cell.col))
                label = str(cell.label or "").strip()
                direct_exact_kind = (
                    _direct_exact_kind(
                        metric=metric,
                        label=label,
                        col_name=str(cell.col_name or ""),
                        context=context,
                        output_type=str(route.get("output_type") or "number"),
                    )
                    if allow_direct_exact
                    else ""
                )
                if direct_exact_kind == "ownership_percentage" and (
                    role is None or role.role != "target_value" or role.confidence < 3
                ):
                    role = ColumnRole(
                        "target_value",
                        "semantic_percentage_column",
                        str(cell.col_name or ""),
                        target_year,
                        3,
                    )
                if role is None or role.role != "target_value" or role.confidence < 3:
                    continue
                label = str(cell.label or "").strip()
                label_norm = norm(label)
                if label_norm in _GENERIC_LABELS or len(label_norm) < 5:
                    rejection_counts["generic_label"] += 1
                    continue
                direct_score = label_metric_score(label, metric)
                context_score = label_metric_score(f"{context} {label}", metric)
                metric_score = max(direct_score, context_score)
                if (
                    not direct_exact_kind
                    and direct_score < 70.0
                    and not (direct_score >= 55.0 and context_score >= 78.0)
                ):
                    rejection_counts["metric_below_threshold"] += 1
                    continue
                raw_value = float(cell.value)
                if not direct_exact_kind and not _strict_metric_label_match(
                    metric, label
                ):
                    rejection_counts["metric_token_mismatch"] += 1
                    continue
                absolute_value = raw_value * unit.effective_scale
                answer_data = _answer_and_scales(
                    route,
                    output_type=str(route.get("output_type") or "number"),
                    raw_value=raw_value,
                    absolute_value=absolute_value,
                    label=label,
                    col_name=str(cell.col_name or ""),
                    effective_scale=unit.effective_scale,
                )
                if answer_data is None:
                    rejection_counts["invalid_answer_or_scale"] += 1
                    continue
                answer, input_scale, output_scale = answer_data
                if answer == 0.0:
                    rejection_counts["zero_candidate"] += 1
                    continue
                supports = _silver_supports(
                    view=view,
                    ticker=ticker,
                    report=report,
                    table=table,
                    target=cell,
                    target_year=target_year,
                    metric=metric,
                    absolute_value=absolute_value,
                )
                evidence_mode = "independent_consensus"
                if not supports and direct_exact_kind:
                    evidence_mode = f"direct_exact:{direct_exact_kind}"

                if not supports and not direct_exact_kind:
                    rejection_counts["no_independent_support"] += 1
                    continue
                candidates.append(
                    CellCandidate(
                        ticker=ticker,
                        report_id=report_id,
                        report_year=int(report.year),
                        doc_type=str(report.doc_type),
                        table_pos=int(table_pos),
                        row=int(cell.row),
                        col=int(cell.col),
                        label=label,
                        row_code=str(getattr(cell, "row_code", "") or ""),
                        col_name=str(cell.col_name or ""),
                        raw_value=raw_value,
                        absolute_value=absolute_value,
                        answer=answer,
                        metric_score=float(metric_score),
                        role=asdict(role),
                        input_scale=input_scale,
                        output_scale=output_scale,
                        unit={
                            "stored_scale": unit.stored_scale,
                            "effective_scale": unit.effective_scale,
                            "stored_source": unit.stored_source,
                            "effective_source": unit.effective_source,
                            "resolution": unit.reason,
                            "terminal_bare_vnd": unit.terminal_bare_vnd,
                        },
                        supports=supports,
                        evidence_mode=evidence_mode,
                    )
                )

    if not candidates:
        return (
            [],
            "no_supported_candidate",
            {
                "candidate_rejections": dict(sorted(rejection_counts.items())),
                "reports": report_ids,
            },
        )
    unique_cells = {candidate.stable_cell: candidate for candidate in candidates}
    candidates = list(unique_cells.values())
    clusters: dict[str, list[CellCandidate]] = defaultdict(list)
    for candidate in candidates:
        clusters[_answer_key(candidate.answer)].append(candidate)
    if len(clusters) != 1:
        return (
            [],
            "competing_answer_clusters",
            {
                "clusters": {
                    key: [_candidate_brief(value) for value in values]
                    for key, values in sorted(clusters.items())
                },
            },
        )
    only = next(iter(clusters.values()))
    only.sort(
        key=lambda value: (
            -value.metric_score,
            -int(value.role.get("confidence") or 0),
            -len(value.supports),
            value.report_id,
            value.table_pos,
            value.row,
            value.col,
        )
    )
    return (
        only,
        "accepted",
        {
            "candidate_count": len(only),
            "answer_cluster": _answer_key(only[0].answer),
        },
    )


def build_single_cell_consensus_overlay(
    primary_path: Path,
    retrieval_path: Path,
    store_dir: Path,
    out_path: Path,
    *,
    mode: str,
    audit_path: Path | None = None,
    expected_selected_ids: set[int] | None = None,
    expected_target_ids: set[int] | None = None,
    expected_primary_signature: str = "",
    expected_primary_sha256: str = "",
    expected_retrieval_sha256: str = "",
) -> dict:
    """Build an immutable complete v5.3a/v5.3b overlay plus audit."""
    if mode not in {"repair", "rescue"}:
        raise ValueError("mode must be repair or rescue")
    primary_path, retrieval_path = Path(primary_path), Path(retrieval_path)
    store_dir, out_path = Path(store_dir), Path(out_path)
    audit_path = Path(audit_path) if audit_path else out_path.with_suffix(".audit.json")
    for path in (out_path, audit_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    primary_sha, retrieval_sha = _sha256(primary_path), _sha256(retrieval_path)
    _check_hash(primary_sha, expected_primary_sha256, "primary")
    _check_hash(retrieval_sha, expected_retrieval_sha256, "retrieval")
    primary_rows, retrieval_rows = read_jsonl(primary_path), read_jsonl(retrieval_path)
    primary = _unique_rows(primary_rows, "primary")
    primary_signature = _one_signature(primary, "primary")
    if expected_primary_signature and primary_signature != expected_primary_signature:
        raise ValueError(
            "primary run_signature mismatch: "
            f"expected={expected_primary_signature}, actual={primary_signature}"
        )

    proposals, discovery = discover_single_cell_consensus(
        primary_rows,
        retrieval_rows,
        store_dir,
        mode=mode,
    )
    selected_ids = [proposal.qid for proposal in proposals]
    target_ids = [int(value) for value in discovery["target_ids"]]
    if expected_selected_ids is not None and set(selected_ids) != {
        int(value) for value in expected_selected_ids
    }:
        raise ValueError(
            "selected id guard mismatch: "
            f"expected={sorted(expected_selected_ids)}, actual={selected_ids}"
        )
    if expected_target_ids is not None and set(target_ids) != {
        int(value) for value in expected_target_ids
    }:
        raise ValueError(
            "target id guard mismatch: "
            f"expected={sorted(expected_target_ids)}, actual={target_ids}"
        )

    policy = POLICY_REPAIR if mode == "repair" else POLICY_RESCUE
    source = SOURCE_REPAIR if mode == "repair" else SOURCE_RESCUE
    selected_tickers = sorted(
        {str(proposal.provenance["selected_cell"]["ticker"]) for proposal in proposals}
    )
    semantic_manifest = _semantic_manifest()
    store_manifest = _store_manifest(store_dir, selected_tickers)
    repair_digest = _json_sha(
        [
            {
                "id": proposal.qid,
                "answer": proposal.answer,
                "query": proposal.query,
                "provenance": proposal.provenance,
            }
            for proposal in proposals
        ]
    )
    run_signature = _json_sha(
        {
            "policy": policy,
            "primary_signature": primary_signature,
            "primary_sha256": primary_sha,
            "retrieval_sha256": retrieval_sha,
            "semantic_manifest": semantic_manifest,
            "store_manifest": store_manifest,
            "repair_digest": repair_digest,
            "target_ids": target_ids,
            "selected_ids": selected_ids,
        }
    )

    by_id = {proposal.qid: proposal for proposal in proposals}
    output: list[dict] = []
    for qid in sorted(primary):
        row = dict(primary[qid])
        row["run_signature"] = run_signature
        proposal = by_id.get(qid)
        if proposal is not None:
            row.update(
                {
                    "answer": proposal.answer,
                    "pandas_query": proposal.query,
                    "used_vars": proposal.used_vars,
                    "source": source,
                    "status": "ok",
                    "detail": f"deterministic consensus {policy}",
                    "single_cell_consensus_provenance": proposal.provenance,
                }
            )
            _validate_repaired_record(row, qid)
        output.append(row)

    _write_jsonl_exclusive(out_path, output)
    output_sha = _sha256(out_path)
    audit = {
        "schema_version": "v53_single_cell_consensus_audit_v1",
        "policy": policy,
        "mode": mode,
        "inputs": {
            "primary": {
                "path": str(primary_path),
                "sha256": primary_sha,
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
            "targets": len(target_ids),
            "selected": len(proposals),
            "unchanged_semantic_rows": len(output) - len(proposals),
        },
        "target_ids": target_ids,
        "selected_ids": selected_ids,
        "selected": [
            {
                "id": proposal.qid,
                "answer": proposal.answer,
                "trigger_reasons": proposal.provenance["trigger_reasons"],
                "support_count": proposal.provenance["support_count"],
                "selected_cell": proposal.provenance["selected_cell"],
            }
            for proposal in proposals
        ],
        "output": {"path": str(out_path), "sha256": output_sha, "rows": len(output)},
    }
    _write_json_exclusive(audit_path, audit)
    return audit


def _proposal_for_fact(
    qid: int,
    row: dict,
    route: dict,
    fact: dict,
    view: _StoreView,
    *,
    mode: str,
    trigger_reasons: list[str],
) -> tuple[ConsensusProposal | None, str, dict]:
    candidates, reason, detail = resolve_single_fact_candidates(
        view=view,
        route=route,
        fact=fact,
        allow_direct_exact=(mode == "rescue"),
    )
    if not candidates:
        return None, reason, detail
    selected = candidates[0]
    try:
        original_answer = float(row.get("answer"))
    except (TypeError, ValueError):
        original_answer = float("nan")
    if mode == "repair" and math.isclose(
        selected.answer, original_answer, rel_tol=0.0, abs_tol=1e-12
    ):
        return (
            None,
            "consensus_matches_primary",
            {
                "answer": selected.answer,
                "candidate_count": len(candidates),
            },
        )

    var = "df1"
    query = (
        f"round(float({var}.loc[({var}['row'] == {selected.row}) & "
        f"({var}['col'] == {selected.col}), 'value'].iloc[0]) * "
        f"{_number_literal(selected.input_scale)} / "
        f"{_number_literal(selected.output_scale)}, 2)"
    )
    binding = {
        "var": var,
        "report_id": selected.report_id,
        "table_pos": selected.table_pos,
    }
    selected_summary = _candidate_brief(selected)
    provenance = {
        "schema_version": 1,
        "policy": POLICY_REPAIR if mode == "repair" else POLICY_RESCUE,
        "mode": mode,
        "trigger_reasons": trigger_reasons,
        "original": {
            "source": row.get("source"),
            "status": row.get("status"),
            "answer": row.get("answer"),
            "query": row.get("pandas_query"),
        },
        "fact": fact,
        "selected_cell": selected_summary,
        "support_count": len(selected.supports),
        "evidence_mode": selected.evidence_mode,
        "supports": selected.supports,
        "same_answer_candidate_cells": [
            _candidate_brief(candidate) for candidate in candidates
        ],
    }
    return (
        ConsensusProposal(
            qid=qid,
            answer=selected.answer,
            query=query,
            used_vars=[binding],
            provenance=provenance,
        ),
        "accepted",
        detail,
    )


def _repair_suspect(
    row: dict, output_type: str, view: _StoreView
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if output_type == "percent":
        reasons.append("lookup_percent_consensus_disagreement")
    parsed = parse_simple_lookup(str(row.get("pandas_query") or ""))
    if parsed is not None:
        bindings = [
            item
            for item in row.get("used_vars") or []
            if isinstance(item, dict) and str(item.get("var")) == parsed.var
        ]
        if len(bindings) == 1:
            binding = bindings[0]
            report_id = str(binding.get("report_id") or "")
            report = view.report(report_id)
            if report is not None:
                try:
                    table_pos = int(binding.get("table_pos"))
                except (TypeError, ValueError):
                    table_pos = -1
                table = view.table(str(report.ticker), report_id, table_pos)
                if table is not None:
                    cells = view.store.cells_of(str(report.ticker), [report_id])
                    selected = cells[
                        (cells.table_pos == table_pos)
                        & cells.label.str.contains(
                            parsed.label_needle,
                            case=False,
                            regex=False,
                            na=False,
                        )
                        & (cells.col == parsed.selected_col)
                    ]
                    if len(selected) != 1:
                        reasons.append("primary_lookup_non_unique")
                    else:
                        selected_cell = selected.iloc[0]
                        same_row = cells[
                            (cells.table_pos == table_pos)
                            & (cells.row == int(selected_cell.row))
                        ]
                        roles = _roles_for_row(
                            table,
                            report,
                            int(report.year),
                            same_row,
                        )
                        role = roles.get(int(selected_cell.col))
                        if role is not None and role.role in {"note", "code"}:
                            reasons.append("primary_selected_non_value_column")
                    unit = resolve_stored_table_unit(
                        table.unit_scale,
                        table.unit_source,
                        str(table.context or ""),
                    )
                    if unit.changed:
                        reasons.append("terminal_vnd_unit_override")
    try:
        refs = all_dataframe_refs(str(row.get("pandas_query") or ""))
    except SyntaxError:
        return False, []
    if len(refs) != 1:
        return False, []
    return bool(reasons), sorted(set(reasons))


def _roles_for_row(table, report, target_year: int, same_row) -> dict:
    base_roles = {}
    value_cols = []
    for cell in same_row.itertuples():
        header = _full_header(table, int(cell.col), str(cell.col_name or ""))
        role = classify_column_role(header, int(report.year), target_year)
        base_roles[int(cell.col)] = role
        if role.role not in {"note", "code"}:
            value_cols.append(int(cell.col))
    first_value_col = min(value_cols) if value_cols else None
    return {
        int(cell.col): classify_column_role(
            _full_header(table, int(cell.col), str(cell.col_name or "")),
            int(report.year),
            target_year,
            positional_current=(int(cell.col) == first_value_col),
        )
        for cell in same_row.itertuples()
    }


def _strict_metric_label_match(metric: str, label: str) -> bool:
    """Require every material routed-metric token in the row label.

    Fuzzy overlap alone confused ``nguyen gia`` (historical cost) with
    ``gia von`` (cost of services). This gate deliberately favours precision;
    narrow audited aliases are handled separately by ``_direct_exact_kind``.
    """
    ignored = {"cua", "tai", "den", "ngay", "nam", "thang", "vao"}
    metric_tokens = {
        token for token in norm(metric).split() if token and token not in ignored
    }
    label_tokens = set(norm(label).split())
    return bool(metric_tokens) and metric_tokens.issubset(label_tokens)


def _direct_exact_kind(
    *, metric: str, label: str, col_name: str, context: str, output_type: str
) -> str:
    """Return a narrow semantic alias class for support-less rescue only."""
    metric_norm, label_norm = norm(metric), norm(label)
    col_norm, context_norm = norm(col_name), norm(context)
    metric_tokens = set(metric_norm.split())
    label_tokens = set(label_norm.split())

    if output_type == "number" and {"ngoai", "te", "usd"}.issubset(metric_tokens):
        if "usd" in label_tokens and (
            {"do", "la", "my"}.issubset(label_tokens) or "ngoai te" in context_norm
        ):
            return "foreign_currency_usd"

    if output_type == "percent" and {"so", "huu"}.issubset(metric_tokens):
        entity_tokens = metric_tokens - {"so", "huu", "ty", "le", "cua"}
        if (
            entity_tokens
            and entity_tokens.issubset(label_tokens)
            and "ty le" in col_norm
            and ("loi ich" in col_norm or "quyen bieu quyet" in col_norm)
        ):
            return "ownership_percentage"
    return ""


def _answer_and_scales(
    route: dict,
    *,
    output_type: str,
    raw_value: float,
    absolute_value: float,
    label: str,
    col_name: str,
    effective_scale: float,
) -> tuple[float, float, float] | None:
    if output_type == "percent":
        answer = round(percent_from_cell(raw_value, label, col_name), 2)
        input_scale = (
            1.0 if cell_is_already_percent(label, col_name, raw_value) else 100.0
        )
        output_scale = 1.0
    elif output_type == "number":
        try:
            output_scale = float(route.get("unit_scale") or 1.0)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(output_scale) or output_scale <= 0:
            return None
        answer = round(absolute_value / output_scale, 2)
        input_scale = float(effective_scale)
    else:
        return None
    if not all(math.isfinite(value) for value in (answer, input_scale, output_scale)):
        return None
    if output_type == "percent" and abs(answer) > 1000:
        return None
    return answer, input_scale, output_scale


def _answer_key(answer: float) -> str:
    return f"{round(float(answer), 2):.2f}"


def _candidate_brief(candidate: CellCandidate) -> dict:
    return {
        "ticker": candidate.ticker,
        "report_id": candidate.report_id,
        "report_year": candidate.report_year,
        "doc_type": candidate.doc_type,
        "table_pos": candidate.table_pos,
        "row": candidate.row,
        "col": candidate.col,
        "label": candidate.label,
        "row_code": candidate.row_code,
        "col_name": candidate.col_name,
        "raw_value": candidate.raw_value,
        "absolute_value": candidate.absolute_value,
        "answer": candidate.answer,
        "metric_score": round(candidate.metric_score, 2),
        "column_role": candidate.role,
        "input_scale": candidate.input_scale,
        "output_scale": candidate.output_scale,
        "unit": candidate.unit,
        "support_count": len(candidate.supports),
        "evidence_mode": candidate.evidence_mode,
    }


def _semantic_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = [
        Path(__file__).resolve(),
        root / "vifinqa" / "codegen" / "semantic_repair.py",
        root / "vifinqa" / "codegen" / "units.py",
        root / "vifinqa" / "codegen" / "hybrid.py",
        root / "vifinqa" / "extraction" / "unit_policy.py",
        root / "vifinqa" / "utils" / "viet_text.py",
        root / "vifinqa" / "codegen" / "semantic.py",
    ]
    return {
        str(path.relative_to(root)).replace("\\", "/"): _sha256(path) for path in paths
    }
