"""Fail-closed multi-operand column/period/unit repair.

V5.2b extends the proven v5.2a policy to a deliberately small set of simple
multi-operand expressions. It never repairs failed/structural-none records and
never repairs ranking, filtering, conditional, or nested programs. Every
corrected operand needs independent signed silver evidence.
"""
from __future__ import annotations

import ast
import itertools
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from ..extraction.unit_policy import resolve_stored_table_unit
from ..utils.io import read_jsonl
from ..utils.viet_text import label_metric_score, norm
from .semantic import all_dataframe_refs, validate_generated_answer
from .semantic_repair import (
    RepairProposal,
    _StoreView,
    _cell_summary,
    _check_hash,
    _full_header,
    _json_sha,
    _number_literal,
    _one_signature,
    _sha256,
    _silver_supports,
    _store_manifest,
    _text_sha,
    _unique_rows,
    _validate_repaired_record,
    _validate_universe,
    _write_json_exclusive,
    _write_jsonl_exclusive,
    classify_column_role,
)


POLICY_VERSION = "v52b_multi_operand_signed_silver_v1"
SOURCE_NAME = "deterministic_v52b"

_SUPPORTED = {
    "difference": ("number", 2, 2),
    "growth_pct": ("percent", 2, 2),
    "ratio": ("percent", 2, 2),
    "average": ("number", 2, 6),
}
_DIFFERENCE_CUE = re.compile(
    r"\b(?:so voi|giua|muc thay doi|chenh lech giua|tu nam .{0,20} den|"
    r"cao hon .{0,30} bao nhieu|thap hon .{0,30} bao nhieu|"
    r"tang .{0,30} bao nhieu|giam .{0,30} bao nhieu)\b"
)
_GROWTH_CUE = re.compile(
    r"\b(?:tang truong|toc do tang|ty le tang|ty le giam|"
    r"tang .{0,30} phan tram|giam .{0,30} phan tram)\b"
)
_RATIO_CUE = re.compile(r"\b(?:ty le|ti le|ty trong|bien loi nhuan|he so|tren)\b")
_AVERAGE_CUE = re.compile(r"\b(?:trung binh|binh quan)\b")


@dataclass(frozen=True)
class CellLeaf:
    var: str
    selected_col: int
    label_needle: str = ""
    selected_row: int | None = None

    @property
    def key(self) -> tuple[str, str, int | None, int]:
        return (self.var, self.label_needle, self.selected_row, self.selected_col)


@dataclass
class ParsedExpression:
    body: ast.AST
    occurrences: list[CellLeaf]
    leaves: list[CellLeaf]


@dataclass
class _Evidence:
    leaf: CellLeaf
    binding: dict
    report: object
    table: object
    selected_cell: object
    same_row: object
    selected_role: object
    unit: object


def parse_multi_operand_expression(
    query: str, operation: str, operand_count: int,
) -> ParsedExpression | None:
    """Parse only known one-expression templates and reject everything else."""
    try:
        tree = ast.parse(str(query or ""), mode="eval")
    except SyntaxError:
        return None
    body = _unwrap_round(tree.body)
    if body is None:
        return None

    calls = [node for node in ast.walk(body) if _is_float_call(node)]
    occurrences: list[CellLeaf] = []
    call_to_key: dict[int, tuple[str, str, int | None, int]] = {}
    for call in calls:
        leaf = _parse_cell_call(call)
        if leaf is None:
            return None
        occurrences.append(leaf)
        call_to_key[id(call)] = leaf.key
    if not occurrences:
        return None

    leaves: list[CellLeaf] = []
    by_key: dict[tuple[str, str, int | None, int], CellLeaf] = {}
    for leaf in occurrences:
        if leaf.key not in by_key:
            by_key[leaf.key] = leaf
            leaves.append(leaf)
    if len(leaves) != int(operand_count):
        return None
    if all_dataframe_refs(str(query or "")) != {leaf.var for leaf in leaves}:
        return None

    symbolic = _LeafReplacer(call_to_key, list(by_key)).visit(body)
    if not _safe_symbolic_expression(symbolic, set(range(len(leaves)))):
        return None
    if not _operation_shape_matches(symbolic, operation, len(leaves)):
        return None
    return ParsedExpression(body=symbolic, occurrences=occurrences, leaves=leaves)


def discover_multi_operand_repairs(
    primary_rows: list[dict],
    retrieval_rows: list[dict],
    store_dir: Path,
) -> tuple[list[RepairProposal], dict]:
    """Discover v5.2b proposals without mutating or writing an artifact."""
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
        plan = route.get("plan") or {}
        counts["rows"] += 1
        if str(row.get("status", "")).lower() != "ok":
            counts["out_of_scope_not_ok"] += 1
            continue
        counts["status_ok"] += 1
        operation = str(plan.get("op") or "")
        facts = plan.get("facts") or []
        supported, normalized_facts, reason = _supported_plan(
            operation, route, facts, str(row.get("question") or ""),
        )
        if not supported:
            counts[reason] += 1
            continue
        counts[f"scope_{operation}"] += 1
        proposal, reason, detail = _proposal_for_row(
            qid, row, route, operation, normalized_facts, view,
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


def build_multi_operand_repair_overlay(
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
    """Build a complete v5.2b overlay, refusing every overwrite or drift."""
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

    proposals, discovery = discover_multi_operand_repairs(
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
        for item in proposal.provenance.get("operands", [])
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
        row = dict(primary[qid])
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
                "semantic_repair_v52b_provenance": proposal.provenance,
            })
            _validate_repaired_record(row, qid)
        output.append(row)

    _write_jsonl_exclusive(out_path, output)
    output_sha = _sha256(out_path)
    audit = {
        "schema_version": "v52b_multi_operand_repair_audit_v1",
        "policy": POLICY_VERSION,
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
            "selected": len(proposals),
            "unchanged_semantic_rows": len(output) - len(proposals),
        },
        "selected_ids": selected_ids,
        "selected": [
            {
                "id": proposal.qid,
                "operation": proposal.provenance["operation"],
                "answer": proposal.answer,
                "query": proposal.query,
                "trigger_reasons": proposal.provenance["trigger_reasons"],
                "operand_support_counts": [
                    int(item["silver_support_count"])
                    for item in proposal.provenance["operands"]
                ],
            }
            for proposal in proposals
        ],
        "output": {"path": str(out_path), "sha256": output_sha, "rows": len(output)},
    }
    _write_json_exclusive(audit_path, audit)
    return audit


def _proposal_for_row(
    qid: int, row: dict, route: dict, operation: str,
    facts: list[dict], view: _StoreView,
):
    parsed = parse_multi_operand_expression(
        str(row.get("pandas_query") or ""), operation, len(facts),
    )
    if parsed is None:
        return None, "query_shape_mismatch", {}

    refs = {leaf.var for leaf in parsed.leaves}
    bindings_by_var: dict[str, dict] = {}
    for item in row.get("used_vars") or []:
        if not isinstance(item, dict):
            continue
        var = str(item.get("var") or "")
        if var in refs:
            if var in bindings_by_var:
                return None, "ambiguous_evidence_binding", {}
            bindings_by_var[var] = dict(item)
    if set(bindings_by_var) != refs:
        return None, "missing_evidence_binding", {}

    evidence: list[_Evidence] = []
    for leaf in parsed.leaves:
        item, reason = _evidence_for_leaf(
            leaf, bindings_by_var[leaf.var], view,
        )
        if item is None:
            return None, reason, {}
        evidence.append(item)

    assignment, reason = _assign_facts(facts, evidence)
    if assignment is None:
        return None, reason, {}

    operands = []
    all_triggers: list[str] = []
    for fact_index, evidence_index in enumerate(assignment):
        resolved, reason, detail = _resolve_operand(
            facts[fact_index], evidence[evidence_index], view,
        )
        if resolved is None:
            if all_triggers or detail.get("repair_triggered"):
                detail = {**detail, "repair_triggered": True}
            return None, reason, detail
        operands.append(resolved)
        all_triggers.extend(resolved["trigger_reasons"])

    triggers = sorted(set(all_triggers))
    if not triggers:
        return None, "no_repair_trigger", {}
    trigger_detail = {"repair_triggered": True, "trigger_reasons": triggers}
    stable_cells = [
        (
            item["ticker"], item["report_id"], item["table_pos"],
            item["cell"]["row"], item["cell"]["col"],
        )
        for item in operands
    ]
    if len(set(stable_cells)) != len(stable_cells):
        return None, "duplicate_repaired_cell", trigger_detail

    try:
        original_answer = float(row.get("answer"))
    except (TypeError, ValueError):
        return None, "invalid_original_answer", trigger_detail
    output_scale = _output_scale(route, operation)
    if output_scale is None:
        return None, "invalid_output_scale", trigger_detail
    values = [float(item["absolute_value"]) for item in operands]
    try:
        answer = _compute_answer(operation, values, output_scale)
    except (ValueError, ZeroDivisionError):
        return None, "invalid_repaired_value", trigger_detail
    if not math.isfinite(answer):
        return None, "invalid_repaired_value", trigger_detail
    if operation in {"ratio", "growth_pct"} and abs(answer) > 10000:
        return None, "implausible_percent", {
            **trigger_detail, "candidate_answer": answer,
        }
    if math.isclose(answer, original_answer, rel_tol=0.0, abs_tol=1e-12):
        return None, "repair_does_not_change_answer", trigger_detail

    terms = [_exact_term(item) for item in operands]
    query = _compile_query(operation, terms, output_scale)
    used_vars = []
    seen_vars = set()
    for item in operands:
        var = str(item["var"])
        if var not in seen_vars:
            used_vars.append(bindings_by_var[var])
            seen_vars.add(var)
    semantic = validate_generated_answer(query, seen_vars, answer, route)
    if not semantic.ok:
        return None, "semantic_validation_failed", {
            **trigger_detail, "semantic": semantic.to_dict(),
        }

    provenance = {
        "schema_version": 1,
        "policy": POLICY_VERSION,
        "operation": operation,
        "trigger_reasons": triggers,
        "original": {
            "source": row.get("source"),
            "status": row.get("status"),
            "answer": original_answer,
            "query": row.get("pandas_query"),
        },
        "output_scale": output_scale,
        "output_unit": route.get("unit_name"),
        "operands": operands,
    }
    return RepairProposal(
        qid=qid,
        answer=answer,
        query=query,
        used_vars=used_vars,
        provenance=provenance,
    ), "accepted", trigger_detail


def _supported_plan(
    operation: str, route: dict, facts: list[dict], question: str,
):
    if operation not in _SUPPORTED:
        return False, facts, "out_of_scope_operation"
    output_type, minimum, maximum = _SUPPORTED[operation]
    if str(route.get("output_type") or "") != output_type:
        return False, facts, "out_of_scope_output_type"
    if not isinstance(facts, list) or not minimum <= len(facts) <= maximum:
        return False, facts, "out_of_scope_fact_arity"
    text = norm(question)
    cue = {
        "difference": _DIFFERENCE_CUE,
        "growth_pct": _GROWTH_CUE,
        "ratio": _RATIO_CUE,
        "average": _AVERAGE_CUE,
    }[operation]
    if not cue.search(text):
        return False, facts, "question_operation_cue_missing"
    normalized = [dict(fact) for fact in facts if isinstance(fact, dict)]
    if len(normalized) != len(facts):
        return False, facts, "invalid_fact"
    if operation == "ratio":
        numerators = [
            fact for fact in normalized if fact.get("role") == "numerator"
        ]
        denominators = [
            fact for fact in normalized if fact.get("role") == "denominator"
        ]
        if len(numerators) != 1 or len(denominators) != 1:
            return False, facts, "ratio_roles_missing"
        normalized = [numerators[0], denominators[0]]
    elif operation == "growth_pct":
        # The router emits chronological facts, while the formula registry is
        # explicitly (end, base). Never inherit the expression's accidental
        # operand order: prove a same-series two-period comparison and order it.
        tickers = {str(fact.get("ticker") or "") for fact in normalized}
        doc_types = {str(fact.get("doc_type") or "") for fact in normalized}
        metrics = {norm(str(fact.get("metric") or "")) for fact in normalized}
        try:
            years = [int(fact.get("year")) for fact in normalized]
        except (TypeError, ValueError):
            return False, facts, "growth_series_invalid"
        if len(tickers) != 1 or len(doc_types) != 1 or len(metrics) != 1 \
                or len(set(years)) != 2:
            return False, facts, "growth_series_invalid"
        normalized.sort(
            key=lambda fact: int(fact["year"]), reverse=True,
        )
    return True, normalized, "supported"


def _evidence_for_leaf(
    leaf: CellLeaf, binding: dict, view: _StoreView,
):
    report_id = str(binding.get("report_id") or "")
    try:
        table_pos = int(binding.get("table_pos"))
    except (TypeError, ValueError):
        return None, "invalid_evidence_binding"
    report = view.report(report_id)
    if report is None:
        return None, "report_not_found"
    ticker = str(report.ticker)
    table = view.table(ticker, report_id, table_pos)
    if table is None:
        return None, "table_not_found"
    cells = view.store.cells_of(ticker, [report_id])
    table_cells = cells[cells.table_pos == table_pos]
    selected = table_cells[table_cells.col == leaf.selected_col]
    if leaf.selected_row is not None:
        selected = selected[selected.row == int(leaf.selected_row)]
    else:
        selected = selected[selected.label.str.contains(
            leaf.label_needle, case=False, regex=False, na=False,
        )]
    if len(selected) != 1:
        return None, "selected_cell_not_unique"
    cell = selected.iloc[0]
    same_row = table_cells[table_cells.row == int(cell.row)].copy()
    if same_row.empty:
        return None, "selected_row_missing"
    selected_role = _role_for_cell(
        cell, same_row, table, int(report.year), int(report.year),
    )
    unit = resolve_stored_table_unit(
        table.unit_scale, table.unit_source, str(table.context or ""),
    )
    return _Evidence(
        leaf=leaf,
        binding=binding,
        report=report,
        table=table,
        selected_cell=cell,
        same_row=same_row,
        selected_role=selected_role,
        unit=unit,
    ), "ok"


def _assign_facts(facts: list[dict], evidence: list[_Evidence]):
    if len(facts) != len(evidence):
        return None, "fact_leaf_count_mismatch"
    scored = []
    for permutation in itertools.permutations(range(len(evidence))):
        scores = []
        valid = True
        for fact_index, evidence_index in enumerate(permutation):
            fact = facts[fact_index]
            item = evidence[evidence_index]
            report = item.report
            try:
                target_year = int(fact.get("year"))
            except (TypeError, ValueError):
                valid = False
                break
            if (
                str(fact.get("ticker") or "") != str(report.ticker)
                or str(fact.get("doc_type") or "") != str(report.doc_type)
                or int(report.year) not in {target_year, target_year + 1}
            ):
                valid = False
                break
            metric_score = label_metric_score(
                str(item.selected_cell.label), str(fact.get("metric") or ""),
            )
            if metric_score < 55.0:
                valid = False
                break
            scores.append(float(metric_score))
        if valid:
            scored.append((round(sum(scores), 8), permutation))
    if not scored:
        return None, "fact_assignment_missing"
    best_score = max(item[0] for item in scored)
    best = [item[1] for item in scored if item[0] == best_score]
    if len(best) != 1:
        return None, "fact_assignment_ambiguous"
    return list(best[0]), "ok"


def _resolve_operand(fact: dict, item: _Evidence, view: _StoreView):
    report = item.report
    table = item.table
    try:
        target_year = int(fact.get("year"))
    except (TypeError, ValueError):
        return None, "invalid_fact_year", {}
    candidates = []
    for cell in item.same_row.itertuples():
        role = _role_for_cell(
            cell, item.same_row, table, int(report.year), target_year,
        )
        if role.role == "target_value":
            candidates.append((role.confidence, int(cell.col), cell, role))
    selected_role = _role_for_cell(
        item.selected_cell, item.same_row, table, int(report.year), target_year,
    )
    triggers = []
    if selected_role.role in {"note", "code"}:
        triggers.append("selected_non_value_column")
    if item.unit.changed:
        triggers.append("terminal_vnd_unit_override")
    detail = {
        "repair_triggered": bool(triggers),
        "trigger_reasons": triggers,
        "selected_role": asdict(selected_role),
    }
    if not candidates:
        return None, "target_column_missing", detail
    confidence = max(value[0] for value in candidates)
    candidates = [value for value in candidates if value[0] == confidence]
    if len(candidates) != 1:
        return None, "target_column_ambiguous", {
            **detail,
            "candidate_cols": sorted(value[1] for value in candidates),
        }
    _confidence, _col, target, target_role = candidates[0]
    metric = str(fact.get("metric") or "")
    metric_score = label_metric_score(str(target.label), metric)
    if metric_score < 55.0:
        return None, "target_metric_mismatch", {
            **detail, "metric_score": round(float(metric_score), 2),
        }
    absolute_value = float(target.value) * item.unit.effective_scale
    if not math.isfinite(absolute_value):
        return None, "invalid_operand_value", detail
    supports = _silver_supports(
        view=view,
        ticker=str(report.ticker),
        report=report,
        table=table,
        target=target,
        target_year=target_year,
        metric=metric,
        absolute_value=absolute_value,
    )
    if not supports:
        return None, "operand_silver_no_signed_support", {
            **detail,
            "fact": fact,
            "candidate_cell": _cell_summary(target, target_role),
            "candidate_absolute_value": absolute_value,
        }
    context = str(table.context or "")
    return {
        "fact": fact,
        "var": item.leaf.var,
        "ticker": str(report.ticker),
        "report_id": str(report.report_id),
        "report_year": int(report.year),
        "doc_type": str(report.doc_type),
        "table_pos": int(table.table_pos),
        "original_selector": {
            "label_needle": item.leaf.label_needle,
            "row": item.leaf.selected_row,
            "col": item.leaf.selected_col,
            "role": asdict(selected_role),
        },
        "cell": _cell_summary(target, target_role),
        "metric_score": round(float(metric_score), 2),
        "trigger_reasons": triggers,
        "absolute_value": absolute_value,
        "unit": {
            "stored_scale": item.unit.stored_scale,
            "effective_scale": item.unit.effective_scale,
            "stored_source": item.unit.stored_source,
            "effective_source": item.unit.effective_source,
            "resolution": item.unit.reason,
            "terminal_bare_vnd": item.unit.terminal_bare_vnd,
            "context_sha256": _text_sha(context),
        },
        "silver_support_count": len(supports),
        "silver_supports": supports,
    }, "ok", detail


def _role_for_cell(
    cell, same_row, table, report_year: int, target_year: int,
):
    value_cols = []
    for row_cell in same_row.itertuples():
        header = _full_header(table, int(row_cell.col), str(row_cell.col_name))
        base = classify_column_role(header, report_year, target_year)
        if base.role not in {"note", "code"}:
            value_cols.append(int(row_cell.col))
    first_value_col = min(value_cols) if value_cols else None
    header = _full_header(table, int(cell.col), str(cell.col_name))
    return classify_column_role(
        header,
        report_year,
        target_year,
        positional_current=(int(cell.col) == first_value_col),
    )


def _output_scale(route: dict, operation: str) -> float | None:
    if operation in {"ratio", "growth_pct"}:
        return 1.0
    try:
        value = float(route.get("unit_scale") or 1.0)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _compute_answer(
    operation: str, values: list[float], output_scale: float,
) -> float:
    if operation == "difference":
        value = (values[0] - values[1]) / output_scale
    elif operation == "growth_pct":
        if values[1] == 0:
            raise ZeroDivisionError
        value = (values[0] - values[1]) / abs(values[1]) * 100.0
    elif operation == "ratio":
        if values[1] == 0:
            raise ZeroDivisionError
        value = values[0] / values[1] * 100.0
    elif operation == "average":
        value = sum(values) / len(values) / output_scale
    else:
        raise ValueError(operation)
    return round(float(value), 2)


def _exact_term(item: dict) -> str:
    var = item["var"]
    cell = item["cell"]
    scale = item["unit"]["effective_scale"]
    return (
        f"(float({var}.loc[({var}['row'] == {int(cell['row'])}) & "
        f"({var}['col'] == {int(cell['col'])}), 'value'].iloc[0]) * "
        f"{_number_literal(scale)})"
    )


def _compile_query(
    operation: str, terms: list[str], output_scale: float,
) -> str:
    if operation == "difference":
        expression = (
            f"({terms[0]} - {terms[1]}) / {_number_literal(output_scale)}"
        )
    elif operation == "growth_pct":
        expression = f"(({terms[0]} - {terms[1]}) / abs({terms[1]})) * 100"
    elif operation == "ratio":
        expression = f"({terms[0]} / {terms[1]}) * 100"
    elif operation == "average":
        expression = (
            f"({' + '.join(terms)}) / {len(terms)} / "
            f"{_number_literal(output_scale)}"
        )
    else:
        raise ValueError(operation)
    return f"round({expression}, 2)"


def _unwrap_round(node: ast.AST) -> ast.AST | None:
    if (
        not isinstance(node, ast.Call)
        or not isinstance(node.func, ast.Name)
        or node.func.id != "round"
        or len(node.args) != 2
        or node.keywords
    ):
        return None
    digits = node.args[1]
    if not isinstance(digits, ast.Constant) or digits.value != 2:
        return None
    return node.args[0]


def _is_float_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "float"
        and len(node.args) == 1
        and not node.keywords
    )


def _parse_cell_call(node: ast.Call) -> CellLeaf | None:
    value = node.args[0]
    if not isinstance(value, ast.Subscript) or _constant(value.slice) != 0:
        return None
    iloc = value.value
    if not isinstance(iloc, ast.Attribute) or iloc.attr != "iloc":
        return None
    loc = iloc.value
    if (
        not isinstance(loc, ast.Subscript)
        or not isinstance(loc.value, ast.Attribute)
        or loc.value.attr != "loc"
        or not isinstance(loc.value.value, ast.Name)
    ):
        return None
    var = loc.value.value.id
    if not re.fullmatch(r"df\d+", var):
        return None
    selector = loc.slice
    if (
        not isinstance(selector, ast.Tuple)
        or len(selector.elts) != 2
        or _constant(selector.elts[1]) != "value"
    ):
        return None
    condition = selector.elts[0]
    columns = []
    rows = []
    labels = []
    for child in ast.walk(condition):
        if (
            isinstance(child, ast.Compare)
            and len(child.ops) == 1
            and isinstance(child.ops[0], ast.Eq)
            and len(child.comparators) == 1
        ):
            key = _df_subscript_key(child.left, var)
            candidate = _constant(child.comparators[0])
            if key == "col" and isinstance(candidate, int) and not isinstance(
                candidate, bool
            ):
                columns.append(int(candidate))
            elif key == "row" and isinstance(candidate, int) and not isinstance(
                candidate, bool
            ):
                rows.append(int(candidate))
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "contains"
            and child.args
        ):
            base = child.func.value
            if (
                isinstance(base, ast.Attribute)
                and base.attr == "str"
                and _df_subscript_key(base.value, var) == "label"
            ):
                label = _constant(child.args[0])
                if isinstance(label, str):
                    labels.append(label)
    unique_cols = set(columns)
    unique_rows = set(rows)
    unique_labels = set(labels)
    if len(unique_cols) != 1 or bool(unique_rows) == bool(unique_labels):
        return None
    if len(unique_rows) > 1 or len(unique_labels) > 1:
        return None
    return CellLeaf(
        var=var,
        selected_col=next(iter(unique_cols)),
        label_needle=next(iter(unique_labels), ""),
        selected_row=next(iter(unique_rows), None),
    )


def _df_subscript_key(node: ast.AST, var: str) -> str | None:
    if (
        not isinstance(node, ast.Subscript)
        or not isinstance(node.value, ast.Name)
        or node.value.id != var
    ):
        return None
    key = _constant(node.slice)
    return key if isinstance(key, str) else None


def _constant(node: ast.AST):
    return node.value if isinstance(node, ast.Constant) else None


class _LeafReplacer(ast.NodeTransformer):
    def __init__(
        self, call_to_key: dict[int, tuple], key_order: list[tuple],
    ):
        self.call_to_key = call_to_key
        self.index = {key: index for index, key in enumerate(key_order)}

    def visit_Call(self, node):  # noqa: N802 - ast API
        key = self.call_to_key.get(id(node))
        if key is not None:
            return ast.copy_location(
                ast.Name(id=f"L{self.index[key]}", ctx=ast.Load()), node,
            )
        return self.generic_visit(node)


def _safe_symbolic_expression(
    node: ast.AST, expected_indices: set[int],
) -> bool:
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            if re.fullmatch(r"L\d+", child.id):
                names.add(int(child.id[1:]))
            elif child.id != "abs":
                return False
        elif isinstance(child, ast.Call):
            if (
                not isinstance(child.func, ast.Name)
                or child.func.id != "abs"
                or len(child.args) != 1
                or child.keywords
            ):
                return False
        elif isinstance(child, ast.BinOp) and not isinstance(
            child.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            return False
        elif isinstance(child, ast.UnaryOp) and not isinstance(
            child.op, (ast.UAdd, ast.USub)
        ):
            return False
        elif isinstance(child, ast.Constant) and (
            isinstance(child.value, bool)
            or not isinstance(child.value, (int, float))
            or not math.isfinite(float(child.value))
        ):
            return False
    return names == expected_indices


def _operation_shape_matches(
    node: ast.AST, operation: str, operand_count: int,
) -> bool:
    operators = [
        child for child in ast.walk(node) if isinstance(child, ast.BinOp)
    ]
    adds = sum(isinstance(child.op, ast.Add) for child in operators)
    subs = sum(isinstance(child.op, ast.Sub) for child in operators)
    leaf_divisions = [
        child
        for child in operators
        if isinstance(child.op, ast.Div) and _contains_leaf(child.right)
    ]
    names = Counter(
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and re.fullmatch(r"L\d+", child.id)
    )
    has_abs = any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == "abs"
        for child in ast.walk(node)
    )
    if operation == "difference":
        return (
            adds == 0
            and subs == 1
            and not leaf_divisions
            and not has_abs
            and sorted(names.values()) == [1, 1]
        )
    if operation == "ratio":
        return (
            adds == 0
            and subs == 0
            and len(leaf_divisions) == 1
            and not has_abs
            and sorted(names.values()) == [1, 1]
        )
    if operation == "growth_pct":
        return (
            adds == 0
            and subs == 1
            and len(leaf_divisions) == 1
            and sorted(names.values()) in ([1, 1], [1, 2])
        )
    if operation == "average":
        has_count_divisor = any(
            isinstance(child.op, ast.Div)
            and isinstance(child.right, ast.Constant)
            and isinstance(child.right.value, (int, float))
            and not isinstance(child.right.value, bool)
            and float(child.right.value) == float(operand_count)
            for child in operators
        )
        return (
            adds == operand_count - 1
            and subs == 0
            and not leaf_divisions
            and not has_abs
            and sorted(names.values()) == [1] * operand_count
            and has_count_divisor
        )
    return False


def _contains_leaf(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Name) and re.fullmatch(r"L\d+", child.id)
        for child in ast.walk(node)
    )


def _semantic_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = [
        Path(__file__).resolve(),
        root / "vifinqa" / "codegen" / "semantic_repair.py",
        root / "vifinqa" / "extraction" / "unit_policy.py",
        root / "vifinqa" / "utils" / "viet_text.py",
        root / "vifinqa" / "codegen" / "semantic.py",
    ]
    return {
        str(path.relative_to(root)).replace("\\", "/"): _sha256(path)
        for path in paths
    }
