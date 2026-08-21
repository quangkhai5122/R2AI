"""Nested selector/target planner compiled through typed_ir."""
from __future__ import annotations

from .typed_ir import (
    SCHEMA_VERSION, FactBinding, IRValidationError, TypedIRAnswer,
    compile_program,
)
from ..utils.viet_text import norm


def _binding(name, value):
    cells = tuple((r.report_id, r.table_pos, r.label, r.col)
                  for r in value.resolved)
    return FactBinding(
        name=name, expr=value.expr, value=value.value,
        scalar_type=value.spec.kind, stable_cell=cells,
        confidence=min(float(r.score) for r in value.resolved),
    )


def try_nested_formula_ir(route, tables, encoder=None, min_score=62.0):
    question = route.get("question", "")
    text = norm(question)
    if str((route.get("plan") or {}).get("op")) != "ranking":
        return TypedIRAnswer(False, detail="nested_ir needs ranking route")
    if any(cue in text for cue in (
            "trung vi", "gia su", "kich ban", "phan nhom", "top ")):
        return TypedIRAnswer(False, detail="nested_ir unsupported filter/scenario")

    from . import formula_solver as fs
    plain = fs._plain(question)
    matches = fs._calculation_matches(question)
    ranked = fs._ranked_match(plain, matches)
    if ranked is None:
        return TypedIRAnswer(False, detail="nested_ir selector missing")
    selector, extreme_start, want_min = ranked
    target = fs._target_match(matches, selector, extreme_start, plain)
    project_year = route.get("output_type") == "year"
    direct_selector_value = False
    if target is None or target.spec.name == selector.spec.name:
        if project_year:
            target = selector
        elif fs._output_accepts_kind(route.get("output_type"), selector.spec.kind):
            target = selector
            direct_selector_value = True
        else:
            return TypedIRAnswer(False, detail="nested_ir distinct target missing")
    if (not project_year
            and not fs._output_accepts_kind(route.get("output_type"), target.spec.kind)):
        return TypedIRAnswer(False, detail="nested_ir target output mismatch")

    tickers = fs._candidate_tickers(route, tables)
    years = sorted(set(fs._route_years(route)))
    if len(tickers) >= 2:
        dimension = [(ticker, years[-1] if years else None) for ticker in tickers]
    elif len(tickers) == 1 and len(years) >= 2:
        dimension = [(tickers[0], year) for year in years]
    else:
        return TypedIRAnswer(False, detail="nested_ir no dimension")

    formula_route = dict(route)
    formula_route.pop("metric_profile_keys", None)
    facts, items, resolved = [], [], []
    for index, (ticker, year) in enumerate(dimension):
        selector_value = fs._evaluate_formula(
            selector.spec, ticker, year, formula_route,
            tables, encoder, min_score)
        target_value = (
            selector_value if direct_selector_value or project_year
            else fs._evaluate_formula(
                target.spec, ticker, year, formula_route,
                tables, encoder, min_score)
        )
        if selector_value is None or target_value is None:
            return TypedIRAnswer(False,
                                 detail=f"nested_ir unresolved {ticker}/{year}")
        if (not fs._value_supports_year(selector_value, year)
                or not fs._value_supports_year(target_value, year)):
            return TypedIRAnswer(False,
                                 detail=f"nested_ir year grounding {ticker}/{year}")

        selector_name = f"selector_{index}"
        facts.append(_binding(selector_name, selector_value))
        if project_year:
            result_node = {"op": "literal", "type": "year", "value": year}
        elif direct_selector_value:
            result_node = {"op": "ref", "fact": selector_name}
        else:
            target_name = f"target_{index}"
            facts.append(_binding(target_name, target_value))
            result_node = {"op": "ref", "fact": target_name}
        items.append({
            "score": {"op": "ref", "fact": selector_name},
            "result": result_node,
        })
        resolved.extend(selector_value.resolved)
        if target_value is not selector_value:
            resolved.extend(target_value.resolved)

    program = {
        "schema_version": SCHEMA_VERSION,
        "output_type": route.get("output_type", "number"),
        "root": {
            "op": "argmin_project" if want_min else "argmax_project",
            "items": items,
        },
    }
    try:
        compiled = compile_program(program, facts, route, question)
    except IRValidationError as exc:
        return TypedIRAnswer(
            False, detail=f"nested_ir {exc.code}: {exc}",
            resolved=tuple(resolved),
        )
    return TypedIRAnswer(
        True, compiled.answer, compiled.query, compiled.confidence,
        f"nested_ir selector={selector.spec.name} "
        f"target={target.spec.name} items={len(items)}",
        compiled, tuple(resolved),
    )
