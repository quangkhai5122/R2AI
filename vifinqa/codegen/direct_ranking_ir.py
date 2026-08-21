"""Direct single-metric ranking planner compiled through typed IR."""
from __future__ import annotations

from ..utils.viet_text import norm
from .fact_resolver import resolve_fact
from .rule_composite import _FactView
from .typed_ir import (
    SCHEMA_VERSION, FactBinding, IRValidationError, TypedIRAnswer,
    compile_program,
)

_UNSUPPORTED = ("trung vi", "gia su", "kich ban", "phan nhom", "top ",
                "trong nhom", "co ty le", "co he so", "ty trong", "ty le",
                "ty suat", "bien loi nhuan", "tren tong", "so voi tong",
                "chiem bao nhieu")


def _binding(name, resolved):
    return FactBinding(
        name=name, expr=resolved.expr_vnd(), value=resolved.value_vnd,
        scalar_type="money",
        stable_cell=(resolved.report_id, resolved.table_pos,
                     resolved.label, resolved.col),
        confidence=float(resolved.score),
    )


def try_direct_ranking_ir(route, tables, encoder=None, min_score=62.0):
    question = route.get("question", "")
    text = norm(question)
    plan = route.get("plan") or {}
    if str(plan.get("op")) != "ranking":
        return TypedIRAnswer(False, detail="direct_rank needs ranking route")
    if any(cue in text for cue in _UNSUPPORTED):
        return TypedIRAnswer(False, detail="direct_rank has filter/scenario")
    from . import formula_solver as fs
    if fs._looks_nested_selector(question):
        return TypedIRAnswer(False, detail="direct_rank selector/target composition")

    tickers = list(dict.fromkeys(route.get("tickers") or []))
    years = sorted(set(int(y) for y in (route.get("years") or [])))
    if len(tickers) == 1 and len(years) >= 2:
        dimension = [(tickers[0], year) for year in years]
    elif len(tickers) >= 2 and len(years) == 1:
        dimension = [(ticker, years[0]) for ticker in tickers]
    else:
        return TypedIRAnswer(False, detail="direct_rank no simple dimension")

    raw_facts = list(plan.get("facts") or [])
    metrics = [norm(f.get("metric", "")) for f in raw_facts
               if norm(f.get("metric", ""))]
    variants = [norm(v) for v in (route.get("metric_variants") or [])
                if norm(v)]
    metric = ""
    if metrics and len(set(metrics)) == 1 and len(metrics[0].split()) >= 2:
        metric = metrics[0]
    if not metric:
        metric = norm(route.get("metric_norm", ""))
    if not variants and metric:
        variants = [metric]
    if not variants or not any(len(v.split()) >= 2 for v in variants):
        return TypedIRAnswer(False, detail="direct_rank metric missing")

    # A direct ranking has one requested financial concept. Multiple canonical
    # formula families indicate selector/target composition, handled elsewhere.
    matches = fs._calculation_matches(question)
    if len({match.spec.name for match in matches}) > 1:
        return TypedIRAnswer(False, detail="direct_rank multiple calculations")

    formula_route = dict(route)
    facts, items, resolved_all = [], [], []
    for index, (ticker, year) in enumerate(dimension):
        fact = _FactView({
            "ticker": ticker, "year": year,
            "doc_type": route.get("doc_type", "consolidated"),
            "metric": metric, "role": "value",
        })
        resolved = resolve_fact(
            fact, tables, variants, encoder, min_score,
            route=formula_route,
        )
        if resolved is None or resolved.year_evidence < 3:
            return TypedIRAnswer(
                False, detail=f"direct_rank unresolved {ticker}/{year}")
        name = f"score_{index}"
        facts.append(_binding(name, resolved))
        if route.get("output_type") == "year":
            result = {"op": "literal", "type": "year", "value": year}
        elif route.get("output_type") == "number":
            result = {"op": "ref", "fact": name}
        else:
            return TypedIRAnswer(False,
                                 detail="direct_rank output unsupported")
        items.append({
            "score": {"op": "ref", "fact": name},
            "result": result,
        })
        resolved_all.append(resolved)

    want_min = any(cue in text for cue in
                   ("thap nhat", "nho nhat", "it nhat", "be nhat"))
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
            False, detail=f"direct_rank {exc.code}: {exc}",
            resolved=tuple(resolved_all),
        )
    return TypedIRAnswer(
        True, compiled.answer, compiled.query, compiled.confidence,
        f"direct_rank metric={metric or variants[0]} items={len(items)}",
        compiled, tuple(resolved_all),
    )
