"""Deterministic formula solver for common financial ratios and conditions.

This module sits between retrieval and LLM codegen.  It handles questions where
the arithmetic is standard once the needed statement rows are located:

* count companies satisfying one or more numeric conditions
* rank companies by a direct formula
* answer direct ratio / margin formulas

The implementation is intentionally conservative.  If every operand for every
candidate cannot be resolved from the retrieved tables, the solver refuses and
the normal composite/LLM path keeps control.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from ..finance.metrics import get_metric
from ..utils.viet_num import parse_vn_number
from ..utils.viet_text import strip_diacritics
from .fact_resolver import ResolvedFact, resolve_fact
from .rule_composite import CompositeAnswer, _FactView
from .units import check_answer_unit


@dataclass(frozen=True)
class Operand:
    metric: str
    variants: tuple[str, ...]
    required_phrases: tuple[str, ...] = ()
    forbidden_phrases: tuple[str, ...] = ()
    expected_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class FormulaSpec:
    name: str
    triggers: tuple[str, ...]
    operands: tuple[Operand, ...]
    kind: str                      # ratio | percent
    value_fn: Callable[[list[float]], float]
    expr_fn: Callable[[list[ResolvedFact]], str]


@dataclass
class FormulaValue:
    spec: FormulaSpec
    ticker: str
    year: int | None
    value: float
    expr: str
    resolved: list[ResolvedFact]
    score: float


@dataclass(frozen=True)
class FormulaMatch:
    spec: FormulaSpec
    start: int
    end: int
    trigger: str


@dataclass(frozen=True)
class Condition:
    spec: FormulaSpec | None
    metric: str
    variants: tuple[str, ...]
    op: str
    threshold: float
    kind: str                      # money | ratio | percent

    @property
    def label(self) -> str:
        return self.spec.name if self.spec else self.metric


@dataclass
class Population:
    tickers: list[str]
    resolved: list[ResolvedFact]
    support_exprs: list[str]


def try_formula_answer(route: dict, tables: list[dict], encoder=None,
                       min_score: float = 62.0) -> CompositeAnswer:
    """Try formula-aware deterministic solving for count/ranking/ratio/margin."""
    plan = route.get("plan") or {}
    op = plan.get("op", "lookup")
    output_type = route.get("output_type", "number")

    if output_type == "count" or op == "count":
        return _try_count(route, tables, encoder, min_score)
    if op == "ranking":
        nested = _try_nested_ranking(route, tables, encoder, min_score)
        if nested.ok:
            return nested
        return _try_formula_ranking(route, tables, encoder, min_score)
    if op in {"difference", "growth_pct", "average"} or output_type == "percentage_point":
        temporal = _try_formula_change(route, tables, encoder, min_score)
        if temporal.ok:
            return temporal
    if op in {"ratio", "margin", "ratio_times"} or output_type in {"ratio", "percent"}:
        if _looks_nested_selector(route.get("question", "")):
            nested = _try_nested_ranking(route, tables, encoder, min_score)
            if nested.ok:
                return nested
        temporal = _try_formula_change(route, tables, encoder, min_score)
        if temporal.ok:
            return temporal
        return _try_direct_formula(route, tables, encoder, min_score)
    return CompositeAnswer(ok=False, detail=f"formula solver skipped op={op}")


def _try_count(route: dict, tables: list[dict], encoder, min_score: float) -> CompositeAnswer:
    question = route.get("question", "")
    text = _plain(question)
    if any(w in text for w in ("gia su", "kich ban", "neu doanh thu")):
        return CompositeAnswer(ok=False, detail="scenario count unsupported")
    if _counts_years(question):
        return _try_count_years(route, tables, encoder, min_score)
    if len(_route_years(route)) >= 2:
        temporal = _try_temporal_entity_count(route, tables, encoder, min_score)
        if temporal.ok:
            return temporal

    conditions = _parsed_conditions(route, question)

    if not conditions:
        c = _direct_metric_condition(route, question)
        if c:
            conditions = [c]
    if not conditions:
        return CompositeAnswer(ok=False, detail="count has no parsed condition")
    expected = _expected_condition_count(question)
    if len(conditions) < expected:
        return CompositeAnswer(
            ok=False,
            detail=f"count parsed only {len(conditions)}/{expected} conditions")

    tickers = _candidate_tickers(route, tables)
    year = _primary_year(route)
    if not tickers:
        return CompositeAnswer(ok=False, detail="count has no candidate tickers")

    population = _maybe_top_n_population(
        question, tickers, year, route, tables, encoder, min_score)
    if population is None:
        return CompositeAnswer(ok=False, detail="top-n population could not be resolved")
    tickers = population.tickers

    terms, resolved, answers = [], list(population.resolved), []
    for ticker in tickers:
        ticker_terms = []
        ticker_passes = []
        for cond in conditions:
            val = _evaluate_condition(cond, ticker, year, route, tables, encoder, min_score)
            if val is None:
                return CompositeAnswer(
                    ok=False,
                    detail=f"count unresolved condition {cond.label} for {ticker}",
                    resolved=resolved)
            resolved.extend(val.resolved)
            passed = _compare(val.value, cond.op, cond.threshold)
            ticker_terms.append(_condition_expr(val.expr, cond.op, cond.threshold))
            ticker_passes.append(passed)
        answers.append(all(ticker_passes))
        terms.append("(" + " and ".join(ticker_terms) + ")")

    answer = float(sum(1 for x in answers if x))
    support = ""
    if population.support_exprs:
        support = f" + 0 * ({' + '.join(population.support_exprs)})"
    query = f"round(float({' + '.join(terms)}){support}, 2)"
    conf = _confidence(resolved, base=92.0 if len(conditions) > 1 else 88.0)
    return CompositeAnswer(ok=True, answer=answer, pandas_query=query,
                           confidence=conf,
                           detail=f"formula_count n={len(tickers)} conditions={len(conditions)}",
                           resolved=resolved)


def _try_count_years(route: dict, tables: list[dict], encoder,
                     min_score: float) -> CompositeAnswer:
    question = route.get("question", "")
    tickers = _candidate_tickers(route, tables)
    years = sorted(set(_route_years(route)))
    if len(tickers) != 1 or len(years) < 2:
        return CompositeAnswer(ok=False, detail="count years needs one ticker and >=2 years")

    median_case = _try_count_years_with_median(
        route, tables, encoder, min_score, tickers[0], years)
    if median_case.ok:
        return median_case

    conditions = _parsed_conditions(route, question)
    if not conditions:
        direct = _direct_metric_condition(route, question)
        conditions = [direct] if direct else []
    if not conditions:
        return CompositeAnswer(ok=False, detail="count years has no parsed condition")

    terms, resolved, passed = [], [], []
    for year in years:
        year_terms, year_passes = [], []
        for cond in conditions:
            val = _evaluate_condition(
                cond, tickers[0], year, route, tables, encoder, min_score)
            if val is None:
                return CompositeAnswer(
                    ok=False,
                    detail=f"count years unresolved {cond.label}/{year}",
                    resolved=resolved)
            if not _value_supports_year(val, year):
                return CompositeAnswer(
                    ok=False,
                    detail=f"count years lacks exact-year evidence {cond.label}/{year}",
                    resolved=resolved)
            resolved.extend(val.resolved)
            year_terms.append(_condition_expr(val.expr, cond.op, cond.threshold))
            year_passes.append(_compare(val.value, cond.op, cond.threshold))
        terms.append("(" + " and ".join(year_terms) + ")")
        passed.append(all(year_passes))

    answer = float(sum(passed))
    query = f"round(float({' + '.join(terms)}), 2)"
    return CompositeAnswer(
        ok=True, answer=answer, pandas_query=query,
        confidence=_confidence(resolved, base=90.0),
        detail=f"formula_count_years n={len(years)} conditions={len(conditions)}",
        resolved=resolved)


def _try_count_years_with_median(route: dict, tables: list[dict], encoder,
                                 min_score: float, ticker: str,
                                 years: list[int]) -> CompositeAnswer:
    text = _plain(route.get("question", ""))
    if "trung vi" not in text or "nam truoc" not in text:
        return CompositeAnswer(ok=False, detail="not a prior-year median count")
    specs = {s.name: s for s in _detected_specs(text)}
    change_spec = specs.get("gross_margin")
    median_spec = specs.get("cfo_margin")
    if change_spec is None or median_spec is None:
        return CompositeAnswer(ok=False, detail="median year count formulas missing")

    change_vals, median_vals, resolved = [], [], []
    for year in years:
        a = _evaluate_formula(
            change_spec, ticker, year, route, tables, encoder, min_score)
        b = _evaluate_formula(
            median_spec, ticker, year, route, tables, encoder, min_score)
        if a is None or b is None:
            return CompositeAnswer(ok=False,
                                   detail=f"median year count unresolved {year}",
                                   resolved=resolved)
        if not _value_supports_year(a, year) or not _value_supports_year(b, year):
            return CompositeAnswer(ok=False,
                                   detail=f"median year count lacks exact year {year}",
                                   resolved=resolved)
        change_vals.append(a)
        median_vals.append(b)
        resolved.extend(a.resolved + b.resolved)

    median = _median([v.value for v in median_vals])
    median_expr = _median_expr([v.expr for v in median_vals])
    terms, outcomes = [], []
    for i in range(1, len(years)):
        terms.append(
            f"(({change_vals[i].expr}) > ({change_vals[i - 1].expr}) and "
            f"({median_vals[i].expr}) > ({median_expr}))")
        outcomes.append(change_vals[i].value > change_vals[i - 1].value
                        and median_vals[i].value > median)
    answer = float(sum(outcomes))
    return CompositeAnswer(
        ok=True, answer=answer,
        pandas_query=f"round(float({' + '.join(terms)}), 2)",
        confidence=_confidence(resolved, base=88.0),
        detail="formula_count_years prior_change+median", resolved=resolved)


def _try_temporal_entity_count(route: dict, tables: list[dict], encoder,
                               min_score: float) -> CompositeAnswer:
    question = route.get("question", "")
    text = _plain(question)
    years = sorted(set(_route_years(route)))
    tickers = _candidate_tickers(route, tables)
    if len(years) < 2 or not tickers:
        return CompositeAnswer(ok=False, detail="temporal count needs periods/entities")
    start, end = years[0], years[-1]
    matches = _calculation_matches(question)
    by_name = _unique_matches(matches)

    compare_growth = ("cao hon toc do tang" in text
                      and "sga_expense" in by_name and "net_revenue" in by_name)
    predicates: list[tuple[str, FormulaSpec, str | None]] = []
    if compare_growth:
        predicates.append(("growth_gt", by_name["sga_expense"].spec,
                           by_name["net_revenue"].spec.name))
    else:
        for match in by_name.values():
            direction = _temporal_direction(text, match)
            if direction:
                predicates.append((direction, match.spec, None))

    if not predicates:
        return CompositeAnswer(ok=False, detail="temporal count predicates missing")
    if len(predicates) < _expected_condition_count(question):
        return CompositeAnswer(ok=False,
                               detail="temporal count parsed too few predicates")

    terms, outcomes, resolved = [], [], []
    for ticker in tickers:
        ticker_terms, ticker_outcomes = [], []
        for kind, spec, other_name in predicates:
            if kind == "growth_gt":
                other = by_name[other_name].spec
                a = _evaluate_change(spec, ticker, start, end, "growth", route,
                                     tables, encoder, min_score)
                b = _evaluate_change(other, ticker, start, end, "growth", route,
                                     tables, encoder, min_score)
                if a is None or b is None:
                    return CompositeAnswer(ok=False,
                                           detail=f"temporal count unresolved {ticker}",
                                           resolved=resolved)
                resolved.extend(a.resolved + b.resolved)
                ticker_terms.append(f"(({a.expr}) > ({b.expr}))")
                ticker_outcomes.append(a.value > b.value)
                continue

            if kind in {"increase", "decrease"}:
                val = _evaluate_change(spec, ticker, start, end, "delta", route,
                                       tables, encoder, min_score)
                if val is None:
                    return CompositeAnswer(ok=False,
                                           detail=f"temporal count unresolved {ticker}",
                                           resolved=resolved)
                resolved.extend(val.resolved)
                op = ">" if kind == "increase" else "<"
                ticker_terms.append(f"(({val.expr}) {op} 0)")
                ticker_outcomes.append(val.value > 0 if op == ">" else val.value < 0)
                continue

            val = _evaluate_formula(spec, ticker, end, route, tables, encoder, min_score)
            if val is None:
                return CompositeAnswer(ok=False,
                                       detail=f"temporal count unresolved {ticker}",
                                       resolved=resolved)
            if not _value_supports_year(val, end):
                return CompositeAnswer(
                    ok=False, detail=f"temporal count lacks exact year {ticker}/{end}",
                    resolved=resolved)
            resolved.extend(val.resolved)
            op = ">" if kind == "positive" else "<"
            ticker_terms.append(f"(({val.expr}) {op} 0)")
            ticker_outcomes.append(val.value > 0 if op == ">" else val.value < 0)
        terms.append("(" + " and ".join(ticker_terms) + ")")
        outcomes.append(all(ticker_outcomes))

    answer = float(sum(outcomes))
    return CompositeAnswer(
        ok=True, answer=answer,
        pandas_query=f"round(float({' + '.join(terms)}), 2)",
        confidence=_confidence(resolved, base=88.0),
        detail=f"formula_count_temporal n={len(tickers)} predicates={len(predicates)}",
        resolved=resolved)


def _try_nested_ranking(route: dict, tables: list[dict], encoder,
                        min_score: float) -> CompositeAnswer:
    question = route.get("question", "")
    text = _plain(question)
    matches = _calculation_matches(question)
    ranked = _ranked_match(text, matches)
    if ranked is None:
        return CompositeAnswer(ok=False, detail="nested ranking selector missing")
    selector, extreme_start, want_min = ranked
    target = _target_match(matches, selector, extreme_start, text)
    if (target is not None
            and _condition_for_match(question, target) is not None
            and _output_accepts_kind(route.get("output_type"), selector.spec.kind)):
        # "revenue lowest among years with net margin > 10%" asks for the
        # ranked revenue itself; the other formula is only a filter.
        target = selector
    if (target is None and _output_accepts_kind(route.get("output_type"),
                                                selector.spec.kind)
            and _selector_is_direct_answer(text, selector)):
        target = selector
    if target is None:
        return CompositeAnswer(ok=False, detail="nested ranking target missing")
    if "trung vi" in text:
        return CompositeAnswer(ok=False, detail="nested ranking median filter unsupported")
    if "tu dau nam den cuoi nam" in text:
        return CompositeAnswer(ok=False, detail="nested ranking needs intra-year columns")
    if (re.search(r"\b\d+\s+(?:doanh nghiep|cong ty|ma)\b", text)
            and any(w in text for w in ("nam giu bao nhieu phan tram tong",
                                         "dong gop bao nhieu phan tram tong",
                                         "ty trong trong tong"))):
        return CompositeAnswer(ok=False,
                               detail="nested top-n group aggregation unsupported")
    if any(w in text for w in ("gia su", "kich ban", "co the tang toi da",
                                "truoc khi", "neu ")):
        return CompositeAnswer(ok=False, detail="nested ranking scenario unsupported")

    tickers = _candidate_tickers(route, tables)
    years = sorted(set(_route_years(route)))
    if len(tickers) >= 2:
        stated_n = _stated_population_size(text)
        if stated_n is not None and stated_n != len(tickers):
            return CompositeAnswer(
                ok=False,
                detail=f"nested ranking population {len(tickers)}/{stated_n}")
        dimension = [(ticker, years[-1] if years else None) for ticker in tickers]
        entity_mode = True
    elif len(tickers) == 1 and len(years) >= 2:
        dimension = [(tickers[0], year) for year in years]
        entity_mode = False
    else:
        return CompositeAnswer(ok=False, detail="nested ranking has no dimension")

    temporal_selector = entity_mode and len(years) >= 2
    selector_mode = _value_mode(text, selector, extreme_start, temporal_selector)
    selector_ctx = text[max(0, selector.start - 65):extreme_start]
    if (not entity_mode
            and any(w in selector_ctx for w in ("tang truong", "toc do tang"))):
        return CompositeAnswer(ok=False,
                               detail="nested year-over-year selector unsupported")
    conditions = [c for c in _parsed_conditions(route, question)
                  if c.label not in {selector.spec.name, target.spec.name}]

    ranked_values, support, resolved = [], [], []
    for ticker, year in dimension:
        if selector_mode == "level":
            sv = _evaluate_formula(
                selector.spec, ticker, year, route, tables, encoder, min_score)
        else:
            sv = _evaluate_change(
                selector.spec, ticker, years[0], years[-1], selector_mode,
                route, tables, encoder, min_score)
        if sv is None:
            return CompositeAnswer(ok=False,
                                   detail=f"nested selector unresolved {ticker}/{year}",
                                   resolved=resolved)
        if selector_mode == "level" and not _value_supports_year(sv, year):
            return CompositeAnswer(
                ok=False, detail=f"nested selector lacks exact year {ticker}/{year}",
                resolved=resolved)
        resolved.extend(sv.resolved)
        support.append(sv.expr)

        passes = True
        for cond in conditions:
            check_years = years if (entity_mode and len(years) >= 2
                                    and "ca hai nam" in text) else [year]
            for check_year in check_years:
                cv = _evaluate_condition(
                    cond, ticker, check_year, route, tables, encoder, min_score)
                if cv is None:
                    return CompositeAnswer(
                        ok=False,
                        detail=f"nested filter unresolved {ticker}/{check_year}",
                        resolved=resolved)
                resolved.extend(cv.resolved)
                support.append(cv.expr)
                if not _value_supports_year(cv, check_year):
                    return CompositeAnswer(
                        ok=False,
                        detail=f"nested filter lacks exact year {ticker}/{check_year}",
                        resolved=resolved)
                passes = passes and _compare(cv.value, cond.op, cond.threshold)
        if passes:
            ranked_values.append((ticker, year, sv))

    if not ranked_values:
        return CompositeAnswer(ok=False, detail="nested ranking filtered all candidates",
                               resolved=resolved)
    chosen = (min(ranked_values, key=lambda x: x[2].value) if want_min
              else max(ranked_values, key=lambda x: x[2].value))
    ticker, selected_year, _selector_value = chosen

    target_mode = (_target_value_mode(text, target, len(years) >= 2)
                   if route.get("output_type") == "percentage_point" else "level")
    if target_mode != "level" and entity_mode:
        tv = _evaluate_change(target.spec, ticker, years[0], years[-1], target_mode,
                              route, tables, encoder, min_score)
    else:
        target_year = selected_year
        if not entity_mode and "nam sau nam" in text:
            target_year = int(selected_year) + 1
        tv = _evaluate_formula(
            target.spec, ticker, target_year, route, tables, encoder, min_score)
    if tv is None:
        return CompositeAnswer(ok=False,
                               detail=f"nested target unresolved {ticker}/{selected_year}",
                               resolved=resolved)
    if target_mode == "level" and not _value_supports_year(tv, target_year):
        return CompositeAnswer(
            ok=False,
            detail=f"nested target lacks exact year {ticker}/{target_year}",
            resolved=resolved)
    resolved.extend(tv.resolved)
    answer, answer_expr = _answer_value(tv, route)
    support_expr = " + ".join(support)
    query = f"round(({answer_expr}) + 0 * ({support_expr}), 2)"
    warn = check_answer_unit(answer, route.get("output_type", tv.spec.kind))
    if warn and "outside plausible range" in warn:
        return CompositeAnswer(ok=False, detail=f"nested unit guard: {warn}",
                               resolved=resolved)
    return CompositeAnswer(
        ok=True, answer=answer, pandas_query=query,
        confidence=_confidence(resolved, base=87.0),
        detail=(f"formula_nested selector={selector.spec.name}/{selector_mode} "
                f"target={target.spec.name}/{target_mode} picked={ticker}/{selected_year}"),
        resolved=resolved)


def _try_formula_change(route: dict, tables: list[dict], encoder,
                        min_score: float) -> CompositeAnswer:
    question = route.get("question", "")
    text = _plain(question)
    aggregates = any(w in text for w in ("binh quan", "trung binh"))
    if _looks_nested_selector(question) and not aggregates:
        return CompositeAnswer(ok=False, detail="formula change looks nested")
    if any(w in text for w in ("gia su", "kich ban", "neu ")):
        return CompositeAnswer(ok=False, detail="formula change scenario unsupported")
    tickers = _candidate_tickers(route, tables)
    years = sorted(set(_route_years(route)))
    matches = [m for m in _calculation_matches(question)
               if m.spec in _FORMULAS]
    by_name = _unique_matches(matches)
    if not by_name:
        return CompositeAnswer(ok=False, detail="formula change has no formulas")

    # Same-period difference between two formula concepts, optionally averaged
    # over a filtered group (for example gross margin minus net margin).
    if (len(years) == 1 and len(by_name) == 2 and len(tickers) >= 1
            and "chenh lech giua" in text):
        specs = _pair_specs_after_difference(text, matches)
        if len(specs) != 2:
            return CompositeAnswer(ok=False, detail="formula pair order unresolved")
        return _formula_pair_difference(route, tables, encoder, min_score,
                                        tickers, years[0], specs)

    if len(years) < 2:
        if len(tickers) == 2 and len(by_name) == 1:
            spec = next(iter(by_name.values())).spec
            vals, resolved = [], []
            for ticker in tickers:
                fv = _evaluate_formula(
                    spec, ticker, years[0] if years else None, route, tables,
                    encoder, min_score)
                if fv is None:
                    return CompositeAnswer(ok=False,
                                           detail=f"formula comparison unresolved {ticker}")
                if years and not _value_supports_year(fv, years[0]):
                    return CompositeAnswer(
                        ok=False,
                        detail=f"formula comparison lacks exact year {ticker}/{years[0]}")
                vals.append(fv)
                resolved.extend(fv.resolved)
            answer = vals[0].value - vals[1].value
            query = f"round(({vals[0].expr}) - ({vals[1].expr}), 2)"
            warn = check_answer_unit(answer, route.get("output_type", spec.kind))
            if warn and "outside plausible range" in warn:
                return CompositeAnswer(ok=False,
                                       detail=f"formula comparison unit guard: {warn}",
                                       resolved=resolved)
            return CompositeAnswer(ok=True, answer=answer, pandas_query=query,
                                   confidence=_confidence(resolved, base=88.0),
                                   detail=f"formula_compare {spec.name}", resolved=resolved)
        return CompositeAnswer(ok=False, detail="formula change needs two periods")

    start, end = years[0], years[-1]
    target = _change_target_match(text, list(by_name.values()))
    if target is None:
        return CompositeAnswer(ok=False, detail="formula change target missing")
    mode = _target_value_mode(text, target, True)
    if mode == "level":
        return CompositeAnswer(ok=False, detail="formula change target is only a filter")

    values, support, resolved = [], [], []
    revenue_filter = ("doanh thu thuan" in text and
                      any(w in text for w in ("doanh thu thuan nam",
                                              "tang truong doanh thu thuan",
                                              "doanh thu thuan tang")) and
                      any(w in text for w in ("tang so voi", "cao hon nam", "duong",
                                              "doanh thu thuan tang")))
    revenue_spec = next(s for s in _METRIC_SPECS if s.name == "net_revenue")
    for ticker in tickers:
        include = True
        if revenue_filter:
            rv = _evaluate_change(revenue_spec, ticker, start, end, "delta", route,
                                  tables, encoder, min_score)
            if rv is None:
                return CompositeAnswer(ok=False,
                                       detail=f"formula change filter unresolved {ticker}",
                                       resolved=resolved)
            include = rv.value > 0
            support.append(rv.expr)
            resolved.extend(rv.resolved)
        if not include:
            continue
        fv = _evaluate_change(target.spec, ticker, start, end, mode, route,
                              tables, encoder, min_score)
        if fv is None:
            return CompositeAnswer(ok=False,
                                   detail=f"formula change unresolved {ticker}",
                                   resolved=resolved)
        values.append(fv)
        support.append(fv.expr)
        resolved.extend(fv.resolved)

    if not values:
        return CompositeAnswer(ok=False, detail="formula change filter empty",
                               resolved=resolved)
    average = len(values) > 1 and any(w in text for w in ("binh quan", "trung binh"))
    if len(values) > 1 and not average:
        return CompositeAnswer(ok=False, detail="formula change ambiguous aggregation",
                               resolved=resolved)
    if average:
        answer = sum(v.value for v in values) / len(values)
        answer_expr = f"(({' + '.join(v.expr for v in values)}) / {len(values)})"
    else:
        answer = values[0].value
        answer_expr = values[0].expr
    query = f"round(({answer_expr}) + 0 * ({' + '.join(support)}), 2)"
    warn = check_answer_unit(answer, route.get("output_type", target.spec.kind))
    if warn and "outside plausible range" in warn:
        return CompositeAnswer(ok=False, detail=f"formula change unit guard: {warn}",
                               resolved=resolved)
    return CompositeAnswer(
        ok=True, answer=answer, pandas_query=query,
        confidence=_confidence(resolved, base=88.0),
        detail=f"formula_change {target.spec.name}/{mode} n={len(values)}",
        resolved=resolved)


def _formula_pair_difference(route: dict, tables: list[dict], encoder,
                             min_score: float, tickers: list[str], year: int,
                             specs: list[FormulaSpec]) -> CompositeAnswer:
    text = _plain(route.get("question", ""))
    values, resolved = [], []
    for ticker in tickers:
        a = _evaluate_formula(specs[0], ticker, year, route, tables, encoder, min_score)
        b = _evaluate_formula(specs[1], ticker, year, route, tables, encoder, min_score)
        if a is None or b is None:
            return CompositeAnswer(ok=False,
                                   detail=f"formula pair unresolved {ticker}",
                                   resolved=resolved)
        if not _value_supports_year(a, year) or not _value_supports_year(b, year):
            return CompositeAnswer(ok=False,
                                   detail=f"formula pair lacks exact year {ticker}/{year}",
                                   resolved=resolved)
        resolved.extend(a.resolved + b.resolved)
        if "duong" in text and b.value <= 0:
            continue
        values.append((a.value - b.value, f"(({a.expr}) - ({b.expr}))"))
    if not values:
        return CompositeAnswer(ok=False, detail="formula pair filter empty",
                               resolved=resolved)
    if len(values) > 1 and not any(w in text for w in ("binh quan", "trung binh")):
        return CompositeAnswer(ok=False, detail="formula pair aggregation missing",
                               resolved=resolved)
    answer = sum(v for v, _ in values) / len(values)
    expr = f"(({' + '.join(e for _, e in values)}) / {len(values)})"
    warn = check_answer_unit(answer, route.get("output_type", "percentage_point"))
    if warn and "outside plausible range" in warn:
        return CompositeAnswer(ok=False, detail=f"formula pair unit guard: {warn}",
                               resolved=resolved)
    return CompositeAnswer(ok=True, answer=answer, pandas_query=f"round({expr}, 2)",
                           confidence=_confidence(resolved, base=87.0),
                           detail=f"formula_pair_difference n={len(values)}",
                           resolved=resolved)


def _try_formula_ranking(route: dict, tables: list[dict], encoder,
                         min_score: float) -> CompositeAnswer:
    question = route.get("question", "")
    if _looks_nested_selector(question):
        return CompositeAnswer(ok=False, detail="ranking looks nested")
    specs = _detected_specs(question)
    if len(specs) != 1:
        return CompositeAnswer(ok=False, detail=f"ranking formulas={len(specs)}")
    spec = specs[0]
    tickers = _candidate_tickers(route, tables)
    year = _primary_year(route)
    if len(tickers) < 2:
        return CompositeAnswer(ok=False, detail="ranking needs >=2 tickers")

    vals, resolved = [], []
    for ticker in tickers:
        fv = _evaluate_formula(spec, ticker, year, route, tables, encoder, min_score)
        if fv is None:
            return CompositeAnswer(ok=False, detail=f"ranking unresolved {ticker}",
                                   resolved=resolved)
        vals.append(fv)
        resolved.extend(fv.resolved)

    want_min = _wants_min(question)
    best = min(vals, key=lambda v: v.value) if want_min else max(vals, key=lambda v: v.value)
    fn = "min" if want_min else "max"
    query = f"round({fn}({', '.join(v.expr for v in vals)}), 2)"
    warn = check_answer_unit(best.value, spec.kind)
    if warn and "outside plausible range" in warn:
        return CompositeAnswer(ok=False, detail=f"formula_ranking unit guard: {warn}",
                               resolved=resolved)
    conf = _confidence(resolved, base=90.0)
    return CompositeAnswer(ok=True, answer=best.value, pandas_query=query,
                           confidence=conf,
                           detail=f"formula_ranking {spec.name}"
                                  + (f" | UNIT-WARN: {warn}" if warn else ""),
                           resolved=resolved)


def _try_direct_formula(route: dict, tables: list[dict], encoder,
                        min_score: float) -> CompositeAnswer:
    question = route.get("question", "")
    if _looks_nested_selector(question) or _has_complex_temporal_selector(question):
        return CompositeAnswer(ok=False, detail="direct formula looks nested")
    if len(_route_years(route)) != 1:
        return CompositeAnswer(ok=False, detail="direct formula needs one year")
    specs = _detected_specs(question)
    if len(specs) != 1:
        return CompositeAnswer(ok=False, detail=f"direct formulas={len(specs)}")
    tickers = _candidate_tickers(route, tables)
    year = _primary_year(route)
    if len(tickers) != 1:
        return CompositeAnswer(ok=False, detail=f"direct formula tickers={len(tickers)}")

    fv = _evaluate_formula(specs[0], tickers[0], year, route, tables, encoder, min_score)
    if fv is None:
        return CompositeAnswer(ok=False, detail=f"direct unresolved {specs[0].name}")
    output_type = route.get("output_type", fv.spec.kind)
    warn = check_answer_unit(fv.value, output_type)
    if warn and "outside plausible range" in warn:
        return CompositeAnswer(ok=False, detail=f"formula_direct unit guard: {warn}",
                               resolved=fv.resolved)
    conf = _confidence(fv.resolved, base=92.0)
    return CompositeAnswer(ok=True, answer=fv.value,
                           pandas_query=f"round({fv.expr}, 2)",
                           confidence=conf,
                           detail=f"formula_direct {fv.spec.name}"
                                  + (f" | UNIT-WARN: {warn}" if warn else ""),
                           resolved=fv.resolved)


def _evaluate_condition(cond: Condition, ticker: str, year: int | None, route: dict,
                        tables: list[dict], encoder, min_score: float) -> FormulaValue | None:
    if cond.spec:
        return _evaluate_formula(cond.spec, ticker, year, route, tables, encoder, min_score)

    fact = _FactView({"ticker": ticker, "year": year, "doc_type": route.get("doc_type"),
                      "metric": cond.metric})
    r = resolve_fact(fact, tables, list(cond.variants), encoder, min_score, route=route)
    if (r is None or not _direct_metric_accepts(cond, r, route)
            or not _resolved_supports_year(r, year)
            or not _resolved_value_sane(r)):
        return None
    return FormulaValue(
        spec=_DIRECT_METRIC_SPEC,
        ticker=ticker,
        year=year,
        value=r.value_vnd,
        expr=r.expr_vnd(),
        resolved=[r],
        score=r.score,
    )


def _direct_metric_accepts(cond: Condition, resolved: ResolvedFact,
                           route: dict) -> bool:
    """Protect count thresholds from high-scoring but contradictory labels."""
    metric = _plain(cond.metric)
    intent = _plain(route.get("question", ""))
    label = _plain(resolved.label)
    required = (
        "thue thu nhap doanh nghiep",
        "lai thuan",
        "ngoai hoi",
        "luu chuyen tien thuan tu hoat dong kinh doanh",
        "dong tien thuan tu hoat dong kinh doanh",
    )
    for phrase in required:
        if (phrase in metric or phrase in intent) and phrase not in label:
            return False
    contradictions = (
        ("doanh nghiep", "ca nhan"),
        ("ngoai hoi", "vang"),
    )
    if any((wanted in metric or wanted in intent) and wrong in label
           for wanted, wrong in contradictions):
        return False
    if "cuoi nam" in intent and "binh quan" in label:
        return False
    return True


def _evaluate_formula(spec: FormulaSpec, ticker: str, year: int | None, route: dict,
                      tables: list[dict], encoder, min_score: float) -> FormulaValue | None:
    resolved = []
    for op in spec.operands:
        fact = _FactView({"ticker": ticker, "year": year, "doc_type": route.get("doc_type"),
                          "metric": op.metric})
        r = resolve_fact(fact, tables, list(op.variants), encoder, min_score, route=route)
        if (r is None or not _operand_accepts(op, r)
                or not _resolved_supports_year(r, year)
                or not _resolved_value_sane(r)):
            return None
        resolved.append(r)
    values = [r.value_vnd for r in resolved]
    try:
        value = spec.value_fn(values)
    except ZeroDivisionError:
        return None
    if value != value or abs(value) == float("inf"):
        return None
    if spec.name in {"quick_ratio", "current_ratio", "debt_assets",
                     "inventory_assets", "fixed_asset_turnover"} and value < 0:
        return None
    return FormulaValue(spec=spec, ticker=ticker, year=year, value=value,
                        expr=spec.expr_fn(resolved), resolved=resolved,
                        score=min(r.score for r in resolved))


def _operand_accepts(operand: Operand, resolved: ResolvedFact) -> bool:
    """Reject fuzzy lookalikes that change a financial formula's meaning."""
    label = _plain(resolved.label)
    if (operand.required_phrases
            and not any(phrase in label for phrase in operand.required_phrases)):
        return False
    if any(phrase in label for phrase in operand.forbidden_phrases):
        return False
    code = re.sub(r"\.0$", "", str(resolved.code or "").strip())
    if operand.expected_codes and code.isdigit() and code not in operand.expected_codes:
        return False
    return True


def _resolved_supports_year(resolved: ResolvedFact, year: int | None) -> bool:
    if year is None or resolved.year_evidence < 3:
        return year is None
    report_year = _report_year(resolved.report_id)
    return report_year in {year, year + 1}


def _resolved_value_sane(resolved: ResolvedFact) -> bool:
    # A 12-digit raw value is already VND. Multiplying it by a million/billion
    # is a common table-level unit extraction error and flips threshold counts.
    return not (abs(resolved.value) >= 1e9 and resolved.unit_scale >= 1e6)


def _evaluate_change(spec: FormulaSpec, ticker: str, start: int, end: int,
                     mode: str, route: dict, tables: list[dict], encoder,
                     min_score: float) -> FormulaValue | None:
    first = _evaluate_formula(spec, ticker, start, route, tables, encoder, min_score)
    last = _evaluate_formula(spec, ticker, end, route, tables, encoder, min_score)
    if first is None or last is None:
        return None
    if not _value_supports_year(first, start) or not _value_supports_year(last, end):
        return None
    if mode == "growth":
        if first.value == 0:
            return None
        value = (last.value - first.value) / abs(first.value) * 100.0
        expr = f"((({last.expr}) - ({first.expr})) / abs({first.expr}) * 100)"
    elif mode == "decrease":
        value = first.value - last.value
        expr = f"(({first.expr}) - ({last.expr}))"
    elif mode == "cagr":
        if first.value <= 0:
            return None
        n = max(1, end - start)
        value = ((last.value / first.value) ** (1.0 / n) - 1.0) * 100.0
        expr = f"((({last.expr}) / ({first.expr})) ** (1/{n}) - 1) * 100"
    else:
        value = last.value - first.value
        expr = f"(({last.expr}) - ({first.expr}))"
    return FormulaValue(spec=spec, ticker=ticker, year=end, value=value, expr=expr,
                        resolved=first.resolved + last.resolved,
                        score=min(first.score, last.score))


def _answer_value(value: FormulaValue, route: dict) -> tuple[float, str]:
    if value.spec.kind != "money":
        return value.value, value.expr
    scale = float(route.get("unit_scale", 1.0) or 1.0)
    return value.value / scale, f"(({value.expr}) / {scale:g})"


def _output_accepts_kind(output_type: str | None, kind: str) -> bool:
    if kind == "money":
        return output_type in {None, "", "number"}
    if kind == "ratio":
        return output_type == "ratio"
    if kind == "percent":
        return output_type in {"percent", "percentage_point"}
    return False


def _value_supports_year(value: FormulaValue, year: int | None) -> bool:
    if year is None:
        return True
    return all(_resolved_supports_year(resolved, year)
               for resolved in value.resolved)


def _report_year(report_id: str) -> int | None:
    found = re.search(r"(?:financial_statements_|_)(20\d{2})(?:_|$)",
                      str(report_id))
    return int(found.group(1)) if found else None


def _direct_metric_condition(route: dict, question: str) -> Condition | None:
    metric = route.get("metric_norm", "")
    if not metric:
        return None
    parsed = _parse_condition_after(question, metric, "money")
    if parsed is None:
        parsed = _parse_condition_after(question, "", "money")
    if parsed is None:
        return None
    op, threshold = parsed
    variants = _clean_metric_variants(route.get("metric_variants") or [metric])
    metric_clean = variants[0] if variants else _clean_condition_metric(metric)
    return Condition(None, metric_clean, tuple(variants or [metric_clean]),
                     op, threshold, "money")


def _parsed_conditions(route: dict, question: str) -> list[Condition]:
    conditions, seen = [], set()
    for match in _calculation_matches(question):
        cond = _condition_for_match(question, match)
        if cond is None or cond.label in seen:
            continue
        conditions.append(cond)
        seen.add(cond.label)
    return conditions


def _condition_for_match(question: str, match: FormulaMatch) -> Condition | None:
    text = _plain(question)
    if match.spec.name == "working_capital":
        phrase = text[match.start:match.end]
        if "thap hon" in phrase:
            return Condition(match.spec, match.spec.name, (), "<", 0.0, "money")
        if "cao hon" in phrase:
            return Condition(match.spec, match.spec.name, (), ">", 0.0, "money")
    parsed = _parse_condition_segment(text[match.end:match.end + 180], match.spec.kind)
    if parsed is None:
        return None
    op, threshold = parsed
    return Condition(match.spec, match.spec.name, (), op, threshold, match.spec.kind)


def _condition_for_spec(question: str, spec: FormulaSpec) -> Condition | None:
    parsed = None
    for trigger in spec.triggers:
        parsed = _parse_condition_after(question, trigger, spec.kind)
        if parsed is not None:
            break
    if parsed is None:
        return None
    op, threshold = parsed
    return Condition(spec, spec.name, (), op, threshold, spec.kind)


def _parse_condition_after(question: str, anchor: str, kind: str) -> tuple[str, float] | None:
    text = _plain(question)
    start = 0
    if anchor:
        pos = text.find(anchor)
        if pos < 0:
            return None
        start = pos + len(anchor)
    return _parse_condition_segment(text[start:start + 180], kind)


def _parse_condition_segment(segment: str, kind: str) -> tuple[str, float] | None:
    patterns = [
        (">=", r">=\s*(" + _NUM_RE + r")"),
        ("<=", r"<=\s*(" + _NUM_RE + r")"),
        (">", r">\s*(" + _NUM_RE + r")"),
        ("<", r"<\s*(" + _NUM_RE + r")"),
        (">=", r"(?:lon hon hoac bang|khong nho hon|toi thieu|it nhat)\s+(" + _NUM_RE + r")"),
        ("<=", r"(?:nho hon hoac bang|khong lon hon|toi da|nhieu nhat)\s+(" + _NUM_RE + r")"),
        (">", r"(?:lon hon|cao hon|vuot|tren|(?<!nho )(?<!thap )hon)\s+(" + _NUM_RE + r")"),
        ("<", r"(?:nho hon|thap hon|duoi|nho hon muc|duoi muc)\s+(" + _NUM_RE + r")"),
        ("=", r"(?:bang|la)\s+(" + _NUM_RE + r")"),
    ]
    for op, pat in patterns:
        m = re.search(pat, segment)
        if not m:
            continue
        raw = m.group(1)
        num = parse_vn_number(raw)
        if num is None:
            continue
        tail = segment[m.end(1):m.end(1) + 60]
        return op, _scale_threshold(float(num), tail, kind)
    if re.search(r"\bkhong\s+am\b", segment):
        return ">=", 0.0
    if re.search(r"\bkhong\s+duong\b", segment):
        return "<=", 0.0
    if re.search(r"\bduong\b", segment):
        return ">", 0.0
    if re.search(r"\bam\b", segment):
        return "<", 0.0
    return None


def _clean_metric_variants(variants) -> list[str]:
    out = []
    for variant in variants:
        cleaned = _clean_condition_metric(str(variant or ""))
        words = cleaned.split()
        if len(words) < 2 or cleaned in {"cac cong ty", "cac doanh nghiep", "trong so"}:
            continue
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out


def _clean_condition_metric(metric: str) -> str:
    text = _plain(metric)
    text = re.sub(r"^(?:ghi nhan|gia tri|so du)\s+", "", text)
    text = re.split(
        r"\s+(?:lon hon|cao hon|vuot|tren|nho hon|thap hon|duoi|duong|am)\b",
        text, maxsplit=1)[0]
    return text.strip()


def _maybe_top_n_population(question: str, tickers: list[str], year: int | None,
                            route: dict, tables: list[dict], encoder,
                            min_score: float) -> Population | None:
    text = _plain(question)
    m = re.search(r"\btrong(?:\s+so)?\s+(\d+)\s+(?:doanh nghiep|cong ty|ma)\b", text)
    if not m:
        return Population(tickers, [], [])
    n = int(m.group(1))
    if n <= 0:
        return Population(tickers, [], [])
    if "doanh thu thuan" not in text or not _wants_max(text):
        return Population(tickers, [], [])

    vals, resolved = [], []
    for ticker in tickers:
        fact = _FactView({"ticker": ticker, "year": year, "doc_type": route.get("doc_type"),
                          "metric": "doanh thu thuan"})
        r = resolve_fact(fact, tables, ["doanh thu thuan"], encoder, min_score, route=route)
        if r is None:
            return None
        resolved.append(r)
        vals.append((ticker, r.value_vnd))
    vals.sort(key=lambda x: x[1], reverse=True)
    return Population([ticker for ticker, _ in vals[:n]], resolved,
                      [r.expr_vnd() for r in resolved])


def _candidate_tickers(route: dict, tables: list[dict]) -> list[str]:
    seen, out = set(), []
    for ticker in route.get("tickers") or []:
        t = str(ticker).upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    if out:
        return out
    for table in tables:
        t = str(table.get("report_id", "")).split("_")[0].upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _stated_population_size(question: str) -> int | None:
    match = re.search(r"\btrong\s+(\d+)\s+(?:doanh nghiep|cong ty|ma)\b",
                      _plain(question))
    return int(match.group(1)) if match else None


def _primary_year(route: dict) -> int | None:
    years = _route_years(route)
    return max(years) if years else None


def _route_years(route: dict) -> list[int]:
    return [int(y) for y in route.get("years") or [] if y is not None]


def _detected_specs(question: str) -> list[FormulaSpec]:
    matches = _all_spec_matches(question, _FORMULAS)
    seen, out = set(), []
    for match in matches:
        if match.spec.name in seen:
            continue
        seen.add(match.spec.name)
        out.append(match.spec)
    return out


def _all_spec_matches(question: str, specs: list[FormulaSpec]) -> list[FormulaMatch]:
    text = _plain(question)
    raw = []
    for spec in specs:
        for trigger in sorted(spec.triggers, key=len, reverse=True):
            for found in re.finditer(re.escape(trigger), text):
                raw.append(FormulaMatch(spec, found.start(), found.end(), trigger))
    raw.sort(key=lambda m: (m.start, -(m.end - m.start)))
    out = []
    for match in raw:
        if any(prev.spec.name == match.spec.name
               and _overlap(prev.start, prev.end, match.start, match.end)
               for prev in out):
            continue
        out.append(match)
    return sorted(out, key=lambda m: (m.start, m.end))


def _calculation_matches(question: str) -> list[FormulaMatch]:
    formulas = _all_spec_matches(question, _FORMULAS)
    metrics = _all_spec_matches(question, _METRIC_SPECS)
    # A one-line metric inside a longer formula phrase is an operand, not a
    # second calculation. A separate occurrence elsewhere remains available as
    # a ranking selector or filter.
    metrics = [m for m in metrics
               if not any(_overlap(m.start, m.end, f.start, f.end) for f in formulas)]
    return sorted(formulas + metrics, key=lambda m: (m.start, m.end))


def _overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) < min(a1, b1)


def _unique_matches(matches: list[FormulaMatch]) -> dict[str, FormulaMatch]:
    out = {}
    for match in matches:
        out.setdefault(match.spec.name, match)
    return out


def _ranked_match(text: str, matches: list[FormulaMatch]):
    extrema = list(re.finditer(r"\b(cao nhat|lon nhat|thap nhat|nho nhat)\b", text))
    choices = []
    for match in matches:
        for extreme in extrema:
            gap = extreme.start() - match.end
            if 0 <= gap <= 90:
                choices.append((gap, match, extreme))
    if not choices:
        return None
    _gap, match, extreme = min(choices, key=lambda x: x[0])
    want_min = extreme.group(1) in {"thap nhat", "nho nhat"}
    return match, extreme.start(), want_min


def _target_match(matches: list[FormulaMatch], selector: FormulaMatch,
                  extreme_start: int, text: str) -> FormulaMatch | None:
    answer_pos = text.find("bao nhieu", extreme_start)
    answer_pos = len(text) if answer_pos < 0 else answer_pos
    after = [m for m in matches
             if m.start > extreme_start and m.start < answer_pos
             and not (m.start == selector.start and m.end == selector.end)]
    if after:
        return max(after, key=lambda m: m.start)
    before = [m for m in matches
              if m.end <= selector.start
              and not (m.start == selector.start and m.end == selector.end)]
    return max(before, key=lambda m: m.end) if before else None


def _selector_is_direct_answer(text: str, selector: FormulaMatch) -> bool:
    prefix = text[:selector.start].strip(" ,:;.-")
    if not prefix:
        return True
    # Allow a short question lead-in ("what is the highest revenue...") but not
    # "metric X at the year/company whose selector Y is highest".
    if len(prefix.split()) <= 4:
        return True
    return not any(w in prefix for w in ("tai nam", "vao nam", "cua doanh nghiep",
                                         "cua cong ty", "o nam"))


def _value_mode(text: str, match: FormulaMatch, extreme_start: int,
                temporal_allowed: bool) -> str:
    if not temporal_allowed:
        return "level"
    ctx = text[max(0, match.start - 70):extreme_start]
    if "cagr" in ctx:
        return "cagr"
    if "muc giam" in ctx:
        return "decrease"
    if any(w in ctx for w in ("tang truong", "toc do tang")):
        return "growth"
    if any(w in ctx for w in ("muc thay doi", "muc tang")):
        return "delta"
    return "level"


def _target_value_mode(text: str, match: FormulaMatch,
                       temporal_allowed: bool) -> str:
    if not temporal_allowed:
        return "level"
    before = text[max(0, match.start - 75):match.start]
    after = text[match.end:min(len(text), match.end + 45)]
    if any(w in before for w in ("muc thay doi", "thay doi", "muc tang")):
        return "delta"
    if "muc giam" in before:
        return "decrease"
    if "cagr" in before:
        return "cagr"
    if any(w in before for w in ("tang truong", "toc do tang")):
        return "growth"
    if re.match(r"\s*(?:co\s+)?(?:muc\s+)?thay doi\b", after):
        return "delta"
    return "level"


def _change_target_match(text: str,
                         matches: list[FormulaMatch]) -> FormulaMatch | None:
    choices = []
    for match in matches:
        clause_start = max(text.rfind(";", 0, match.start),
                           text.rfind(".", 0, match.start))
        before = text[max(clause_start + 1, match.start - 80):match.start]
        after = text[match.end:min(len(text), match.end + 45)]
        if (any(w in before for w in ("muc thay doi", "thay doi", "muc tang",
                                      "muc giam", "tang truong", "toc do tang"))
                or re.match(r"\s*(?:co\s+)?(?:muc\s+)?(?:thay doi|tang|giam)\b",
                            after)):
            choices.append(match)
    unique = _unique_matches(choices)
    return next(iter(unique.values())) if len(unique) == 1 else None


def _pair_specs_after_difference(text: str,
                                 matches: list[FormulaMatch]) -> list[FormulaSpec]:
    pos = text.find("chenh lech giua")
    if pos < 0:
        return []
    seen, specs = set(), []
    for match in matches:
        if match.start <= pos or match.spec.name in seen:
            continue
        seen.add(match.spec.name)
        specs.append(match.spec)
    return specs[:2]


def _temporal_direction(text: str, match: FormulaMatch) -> str | None:
    before = text[max(0, match.start - 45):match.start]
    after = text[match.end:min(len(text), match.end + 80)]
    if re.search(r"(?:tang|cai thien)\s+$", before) or re.match(
            r"(?:\s+\w+){0,3}\s+(?:tang|cao hon)\b", after):
        return "increase"
    if re.search(r"giam\s+$", before) or re.match(
            r"(?:\s+\w+){0,3}\s+(?:giam|thap hon)\b", after):
        return "decrease"
    if re.match(r"(?:\s+\w+){0,4}\s+duong\b", after):
        return "positive"
    if re.match(r"(?:\s+\w+){0,4}\s+am\b", after):
        return "negative"
    return None


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def _median_expr(exprs: list[str]) -> str:
    ordered = f"sorted([{', '.join(exprs)}])"
    mid = len(exprs) // 2
    if len(exprs) % 2:
        return f"({ordered}[{mid}])"
    return f"(({ordered}[{mid - 1}] + {ordered}[{mid}]) / 2)"


def _condition_expr(expr: str, op: str, threshold: float) -> str:
    return f"(({expr}) {op} {threshold:g})"


def _compare(value: float, op: str, threshold: float) -> bool:
    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    if op == "=":
        return abs(value - threshold) <= max(1e-9, abs(threshold) * 1e-9)
    return False


def _confidence(resolved: list[ResolvedFact], base: float) -> float:
    if not resolved:
        return 0.0
    weakest = min(r.score for r in resolved)
    keys = {(r.report_id, r.table_pos, r.label, r.col) for r in resolved}
    conf = min(base, weakest + 8.0)
    if len(keys) < len(resolved):
        conf = min(conf, 35.0)
    return max(0.0, conf)


def _scale_threshold(value: float, tail: str, kind: str) -> float:
    if kind in {"ratio", "percent"}:
        return value
    tail = _plain(tail)
    if "nghin ty" in tail or "ngan ty" in tail:
        return value * 1e12
    if "tram ty" in tail:
        return value * 1e11
    if "ty" in tail:
        return value * 1e9
    if "trieu" in tail:
        return value * 1e6
    if "nghin" in tail or "ngan" in tail:
        return value * 1e3
    return value


def _plain(text: str) -> str:
    text = strip_diacritics(str(text or "")).lower()
    text = text.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", text).strip()


def _wants_min(question: str) -> bool:
    text = _plain(question)
    return any(w in text for w in (
        "nho nhat", "thap nhat", "it nhat", "thap hon ca", "be nhat",
        "toi thieu"))


def _wants_max(question: str) -> bool:
    text = _plain(question)
    return any(w in text for w in (
        "lon nhat", "cao nhat", "nhieu nhat", "toi da", "dung dau"))


def _looks_nested_selector(question: str) -> bool:
    text = _plain(question)
    if re.search(r"\b(?:cua|cho|tai)\s+(?:cong ty|doanh nghiep|ma|to chuc)\s+co\b",
                 text):
        return True
    if re.search(r"\b(?:cong ty|doanh nghiep|ma|to chuc)\s+co\b", text):
        return True
    return bool(re.search(r"\bnam\s+(?:ma|co|ghi nhan)\b", text))


def _has_complex_temporal_selector(question: str) -> bool:
    text = _plain(question)
    return any(w in text for w in (
        "giai doan", "tu nam", "den nam", "nam dau tien", "nam lien truoc",
        "ngay sau", "sau nam", "truoc nam"))


def _counts_years(question: str) -> bool:
    return "bao nhieu nam" in _plain(question)


def _expected_condition_count(question: str) -> int:
    text = _plain(question)
    expected = 1
    expected = max(expected, text.count("vua co"))
    if "dong thoi" in text and " va " in text:
        expected = max(expected, 2)
    return expected


def _div(a: float, b: float) -> float:
    if b == 0:
        raise ZeroDivisionError("formula denominator is zero")
    return a / b


def _ratio_expr(num: ResolvedFact, den: ResolvedFact) -> str:
    return f"({num.expr_vnd()} / {den.expr_vnd()})"


def _percent_expr(num: ResolvedFact, den: ResolvedFact) -> str:
    return f"({_ratio_expr(num, den)} * 100)"


def _quick_expr(rs: list[ResolvedFact]) -> str:
    return f"(({rs[0].expr_vnd()} - {rs[1].expr_vnd()}) / {rs[2].expr_vnd()})"


def _sum_margin_expr(rs: list[ResolvedFact]) -> str:
    return f"(({rs[0].expr_vnd()} + {rs[1].expr_vnd()}) / {rs[2].expr_vnd()} * 100)"


def _inventory_assets_expr(rs: list[ResolvedFact]) -> str:
    return f"({rs[0].expr_vnd()} / {rs[1].expr_vnd()} * 100)"


def _working_capital_expr(rs: list[ResolvedFact]) -> str:
    return f"({rs[0].expr_vnd()} - {rs[1].expr_vnd()})"


def _interest_coverage_expr(rs: list[ResolvedFact]) -> str:
    return (f"(({rs[0].expr_vnd()} + abs({rs[1].expr_vnd()})) "
            f"/ abs({rs[1].expr_vnd()}))")


def _sum_expr(rs: list[ResolvedFact]) -> str:
    return f"({rs[0].expr_vnd()} + {rs[1].expr_vnd()})"


def _identity_expr(rs: list[ResolvedFact]) -> str:
    return rs[0].expr_vnd()


_NUM_RE = r"[-+]?\d+(?:[\.,]\d+)?"

def _canonical_operand(key: str, strict_codes: bool = True) -> Operand:
    metric = get_metric(key)
    return Operand(
        metric=metric.label,
        variants=metric.variants,
        required_phrases=metric.required_phrases,
        forbidden_phrases=metric.forbidden_phrases,
        expected_codes=metric.codes if strict_codes else (),
    )


_CURRENT_ASSETS = _canonical_operand("current_assets")
_INVENTORY = _canonical_operand("inventory")
_CURRENT_LIABILITIES = _canonical_operand("current_liabilities")
_LIABILITIES = _canonical_operand("liabilities")
_EQUITY = _canonical_operand("equity")
_NET_REVENUE = _canonical_operand("net_revenue")
_GROSS_PROFIT = _canonical_operand("gross_profit")
_NET_PROFIT = _canonical_operand("net_profit")
_CFO = _canonical_operand("cfo", strict_codes=False)
_TOTAL_ASSETS = _canonical_operand("total_assets")
_SELLING_EXP = _canonical_operand("selling_expense")
_ADMIN_EXP = _canonical_operand("administrative_expense")
_FIXED_ASSETS = _canonical_operand("fixed_assets")
_PRETAX_PROFIT = _canonical_operand("pretax_profit")
_INTEREST_EXPENSE = _canonical_operand("interest_expense", strict_codes=False)
_OPERATING_PROFIT = _canonical_operand("operating_profit")

_DIRECT_METRIC_SPEC = FormulaSpec(
    "direct_metric", (), (), "money", lambda vals: vals[0], lambda rs: rs[0].expr_vnd())

_FORMULAS = [
    FormulaSpec(
        "quick_ratio",
        ("he so thanh toan nhanh", "ty so thanh toan nhanh", "quick ratio"),
        (_CURRENT_ASSETS, _INVENTORY, _CURRENT_LIABILITIES),
        "ratio",
        lambda v: _div(v[0] - v[1], v[2]),
        _quick_expr,
    ),
    FormulaSpec(
        "current_ratio",
        ("he so thanh toan hien hanh", "ty so thanh toan hien hanh",
         "current ratio"),
        (_CURRENT_ASSETS, _CURRENT_LIABILITIES),
        "ratio",
        lambda v: _div(v[0], v[1]),
        lambda rs: _ratio_expr(rs[0], rs[1]),
    ),
    FormulaSpec(
        "debt_equity",
        ("he so no phai tra tren von chu so huu", "no phai tra tren von chu so huu",
         "d/e", "debt/equity"),
        (_LIABILITIES, _EQUITY),
        "ratio",
        lambda v: _div(v[0], v[1]),
        lambda rs: _ratio_expr(rs[0], rs[1]),
    ),
    FormulaSpec(
        "debt_assets",
        ("no phai tra tren tong tai san", "debt/assets", "he so no tren tai san"),
        (_LIABILITIES, _TOTAL_ASSETS),
        "ratio",
        lambda v: _div(v[0], v[1]),
        lambda rs: _ratio_expr(rs[0], rs[1]),
    ),
    FormulaSpec(
        "gross_margin",
        ("bien loi nhuan gop", "gross margin",
         "loi nhuan gop tren doanh thu thuan"),
        (_GROSS_PROFIT, _NET_REVENUE),
        "percent",
        lambda v: _div(v[0], v[1]) * 100.0,
        lambda rs: _percent_expr(rs[0], rs[1]),
    ),
    FormulaSpec(
        "net_margin",
        ("bien loi nhuan sau thue", "bien loi nhuan rong", "net margin",
         "loi nhuan sau thue tren doanh thu thuan"),
        (_NET_PROFIT, _NET_REVENUE),
        "percent",
        lambda v: _div(v[0], v[1]) * 100.0,
        lambda rs: _percent_expr(rs[0], rs[1]),
    ),
    FormulaSpec(
        "cfo_margin",
        ("bien dong tien tu hoat dong kinh doanh", "cfo margin",
         "luu chuyen tien thuan tu hoat dong kinh doanh tren doanh thu thuan",
         "dong tien thuan tu hoat dong kinh doanh tren doanh thu thuan"),
        (_CFO, _NET_REVENUE),
        "percent",
        lambda v: _div(v[0], v[1]) * 100.0,
        lambda rs: _percent_expr(rs[0], rs[1]),
    ),
    FormulaSpec(
        "cfo_net_profit",
        ("cfo tren lnst", "cfo/lnst",
         "luu chuyen tien thuan tu hoat dong kinh doanh tren loi nhuan sau thue",
         "dong tien thuan tu hoat dong kinh doanh tren loi nhuan sau thue"),
        (_CFO, _NET_PROFIT),
        "ratio",
        lambda v: _div(v[0], v[1]),
        lambda rs: _ratio_expr(rs[0], rs[1]),
    ),
    FormulaSpec(
        "cfo_operating_profit",
        ("luu chuyen tien thuan tu hoat dong kinh doanh tren loi nhuan thuan tu hoat dong kinh doanh",
         "dong tien thuan tu hoat dong kinh doanh tren loi nhuan thuan tu hoat dong kinh doanh"),
        (_CFO, _OPERATING_PROFIT),
        "ratio",
        lambda v: _div(v[0], v[1]),
        lambda rs: _ratio_expr(rs[0], rs[1]),
    ),
    FormulaSpec(
        "cfo_current_liabilities",
        ("cfo tren no ngan han", "cfo/no ngan han",
         "luu chuyen tien thuan tu hoat dong kinh doanh tren no ngan han",
         "dong tien thuan tu hoat dong kinh doanh tren no ngan han",
         "he so dong tien hoat dong tren no ngan han",
         "dong tien hoat dong tren no ngan han"),
        (_CFO, _CURRENT_LIABILITIES),
        "ratio",
        lambda v: _div(v[0], v[1]),
        lambda rs: _ratio_expr(rs[0], rs[1]),
    ),
    FormulaSpec(
        "roa",
        ("roa", "loi nhuan sau thue tren tong tai san"),
        (_NET_PROFIT, _TOTAL_ASSETS),
        "percent",
        lambda v: _div(v[0], v[1]) * 100.0,
        lambda rs: _percent_expr(rs[0], rs[1]),
    ),
    FormulaSpec(
        "roe",
        ("roe", "loi nhuan sau thue tren von chu so huu"),
        (_NET_PROFIT, _EQUITY),
        "percent",
        lambda v: _div(v[0], v[1]) * 100.0,
        lambda rs: _percent_expr(rs[0], rs[1]),
    ),
    FormulaSpec(
        "sga_intensity",
        ("ty trong chi phi ban hang va quan ly doanh nghiep",
         "chi phi ban hang va quan ly doanh nghiep tren doanh thu thuan",
         "sga intensity", "sg&a intensity"),
        (_SELLING_EXP, _ADMIN_EXP, _NET_REVENUE),
        "percent",
        lambda v: _div(v[0] + v[1], v[2]) * 100.0,
        _sum_margin_expr,
    ),
    FormulaSpec(
        "fixed_asset_turnover",
        ("vong quay tai san co dinh", "doanh thu thuan tren tai san co dinh",
         "fixed asset turnover"),
        (_NET_REVENUE, _FIXED_ASSETS),
        "ratio",
        lambda v: _div(v[0], v[1]),
        lambda rs: _ratio_expr(rs[0], rs[1]),
    ),
    FormulaSpec(
        "inventory_assets",
        ("hang ton kho tren tong tai san", "ty trong hang ton kho"),
        (_INVENTORY, _TOTAL_ASSETS),
        "percent",
        lambda v: _div(v[0], v[1]) * 100.0,
        _inventory_assets_expr,
    ),
    FormulaSpec(
        "working_capital",
        ("von luu dong rong", "tai san ngan han thap hon no ngan han",
         "tai san ngan han cao hon no ngan han"),
        (_CURRENT_ASSETS, _CURRENT_LIABILITIES),
        "money",
        lambda v: v[0] - v[1],
        _working_capital_expr,
    ),
    FormulaSpec(
        "interest_coverage",
        ("he so kha nang thanh toan lai vay", "kha nang thanh toan lai vay",
         "loi nhuan truoc lai vay va thue", "ebit tren chi phi lai vay"),
        (_PRETAX_PROFIT, _INTEREST_EXPENSE),
        "ratio",
        lambda v: _div(v[0] + abs(v[1]), abs(v[1])),
        _interest_coverage_expr,
    ),
]

# One-line statement metrics are useful as selectors and filters.  They stay
# separate from _FORMULAS so a direct lookup question is not mistaken for a
# formula question merely because it mentions revenue or inventory.
_METRIC_SPECS = [
    FormulaSpec("net_revenue", ("doanh thu thuan",), (_NET_REVENUE,),
                "money", lambda v: v[0], _identity_expr),
    FormulaSpec("gross_profit", ("loi nhuan gop",), (_GROSS_PROFIT,),
                "money", lambda v: v[0], _identity_expr),
    FormulaSpec("net_profit", ("loi nhuan sau thue", "lnst"), (_NET_PROFIT,),
                "money", lambda v: v[0], _identity_expr),
    FormulaSpec(
        "cfo",
        ("luu chuyen tien thuan tu hoat dong kinh doanh",
         "dong tien thuan tu hoat dong kinh doanh", "cfo"),
        (_CFO,), "money", lambda v: v[0], _identity_expr),
    FormulaSpec("inventory", ("gia tri hang ton kho", "hang ton kho"),
                (_INVENTORY,), "money", lambda v: v[0], _identity_expr),
    FormulaSpec(
        "sga_expense",
        ("tong chi phi ban hang va chi phi quan ly doanh nghiep",
         "chi phi ban hang va chi phi quan ly doanh nghiep"),
        (_SELLING_EXP, _ADMIN_EXP), "money", lambda v: v[0] + v[1], _sum_expr),
    FormulaSpec("pretax_profit", ("loi nhuan truoc thue", "lntt"),
                (_PRETAX_PROFIT,), "money", lambda v: v[0], _identity_expr),
    FormulaSpec("interest_expense", ("chi phi lai vay",), (_INTEREST_EXPENSE,),
                "money", lambda v: v[0], _identity_expr),
]
