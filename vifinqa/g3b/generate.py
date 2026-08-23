"""Program-first G3B question families using the Selection-v2 IR."""
from __future__ import annotations

from collections import defaultdict

from .common import canonical_sha256
from .source import DISPLAY, Fact, statement_facts


def fact_nodes(
    facts: list[Fact], names: list[str], roles: list[str]
) -> dict:
    return {
        name: {
            "ref": index,
            "as": "money",
            "role": role,
            "slot": f"F{index}",
        }
        for index, (name, role) in enumerate(zip(names, roles), 1)
    }


def program(
    facts: list[Fact],
    names: list[str],
    roles: list[str],
    root: dict,
    output_type: str,
    bindings: dict | None = None,
) -> dict:
    return {
        "schema_version": 2,
        "output_type": output_type,
        "facts": fact_nodes(facts, names, roles),
        "bindings": bindings or {},
        "root": root,
    }


def shape_node(node: dict, typed_program: dict) -> object:
    if "var" in node:
        name = str(node["var"])
        if name in typed_program["facts"]:
            return {"fact": typed_program["facts"][name]["role"]}
        return shape_node(typed_program["bindings"][name], typed_program)
    if "year" in node:
        return {"year": "YEAR"}
    if "literal" in node:
        return {"literal": node.get("type", "number")}
    if node.get("op") in {"argmax_project", "argmin_project"}:
        return {
            "op": node["op"],
            "items": [
                {
                    "score": shape_node(item["score"], typed_program),
                    "result": shape_node(item["result"], typed_program),
                }
                for item in node["items"]
            ],
        }
    return {
        "op": node.get("op"),
        "args": [
            shape_node(arg, typed_program)
            for arg in node.get("args", [])
        ],
    }


def primitive_ops(node: object) -> set[str]:
    if not isinstance(node, dict):
        return set()
    output = {str(node["op"])} if node.get("op") else set()
    for value in node.values():
        children = value if isinstance(value, list) else [value]
        for child in children:
            if isinstance(child, dict):
                output.update(primitive_ops(child))
    return output


def candidate(
    family: str,
    question: str,
    facts: list[Fact],
    answer: float,
    output_type: str,
    typed_program: dict,
    *,
    metric_family: str,
    stress_tags: list[str] | None = None,
) -> dict:
    fact_ids = [fact.fact_id for fact in facts]
    documents = list(dict.fromkeys(fact.report_id for fact in facts))
    tables = list(dict.fromkeys(
        f"{fact.report_id}|{fact.table_line}" for fact in facts
    ))
    shape = shape_node(typed_program["root"], typed_program)
    return {
        "family": family,
        "operator": str(typed_program["root"].get("op") or "lookup"),
        "question": question,
        "facts": facts,
        "fact_ids": fact_ids,
        "answer": round(float(answer), 2),
        "output_type": output_type,
        "unit_scale": 1e6 if output_type == "number" else 1.0,
        "tolerance": 0.01,
        "program": typed_program,
        "metric_family": metric_family,
        "tree_shape": canonical_sha256(shape),
        "tree_shape_value": shape,
        "primitive_ops": sorted(primitive_ops(shape)),
        "relevant_docs": documents,
        "relevant_tables": tables,
        "tickers": sorted({fact.ticker for fact in facts}),
        "years": sorted({fact.period_year for fact in facts}),
        "report_ids": sorted(documents),
        "scopes": sorted({fact.doc_type for fact in facts}),
        "stress_tags": sorted(set(stress_tags or [])),
        "fact_group": canonical_sha256(sorted(fact_ids)),
        "program_key": canonical_sha256({
            "family": family,
            "fact_ids": sorted(fact_ids),
            "program": typed_program,
        }),
    }


def series_candidates(facts: list[Fact]) -> list[dict]:
    output: list[dict] = []
    groups: dict[tuple[str, str, str], list[Fact]] = defaultdict(list)
    for fact in statement_facts(facts):
        groups[(fact.ticker, fact.doc_type, fact.metric_key)].append(fact)
    for (_ticker, _scope, metric_key), values in sorted(groups.items()):
        unique = {}
        for fact in sorted(
            values, key=lambda item: (item.period_year, item.fact_id)
        ):
            unique.setdefault(fact.period_year, fact)
        years = sorted(unique)
        for index in range(len(years) - 1):
            earlier = unique[years[index]]
            later = unique[years[index + 1]]
            if later.period_year - earlier.period_year > 3:
                continue
            label = DISPLAY.get(metric_key, earlier.label.lower())
            average = program(
                [earlier, later],
                ["earlier", "later"],
                ["value", "value"],
                {
                    "op": "average",
                    "args": [
                        {"var": "earlier"},
                        {"var": "later"},
                    ],
                },
                "number",
            )
            output.append(candidate(
                "simple_average",
                f"Giá trị bình quân {label} của {earlier.ticker} theo "
                f"báo cáo {earlier.scope} trong hai năm "
                f"{earlier.period_year} và {later.period_year} là bao "
                f"nhiêu triệu đồng?",
                [earlier, later],
                (earlier.base_value + later.base_value) / 2e6,
                "number",
                average,
                metric_family=metric_key,
            ))
            periods = later.period_year - earlier.period_year
            if (
                periods >= 2
                and earlier.base_value > 1
                and later.base_value > 1
            ):
                cagr = program(
                    [later, earlier],
                    ["end", "base"],
                    ["end", "base"],
                    {
                        "op": "cagr_percent",
                        "periods": periods,
                        "args": [{"var": "end"}, {"var": "base"}],
                    },
                    "percent",
                )
                output.append(candidate(
                    "cagr",
                    f"CAGR của {label} {earlier.ticker} theo báo cáo "
                    f"{earlier.scope} từ năm {earlier.period_year} đến "
                    f"năm {later.period_year} ({periods} kỳ) là bao nhiêu "
                    f"phần trăm?",
                    [later, earlier],
                    (
                        (later.base_value / earlier.base_value)
                        ** (1 / periods)
                        - 1
                    ) * 100,
                    "percent",
                    cagr,
                    metric_family=metric_key,
                ))
        for index in range(len(years) - 2):
            selected_years = years[index:index + 3]
            if selected_years[-1] - selected_years[0] > 5:
                continue
            selected = [unique[year] for year in selected_years]
            label = DISPLAY.get(metric_key, selected[0].label.lower())
            names = [f"value_{year}" for year in selected_years]
            items = [
                {
                    "score": {"var": name},
                    "result": {"year": year},
                }
                for name, year in zip(names, selected_years)
            ]
            for family, op, picker, adjective in (
                ("ranking_argmax", "argmax_project", max, "cao nhất"),
                ("ranking_argmin", "argmin_project", min, "thấp nhất"),
            ):
                typed = {
                    "schema_version": 2,
                    "output_type": "year",
                    "facts": fact_nodes(selected, names, ["rank"] * 3),
                    "bindings": {},
                    "root": {"op": op, "items": items},
                }
                answer = picker(
                    selected, key=lambda item: item.base_value
                ).period_year
                output.append(candidate(
                    family,
                    f"Trong các năm {', '.join(map(str, selected_years))}, "
                    f"năm nào {label} của {selected[0].ticker} theo "
                    f"báo cáo {selected[0].scope} {adjective}?",
                    selected,
                    answer,
                    "year",
                    typed,
                    metric_family=metric_key,
                ))
    return output


def count_candidates(facts: list[Fact]) -> list[dict]:
    output: list[dict] = []
    groups: dict[tuple[int, str, str], list[Fact]] = defaultdict(list)
    for fact in statement_facts(facts):
        groups[(fact.period_year, fact.doc_type, fact.metric_key)].append(
            fact
        )
    for (year, _scope, metric_key), values in sorted(groups.items()):
        unique = {}
        for fact in sorted(values, key=lambda item: item.fact_id):
            unique.setdefault(fact.ticker, fact)
        ordered = list(unique.values())
        for start in range(0, len(ordered) - 2, 3):
            selected = ordered[start:start + 3]
            names = [f"value_{index}" for index in range(1, 4)]
            conditions = [
                {
                    "op": "gt",
                    "args": [
                        {"var": name},
                        {"literal": 0, "type": "number"},
                    ],
                }
                for name in names
            ]
            typed = program(
                selected,
                names,
                ["filter"] * 3,
                {"op": "count_true", "args": conditions},
                "count",
            )
            label = DISPLAY.get(metric_key, selected[0].label.lower())
            output.append(candidate(
                "count_positive",
                f"Trong {', '.join(item.ticker for item in selected)}, "
                f"có bao nhiêu doanh nghiệp có {label} dương theo báo "
                f"cáo {selected[0].scope} năm {year}, so với ngưỡng 0?",
                selected,
                sum(item.base_value > 0 for item in selected),
                "count",
                typed,
                metric_family=metric_key,
            ))
    return output


def ratio_candidates(facts: list[Fact]) -> list[dict]:
    output: list[dict] = []
    by_report: dict[str, dict[str, Fact]] = defaultdict(dict)
    for fact in statement_facts(facts):
        by_report[fact.report_id][fact.metric_key] = fact
    margins: dict[tuple[str, str], list[tuple[Fact, Fact]]] = defaultdict(
        list
    )
    for values in by_report.values():
        revenue = values.get("net_revenue")
        profit = values.get("net_profit")
        if revenue and profit and abs(revenue.base_value) > 1:
            margins[(profit.ticker, profit.doc_type)].append(
                (profit, revenue)
            )
        debt = values.get("liabilities")
        assets = values.get("total_assets")
        if debt and assets and abs(assets.base_value) > 1:
            typed = program(
                [debt, assets],
                ["debt", "assets"],
                ["numerator", "denominator"],
                {
                    "op": "divide",
                    "args": [{"var": "debt"}, {"var": "assets"}],
                },
                "ratio",
            )
            output.append(candidate(
                "debt_assets_ratio",
                f"Nợ phải trả bằng bao nhiêu lần tổng tài sản của "
                f"{assets.ticker} theo báo cáo {assets.scope} năm "
                f"{assets.period_year}?",
                [debt, assets],
                debt.base_value / assets.base_value,
                "ratio",
                typed,
                metric_family="debt_assets",
            ))
    for (_ticker, _scope), values in sorted(margins.items()):
        ordered = sorted(
            values, key=lambda pair: pair[0].period_year
        )
        for previous, current in zip(ordered, ordered[1:]):
            profit_before, revenue_before = previous
            profit_now, revenue_now = current
            if profit_now.period_year - profit_before.period_year != 1:
                continue
            selected = [
                profit_now,
                revenue_now,
                profit_before,
                revenue_before,
            ]
            names = [
                "profit_now",
                "revenue_now",
                "profit_before",
                "revenue_before",
            ]
            roles = [
                "numerator",
                "denominator",
                "numerator",
                "denominator",
            ]
            bindings = {
                "margin_now": {
                    "op": "divide",
                    "args": [
                        {"var": "profit_now"},
                        {"var": "revenue_now"},
                    ],
                },
                "margin_before": {
                    "op": "divide",
                    "args": [
                        {"var": "profit_before"},
                        {"var": "revenue_before"},
                    ],
                },
            }
            margin_now = profit_now.base_value / revenue_now.base_value
            margin_before = (
                profit_before.base_value / revenue_before.base_value
            )
            pp = program(
                selected,
                names,
                roles,
                {
                    "op": "percentage_point_change",
                    "args": [
                        {"var": "margin_now"},
                        {"var": "margin_before"},
                    ],
                },
                "percentage_point",
                bindings,
            )
            output.append(candidate(
                "percentage_point_change",
                f"Biên lợi nhuận sau thuế của {profit_now.ticker} theo "
                f"báo cáo {profit_now.scope} thay đổi bao nhiêu điểm "
                f"phần trăm từ năm {profit_before.period_year} đến năm "
                f"{profit_now.period_year}?",
                selected,
                (margin_now - margin_before) * 100,
                "percentage_point",
                pp,
                metric_family="net_margin",
                stress_tags=["period", "compositional"],
            ))
            nested = program(
                selected,
                names,
                roles,
                {
                    "op": "average",
                    "args": [
                        {"var": "margin_before"},
                        {"var": "margin_now"},
                    ],
                },
                "percent",
                bindings,
            )
            output.append(candidate(
                "nested_margin_average",
                f"Biên lợi nhuận sau thuế bình quân của "
                f"{profit_now.ticker} theo báo cáo {profit_now.scope} "
                f"trong hai năm {profit_before.period_year} và "
                f"{profit_now.period_year} là bao nhiêu phần trăm?",
                selected,
                (margin_before + margin_now) / 2 * 100,
                "percent",
                nested,
                metric_family="net_margin",
                stress_tags=["period", "compositional"],
            ))
    return output


def stress_candidates(facts: list[Fact]) -> list[dict]:
    output: list[dict] = []
    by_scope: dict[
        tuple[str, int, str], dict[str, Fact]
    ] = defaultdict(dict)
    for fact in statement_facts(facts):
        by_scope[
            (fact.ticker, fact.report_year, fact.metric_key)
        ][fact.doc_type] = fact
    for (_ticker, _year, metric_key), values in sorted(by_scope.items()):
        consolidated = values.get("consolidated")
        separate = values.get("separate")
        if not consolidated or not separate:
            continue
        typed = program(
            [consolidated, separate],
            ["consolidated", "separate"],
            ["value", "value"],
            {
                "op": "subtract",
                "args": [
                    {"var": "consolidated"},
                    {"var": "separate"},
                ],
            },
            "number",
        )
        output.append(candidate(
            "scope_delta",
            f"{DISPLAY.get(metric_key, consolidated.label.lower()).capitalize()} "
            f"hợp nhất cao hơn hoặc thấp hơn số liệu riêng của công ty "
            f"mẹ {consolidated.ticker} năm {consolidated.period_year} "
            f"bao nhiêu triệu đồng? Tính hợp nhất trừ riêng.",
            [consolidated, separate],
            (consolidated.base_value - separate.base_value) / 1e6,
            "number",
            typed,
            metric_family=metric_key,
            stress_tags=["scope", "ambiguous_scope"],
        ))
    for fact in statement_facts(facts, current=False):
        typed = program(
            [fact],
            ["opening_value"],
            ["value"],
            {"op": "lookup", "args": [{"var": "opening_value"}]},
            "number",
        )
        output.append(candidate(
            "prior_period_lookup",
            f"Trong báo cáo {fact.scope} năm {fact.report_year} của "
            f"{fact.ticker}, số đầu kỳ tương ứng cuối năm "
            f"{fact.period_year} của "
            f"{DISPLAY.get(fact.metric_key, fact.label.lower())} là "
            f"bao nhiêu triệu đồng?",
            [fact],
            fact.base_value / 1e6,
            "number",
            typed,
            metric_family=fact.metric_key,
            stress_tags=["period", "prior_period"],
        ))
    for fact in facts:
        if (
            fact.source_kind != "note_table"
            or fact.period_year != fact.report_year
        ):
            continue
        typed = program(
            [fact],
            ["note_value"],
            ["value"],
            {"op": "lookup", "args": [{"var": "note_value"}]},
            "number",
        )
        output.append(candidate(
            "note_lookup",
            f"Theo bảng thuyết minh trong báo cáo {fact.scope} năm "
            f"{fact.report_year} của {fact.ticker}, "
            f"{fact.label.lower()} là bao nhiêu triệu đồng?",
            [fact],
            fact.base_value / 1e6,
            "number",
            typed,
            metric_family=fact.metric_key,
            stress_tags=["note_table"],
        ))
    return output


def build_candidates(facts: list[Fact]) -> list[dict]:
    values = [
        *series_candidates(facts),
        *count_candidates(facts),
        *ratio_candidates(facts),
        *stress_candidates(facts),
    ]
    unique = {}
    for item in values:
        unique.setdefault(item["program_key"], item)
    return sorted(unique.values(), key=lambda item: item["program_key"])
