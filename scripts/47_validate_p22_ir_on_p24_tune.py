"""Validate Selection v2 expressiveness on the labelled P2.4 tune set.

This is an oracle representability test, not a model-accuracy result: gold
evidence/AST is translated into the production IR and compiled/replayed.  The
script explicitly rejects locked paths and is never copied into Kaggle runtime.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.codegen.executor import run_code
from vifinqa.codegen.selection_v2 import IRValidationError, compile_program
from vifinqa.utils.io import read_jsonl, setup_stdout


def _strip_output_wrapper(ast: dict, output: dict) -> dict:
    output_type = output.get("type")
    args = ast.get("args") if isinstance(ast, dict) else None
    if output_type == "number" and ast.get("op") == "divide" \
            and isinstance(args, list) and len(args) == 2 \
            and args[1].get("kind") == "literal" \
            and float(args[1].get("value")) == float(output.get("scale") or 1.0):
        return args[0]
    if output_type == "percent" and ast.get("op") == "multiply" \
            and isinstance(args, list) and len(args) == 2:
        for pos in (0, 1):
            literal, other = args[pos], args[1 - pos]
            if literal.get("kind") == "literal" and float(literal.get("value")) == 100.0:
                return other
    return ast


def _evidence_ids(ast) -> set[str]:
    if not isinstance(ast, dict):
        return set()
    out = {str(ast["evidence_id"])} if ast.get("kind") == "evidence" else set()
    for value in ast.values():
        if isinstance(value, dict):
            out.update(_evidence_ids(value))
        elif isinstance(value, list):
            for item in value:
                out.update(_evidence_ids(item))
    return out


def _contains_op(ast: dict, op: str) -> bool:
    if not isinstance(ast, dict):
        return False
    if ast.get("op") == op:
        return True
    return any(_contains_op(item, op) for item in ast.get("args") or [])


def _convert(ast: dict, output_type: str, facts: dict[str, str]) -> dict:
    kind = ast.get("kind")
    if kind == "evidence":
        return {"var": facts[str(ast["evidence_id"])]}
    if kind == "literal":
        value = float(ast["value"])
        if value == int(value) and 1900 <= value <= 2100:
            return {"year": int(value)}
        return {"literal": int(value) if value == int(value) else value,
                "type": "ratio"}
    if kind != "op":
        raise ValueError(f"unknown gold AST kind {kind!r}")
    op = str(ast.get("op"))
    args = list(ast.get("args") or [])
    mapped = {
        "lookup": "lookup", "add": "add", "subtract": "subtract",
        "sum": "sum", "average": "average", "min": "min", "max": "max",
        "median": "median", "multiply": "multiply", "divide": "divide",
        "power": "power", "abs": "abs", "negate": "negate", "count_true": "count_true",
        "gt": "gt", "ge": "ge", "lt": "lt", "le": "le",
        "eq": "eq", "ne": "ne", "and": "and", "or": "or",
        "if_else": "if_else",
    }
    if op in {"argmax_project", "argmin_project"}:
        if len(args) < 4 or len(args) % 2:
            raise ValueError(f"{op} gold args are not score/result pairs")
        return {"op": op, "items": [
            {"score": _convert(args[i], output_type, facts),
             "result": _convert(args[i + 1], output_type, facts)}
            for i in range(0, len(args), 2)
        ]}
    target_op = mapped.get(op)
    if target_op is None:
        raise ValueError(f"unsupported gold op {op!r}")
    if output_type == "percentage_point" and op == "subtract":
        target_op = "percentage_point_change"
    return {"op": target_op,
            "args": [_convert(arg, output_type, facts) for arg in args]}


def _candidate(evidence: dict, fact_type: str):
    del fact_type
    report = str(evidence["report_id"])
    match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", str(evidence.get("col_name", "")))
    if match is None:
        match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", report)
    year = int(match.group(1)) if match else None
    return SimpleNamespace(
        var=str(evidence["variable"]), report_id=report,
        table_pos=int(evidence["table_pos"]), row=int(evidence["row"]),
        col=int(evidence["col"]), label=str(evidence.get("label", "")),
        code=str(evidence.get("code", "")), col_name=str(evidence.get("col_name", "")),
        value=float(evidence["value"]), unit_scale=float(evidence["unit_scale"]),
        score=100.0, rescue=False, fact_year=year, report_year=year,
        fact_slot="", fact_role="value", fact_metric="", ticker=report.split("_", 1)[0],
    )


def _frames(evidence: list[dict]) -> dict[str, pd.DataFrame]:
    grouped: dict[str, list[dict]] = {}
    seen = set()
    for item in evidence:
        key = (item["variable"], int(item["row"]), int(item["col"]))
        if key in seen:
            continue
        seen.add(key)
        grouped.setdefault(str(item["variable"]), []).append({
            "row": int(item["row"]), "col": int(item["col"]),
            "value": float(item["value"]), "unit_scale": float(item["unit_scale"]),
            "label": str(item.get("label", "")), "col_name": str(item.get("col_name", "")),
        })
    return {name: pd.DataFrame(rows) for name, rows in grouped.items()}


def validate(path: Path) -> dict:
    if "locked" in path.name.lower() or "locked" in str(path.parent).lower():
        raise SystemExit("this development validator refuses locked paths")
    rows = read_jsonl(path)
    outcomes, failures = Counter(), []
    for row in rows:
        ast = _strip_output_wrapper(dict(row["ast"]), row["output"])
        used = _evidence_ids(ast)
        evidence = [item for item in row["evidence"] if str(item["evidence_id"]) in used]
        output_type = str(row["output"]["type"])
        direct_percent = output_type in {"percent", "percentage_point"} \
            and not _contains_op(ast, "divide")
        fact_type = "percent" if direct_percent else "number"
        candidates = [_candidate(item, fact_type) for item in evidence]
        index = {str(item["evidence_id"]): i for i, item in enumerate(evidence, 1)}
        facts = {f"e{i}": {"ref": i, "as": fact_type, "role": "value"}
                 for i in range(1, len(evidence) + 1)}
        fact_names = {eid: f"e{idx}" for eid, idx in index.items()}
        program = {
            "schema_version": 2,
            "output_type": output_type,
            "facts": facts,
            "bindings": {},
            "root": _convert(ast, output_type, fact_names),
        }
        route = {
            "output_type": output_type,
            "unit_scale": (float(row["output"].get("scale") or 1.0)
                           if output_type == "number" else 1.0),
            "years": [c.fact_year for c in candidates if c.fact_year],
        }
        try:
            compiled = compile_program(program, candidates, route, row["question"])
            replay = run_code(compiled.query, _frames(evidence))
            if replay.get("status") != "ok":
                raise ValueError(str(replay.get("error") or replay.get("status")))
            expected = float(row["output"]["value"])
            if abs(float(replay["value"]) - expected) > 0.011:
                raise ValueError(f"replay {replay['value']} != gold {expected}")
        except (IRValidationError, ValueError, KeyError, TypeError) as exc:
            code = getattr(exc, "code", type(exc).__name__)
            outcomes[str(code)] += 1
            failures.append({"id": int(row["id"]), "reason_code": str(code),
                             "reason": str(exc)[:500], "output_type": output_type})
        else:
            outcomes["verified"] += 1
    return {
        "schema_version": "p22_ir_p24_tune_representability_v1",
        "scope": "oracle evidence/AST structural gate; not model accuracy",
        "input": str(path),
        "counts": {"rows": len(rows), "verified": outcomes["verified"],
                   "failed": len(failures), "by_outcome": dict(outcomes)},
        "failures": failures,
    }


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/devset_p24/p24_tune_gold.final.jsonl")
    parser.add_argument("--out", default="artifacts/p22_targets/p22_ir_p24_tune_audit.json")
    args = parser.parse_args()
    report = validate(Path(args.gold))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    print(f"OK -> {output}")


if __name__ == "__main__":
    main()

