"""Build exact P2.4 tune gold from compact cell-and-expression specs.

An evidence name (``E1``, ``E2``, ...) evaluates to the referenced cell value
times its unit scale. The builder derives a typed AST, exact-cell pandas replay,
answer and hashes, and has no code path that reads the locked split.
"""
from __future__ import annotations

import ast as py_ast
import math
import re
from pathlib import Path
from statistics import median
from typing import Any

from ..codegen.executor import run_code
from ..utils.io import read_jsonl, write_jsonl
from .p24 import P24ValidationError, StoreTableLoader, canonical_sha256, validate_gold_records

AUTHORING_SCHEMA = "p24_tune_authoring_v1"
_EID = re.compile(r"^E[1-9][0-9]*$")
_BIN = {py_ast.Add: "add", py_ast.Sub: "subtract", py_ast.Mult: "multiply",
        py_ast.Div: "divide", py_ast.Pow: "power"}
_CMP = {py_ast.Gt: "gt", py_ast.GtE: "ge", py_ast.Lt: "lt",
        py_ast.LtE: "le", py_ast.Eq: "eq", py_ast.NotEq: "ne"}
_CALL = {name: name for name in (
    "sum", "average", "min", "max", "median", "count_true", "abs",
    "argmax_project", "argmin_project",
)}


class P24AuthoringError(ValueError):
    pass


def _fail(message: str):
    raise P24AuthoringError(message)


def _parse(expression: str) -> py_ast.AST:
    if not isinstance(expression, str) or not expression.strip():
        _fail("expression must be non-empty")
    try:
        return py_ast.parse(expression, mode="eval").body
    except SyntaxError as exc:
        raise P24AuthoringError(f"invalid expression: {exc.msg}") from exc


def expression_to_gold_ast(node: py_ast.AST) -> dict[str, Any]:
    if isinstance(node, py_ast.Name) and _EID.fullmatch(node.id):
        return {"kind": "evidence", "evidence_id": node.id}
    if isinstance(node, py_ast.Constant) and isinstance(node.value, (int, float)):
        value = float(node.value)
        if not math.isfinite(value):
            _fail("literal must be finite")
        return {"kind": "literal", "value": value}
    if isinstance(node, py_ast.BinOp) and type(node.op) in _BIN:
        return {"kind": "op", "op": _BIN[type(node.op)], "args": [
            expression_to_gold_ast(node.left), expression_to_gold_ast(node.right)]}
    if isinstance(node, py_ast.UnaryOp) and isinstance(node.op, py_ast.USub):
        return {"kind": "op", "op": "negate", "args": [expression_to_gold_ast(node.operand)]}
    if isinstance(node, py_ast.Compare) and len(node.ops) == len(node.comparators) == 1:
        op = _CMP.get(type(node.ops[0]))
        if op is None:
            _fail("unsupported comparison")
        return {"kind": "op", "op": op, "args": [
            expression_to_gold_ast(node.left), expression_to_gold_ast(node.comparators[0])]}
    if isinstance(node, py_ast.BoolOp) and isinstance(node.op, (py_ast.And, py_ast.Or)):
        return {"kind": "op", "op": "and" if isinstance(node.op, py_ast.And) else "or",
                "args": [expression_to_gold_ast(item) for item in node.values]}
    if isinstance(node, py_ast.IfExp):
        return {"kind": "op", "op": "if_else", "args": [
            expression_to_gold_ast(node.test), expression_to_gold_ast(node.body),
            expression_to_gold_ast(node.orelse)]}
    if isinstance(node, py_ast.Call) and isinstance(node.func, py_ast.Name):
        op = _CALL.get(node.func.id)
        if op is None or node.keywords:
            _fail(f"unsupported function {node.func.id!r}")
        args = [expression_to_gold_ast(arg) for arg in node.args]
        if op in {"argmax_project", "argmin_project"} and (len(args) < 4 or len(args) % 2):
            _fail(f"{op} requires score/project pairs")
        return {"kind": "op", "op": op, "args": args}
    _fail(f"unsupported expression node {type(node).__name__}")


def _eval(node: py_ast.AST, values: dict[str, float]) -> float:
    if isinstance(node, py_ast.Name) and node.id in values:
        return float(values[node.id])
    if isinstance(node, py_ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, py_ast.BinOp) and type(node.op) in _BIN:
        a, b = _eval(node.left, values), _eval(node.right, values)
        if isinstance(node.op, py_ast.Add): return a + b
        if isinstance(node.op, py_ast.Sub): return a - b
        if isinstance(node.op, py_ast.Mult): return a * b
        if isinstance(node.op, py_ast.Div):
            if abs(b) < 1e-18: _fail("division by zero")
            return a / b
        return a ** b
    if isinstance(node, py_ast.UnaryOp) and isinstance(node.op, py_ast.USub):
        return -_eval(node.operand, values)
    if isinstance(node, py_ast.Compare) and len(node.ops) == len(node.comparators) == 1:
        a, b, op = _eval(node.left, values), _eval(node.comparators[0], values), node.ops[0]
        result = (a > b if isinstance(op, py_ast.Gt) else
                  a >= b if isinstance(op, py_ast.GtE) else
                  a < b if isinstance(op, py_ast.Lt) else
                  a <= b if isinstance(op, py_ast.LtE) else
                  a == b if isinstance(op, py_ast.Eq) else a != b)
        return float(result)
    if isinstance(node, py_ast.BoolOp):
        bits = [bool(_eval(item, values)) for item in node.values]
        return float(all(bits) if isinstance(node.op, py_ast.And) else any(bits))
    if isinstance(node, py_ast.IfExp):
        return _eval(node.body if bool(_eval(node.test, values)) else node.orelse, values)
    if isinstance(node, py_ast.Call) and isinstance(node.func, py_ast.Name):
        args, name = [_eval(arg, values) for arg in node.args], node.func.id
        if name == "sum": return float(sum(args))
        if name == "average": return float(sum(args) / len(args))
        if name == "min": return float(min(args))
        if name == "max": return float(max(args))
        if name == "median": return float(median(args))
        if name == "count_true": return float(sum(bool(arg) for arg in args))
        if name == "abs": return float(abs(args[0]))
        if name in {"argmax_project", "argmin_project"}:
            pairs = [(args[i], args[i + 1]) for i in range(0, len(args), 2)]
            chooser = max if name == "argmax_project" else min
            return chooser(enumerate(pairs), key=lambda x: (x[1][0], -x[0]))[1][1]
    _fail(f"cannot evaluate {type(node).__name__}")


def _leaf(var: str, row: int, col: int) -> str:
    mask = f"({var}['row'] == {row}) & ({var}['col'] == {col})"
    return (f"(float({var}.loc[{mask}, 'value'].iloc[0]) * "
            f"float({var}.loc[{mask}, 'unit_scale'].iloc[0]))")


def _compile(node: py_ast.AST, leaves: dict[str, str]) -> str:
    if isinstance(node, py_ast.Name) and node.id in leaves: return leaves[node.id]
    if isinstance(node, py_ast.Constant) and isinstance(node.value, (int, float)):
        return repr(float(node.value))
    if isinstance(node, py_ast.BinOp) and type(node.op) in _BIN:
        symbol = {py_ast.Add: "+", py_ast.Sub: "-", py_ast.Mult: "*",
                  py_ast.Div: "/", py_ast.Pow: "**"}[type(node.op)]
        return f"({_compile(node.left, leaves)} {symbol} {_compile(node.right, leaves)})"
    if isinstance(node, py_ast.UnaryOp) and isinstance(node.op, py_ast.USub):
        return f"(-{_compile(node.operand, leaves)})"
    if isinstance(node, py_ast.Compare):
        symbol = {py_ast.Gt: ">", py_ast.GtE: ">=", py_ast.Lt: "<",
                  py_ast.LtE: "<=", py_ast.Eq: "==", py_ast.NotEq: "!="}[type(node.ops[0])]
        return f"({_compile(node.left, leaves)} {symbol} {_compile(node.comparators[0], leaves)})"
    if isinstance(node, py_ast.BoolOp):
        symbol = " and " if isinstance(node.op, py_ast.And) else " or "
        return "(" + symbol.join(_compile(item, leaves) for item in node.values) + ")"
    if isinstance(node, py_ast.IfExp):
        return f"({_compile(node.body, leaves)} if {_compile(node.test, leaves)} else {_compile(node.orelse, leaves)})"
    if isinstance(node, py_ast.Call) and isinstance(node.func, py_ast.Name):
        args, name = [_compile(arg, leaves) for arg in node.args], node.func.id
        if name in {"sum", "min", "max"}: return f"{name}([{', '.join(args)}])"
        if name == "average": return f"(sum([{', '.join(args)}]) / {len(args)})"
        if name == "median":
            ordered, n = f"sorted([{', '.join(args)}])", len(args)
            return (f"{ordered}[{n//2}]" if n % 2 else
                    f"(({ordered}[{n//2-1}] + {ordered}[{n//2}]) / 2.0)")
        if name == "count_true": return f"sum([{', '.join(f'int({x})' for x in args)}])"
        if name == "abs": return f"abs({args[0]})"
        if name in {"argmax_project", "argmin_project"}:
            scores, projects = args[0::2], args[1::2]
            chooser, score_list = ("max" if name == "argmax_project" else "min"), f"[{', '.join(scores)}]"
            return f"[{', '.join(projects)}][{score_list}.index({chooser}({score_list}))]"
    _fail(f"cannot compile {type(node).__name__}")


def build_tune_gold_records(specs: list[dict], questions: list[dict], templates: list[dict],
                            *, table_loader: StoreTableLoader) -> list[dict]:
    qmap, tmap = {int(x["id"]): x for x in questions}, {int(x["id"]): x for x in templates}
    smap = {int(x["id"]): x for x in specs}
    if len(smap) != len(specs) or set(smap) != set(qmap) or set(tmap) != set(qmap):
        _fail(f"spec ids must exactly equal tune ids; missing={sorted(set(qmap)-set(smap))}")
    out = []
    for question in questions:
        qid, spec, template = int(question["id"]), smap[int(question["id"])], tmap[int(question["id"])]
        if spec.get("schema_version") != AUTHORING_SCHEMA: _fail(f"id {qid}: schema mismatch")
        cells = spec.get("cells")
        if not isinstance(cells, list) or not cells: _fail(f"id {qid}: cells required")
        parsed = _parse(str(spec.get("expression", "")))
        gold_ast = expression_to_gold_ast(parsed)
        if gold_ast.get("kind") != "op": gold_ast = {"kind": "op", "op": "lookup", "args": [gold_ast]}
        bindings, evidence, values, leaves = {}, [], {}, {}
        for index, ref in enumerate(cells, 1):
            if not isinstance(ref, dict) or set(ref) != {"report_id", "table_pos", "row", "col"}:
                _fail(f"id {qid} E{index}: invalid cell ref")
            report, pos, row, col = str(ref["report_id"]), int(ref["table_pos"]), int(ref["row"]), int(ref["col"])
            df = table_loader(report, pos); hit = df[(df.row == row) & (df.col == col)]
            if len(hit) != 1: _fail(f"id {qid} E{index}: cell found {len(hit)} times")
            actual, key, eid = hit.iloc[0], (report, pos), f"E{index}"
            if key not in bindings: bindings[key] = f"df{len(bindings)+1}"
            var, raw, scale = bindings[key], float(actual.value), float(actual.unit_scale)
            values[eid], leaves[eid] = raw * scale, _leaf(var, row, col)
            evidence.append({"evidence_id": eid, "variable": var, "report_id": report,
                "table_pos": pos, "row": row, "col": col,
                "label": "" if actual.label != actual.label else str(actual.label),
                "code": "" if actual.code != actual.code else str(actual.code),
                "col_name": "" if actual.col_name != actual.col_name else str(actual.col_name),
                "value": raw, "unit_scale": scale})
        decimals, answer = int(template["output"]["round_decimals"]), _eval(parsed, values)
        if not math.isfinite(answer): _fail(f"id {qid}: non-finite answer")
        answer = round(float(answer), decimals)
        if template["output"]["type"] in {"count", "year"}:
            if not math.isclose(answer, round(answer), rel_tol=0, abs_tol=1e-9): _fail(f"id {qid}: non-integral output")
            answer = float(int(round(answer)))
        query = f"round(float({_compile(parsed, leaves)}), {decimals})"
        dfs = {var: table_loader(report, pos) for (report, pos), var in bindings.items()}
        replay = run_code(query, dfs, timeout=10)
        if replay.get("status") != "ok" or not math.isclose(float(replay["value"]), answer, rel_tol=0, abs_tol=10**(-decimals)):
            _fail(f"id {qid}: replay mismatch {replay}")
        used_vars = [{"var": var, "report_id": report, "table_pos": pos}
                     for (report, pos), var in bindings.items()]
        out.append({"schema_version": template["schema_version"], "split": "tune",
            "id": qid, "question": question["question"], "question_sha256": question["question_sha256"],
            "stratum": question["stratum"], "label_status": "verified", "evidence": evidence,
            "output": {**template["output"], "value": answer}, "ast": gold_ast,
            "replay": {"pandas_query": query, "used_vars": used_vars, "expected_answer": answer,
                "tolerance": max(1e-8, 10**(-decimals)), "status": "verified",
                "evidence_sha256": canonical_sha256(evidence), "ast_sha256": canonical_sha256(gold_ast)},
            "annotator_notes": str(spec.get("notes", ""))})
    try:
        validate_gold_records(out, questions, "tune", table_loader=table_loader, require_complete=True)
    except P24ValidationError as exc:
        raise P24AuthoringError(f"strict validation failed: {exc}") from exc
    return out


def build_tune_gold_file(specs_path: Path | str, bundle_dir: Path | str,
                         output_path: Path | str, store_dir: Path | str) -> dict[str, Any]:
    bundle, output = Path(bundle_dir), Path(output_path)
    if output.exists(): _fail(f"refusing to overwrite {output}")
    records = build_tune_gold_records(read_jsonl(specs_path),
        read_jsonl(bundle / "p24_tune_questions.jsonl"),
        read_jsonl(bundle / "p24_tune_gold.template.jsonl"),
        table_loader=StoreTableLoader(store_dir))
    write_jsonl(output, records)
    return {"count": len(records), "output": str(output), "records_sha256": canonical_sha256(records)}
