"""Structured Selection v2: named evidence bindings + a typed nested IR.

The model is deliberately kept away from pandas and arithmetic values.  It
binds semantic fact names to numbered shortlist cells, then composes those
facts with a small expression language.  This module validates the complete
program, infers scalar types, compiles a single replayable pandas expression,
and records an auditable rejection reason when any invariant fails.

The v1 selector remains in :mod:`vifinqa.codegen.selection`; this module is an
opt-in experiment selected with ``--llm-mode select_v2``.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable

from ..extraction.unit_policy import (
    RESOLUTION_OVERRIDE,
    RESOLUTION_STORED,
)
from ..utils.viet_text import norm, strip_diacritics
from .selection import _candidate_year, _stable_cell_key, cell_value_expr
from .units import cell_is_already_percent


SCHEMA_VERSION = 2
POLICY_VERSION = "typed_nested_ir_v2_semantic_grounded_v5_1_unit"

MAX_FACTS = 36
MAX_BINDINGS = 48
MAX_NODES = 128
MAX_DEPTH = 14
MAX_QUERY_CHARS = 60_000
MAX_RAW_TRACE_CHARS = 12_000
MAX_REASON_CHARS = 700

SCALAR_TYPES = {
    "money", "number", "ratio", "percent", "percentage_point",
    "count", "year", "bool",
}
REF_TYPES = {"auto", "money", "number", "ratio", "percent"}
OUTPUT_TYPES = {"number", "ratio", "percent", "percentage_point", "count", "year"}
FACT_ROLES = {
    "value", "numerator", "denominator", "base", "end", "filter",
    "rank", "project", "threshold", "weight",
}
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,39}\Z")
_PERCENT_CUE = re.compile(
    r"(?:%|phan tram|ty le|ty suat|lai suat|bien loi nhuan|thue suat)"
)
_RATIO_CUE = re.compile(r"(?:he so|so vong|vong quay|lan)\b")
_MONEY_CUE = re.compile(r"(?:dong|vnd|usd)\b")
_HARD_ABS_LIMIT = {
    "percent": 1_000_000.0,
    "percentage_point": 10_000.0,
    "ratio": 1_000_000.0,
}
_COMPARATIVE_GAP_CUE = re.compile(
    r"(?:be|nho|it|thap|kem|lon|cao|nhieu) hon"
)
_SINGLE_METRIC_PLAN_OPS = {
    "lookup", "difference", "average", "growth", "growth_pct", "cagr",
    "ranking",
}
# These anchors are deliberately small and high precision.  They are only
# enforced for plans whose atomic slots all ask for the same metric, so a
# composite formula such as ROA is not forced to label every operand as total
# assets.
_LABEL_ANCHORS = (
    (
        "total_assets",
        re.compile(r"\btong(?: cong)? tai san\b"),
        re.compile(r"\btong(?: cong)? tai san\b"),
    ),
    (
        "income_tax_payable",
        re.compile(r"\bthue thu nhap(?: doanh nghiep)? phai nop\b"),
        re.compile(r"\bthue thu nhap(?: doanh nghiep)? phai nop\b"),
    ),
)


class IRValidationError(ValueError):
    """A fail-closed validation error with a stable trace taxonomy."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class _Node:
    expr: str
    scalar_type: str
    refs: frozenset[int]
    constant: float | None = None

@dataclass(frozen=True)
class UnitUse:
    candidate_index: int
    stored_scale: float
    effective_scale: float
    stored_source: str
    effective_source: str
    resolution: str
    terminal_bare_vnd: bool
    context_sha256: str



@dataclass(frozen=True)
class CompiledProgram:
    query: str
    output_type: str
    inferred_type: str
    referenced_indices: tuple[int, ...]
    used_fact_names: tuple[str, ...]
    used_binding_names: tuple[str, ...]
    node_count: int
    max_depth: int
    root_op: str
    unit_provenance: tuple[UnitUse, ...]


@dataclass(frozen=True)
class V2Decision:
    answer: float
    query: str
    confidence: float
    compiled: CompiledProgram


def _bounded(value, limit: int) -> tuple[str, int, bool, str]:
    raw = "" if value is None else str(value)
    encoded = raw.encode("utf-8", errors="replace")
    clean = encoded.decode("utf-8", errors="replace")
    clean = "".join(
        ch if ch in "\n\r\t" or ord(ch) >= 32 else "\ufffd" for ch in clean
    )
    return clean[:limit], len(clean), len(clean) > limit, hashlib.sha256(encoded).hexdigest()


def _json_objects(text: str) -> Iterable[dict]:
    """Yield nested JSON objects from prose/fences without a flat-brace regex."""
    if not text:
        return
    blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, re.S | re.I) or [text]
    decoder = json.JSONDecoder()
    for block in blocks:
        starts = [i for i, ch in enumerate(block) if ch == "{"]
        if block.lstrip().startswith("{"):
            starts = [len(block) - len(block.lstrip()), *starts]
        seen = set()
        for start in starts:
            if start in seen:
                continue
            seen.add(start)
            try:
                obj, _end = decoder.raw_decode(block[start:])
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(obj, dict):
                yield obj


def parse_program(text: str) -> dict | None:
    """Return the first object that advertises the v2 contract."""
    fallback = None
    for obj in _json_objects(text):
        version = obj.get("schema_version", obj.get("version", obj.get("v")))
        if version in {SCHEMA_VERSION, str(SCHEMA_VERSION), "selection_v2"}:
            return obj
        if fallback is None and any(key in obj for key in ("root", "expr", "facts")):
            fallback = obj
    return fallback


def is_model_none(program: dict | None) -> bool:
    if not isinstance(program, dict):
        return False
    root = program.get("root", program.get("expr"))
    return isinstance(root, dict) and str(root.get("op", "")).lower() == "none"


class _Compiler:
    def __init__(self, program: dict, candidates, route: dict, question: str,
                 atomic_facts: list[dict] | None = None):
        self.program = program
        self.candidates = list(candidates)
        self.route = route or {}
        self.question = question or ""
        self.atomic_facts = [dict(fact) for fact in (atomic_facts or [])
                             if isinstance(fact, dict)]
        self.facts: dict[str, dict] = {}
        self.bindings: dict[str, dict] = {}
        self.definitions: dict[str, dict] = {}
        self.cache: dict[str, _Node] = {}
        self.stack: list[str] = []
        self.py_names: dict[str, str] = {}
        self.topo: list[str] = []
        self.node_count = 0
        self.max_depth = 0
        self.used_names: set[str] = set()
        self.required_slots: tuple[str, ...] = ()
        self.selected_slots: tuple[str, ...] = ()
        self.unit_uses: dict[int, UnitUse] = {}

    def compile(self) -> CompiledProgram:
        self._validate_program_header()
        root = self.program.get("root", self.program.get("expr"))
        if not isinstance(root, dict):
            raise IRValidationError("schema_error", "root must be an IR object")
        if str(root.get("op", "")).lower() == "none":
            raise IRValidationError("model_none", "model explicitly returned op=none")

        root_node = self._compile_node(root, depth=1, allow_direct_ref=False)
        self._validate_usage(root_node)
        self._validate_unique_fact_refs()
        self._validate_atomic_grounding(root_node.refs)
        self._validate_routed_operation(root)
        self._validate_stable_cells(root_node.refs)
        self._validate_metric_anchor(root_node.refs)
        final_expr, expected = self._normalise_root(root_node)

        # Bind each named expression once with nested lambdas.  This keeps large
        # filter/rank programs compact while remaining a single eval expression.
        wrapped = final_expr
        for name in reversed(self.topo):
            if name not in self.used_names:
                continue
            node = self.cache[name]
            wrapped = f"(lambda {self.py_names[name]}: {wrapped})({node.expr})"
        query = f"round(float({wrapped}), 2)"
        if len(query) > MAX_QUERY_CHARS:
            raise IRValidationError(
                "compile_error", f"compiled query exceeds {MAX_QUERY_CHARS} characters",
            )
        try:
            compile(query, "<selection_v2>", "eval")
        except SyntaxError as exc:
            raise IRValidationError("compile_error", f"invalid compiled syntax: {exc.msg}") from exc

        used_facts = tuple(name for name in self.facts if name in self.used_names)
        used_bindings = tuple(name for name in self.bindings if name in self.used_names)
        return CompiledProgram(
            query=query,
            output_type=expected,
            inferred_type=root_node.scalar_type,
            referenced_indices=tuple(sorted(root_node.refs)),
            used_fact_names=used_facts,
            used_binding_names=used_bindings,
            node_count=self.node_count,
            max_depth=self.max_depth,
            root_op=_root_op(root),
            unit_provenance=tuple(
                self.unit_uses[index] for index in sorted(root_node.refs)
            ),
        )

    def _validate_program_header(self) -> None:
        version = self.program.get("schema_version", self.program.get("version", self.program.get("v")))
        if version not in {SCHEMA_VERSION, str(SCHEMA_VERSION), "selection_v2"}:
            raise IRValidationError(
                "schema_error", f"schema_version must be {SCHEMA_VERSION}, got {version!r}",
            )
        expected = str(self.route.get("output_type") or "number")
        declared = str(self.program.get("output_type", self.program.get("output", "")))
        if expected not in OUTPUT_TYPES:
            raise IRValidationError("schema_error", f"unsupported routed output_type {expected!r}")
        if declared != expected:
            raise IRValidationError(
                "output_type_error",
                f"declared output_type {declared!r} does not match route {expected!r}",
            )

        facts = self.program.get("facts")
        bindings = self.program.get("bindings", {})
        if not isinstance(facts, dict) or not facts:
            raise IRValidationError("schema_error", "facts must be a non-empty object")
        if not isinstance(bindings, dict):
            raise IRValidationError("schema_error", "bindings must be an object")
        if len(facts) > MAX_FACTS:
            raise IRValidationError("limit_error", f"too many facts: {len(facts)} > {MAX_FACTS}")
        if len(bindings) > MAX_BINDINGS:
            raise IRValidationError(
                "limit_error", f"too many bindings: {len(bindings)} > {MAX_BINDINGS}",
            )
        overlap = set(facts) & set(bindings)
        if overlap:
            raise IRValidationError("binding_error", f"duplicate fact/binding names: {sorted(overlap)}")
        for name in [*facts, *bindings]:
            if not isinstance(name, str) or not _NAME.fullmatch(name):
                raise IRValidationError("binding_error", f"invalid binding name {name!r}")
        for name, leaf in facts.items():
            if not isinstance(leaf, dict) or set(leaf) - {"ref", "as", "role", "slot"}:
                raise IRValidationError(
                    "schema_error", f"fact {name!r} must be one atomic ref object",
                )
            if "ref" not in leaf:
                raise IRValidationError("schema_error", f"fact {name!r} has no ref")
            role = str(leaf.get("role", "value"))
            if role not in FACT_ROLES:
                raise IRValidationError("schema_error", f"fact {name!r} has invalid role {role!r}")
        for name, node in bindings.items():
            if not isinstance(node, dict):
                raise IRValidationError("schema_error", f"binding {name!r} must be an IR object")

        self.facts = dict(facts)
        self.bindings = dict(bindings)
        self.definitions = {**self.facts, **self.bindings}
        self.py_names = {name: f"_v2_{i}" for i, name in enumerate(self.definitions)}

    def _compile_name(self, name: str, depth: int) -> _Node:
        if name not in self.definitions:
            raise IRValidationError("binding_error", f"unknown binding {name!r}")
        self.used_names.add(name)
        if name in self.cache:
            return _Node(self.py_names[name], self.cache[name].scalar_type,
                         self.cache[name].refs, self.cache[name].constant)
        if name in self.stack:
            cycle = " -> ".join([*self.stack, name])
            raise IRValidationError("binding_error", f"binding cycle: {cycle}")
        self.stack.append(name)
        is_fact = name in self.facts
        node = self._compile_node(
            self.definitions[name], depth=depth, allow_direct_ref=is_fact,
        )
        self.stack.pop()
        self.cache[name] = node
        self.topo.append(name)
        return _Node(self.py_names[name], node.scalar_type, node.refs, node.constant)

    def _compile_node(self, obj: dict, depth: int, allow_direct_ref: bool) -> _Node:
        self.node_count += 1
        self.max_depth = max(self.max_depth, depth)
        if self.node_count > MAX_NODES:
            raise IRValidationError("limit_error", f"IR exceeds {MAX_NODES} nodes")
        if depth > MAX_DEPTH:
            raise IRValidationError("limit_error", f"IR exceeds depth {MAX_DEPTH}")
        if not isinstance(obj, dict):
            raise IRValidationError("schema_error", "every IR node must be an object")

        if "var" in obj:
            if set(obj) != {"var"}:
                raise IRValidationError("schema_error", "var nodes may only contain 'var'")
            return self._compile_name(str(obj["var"]), depth + 1)
        if "ref" in obj:
            if not allow_direct_ref:
                raise IRValidationError(
                    "grounding_error", "candidate refs are allowed only inside top-level facts",
                )
            return self._compile_ref(obj)
        if "year" in obj:
            if set(obj) != {"year"}:
                raise IRValidationError("schema_error", "year nodes may only contain 'year'")
            return self._compile_literal(float(obj["year"]), "year")
        if "literal" in obj:
            allowed = {"literal", "type"}
            if set(obj) - allowed:
                raise IRValidationError("schema_error", "literal node has unsupported fields")
            return self._compile_literal(obj["literal"], str(obj.get("type", "number")))

        op = str(obj.get("op", "")).strip().lower()
        if not op:
            raise IRValidationError("schema_error", "IR node has no op/ref/var/literal")
        if op in {"argmax_project", "argmin_project"}:
            return self._compile_projection(op, obj, depth)
        args = obj.get("args")
        if not isinstance(args, list):
            raise IRValidationError("schema_error", f"op {op} requires args list")
        children = [self._compile_node(x, depth + 1, False) for x in args]
        return self._compile_op(op, children, obj)

    def _compile_ref(self, obj: dict) -> _Node:
        try:
            index = int(obj["ref"])
        except (TypeError, ValueError) as exc:
            raise IRValidationError("grounding_error", f"invalid candidate ref {obj.get('ref')!r}") from exc
        if not 1 <= index <= len(self.candidates):
            raise IRValidationError(
                "grounding_error", f"candidate ref {index} outside 1..{len(self.candidates)}",
            )
        declared = str(obj.get("as", "auto")).lower()
        if declared not in REF_TYPES:
            raise IRValidationError("type_error", f"invalid ref type {declared!r}")
        candidate = self.candidates[index - 1]
        scalar_type = _auto_ref_type(candidate, self.route) if declared == "auto" else declared
        if scalar_type == "auto":
            scalar_type = "money"
        raw = cell_value_expr(candidate)
        unit_use = self._unit_use(index, candidate)
        self.unit_uses[index] = unit_use
        scale = unit_use.effective_scale
        if scalar_type in {"money", "number"}:
            expr = f"({raw} * {scale:g})"
        elif scalar_type == "percent":
            already = cell_is_already_percent(
                str(getattr(candidate, "label", "")),
                str(getattr(candidate, "col_name", "")),
                float(getattr(candidate, "value", 0.0)),
            )
            expr = raw if already else f"({raw} * 100)"
        elif scalar_type == "ratio":
            expr = raw
        else:  # pragma: no cover - REF_TYPES keeps this unreachable
            raise IRValidationError("type_error", f"unsupported ref type {scalar_type!r}")

        slot = str(obj.get("slot", "")).strip()
        actual_slot = str(getattr(candidate, "fact_slot", "") or "")
        if slot and slot != actual_slot:
            raise IRValidationError(
                "grounding_error", f"candidate {index} is {actual_slot or '-'}, not declared slot {slot}",
            )
        return _Node(expr, scalar_type, frozenset({index}))
    def _unit_use(self, index: int, candidate) -> UnitUse:
        try:
            effective = float(getattr(candidate, "unit_scale", 1.0))
        except (TypeError, ValueError) as exc:
            raise IRValidationError(
                "unit_provenance_error",
                f"candidate {index} has invalid effective unit_scale",
            ) from exc
        stored_raw = getattr(candidate, "unit_original_scale", None)
        try:
            stored = effective if stored_raw is None else float(stored_raw)
        except (TypeError, ValueError) as exc:
            raise IRValidationError(
                "unit_provenance_error",
                f"candidate {index} has invalid stored unit_scale",
            ) from exc
        if (not math.isfinite(effective) or effective <= 0
                or not math.isfinite(stored) or stored <= 0):
            raise IRValidationError(
                "unit_provenance_error",
                f"candidate {index} has non-positive/non-finite unit scale",
            )

        source = str(getattr(candidate, "unit_source", "unknown") or "unknown")
        effective_source = str(
            getattr(candidate, "unit_effective_source", source) or source
        )
        resolution = str(
            getattr(candidate, "unit_resolution", RESOLUTION_STORED)
            or RESOLUTION_STORED
        )
        terminal = bool(
            getattr(candidate, "unit_context_terminal_vnd", False)
        )
        context_sha = str(
            getattr(candidate, "unit_context_sha256", "") or ""
        )
        if context_sha and not re.fullmatch(r"[0-9a-f]{64}", context_sha):
            raise IRValidationError(
                "unit_provenance_error",
                f"candidate {index} has invalid unit context digest",
            )

        changed = not math.isclose(
            stored, effective, rel_tol=0.0, abs_tol=0.0,
        )
        if resolution == RESOLUTION_OVERRIDE:
            valid_override = (
                source == "sticky"
                and effective_source == "terminal_vnd"
                and terminal
                and changed
                and math.isclose(effective, 1.0, rel_tol=0.0, abs_tol=0.0)
            )
            if not valid_override:
                raise IRValidationError(
                    "unit_provenance_error",
                    f"candidate {index} has an invalid terminal-VND override",
                )
        elif changed:
            raise IRValidationError(
                "unit_provenance_error",
                f"candidate {index} has an unexplained unit-scale change",
            )
        elif (source == "sticky" and terminal
              and not math.isclose(stored, 1.0, rel_tol=0.0, abs_tol=0.0)):
            raise IRValidationError(
                "unit_provenance_error",
                f"candidate {index} keeps sticky x{stored:g} despite terminal VND",
            )

        return UnitUse(
            candidate_index=index,
            stored_scale=stored,
            effective_scale=effective,
            stored_source=source,
            effective_source=effective_source,
            resolution=resolution,
            terminal_bare_vnd=terminal,
            context_sha256=context_sha,
        )

    def _compile_literal(self, raw, scalar_type: str) -> _Node:
        if scalar_type not in {"money", "number", "ratio", "percent", "year"}:
            raise IRValidationError("type_error", f"literal type {scalar_type!r} is not allowed")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise IRValidationError("schema_error", f"literal {raw!r} is not numeric") from exc
        if not math.isfinite(value):
            raise IRValidationError("schema_error", "literal must be finite")
        if scalar_type == "year":
            if value != round(value) or not 1900 <= value <= 2100:
                raise IRValidationError("type_error", f"invalid year literal {value:g}")
            allowed = set()
            for y in self.route.get("years") or []:
                try:
                    y = int(y)
                except (TypeError, ValueError):
                    continue
                allowed.update({y - 1, y, y + 1})
            if allowed and int(value) not in allowed:
                raise IRValidationError("grounding_error", f"year {int(value)} is outside routed periods")
        elif not _literal_is_grounded(value, self.question, scalar_type):
            raise IRValidationError(
                "grounding_error",
                f"literal {value:g} is neither a safe identity nor present in the question",
            )
        expr = str(int(value)) if value == int(value) else repr(value)
        return _Node(expr, scalar_type, frozenset(), value)

    def _compile_op(self, op: str, args: list[_Node], obj: dict) -> _Node:
        refs = frozenset().union(*(node.refs for node in args)) if args else frozenset()

        if op in {"lookup", "abs", "negate", "not"}:
            _arity(op, args, exact=1)
            a = args[0]
            if op == "lookup":
                _numeric(a, op)
                return _Node(a.expr, a.scalar_type, refs, a.constant)
            if op == "abs":
                _numeric(a, op)
                return _Node(f"abs({a.expr})", a.scalar_type, refs,
                             abs(a.constant) if a.constant is not None else None)
            if op == "negate":
                _numeric(a, op)
                return _Node(f"(-({a.expr}))", a.scalar_type, refs,
                             -a.constant if a.constant is not None else None)
            _require_type(a, "bool", op)
            return _Node(f"(not ({a.expr}))", "bool", refs)

        if op in {"add", "sum", "average", "min", "max", "median"}:
            minimum = 2 if op in {"add", "sum", "average", "median"} else 2
            _arity(op, args, minimum=minimum)
            scalar_type = _common_numeric_type(args, op)
            exprs = ", ".join(a.expr for a in args)
            if op in {"add", "sum"}:
                expr = f"sum([{exprs}])"
            elif op == "average":
                expr = f"(sum([{exprs}]) / {len(args)})"
            elif op in {"min", "max"}:
                expr = f"{op}([{exprs}])"
            else:
                n = len(args)
                if n % 2:
                    body = f"sorted(_xs)[{n // 2}]"
                else:
                    body = f"((sorted(_xs)[{n // 2 - 1}] + sorted(_xs)[{n // 2}]) / 2)"
                expr = f"(lambda _xs: {body})([{exprs}])"
            constant = None
            if all(a.constant is not None for a in args):
                vals = [float(a.constant) for a in args]
                if op in {"add", "sum"}:
                    constant = sum(vals)
                elif op == "average":
                    constant = sum(vals) / len(vals)
                elif op == "min":
                    constant = min(vals)
                elif op == "max":
                    constant = max(vals)
                else:
                    vals = sorted(vals)
                    constant = vals[len(vals) // 2] if len(vals) % 2 else (
                        vals[len(vals) // 2 - 1] + vals[len(vals) // 2]
                    ) / 2
            return _Node(expr, scalar_type, refs, constant)

        if op == "subtract":
            _arity(op, args, exact=2)
            scalar_type = _common_numeric_type(args, op)
            constant = (args[0].constant - args[1].constant
                        if all(a.constant is not None for a in args) else None)
            return _Node(f"({args[0].expr} - {args[1].expr})", scalar_type, refs, constant)

        if op == "multiply":
            _arity(op, args, exact=2)
            scalar_type = _multiply_type(args[0].scalar_type, args[1].scalar_type)
            constant = (args[0].constant * args[1].constant
                        if all(a.constant is not None for a in args) else None)
            return _Node(f"({args[0].expr} * {args[1].expr})", scalar_type, refs, constant)

        if op in {"divide", "ratio"}:
            _arity(op, args, exact=2)
            scalar_type = _divide_type(args[0].scalar_type, args[1].scalar_type)
            if args[1].constant == 0:
                raise IRValidationError("type_error", "division by literal zero")
            constant = None
            if args[0].constant is not None and args[1].constant not in {None, 0}:
                constant = args[0].constant / args[1].constant
            return _Node(f"({args[0].expr} / {args[1].expr})", scalar_type, refs, constant)

        if op in {"growth_percent", "growth_pct"}:
            _arity(op, args, exact=2)
            _same_quantity(args, op)
            if args[1].constant == 0:
                raise IRValidationError("type_error", "growth base is literal zero")
            return _Node(
                f"(({args[0].expr} - {args[1].expr}) / abs({args[1].expr}) * 100)",
                "percent", refs,
            )

        if op == "cagr_percent":
            _arity(op, args, exact=2)
            _same_quantity(args, op)
            try:
                periods = int(obj.get("periods"))
            except (TypeError, ValueError) as exc:
                raise IRValidationError("schema_error", "cagr_percent requires integer periods") from exc
            if not 1 <= periods <= 30 or not _period_is_grounded(periods, self.route, self.question):
                raise IRValidationError("grounding_error", f"CAGR periods={periods} is not grounded")
            return _Node(
                f"((({args[0].expr} / {args[1].expr}) ** (1 / {periods})) - 1) * 100",
                "percent", refs,
            )

        if op in {"percentage_point", "percentage_point_change"}:
            _arity(op, args, exact=2)
            types = {a.scalar_type for a in args}
            if types == {"percent"}:
                expr = f"({args[0].expr} - {args[1].expr})"
            elif types == {"ratio"}:
                expr = f"(({args[0].expr} - {args[1].expr}) * 100)"
            else:
                raise IRValidationError(
                    "type_error", f"op {op} needs two percent values or two ratios",
                )
            return _Node(expr, "percentage_point", refs)

        if op in {"apply_percent_change", "increase_percent", "decrease_percent"}:
            _arity(op, args, exact=2)
            base, change = args
            _numeric(base, op)
            if base.scalar_type in {"percentage_point", "count"}:
                raise IRValidationError("type_error", f"op {op} cannot change {base.scalar_type}")
            if change.scalar_type != "percent":
                raise IRValidationError("type_error", f"op {op} needs a percent second arg")
            sign = "-" if op == "decrease_percent" else "+"
            if op == "apply_percent_change":
                expr = f"({base.expr} * (1 + {change.expr} / 100))"
            else:
                expr = f"({base.expr} * (1 {sign} {change.expr} / 100))"
            return _Node(expr, base.scalar_type, refs)

        if op == "power":
            _arity(op, args, exact=2)
            base, exponent = args
            if base.scalar_type not in {"number", "ratio"}:
                raise IRValidationError("type_error", f"power base cannot be {base.scalar_type}")
            if exponent.scalar_type not in {"number", "ratio"} or exponent.constant is None:
                raise IRValidationError("type_error", "power exponent must be a grounded literal")
            if abs(exponent.constant) > 8:
                raise IRValidationError("type_error", "power exponent magnitude exceeds 8")
            return _Node(f"({base.expr} ** {exponent.expr})", base.scalar_type, refs)

        if op in {"gt", "ge", "lt", "le", "eq", "ne"}:
            _arity(op, args, exact=2)
            _comparable(args[0], args[1], op)
            symbol = {"gt": ">", "ge": ">=", "lt": "<", "le": "<=", "eq": "==", "ne": "!="}[op]
            return _Node(f"({args[0].expr} {symbol} {args[1].expr})", "bool", refs)

        if op in {"and", "or"}:
            _arity(op, args, minimum=2)
            for a in args:
                _require_type(a, "bool", op)
            glue = " and " if op == "and" else " or "
            return _Node("(" + glue.join(f"({a.expr})" for a in args) + ")", "bool", refs)

        if op == "count_true":
            _arity(op, args, minimum=1)
            for a in args:
                _require_type(a, "bool", op)
            return _Node(
                "sum([" + ", ".join(f"int({a.expr})" for a in args) + "])",
                "count", refs,
            )

        if op == "if_else":
            _arity(op, args, exact=3)
            cond, yes, no = args
            _require_type(cond, "bool", op)
            scalar_type = _branch_type(yes, no, op)
            return _Node(f"({yes.expr} if {cond.expr} else {no.expr})", scalar_type, refs)

        raise IRValidationError("schema_error", f"unknown v2 op {op!r}")

    def _compile_projection(self, op: str, obj: dict, depth: int) -> _Node:
        if set(obj) - {"op", "items"}:
            raise IRValidationError("schema_error", f"{op} only supports op/items")
        items = obj.get("items")
        if not isinstance(items, list) or len(items) < 2:
            raise IRValidationError("schema_error", f"{op} needs at least two items")
        scores, results, conditions = [], [], []
        for pos, item in enumerate(items, 1):
            if (not isinstance(item, dict)
                    or set(item) not in ({"score", "result"},
                                         {"score", "result", "when"})):
                raise IRValidationError(
                    "schema_error", f"{op} item {pos} must contain score/result and optional when",
                )
            score = self._compile_node(item["score"], depth + 1, False)
            result = self._compile_node(item["result"], depth + 1, False)
            condition = (self._compile_node(item["when"], depth + 1, False)
                         if "when" in item else None)
            if condition is not None:
                _require_type(condition, "bool", op)
            _numeric(score, op)
            if score.scalar_type in {"bool", "year"}:
                raise IRValidationError("type_error", f"{op} score cannot be {score.scalar_type}")
            if result.scalar_type == "year" and result.constant is not None:
                grounded_years = {
                    _candidate_year(self.candidates[i - 1], self.route) for i in score.refs
                }
                if int(result.constant) not in grounded_years:
                    raise IRValidationError(
                        "grounding_error",
                        f"{op} projected year {int(result.constant)} is not grounded by its score refs",
                    )
            scores.append(score)
            results.append(result)
            conditions.append(condition)
        score_type = _common_numeric_type(scores, op)
        del score_type  # only the compatibility check is needed
        result_type = _common_projection_type(results, op)
        condition_nodes = [x for x in conditions if x is not None]
        refs = frozenset().union(*(x.refs for x in [*scores, *results, *condition_nodes]))
        score_exprs = ", ".join(x.expr for x in scores)
        result_exprs = ", ".join(x.expr for x in results)
        fn = "max" if op == "argmax_project" else "min"
        if condition_nodes:
            item_exprs = ", ".join(
                f"({condition.expr if condition is not None else 'True'}, {score.expr}, {result.expr})"
                for condition, score, result in zip(conditions, scores, results)
            )
            expr = (
                "(lambda _items: (lambda _valid: "
                f"_valid[[x[1] for x in _valid].index({fn}([x[1] for x in _valid]))][2])"
                "([x for x in _items if x[0]]))"
                f"([{item_exprs}])"
            )
        else:
            expr = (
                f"(lambda _scores, _results: _results[_scores.index({fn}(_scores))])"
                f"([{score_exprs}], [{result_exprs}])"
            )
        return _Node(expr, result_type, refs)

    def _validate_usage(self, root: _Node) -> None:
        if not root.refs:
            raise IRValidationError("grounding_error", "root is not derived from any candidate")
        unused_facts = sorted(set(self.facts) - self.used_names)
        unused_bindings = sorted(set(self.bindings) - self.used_names)
        if unused_facts or unused_bindings:
            raise IRValidationError(
                "binding_error",
                f"unused definitions: facts={unused_facts}, bindings={unused_bindings}",
            )

    def _validate_unique_fact_refs(self) -> None:
        """One physical candidate may have one semantic fact name only.

        Reusing the same fact through ``var`` is valid; inventing several fact
        names for one ref silently fabricates independent observations and was
        the main failure mode in the interrupted Stage-B checkpoint.
        """
        by_ref: dict[int, list[str]] = {}
        for name, leaf in self.facts.items():
            try:
                index = int(leaf.get("ref"))
            except (TypeError, ValueError):
                continue  # _compile_ref emits the canonical validation error.
            by_ref.setdefault(index, []).append(name)
        duplicates = {index: names for index, names in by_ref.items()
                      if len(names) > 1}
        if duplicates:
            raise IRValidationError(
                "grounding_error",
                f"one candidate ref is bound to multiple facts: {duplicates}",
            )

    def _validate_atomic_grounding(self, refs: frozenset[int]) -> None:
        """Require a complete, one-to-one binding of routed atomic fact slots."""
        if not self.atomic_facts:
            return
        ungrounded_routes = [
            f"F{index}" for index, fact in enumerate(self.atomic_facts, 1)
            if not bool(fact.get("route_grounded", True))
        ]
        if ungrounded_routes:
            raise IRValidationError(
                "route_grounding_error",
                f"atomic routes are not entity-grounded: {ungrounded_routes}",
            )
        required = tuple(f"F{i}" for i in range(1, len(self.atomic_facts) + 1))
        required_set = set(required)
        available = {
            str(getattr(candidate, "fact_slot", "") or "").strip()
            for candidate in self.candidates
        }
        missing_from_shortlist = sorted(required_set - available)
        if missing_from_shortlist:
            raise IRValidationError(
                "grounding_error",
                "shortlist lacks required atomic slots: "
                f"{missing_from_shortlist}",
            )

        selected: dict[str, list[int]] = {}
        for index in sorted(refs):
            candidate = self.candidates[index - 1]
            slot = str(getattr(candidate, "fact_slot", "") or "").strip()
            if slot not in required_set:
                raise IRValidationError(
                    "grounding_error",
                    f"candidate {index} has no required atomic slot ({slot or '-'})",
                )
            selected.setdefault(slot, []).append(index)
            fact = self.atomic_facts[int(slot[1:]) - 1]
            self._validate_candidate_provenance(index, candidate, slot, fact)

        missing = sorted(required_set - set(selected))
        repeated = {slot: indices for slot, indices in selected.items()
                    if len(indices) != 1}
        if missing or repeated:
            raise IRValidationError(
                "grounding_error",
                "atomic slot binding must be complete and one-to-one: "
                f"missing={missing}, repeated={repeated}",
            )
        self.required_slots = required
        self.selected_slots = tuple(sorted(selected, key=lambda x: int(x[1:])))

    @staticmethod
    def _validate_candidate_provenance(index: int, candidate, slot: str,
                                       fact: dict) -> None:
        if not bool(getattr(candidate, "metric_grounded", True)):
            reason = str(getattr(candidate, "metric_grounding_reason", "") or "")
            raise IRValidationError(
                "metric_grounding_error",
                f"candidate {index}/{slot} is not semantically grounded"
                + (f": {reason}" if reason else ""),
            )
        expected_ticker = str(fact.get("ticker") or "").strip().upper()
        actual_ticker = str(getattr(candidate, "ticker", "") or "").strip().upper()
        if expected_ticker and actual_ticker != expected_ticker:
            raise IRValidationError(
                "grounding_error",
                f"candidate {index}/{slot} ticker={actual_ticker or '-'}, "
                f"expected {expected_ticker}",
            )
        expected_year = _safe_year(fact.get("year"))
        actual_year = _safe_year(getattr(candidate, "fact_year", None))
        if actual_year is None:
            actual_year = _safe_year(_candidate_year(candidate))
        if expected_year is not None and actual_year != expected_year:
            raise IRValidationError(
                "grounding_error",
                f"candidate {index}/{slot} year={actual_year or '-'}, "
                f"expected {expected_year}",
            )

    def _validate_routed_operation(self, root: dict) -> None:
        routed = str((self.route.get("plan") or {}).get("op") or "lookup")
        root_op = _root_op(root)
        if routed == "ranking" and root_op not in {
            "argmax_project", "argmin_project",
        }:
            raise IRValidationError(
                "operation_error",
                "routed ranking requires argmax_project or argmin_project at root; "
                f"got {root_op}",
            )

    def _validate_metric_anchor(self, refs: frozenset[int]) -> None:
        """Apply exact high-confidence label anchors for single-metric plans."""
        if not self.atomic_facts:
            return
        routed = str((self.route.get("plan") or {}).get("op") or "lookup")
        if routed not in _SINGLE_METRIC_PLAN_OPS:
            return
        metrics = {norm(str(fact.get("metric") or ""))
                   for fact in self.atomic_facts}
        metrics.discard("")
        if len(metrics) != 1:
            return
        question = norm(self.question)
        for name, question_pattern, label_pattern in _LABEL_ANCHORS:
            if not question_pattern.search(question):
                continue
            mismatches = []
            for index in sorted(refs):
                candidate = self.candidates[index - 1]
                label = norm(" ".join([
                    str(getattr(candidate, "label", "") or ""),
                    str(getattr(candidate, "code", "") or ""),
                ]))
                if not label_pattern.search(label):
                    mismatches.append(index)
            if mismatches:
                raise IRValidationError(
                    "metric_anchor_error",
                    f"question anchor {name} is absent from candidate labels "
                    f"at refs {mismatches}",
                )

    def _validate_stable_cells(self, refs: frozenset[int]) -> None:
        by_cell: dict[tuple[str, int, int], list[int]] = {}
        for index in refs:
            try:
                key = _stable_cell_key(self.candidates[index - 1])
            except ValueError as exc:
                raise IRValidationError("grounding_error", str(exc)) from exc
            by_cell.setdefault(key, []).append(index)
        aliases = {key: ids for key, ids in by_cell.items() if len(ids) > 1}
        if aliases:
            raise IRValidationError(
                "grounding_error", f"distinct refs alias the same stable cell: {aliases}",
            )

    def _normalise_root(self, root: _Node) -> tuple[str, str]:
        expected = str(self.route.get("output_type") or "number")
        q_scale = float(self.route.get("unit_scale", 1.0) or 1.0)
        if not math.isfinite(q_scale) or q_scale <= 0:
            raise IRValidationError("output_type_error", "route unit_scale must be positive")
        if expected == "number":
            if root.scalar_type not in {"money", "number"}:
                raise IRValidationError(
                    "output_type_error", f"number output cannot consume {root.scalar_type}",
                )
            return f"({root.expr} / {q_scale:g})", expected
        if expected == "percent":
            if root.scalar_type == "ratio":
                return f"({root.expr} * 100)", expected
            if root.scalar_type == "percent":
                return root.expr, expected
            raise IRValidationError(
                "output_type_error", f"percent output needs ratio/percent, got {root.scalar_type}",
            )
        wanted = {
            "ratio": "ratio", "percentage_point": "percentage_point",
            "count": "count", "year": "year",
        }[expected]
        if root.scalar_type != wanted:
            raise IRValidationError(
                "output_type_error", f"{expected} output needs {wanted}, got {root.scalar_type}",
            )
        return root.expr, expected


def compile_program(program: dict, candidates, route: dict, question: str = "",
                    *, atomic_facts: list[dict] | None = None) -> CompiledProgram:
    return _Compiler(
        program, candidates, route, question, atomic_facts=atomic_facts,
    ).compile()


def confidence(compiled: CompiledProgram, candidates) -> float:
    picks = [candidates[i - 1] for i in compiled.referenced_indices]
    if not picks:
        return 0.0
    base = min(float(getattr(c, "score", 0.0)) for c in picks)
    if any(bool(getattr(c, "rescue", False)) for c in picks):
        base -= 5.0
    base -= max(0, compiled.max_depth - 5) * 1.5
    return max(0.0, min(95.0, base))


def validate_output_value(value: float, output_type: str,
                          question: str = "") -> str | None:
    if not math.isfinite(float(value)):
        return "answer is not finite"
    if output_type == "year":
        if abs(value - round(value)) > 1e-6 or not 1900 <= round(value) <= 2100:
            return "year answer is not an in-range integer"
    elif output_type == "count":
        if value < 0 or abs(value - round(value)) > 1e-6:
            return "count answer is not a non-negative integer"
    hard = _HARD_ABS_LIMIT.get(output_type)
    if hard is not None and abs(value) > hard:
        return f"{output_type} magnitude {value:g} exceeds hard limit {hard:g}"
    question_norm = norm(question)
    if (value < 0 and "bao nhieu" in question_norm
            and _COMPARATIVE_GAP_CUE.search(question_norm)):
        return "comparative gap phrased with 'bao nhieu' must be non-negative"
    return None


def evaluate_samples(samples, candidates, route: dict, question: str,
                     execute: Callable[[str], dict], *,
                     atomic_facts: list[dict] | None = None,
                     ) -> tuple[V2Decision | None, dict]:
    """Parse/compile/execute the first valid v2 sample and return an audit trace."""
    samples = list(samples or [])
    candidates = list(candidates or [])
    attempts: list[dict] = []
    accepted: V2Decision | None = None
    accepted_index: int | None = None

    if not candidates:
        for index, raw in enumerate(samples, 1):
            attempt = _attempt(index, raw)
            attempt.update(stage="not_evaluated", reason_code="no_candidates",
                           reason="selection_v2 shortlist is empty")
            attempts.append(attempt)
        return None, _trace(0, samples, attempts, "no_candidates")

    for index, raw in enumerate(samples, 1):
        attempt = _attempt(index, raw)
        program = parse_program(str(raw or ""))
        if program is None:
            hit_limit = bool(attempt.get("generation_truncated"))
            attempt.update(
                stage="generation" if hit_limit else "parse",
                reason_code="generation_truncated" if hit_limit else "parse_error",
                reason=("generation ended at max_tokens before valid JSON"
                        if hit_limit else "response has no parseable v2 JSON object"),
            )
            attempts.append(attempt)
            continue
        if is_model_none(program):
            attempt.update(stage="program", reason_code="model_none",
                           reason="model explicitly returned op=none")
            attempts.append(attempt)
            continue
        try:
            compiled = compile_program(
                program, candidates, route, question,
                atomic_facts=atomic_facts,
            )
        except IRValidationError as exc:
            attempt.update(stage="validation", reason_code=exc.code,
                           reason=str(exc)[:MAX_REASON_CHARS])
            attempts.append(attempt)
            continue
        except Exception as exc:  # fail closed on an unexpected compiler defect
            attempt.update(stage="compile", reason_code="compile_error",
                           reason=f"{type(exc).__name__}: {exc}"[:MAX_REASON_CHARS])
            attempts.append(attempt)
            continue

        attempt["program"] = {
            "output_type": compiled.output_type,
            "root_op": compiled.root_op,
            "referenced_indices": list(compiled.referenced_indices),
            "used_fact_names": list(compiled.used_fact_names),
            "used_binding_names": list(compiled.used_binding_names),
            "inferred_type": compiled.inferred_type,
            "node_count": compiled.node_count,
            "max_depth": compiled.max_depth,
            "unit_provenance": [
                {
                    "candidate_index": unit.candidate_index,
                    "stored_scale": unit.stored_scale,
                    "effective_scale": unit.effective_scale,
                    "stored_source": unit.stored_source,
                    "effective_source": unit.effective_source,
                    "resolution": unit.resolution,
                    "terminal_bare_vnd": unit.terminal_bare_vnd,
                    "context_sha256": unit.context_sha256,
                }
                for unit in compiled.unit_provenance
            ],
        }
        ex = execute(compiled.query)
        attempt["execution"] = {
            "status": str(ex.get("status", "")),
            "value": ex.get("value"),
            "error": str(ex.get("error") or "")[:MAX_REASON_CHARS],
        }
        if ex.get("status") != "ok":
            semantic = ex.get("status") == "semantic_error"
            attempt.update(
                stage="semantic" if semantic else "execution",
                reason_code="semantic_validation_failed" if semantic else "execution_failed",
                reason=str(ex.get("error") or ex.get("status") or "execution failed")[:MAX_REASON_CHARS],
            )
            attempts.append(attempt)
            continue
        answer = round(float(ex["value"]), 2)
        guard = validate_output_value(answer, compiled.output_type, question)
        if guard:
            attempt.update(stage="output_guard", reason_code="output_guard_failed",
                           reason=guard[:MAX_REASON_CHARS])
            attempts.append(attempt)
            continue
        accepted = V2Decision(
            answer=answer,
            query=compiled.query,
            confidence=confidence(compiled, candidates),
            compiled=compiled,
        )
        accepted_index = index
        attempt.update(stage="accepted", accepted=True, reason_code="", reason="",
                       answer=answer)
        attempts.append(attempt)
        break

    if accepted_index is not None:
        for index, raw in enumerate(samples[accepted_index:], accepted_index + 1):
            attempt = _attempt(index, raw)
            attempt.update(stage="not_evaluated",
                           reason_code="not_evaluated_after_acceptance",
                           reason=f"attempt {accepted_index} was already accepted")
            attempts.append(attempt)
        outcome = "accepted"
    elif not samples:
        outcome = "no_samples"
    else:
        outcome = "rejected"
    return accepted, _trace(
        len(candidates), samples, attempts, outcome, accepted_index,
    )


def _attempt(index: int, raw) -> dict:
    preview, chars, truncated, digest = _bounded(raw, MAX_RAW_TRACE_CHARS)
    finish_reason = str(getattr(raw, "finish_reason", "") or "")
    generation_tokens = getattr(raw, "token_count", None)
    generation_max_tokens = getattr(raw, "max_tokens", None)
    try:
        generation_tokens = (None if generation_tokens is None
                             else int(generation_tokens))
    except (TypeError, ValueError):
        generation_tokens = None
    try:
        generation_max_tokens = (None if generation_max_tokens is None
                                 else int(generation_max_tokens))
    except (TypeError, ValueError):
        generation_max_tokens = None
    generation_truncated = bool(
        getattr(raw, "hit_max_tokens", False) or finish_reason == "length"
    )
    return {
        "index": index,
        "raw_response": preview,
        "raw_chars": chars,
        "raw_truncated": truncated,
        "raw_sha256": digest,
        "generation_finish_reason": finish_reason or "unknown",
        "generation_tokens": generation_tokens,
        "generation_max_tokens": generation_max_tokens,
        "generation_truncated": generation_truncated,
        "stage": "parse",
        "accepted": False,
        "reason_code": "",
        "reason": "",
    }


def _trace(candidate_count: int, samples: list, attempts: list[dict], outcome: str,
           accepted_attempt: int | None = None) -> dict:
    rejected = Counter(
        a.get("reason_code") for a in attempts
        if a.get("reason_code") and not a.get("accepted")
        and a.get("stage") != "not_evaluated"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "select_v2",
        "policy_version": POLICY_VERSION,
        "outcome": outcome,
        "candidate_count": int(candidate_count),
        "samples_received": len(samples),
        "attempts_evaluated": sum(a.get("stage") != "not_evaluated" for a in attempts),
        "accepted_attempt": accepted_attempt,
        "rejection_counts": dict(rejected),
        "attempts": attempts,
    }


def _root_op(root: dict) -> str:
    if "op" in root:
        return str(root["op"]).lower()
    if "var" in root:
        return "var"
    if "year" in root:
        return "year"
    if "literal" in root:
        return "literal"
    return "unknown"


def _safe_year(value) -> int | None:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if 1900 <= value <= 2100 else None


def _auto_ref_type(candidate, route: dict) -> str:
    text = norm(
        " ".join([
            str(getattr(candidate, "label", "")),
            str(getattr(candidate, "col_name", "")),
            str(getattr(candidate, "fact_metric", "")),
        ])
    )
    if _PERCENT_CUE.search(text):
        return "percent"
    if _RATIO_CUE.search(text):
        return "ratio"
    unit_name = norm(str(route.get("unit_name", "")))
    return "money" if _MONEY_CUE.search(unit_name) else "number"


def _literal_is_grounded(value: float, question: str,
                         scalar_type: str = "number") -> bool:
    if value in {-1.0, 0.0, 1.0, 2.0, 100.0}:
        return True
    variants = {f"{value:g}", f"{value:g}".replace(".", ",")}
    text = str(question)
    # A decimal comma is often followed by normal punctuation ("1,5, doanh").
    # Reject only a following digit or a decimal separator followed by a digit,
    # so that 1 cannot ground 1,5 and 1,5 cannot ground 1,50.
    if any(re.search(rf"(?<![\d.,]){re.escape(v)}(?!\d|[.,]\d)", text)
           for v in variants):
        return True
    return scalar_type == "money" and any(
        math.isclose(value, grounded, rel_tol=0.0,
                     abs_tol=max(1e-9, abs(value) * 1e-12))
        for grounded in _question_money_literals(question)
    )


def _question_money_literals(question: str) -> set[float]:
    multipliers = {
        "nghin ty": 1e12, "tram ty": 1e11, "ty": 1e9,
        "trieu": 1e6, "nghin": 1e3,
    }
    pattern = re.compile(
        r"(?<![\d.,])(?P<value>[-+]?\d+(?:[.,]\d+)?)\s*"
        r"(?P<unit>nghin\s+ty|tram\s+ty|ty|trieu|nghin)"
        r"(?:\s*(?:dong|vnd))?(?![0-9a-z])"
    )
    out = set()
    source = strip_diacritics(str(question or "")).lower()
    for match in pattern.finditer(source):
        amount = float(match.group("value").replace(",", "."))
        unit = " ".join(match.group("unit").split())
        out.add(amount * multipliers[unit])
    return out


def _period_is_grounded(periods: int, route: dict, question: str) -> bool:
    years = []
    for raw in route.get("years") or []:
        try:
            years.append(int(raw))
        except (TypeError, ValueError):
            pass
    if len(years) >= 2 and max(years) - min(years) == periods:
        return True
    return _literal_is_grounded(float(periods), question)


def _arity(op: str, args: list[_Node], exact: int | None = None,
           minimum: int | None = None) -> None:
    if exact is not None and len(args) != exact:
        raise IRValidationError("schema_error", f"op {op} needs exactly {exact} args")
    if minimum is not None and len(args) < minimum:
        raise IRValidationError("schema_error", f"op {op} needs at least {minimum} args")


def _numeric(node: _Node, op: str) -> None:
    if node.scalar_type in {"bool", "year"}:
        raise IRValidationError("type_error", f"op {op} cannot consume {node.scalar_type}")


def _require_type(node: _Node, wanted: str, op: str) -> None:
    if node.scalar_type != wanted:
        raise IRValidationError(
            "type_error", f"op {op} needs {wanted}, got {node.scalar_type}",
        )


def _common_numeric_type(args: list[_Node], op: str) -> str:
    for node in args:
        _numeric(node, op)
    types = {node.scalar_type for node in args if node.constant != 0}
    if not types:
        types = {args[0].scalar_type}
    if len(types) != 1:
        raise IRValidationError("type_error", f"op {op} has incompatible types {sorted(types)}")
    return next(iter(types))


def _same_quantity(args: list[_Node], op: str) -> None:
    if len(args) != 2 or args[0].scalar_type != args[1].scalar_type \
            or args[0].scalar_type not in {"money", "number"}:
        raise IRValidationError(
            "type_error", f"op {op} needs two equal quantity types",
        )


def _multiply_type(left: str, right: str) -> str:
    if left == "ratio" and right == "ratio":
        return "ratio"
    if left in {"money", "number", "percent"} and right == "ratio":
        return left
    if right in {"money", "number", "percent"} and left == "ratio":
        return right
    if left == right == "number":
        return "number"
    raise IRValidationError("type_error", f"cannot multiply {left} by {right}")


def _divide_type(left: str, right: str) -> str:
    if left == right and left in {"money", "number", "ratio", "percent", "count"}:
        return "ratio"
    if left == "money" and right == "number":
        return "money"
    if left in {"money", "number", "percent"} and right == "ratio":
        return left
    if right == "count" and left in {"money", "number", "ratio", "percent"}:
        return left
    raise IRValidationError("type_error", f"cannot divide {left} by {right}")


def _comparable(left: _Node, right: _Node, op: str) -> None:
    if left.constant == 0 and left.scalar_type in {"number", "ratio"}:
        return
    if right.constant == 0 and right.scalar_type in {"number", "ratio"}:
        return
    if left.scalar_type != right.scalar_type or left.scalar_type == "bool":
        raise IRValidationError(
            "type_error", f"op {op} cannot compare {left.scalar_type} and {right.scalar_type}",
        )


def _branch_type(yes: _Node, no: _Node, op: str) -> str:
    if yes.scalar_type == no.scalar_type:
        return yes.scalar_type
    if yes.constant == 0 and yes.scalar_type in {"number", "ratio"}:
        return no.scalar_type
    if no.constant == 0 and no.scalar_type in {"number", "ratio"}:
        return yes.scalar_type
    raise IRValidationError(
        "type_error", f"op {op} branches differ: {yes.scalar_type}/{no.scalar_type}",
    )


def _common_projection_type(results: list[_Node], op: str) -> str:
    types = {node.scalar_type for node in results}
    if len(types) != 1 or "bool" in types:
        raise IRValidationError(
            "type_error", f"{op} results have incompatible types {sorted(types)}",
        )
    return next(iter(types))

