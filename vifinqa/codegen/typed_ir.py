"""Typed nested financial IR and deterministic pandas compiler.

The IR contains named, provenance-bearing facts and a small expression tree. It
never accepts raw pandas code. Compilation is fail-closed on schema, arity,
types, duplicate evidence cells, output units and ungrounded literals.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..utils.viet_text import norm
from .units import check_answer_unit

SCHEMA_VERSION = 1
POLICY_VERSION = "typed_nested_financial_ir_v1"
MAX_FACTS = 36
MAX_NODES = 128
MAX_DEPTH = 14
_NESTED_CUES = (
    "nam ma", "nam co", "doanh nghiep co", "cong ty co", "tai nam",
    "sau khi", "trong phan nhom", "trung vi", "phan nhom", "gia su",
    "kich ban", "neu ",
)


class IRValidationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class FactBinding:
    name: str
    expr: str
    value: float
    scalar_type: str
    stable_cell: tuple
    confidence: float = 80.0


@dataclass(frozen=True)
class CompiledIR:
    query: str
    answer: float
    output_type: str
    inferred_type: str
    referenced_facts: tuple[str, ...]
    root_op: str
    node_count: int
    max_depth: int
    confidence: float


@dataclass(frozen=True)
class _Node:
    expr: str
    scalar_type: str
    refs: frozenset[str]
    value: float


def _same_type(args, op, allowed):
    types={node.scalar_type for node in args}
    if len(types)!=1 or next(iter(types)) not in allowed:
        raise IRValidationError("type", f"{op} has incompatible types {sorted(types)}")
    return next(iter(types))


class Compiler:
    def __init__(self, program: dict, facts: list[FactBinding], route: dict,
                 question: str=""):
        self.program=program
        self.facts={fact.name:fact for fact in facts}
        self.route=route or {}
        self.question=question or ""
        self.node_count=0
        self.max_depth=0

    def compile(self) -> CompiledIR:
        if self.program.get("schema_version") != SCHEMA_VERSION:
            raise IRValidationError("schema", "schema_version mismatch")
        if not self.facts or len(self.facts)>MAX_FACTS:
            raise IRValidationError("facts", "invalid fact count")
        stable=[]
        for fact in self.facts.values():
            cell=fact.stable_cell
            if (isinstance(cell, tuple) and cell
                    and all(isinstance(item, tuple) for item in cell)):
                stable.extend(cell)
            else:
                stable.append(cell)
        if len(stable)!=len(set(stable)):
            raise IRValidationError("grounding", "duplicate stable evidence cells")
        root=self.program.get("root")
        node=self._node(root,1)
        declared=str(self.program.get("output_type", ""))
        expected=str(self.route.get("output_type") or "number")
        if declared != expected:
            raise IRValidationError("output", f"declared {declared} != route {expected}")
        expr,answer=self._normalise(node,expected)
        query=f"round(float({expr}), 2)"
        try: compile(query,"<typed_ir>","eval")
        except SyntaxError as exc: raise IRValidationError("compile",str(exc)) from exc
        if not math.isfinite(answer): raise IRValidationError("value","non-finite answer")
        warning=check_answer_unit(answer,expected)
        if warning: raise IRValidationError("unit",warning)
        used=tuple(sorted(node.refs))
        if set(used)!=set(self.facts):
            missing=sorted(set(self.facts)-set(used))
            raise IRValidationError("grounding",f"unused facts {missing}")
        confidence=max(0.0,min(99.0,min(self.facts[name].confidence for name in used)
                                - max(0,self.max_depth-5)*1.5))
        return CompiledIR(query,round(answer,2),expected,node.scalar_type,used,
                          str(root.get("op","")),self.node_count,self.max_depth,confidence)

    def _node(self,obj,depth):
        self.node_count+=1; self.max_depth=max(self.max_depth,depth)
        if self.node_count>MAX_NODES: raise IRValidationError("limit","too many nodes")
        if depth>MAX_DEPTH: raise IRValidationError("limit","IR too deep")
        if not isinstance(obj,dict): raise IRValidationError("schema","node must be object")
        op=str(obj.get("op","")).lower()
        if op=="ref":
            name=str(obj.get("fact",""))
            if name not in self.facts: raise IRValidationError("grounding",f"unknown fact {name}")
            fact=self.facts[name]
            return _Node(fact.expr,fact.scalar_type,frozenset({name}),float(fact.value))
        if op=="literal":
            typ=str(obj.get("type","number")); value=float(obj.get("value"))
            if typ not in {"number","ratio","percent","year","money"}: raise IRValidationError("type",f"bad literal type {typ}")
            if not math.isfinite(value): raise IRValidationError("value","non-finite literal")
            return _Node(repr(value),typ,frozenset(),value)
        if op in {"argmax_project","argmin_project"}:
            return self._projection(op,obj,depth)
        args=obj.get("args")
        if not isinstance(args,list): raise IRValidationError("schema",f"{op} needs args")
        children=[self._node(child,depth+1) for child in args]
        return self._operation(op,children)


    def _operation(self, op, args):
        if op in ("add", "sub", "difference"):
            if len(args) != 2: raise IRValidationError("arity", f"{op} needs 2 args")
            types={x.scalar_type for x in args}
            if len(types)!=1 or next(iter(types)) not in {"money","number","percent","percentage_point"}:
                raise IRValidationError("type", f"{op} type mismatch")
            value=args[0].value + args[1].value if op=="add" else args[0].value-args[1].value
            symbol='+' if op=='add' else '-'
            return _Node(f"({args[0].expr} {symbol} {args[1].expr})",args[0].scalar_type,args[0].refs|args[1].refs,value)
        if op in ("average", "sum"):
            if len(args)<2: raise IRValidationError("arity", f"{op} needs 2+ args")
            types={x.scalar_type for x in args}
            if len(types)!=1: raise IRValidationError("type", f"{op} type mismatch")
            value=sum(x.value for x in args)/(len(args) if op=="average" else 1)
            inner=' + '.join(f"({x.expr})" for x in args)
            expr=f"(({inner}) / {len(args)})" if op=="average" else f"({inner})"
            return _Node(expr,args[0].scalar_type,frozenset().union(*(x.refs for x in args)),value)
        if op in ("growth_pct","ratio_pct","ratio_times","percentage_point"):
            if len(args)!=2: raise IRValidationError("arity", f"{op} needs 2 args")
            if args[1].value==0: raise IRValidationError("zero", "division by zero")
            if op=="growth_pct":
                value=(args[0].value-args[1].value)/abs(args[1].value)*100; typ="percent"; expr=f"(({args[0].expr} - {args[1].expr}) / abs({args[1].expr}) * 100)"
            elif op=="percentage_point":
                value=args[0].value-args[1].value; typ="percentage_point"; expr=f"({args[0].expr} - {args[1].expr})"
            else:
                value=args[0].value/args[1].value; typ="percent" if op=="ratio_pct" else "ratio"; expr=f"({args[0].expr} / {args[1].expr})" + (" * 100" if op=="ratio_pct" else "")
            return _Node(expr,typ,args[0].refs|args[1].refs,value)
        raise IRValidationError("op", f"unsupported operation {op}")

    def _projection(self, op, obj, depth):
        items=obj.get("items")
        if not isinstance(items,list) or len(items)<2: raise IRValidationError("arity", f"{op} needs 2+ items")
        pairs=[]
        for item in items:
            if not isinstance(item,dict): raise IRValidationError("schema", "projection item must be object")
            score=self._node(item.get("score"),depth+1); result=self._node(item.get("result"),depth+1)
            if score.scalar_type not in {"money","number","ratio","percent"}: raise IRValidationError("type", "ranking score is not numeric")
            pairs.append((score,result))
        types={result.scalar_type for _,result in pairs}
        if len(types)!=1: raise IRValidationError("type", "projection result type mismatch")
        chosen=min(pairs,key=lambda p:p[0].value) if op=="argmin_project" else max(pairs,key=lambda p:p[0].value)
        refs=frozenset().union(*(score.refs|result.refs for score,result in pairs))
        support=" + ".join(
            [f"({score.expr})" for score,_result in pairs]
            + [f"({result.expr})" for _score,result in pairs]
        )
        expr=f"(({chosen[1].expr}) + 0 * ({support}))"
        return _Node(expr,chosen[1].scalar_type,refs,chosen[1].value)

    def _normalise(self, node, output):
        scale=float(self.route.get("unit_scale") or 1.0)
        if output=="number":
            if node.scalar_type not in {"money","number"}: raise IRValidationError("output", "number needs money/number")
            return f"({node.expr}) / {scale:g}",node.value/scale
        if output=="percent" and node.scalar_type in {"percent","percentage_point"}: return node.expr,node.value
        if output=="percentage_point" and node.scalar_type=="percentage_point": return node.expr,node.value
        if output=="ratio" and node.scalar_type=="ratio": return node.expr,node.value
        if output=="year" and node.scalar_type=="year": return node.expr,node.value
        if output=="count" and node.scalar_type in {"number","money"}: return node.expr,node.value
        raise IRValidationError("output", f"{output} incompatible with {node.scalar_type}")


def compile_program(program: dict, facts: list[FactBinding], route: dict, question: str="") -> CompiledIR:
    return Compiler(program,facts,route,question).compile()


def binding_from_resolved(name: str, resolved) -> FactBinding:
    return FactBinding(name=name,expr=resolved.expr_vnd(),value=resolved.value_vnd,scalar_type="money",stable_cell=(resolved.report_id,resolved.table_pos,resolved.label,resolved.col),confidence=float(resolved.score))


def program_for_operation(op: str, names: list[str], route: dict) -> dict:
    refs=[{"op":"ref","fact":name} for name in names]
    if op=="difference": root={"op":"difference","args":refs[:2]}
    elif op in {"growth_pct","cagr"}: root={"op":"growth_pct","args":refs[:2]}
    elif op in {"ratio","margin"}: root={"op":"ratio_pct" if route.get("output_type")=="percent" else "ratio_times","args":refs[:2]}
    elif op=="average": root={"op":"average","args":refs}
    elif op=="sum": root={"op":"sum","args":refs}
    elif op=="ranking": root={"op":"argmin_project" if any(x in str(route.get("question","")).lower() for x in ("thấp nhất","nhỏ nhất","ít nhất")) else "argmax_project","items":[{"score":ref,"result":ref} for ref in refs]}
    else: return None
    return {"schema_version":SCHEMA_VERSION,"output_type":route.get("output_type","number"),"root":root}


@dataclass(frozen=True)
class TypedIRAnswer:
    ok: bool
    answer: float = 0.0
    pandas_query: str = ""
    confidence: float = 0.0
    detail: str = ""
    compiled: CompiledIR | None = None
    resolved: tuple = ()


def try_typed_ir_answer(route: dict, tables: list[dict], encoder=None,
                        min_score: float = 62.0) -> TypedIRAnswer:
    plan=route.get("plan") or {}; op=str(plan.get("op","lookup")); raw_facts=plan.get("facts") or []
    if len(raw_facts)<2 or op not in {"difference","growth_pct","ratio","margin","average","sum","ranking"}:
        return TypedIRAnswer(False,detail="typed_ir route unsupported")
    question=norm(route.get("question",""))
    if any(cue in question for cue in _NESTED_CUES):
        return TypedIRAnswer(False,detail="typed_ir nested route needs explicit program")
    from .rule_composite import _FactView
    from .fact_resolver import resolve_fact
    resolved=[]
    for raw in raw_facts:
        fact=_FactView(raw); variants=[fact.metric] if fact.metric else route.get("metric_variants",[])
        found=resolve_fact(fact,tables,variants,encoder,min_score,route=route)
        if found is None: return TypedIRAnswer(False,detail=f"typed_ir unresolved {fact.metric}")
        resolved.append(found)
    names=[f"f{i}" for i in range(len(resolved))]
    facts=[binding_from_resolved(name,fact) for name,fact in zip(names,resolved)]
    program=program_for_operation(op,names,route)
    if program is None: return TypedIRAnswer(False,detail="typed_ir no program")
    try: compiled=compile_program(program,facts,route,route.get("question",""))
    except IRValidationError as exc:
        return TypedIRAnswer(False,detail=f"typed_ir {exc.code}: {exc}",resolved=tuple(resolved))
    return TypedIRAnswer(True,compiled.answer,compiled.query,compiled.confidence,
                         f"typed_ir op={compiled.root_op} facts={len(facts)}",
                         compiled,tuple(resolved))
