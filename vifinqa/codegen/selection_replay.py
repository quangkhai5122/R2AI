"""Conservative, deterministic replay of saved Selection responses.

P2.1 stored the raw/parsed Selection attempts even when synthesis was rejected.
This module rebuilds the exact same shortlist, runs the current deterministic
synthesiser, executes the resulting expression, and writes a *new* codegen
artifact.  The input artifact is never modified.

The default policy replaces only the generator's structural ``source=none``
placeholder.  Successful LLM/rule answers therefore remain authoritative.
"""
from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections import Counter
from pathlib import Path

from ..extraction.build_store import Store
from ..utils.io import read_jsonl, write_json, write_jsonl
from .executor import run_code
from .generate import QuestionBundle
from .hybrid import is_structural_none
from .selection import Selection, confidence, parse_selection, synthesize
from .semantic import validate_generated_answer


POLICY = "p21r_saved_selection_replay_v3"
REPLACE_POLICIES = {"none_only", "trace_failures"}
VALID_OUTPUT_TYPES = {
    "number", "percent", "percentage_point", "ratio", "year", "count",
}


def _semantic_component_paths() -> dict[str, Path]:
    """Code files whose contents determine operand rebuilding/replay."""
    package = Path(__file__).resolve().parents[1]
    return {
        "codegen/executor.py": package / "codegen" / "executor.py",
        "codegen/generate.py": package / "codegen" / "generate.py",
        "codegen/hybrid.py": package / "codegen" / "hybrid.py",
        "codegen/selection.py": package / "codegen" / "selection.py",
        "codegen/selection_replay.py": package / "codegen" / "selection_replay.py",
        "codegen/semantic.py": package / "codegen" / "semantic.py",
        "codegen/units.py": package / "codegen" / "units.py",
        "config.py": package / "config.py",
        "extraction/build_store.py": package / "extraction" / "build_store.py",
        "retrieval/serialize.py": package / "retrieval" / "serialize.py",
        "retrieval/shortlist.py": package / "retrieval" / "shortlist.py",
        "router/metric_phrase.py": package / "router" / "metric_phrase.py",
        "utils/viet_text.py": package / "utils" / "viet_text.py",
    }


def replay_selection_artifact(
    retrieval_path: Path,
    codegen_path: Path,
    store_dir: Path,
    out_path: Path,
    audit_path: Path | None = None,
    *,
    k: int = 0,
    top_n: int = 12,
    replace_policy: str = "none_only",
    output_types: set[str] | list[str] | tuple[str, ...] | str | None = None,
) -> dict:
    """Replay saved Selection attempts and return an audit dictionary.

    ``none_only`` is the safe default. ``trace_failures`` is an explicit
    ablation that may replace a rule answer when its associated Selection trace
    was rejected; it is intentionally never selected implicitly.
    """
    retrieval_path = Path(retrieval_path)
    codegen_path = Path(codegen_path)
    store_dir = Path(store_dir)
    out_path = Path(out_path)
    audit_path = (Path(audit_path) if audit_path else
                  out_path.with_suffix(".audit.json"))
    _validate_policy(replace_policy)
    selected_output_types = _normalize_output_types(output_types)
    if k < 0:
        raise ValueError("k must be >= 0 (0 = route.evidence_budget)")
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    if _same_path(codegen_path, out_path):
        raise ValueError("refusing to overwrite the input codegen artifact")
    if _same_path(codegen_path, audit_path):
        raise ValueError("audit path must not overwrite the input artifact")

    retrieval = _read_unique(retrieval_path, "retrieval")
    original = _read_unique(codegen_path, "codegen")
    _validate_same_universe(original, retrieval)
    original_signature = _one_signature(original)
    input_sha256 = _sha256(codegen_path)
    retrieval_sha256 = _sha256(retrieval_path)
    compiler_path = Path(__file__).with_name("selection.py")
    compiler_sha256 = _sha256(compiler_path)
    semantic_inputs = _semantic_inputs_identity()
    store_identity = _store_identity(store_dir, retrieval)
    replay_signature = _replay_signature(
        original_signature, input_sha256, retrieval_sha256,
        semantic_inputs_sha256=semantic_inputs["manifest_sha256"],
        store_manifest_sha256=store_identity["manifest_sha256"],
        k=k, top_n=top_n, replace_policy=replace_policy,
        output_types=selected_output_types,
    )

    store = Store(store_dir, cache_size=6)
    output = []
    counts = Counter()
    replayed_ids: list[int] = []
    replayed_records: list[dict] = []
    skipped_output_type_ids: list[int] = []
    unresolved_ids: list[int] = []
    unresolved_reasons = Counter()

    for qid in sorted(original):
        old = original[qid]
        eligible = _eligible(old, replace_policy)
        route_output_type = str(
            (retrieval[qid].get("route") or {}).get("output_type", "number")
        ).strip().lower()
        selected = (selected_output_types is None or
                    route_output_type in selected_output_types)
        if eligible and selected:
            bundle = QuestionBundle(
                retrieval[qid], store, k=k, run_signature=replay_signature,
            )
            row, outcome, replay_meta = replay_selection_record(
                old, bundle, top_n=top_n, replace_policy=replace_policy,
            )
        elif eligible:
            row, outcome, replay_meta = (
                dict(old), "skipped_by_output_type", None,
            )
        else:
            row, outcome, replay_meta = dict(old), "not_eligible", None

        if outcome == "replayed":
            counts["replayed"] += 1
            replayed_ids.append(qid)
            route = retrieval[qid].get("route") or {}
            replayed_records.append({
                "id": qid,
                "question": old.get("question", ""),
                "operation": (route.get("plan") or {}).get("op", "lookup"),
                "output_type": route.get("output_type", "number"),
                "answer": row.get("answer"),
                "pandas_query": row.get("pandas_query", ""),
                "used_vars": row.get("used_vars") or [],
                "replay": replay_meta or {},
            })
            selected_from = "replayed_selection"
        elif outcome == "skipped_by_output_type":
            counts["skipped_by_output_type"] += 1
            skipped_output_type_ids.append(qid)
            selected_from = "original_filtered_output_type"
        elif eligible:
            counts["unresolved"] += 1
            unresolved_ids.append(qid)
            unresolved_reasons[outcome] += 1
            selected_from = "original_unresolved"
        else:
            counts["kept_non_eligible"] += 1
            selected_from = "original"

        row["run_signature"] = replay_signature
        provenance = dict(row.get("p21r_provenance") or {})
        provenance.update({
            "schema_version": 3,
            "policy": POLICY,
            "replace_policy": replace_policy,
            "selected_from": selected_from,
            "original_run_signature": original_signature,
            "original_source": old.get("source"),
            "original_status": old.get("status"),
            "decision": outcome,
        })
        if replay_meta:
            provenance["replay"] = replay_meta
        row["p21r_provenance"] = provenance
        output.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, output)
    # A read-only input integrity check makes accidental in-place mutation loud.
    if _sha256(codegen_path) != input_sha256:
        raise RuntimeError("input codegen artifact changed during replay")

    audit = {
        "schema_version": 2,
        "policy": POLICY,
        "replace_policy": replace_policy,
        "parameters": {
            "k": int(k),
            "top_n": int(top_n),
            "output_types": _output_types_for_audit(selected_output_types),
        },
        "compiler": {
            "path": str(compiler_path),
            "sha256": compiler_sha256,
        },
        "semantic_inputs": semantic_inputs,
        "store": store_identity,
        "input": {
            "path": str(codegen_path),
            "sha256": input_sha256,
            "run_signature": original_signature,
            "rows": len(original),
        },
        "retrieval": {
            "path": str(retrieval_path),
            "sha256": retrieval_sha256,
            "rows": len(retrieval),
        },
        "output": {
            "path": str(out_path),
            "sha256": _sha256(out_path),
            "run_signature": replay_signature,
            "rows": len(output),
        },
        "counts": {
            "total": len(output),
            "kept_non_eligible": counts["kept_non_eligible"],
            "skipped_by_output_type": counts["skipped_by_output_type"],
            "replayed": counts["replayed"],
            "unresolved": counts["unresolved"],
        },
        "unresolved_reasons": dict(sorted(unresolved_reasons.items())),
        "replayed_ids": replayed_ids,
        "replayed_records": replayed_records,
        "skipped_by_output_type_ids": skipped_output_type_ids,
        "unresolved_ids": unresolved_ids,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(audit_path, audit)
    return audit


def replay_selection_record(
    original: dict,
    bundle,
    *,
    top_n: int = 12,
    replace_policy: str = "none_only",
) -> tuple[dict, str, dict | None]:
    """Replay one record; exposed separately for focused unit tests."""
    _validate_policy(replace_policy)
    if not _eligible(original, replace_policy):
        return dict(original), "not_eligible", None

    trace = original.get("selection_trace") or {}
    attempts = trace.get("attempts") or []
    candidates = bundle.shortlist(None, top_n=top_n)
    if not candidates:
        return dict(original), "no_candidates", None

    saw_selection = False
    failures = Counter()
    for attempt in attempts:
        selection = _selection_from_attempt(attempt)
        if selection is None:
            continue
        saw_selection = True
        answer, query, error = synthesize(selection, candidates, bundle.route)
        if error or answer is None:
            failures[f"synthesis:{error or 'no answer'}"] += 1
            continue

        executed = run_code(query, bundle.dfs)
        if executed.get("status") != "ok":
            failures[f"execution:{executed.get('status')}"] += 1
            continue
        check = validate_generated_answer(
            query, bundle.dfs.keys(), executed["value"], route=bundle.route,
        )
        if not check.ok:
            failures["semantic:" + "; ".join(check.errors)] += 1
            continue
        if not math.isclose(
            float(executed["value"]), float(answer),
            rel_tol=0.0, abs_tol=0.011,
        ):
            failures["answer_mismatch"] += 1
            continue

        row = dict(original)
        row.update({
            "answer": round(float(executed["value"]), 2),
            "pandas_query": query,
            "used_vars": bundle.used_vars(query),
            "inline_error": "",
            "status": "ok",
            "source": "llm_select_p21r",
            "votes": 1,
            "n_ok": 1,
            "detail": (f"P2.1r replay op={selection.op} "
                       f"operands={selection.operands}"),
            "detail_conf": confidence(
                selection, candidates, float(answer), bundle.route,
            ),
            "semantic": check.to_dict(),
        })
        meta = {
            "attempt_index": int(attempt.get("index", 0) or 0),
            "original_reason_code": str(attempt.get("reason_code", "")),
            "selection": {
                "op": selection.op,
                "operands": list(selection.operands),
            },
        }
        return row, "replayed", meta

    if not saw_selection:
        return dict(original), "no_saved_selection", None
    if not failures:
        return dict(original), "no_replay_result", None
    # Bounded, deterministic reason suitable for aggregate audit counts.
    reason = failures.most_common(1)[0][0]
    return dict(original), reason[:200], None


def _selection_from_attempt(attempt) -> Selection | None:
    if not isinstance(attempt, dict):
        return None
    saved = attempt.get("selection")
    if isinstance(saved, dict):
        raw_operands = saved.get("operands")
        if isinstance(raw_operands, list):
            try:
                operands = [int(value) for value in raw_operands]
            except (TypeError, ValueError):
                operands = []
            op = str(saved.get("op", "")).strip().lower()
            if op and operands:
                return Selection(op, operands, str(saved.get("note", ""))[:120])
    return parse_selection(str(attempt.get("raw_response", "")))


def _eligible(row: dict, replace_policy: str) -> bool:
    if replace_policy == "none_only":
        return is_structural_none(row)
    trace = row.get("selection_trace") or {}
    return str(trace.get("outcome", "")) == "rejected"


def _validate_policy(value: str) -> None:
    if value not in REPLACE_POLICIES:
        raise ValueError(
            f"replace_policy must be one of {sorted(REPLACE_POLICIES)}, got {value!r}"
        )


def _read_unique(path: Path, label: str) -> dict[int, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    out: dict[int, dict] = {}
    duplicates = []
    for row in read_jsonl(path):
        if "id" not in row:
            raise ValueError(f"{label} row has no id")
        qid = int(row["id"])
        if qid in out:
            duplicates.append(qid)
        out[qid] = row
    if duplicates:
        raise ValueError(
            f"{label} contains duplicate ids: {sorted(set(duplicates))[:10]}"
        )
    if not out:
        raise ValueError(f"{label} file is empty: {path}")
    return out


def _validate_same_universe(codegen: dict[int, dict], retrieval: dict[int, dict]) -> None:
    if set(codegen) != set(retrieval):
        raise ValueError(
            "codegen/retrieval id sets differ: "
            f"missing={sorted(set(codegen) - set(retrieval))[:10]}, "
            f"extra={sorted(set(retrieval) - set(codegen))[:10]}"
        )
    mismatches = []
    for qid in codegen:
        a = unicodedata.normalize("NFC", str(codegen[qid].get("question", ""))).strip()
        b = unicodedata.normalize("NFC", str(retrieval[qid].get("question", ""))).strip()
        if not a or a != b:
            mismatches.append(qid)
    if mismatches:
        raise ValueError(f"codegen/retrieval questions differ: ids={mismatches[:10]}")


def _one_signature(rows: dict[int, dict]) -> str:
    values = {str(row.get("run_signature", "")).strip() for row in rows.values()}
    if "" in values or len(values) != 1:
        raise ValueError("codegen must contain exactly one non-empty run_signature")
    return next(iter(values))


def _same_path(a: Path, b: Path) -> bool:
    return a.resolve(strict=False) == b.resolve(strict=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _semantic_inputs_identity() -> dict:
    files = []
    manifest = []
    for name, path in sorted(_semantic_component_paths().items()):
        digest = _sha256(path)
        size = path.stat().st_size
        files.append({
            "name": name,
            "path": str(path),
            "sha256": digest,
            "bytes": size,
        })
        manifest.append({"name": name, "sha256": digest, "bytes": size})
    return {"manifest_sha256": _json_sha256(manifest), "files": files}


def _store_identity(store_dir: Path, retrieval: dict[int, dict]) -> dict:
    """Hash the store files that can be read while rebuilding shortlists."""
    store_dir = Path(store_dir)
    tickers = set()
    for rec in retrieval.values():
        for candidate in rec.get("candidates") or []:
            ticker = str(candidate.get("ticker") or "").strip().upper()
            if not ticker:
                ticker = str(candidate.get("report_id") or "").split("_", 1)[0]
                ticker = ticker.strip().upper()
            if ticker:
                tickers.add(ticker)

    required = [store_dir / "reports.parquet"]
    required.extend(store_dir / "tables" / f"{ticker}.parquet"
                    for ticker in sorted(tickers))
    files = []
    manifest = []
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"required replay store file not found: {path}")
        relative = path.relative_to(store_dir).as_posix()
        digest = _sha256(path)
        size = path.stat().st_size
        files.append({
            "path": relative,
            "sha256": digest,
            "bytes": size,
        })
        manifest.append({"path": relative, "sha256": digest, "bytes": size})
    return {
        "path": str(store_dir),
        "manifest_sha256": _json_sha256(manifest),
        "required_tickers": sorted(tickers),
        "files": files,
    }


def _normalize_output_types(
    values: set[str] | list[str] | tuple[str, ...] | str | None,
) -> frozenset[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        raw_values = values.split(",")
    else:
        raw_values = values
    normalized = {
        str(value).strip().lower() for value in raw_values
        if str(value).strip()
    }
    if not normalized or normalized == {"all"}:
        return None
    if "all" in normalized:
        raise ValueError("output_types='all' cannot be combined with other values")
    unknown = sorted(normalized - VALID_OUTPUT_TYPES)
    if unknown:
        raise ValueError(
            f"unknown output_types {unknown}; expected {sorted(VALID_OUTPUT_TYPES)}"
        )
    return frozenset(normalized)


def _output_types_for_audit(values: frozenset[str] | None):
    return "all" if values is None else sorted(values)


def _replay_signature(
    original_signature: str,
    input_sha256: str,
    retrieval_sha256: str,
    *,
    semantic_inputs_sha256: str,
    store_manifest_sha256: str,
    k: int,
    top_n: int,
    replace_policy: str,
    output_types: frozenset[str] | None,
) -> str:
    payload = json.dumps({
        "policy": POLICY,
        "replace_policy": replace_policy,
        "original_run_signature": original_signature,
        "input_sha256": input_sha256,
        "retrieval_sha256": retrieval_sha256,
        "semantic_inputs_sha256": semantic_inputs_sha256,
        "store_manifest_sha256": store_manifest_sha256,
        "k": int(k),
        "top_n": int(top_n),
        "output_types": _output_types_for_audit(output_types),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
