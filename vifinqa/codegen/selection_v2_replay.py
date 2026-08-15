"""CPU-only replay of saved Structured Selection v2 responses.

The interrupted Kaggle checkpoint contains every raw model response.  This
module reconstructs the exact shortlist from the frozen retrieval/store, runs
the current deterministic compiler and semantic validator, and writes a new
complete artifact derived from the frozen control.  Neither input is modified.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

from ..extraction.build_store import Store
from ..utils.io import read_jsonl, write_json, write_jsonl
from .generate import (
    QuestionBundle,
    _final,
    _mark_llm_attempt_completed,
    _run_validated,
)
from .hybrid import is_structural_none
from .llm_client import GenerationSample
from .selection_v2 import POLICY_VERSION, evaluate_samples


POLICY = "p22_saved_response_replay_v2_subset_safe_v5_1"


def replay_checkpoint(
    retrieval_path: Path,
    checkpoint_path: Path,
    control_path: Path,
    mask_path: Path,
    store_dir: Path,
    out_path: Path,
    audit_path: Path | None = None,
    *,
    k: int = 0,
    top_n: int = 24,
    rescue_no_candidates: bool = False,
    rescue_table_k: int = 20,
    rescue_min_score: float = 28.0,
    allow_checkpoint_superset: bool = False,
) -> dict:
    """Replay every completed response in a masked, possibly partial checkpoint."""
    retrieval_path = Path(retrieval_path)
    checkpoint_path = Path(checkpoint_path)
    control_path = Path(control_path)
    mask_path = Path(mask_path)
    store_dir = Path(store_dir)
    out_path = Path(out_path)
    audit_path = Path(audit_path) if audit_path else out_path.with_suffix(".audit.json")
    if k < 0:
        raise ValueError("k must be >= 0 (0 = route.evidence_budget)")
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    if rescue_table_k < 1:
        raise ValueError("rescue_table_k must be >= 1")
    if not (-1_000_000 < float(rescue_min_score) < 1_000_000):
        raise ValueError("rescue_min_score must be finite and bounded")
    for source in (retrieval_path, checkpoint_path, control_path, mask_path):
        if _same_path(source, out_path) or _same_path(source, audit_path):
            raise ValueError("replay outputs must not overwrite any input")

    retrieval = _read_unique(retrieval_path, "retrieval")
    checkpoint = _read_unique(checkpoint_path, "checkpoint")
    control = _read_unique(control_path, "control")
    _validate_same_universe(retrieval, checkpoint, "retrieval/checkpoint")
    _validate_same_universe(retrieval, control, "retrieval/control")
    target = _read_mask(mask_path)
    if not target <= set(retrieval):
        raise ValueError(f"mask contains unknown IDs: {sorted(target - set(retrieval))[:10]}")
    non_structural = sorted(qid for qid in target if not is_structural_none(control[qid]))
    if non_structural:
        raise ValueError(
            "P2.2 replay mask must target frozen-control structural none only: "
            f"ids={non_structural[:10]}"
        )

    checkpoint_signature = _one_signature(checkpoint, "checkpoint")
    control_signature = _one_signature(control, "control")
    input_hashes = {
        "retrieval": _sha256(retrieval_path),
        "checkpoint": _sha256(checkpoint_path),
        "control": _sha256(control_path),
        "mask": _sha256(mask_path),
    }
    semantic_inputs = _semantic_identity()
    store_identity = _store_identity(store_dir, retrieval)
    replay_signature = _signature(
        input_hashes, semantic_inputs["manifest_sha256"],
        store_identity["manifest_sha256"], checkpoint_signature,
        control_signature, k=k, top_n=top_n,
        rescue_no_candidates=rescue_no_candidates,
        rescue_table_k=rescue_table_k, rescue_min_score=rescue_min_score,
        allow_checkpoint_superset=allow_checkpoint_superset,
    )

    checkpoint_attempted = set()
    for qid, row in checkpoint.items():
        marker = row.get("llm_attempt_status") == "completed"
        trace = row.get("selection_trace")
        v2_trace = (
            isinstance(trace, dict)
            and trace.get("mode") == "select_v2"
            and int(trace.get("schema_version", 0)) == 2
        )
        if marker or v2_trace:
            if not (marker and v2_trace):
                raise ValueError(f"incomplete checkpoint marker/trace at id={qid}")
            checkpoint_attempted.add(qid)
    attempted, ignored_checkpoint_attempted = _scope_attempted(
        checkpoint_attempted, target,
        allow_checkpoint_superset=allow_checkpoint_superset,
    )

    store = Store(store_dir, cache_size=6)
    output = []
    counts = Counter()
    rejection_counts = Counter()
    transitions = Counter()
    accepted_ids = []
    rejected_ids = []
    pending_ids = sorted(target - attempted)
    records = []

    for qid in sorted(control):
        base = dict(control[qid])
        old = checkpoint[qid]
        old_trace = old.get("selection_trace") or {}
        old_outcome = str(old_trace.get("outcome") or "not_attempted")
        new_trace = None
        selected_from = "control_outside_mask"

        if qid in attempted:
            bundle = QuestionBundle(
                retrieval[qid], store, k=k, run_signature=replay_signature,
                rescue_no_candidates=rescue_no_candidates,
                rescue_table_k=rescue_table_k,
                rescue_min_score=rescue_min_score,
            )
            samples = _saved_samples(old, qid)
            candidates = bundle.shortlist_v2(None, top_n=top_n)
            decision, new_trace = evaluate_samples(
                samples, candidates, bundle.route, bundle.question,
                lambda query, b=bundle: _run_validated(b, query),
                atomic_facts=bundle.atomic_slots(),
            )
            new_trace["shortlist"] = dict(bundle.shortlist_trace)
            new_outcome = str(new_trace.get("outcome") or "unknown")
            rejection_counts.update(new_trace.get("rejection_counts") or {})
            transitions[f"{old_outcome}->{new_outcome}"] += 1
            if decision is not None:
                row = _final(
                    bundle, decision.answer, decision.query, "llm_select_v2",
                )
                row["detail"] = (
                    f"CPU replay {POLICY_VERSION}; root={decision.compiled.root_op} "
                    f"refs={list(decision.compiled.referenced_indices)}"
                )
                row["detail_conf"] = decision.confidence
                row["selection_trace"] = new_trace
                _mark_llm_attempt_completed(row)
                accepted_ids.append(qid)
                counts["accepted"] += 1
                selected_from = "replayed_selection"
            else:
                row = base
                row["selection_trace"] = new_trace
                _mark_llm_attempt_completed(row)
                rejected_ids.append(qid)
                counts["rejected"] += 1
                selected_from = "control_after_rejection"
            records.append({
                "id": qid,
                "old_outcome": old_outcome,
                "new_outcome": new_outcome,
                "old_answer": old.get("answer"),
                "new_answer": row.get("answer"),
                "new_source": row.get("source"),
                "accepted_attempt": new_trace.get("accepted_attempt"),
                "rejection_counts": new_trace.get("rejection_counts") or {},
                "attempts": [
                    {
                        "index": attempt.get("index"),
                        "stage": attempt.get("stage"),
                        "reason_code": attempt.get("reason_code"),
                        "reason": attempt.get("reason"),
                    }
                    for attempt in new_trace.get("attempts") or []
                ],
            })
        elif qid in target:
            row = base
            counts["pending"] += 1
            selected_from = "control_pending_no_response"
        else:
            row = base
            counts["outside_mask"] += 1

        row["run_signature"] = replay_signature
        row["p22_replay_provenance"] = {
            "schema_version": 1,
            "policy": POLICY,
            "compiler_policy": POLICY_VERSION,
            "selected_from": selected_from,
            "checkpoint_run_signature": checkpoint_signature,
            "control_run_signature": control_signature,
        }
        output.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, output)
    for name, path in {
        "retrieval": retrieval_path, "checkpoint": checkpoint_path,
        "control": control_path, "mask": mask_path,
    }.items():
        if _sha256(path) != input_hashes[name]:
            raise RuntimeError(f"{name} input changed during replay")

    audit = {
        "schema_version": "p22_saved_response_replay_v2_subset_safe",
        "policy": POLICY,
        "compiler_policy": POLICY_VERSION,
        "parameters": {
            "k": int(k), "top_n": int(top_n),
            "rescue_no_candidates": bool(rescue_no_candidates),
            "rescue_table_k": int(rescue_table_k),
            "rescue_min_score": float(rescue_min_score),
            "allow_checkpoint_superset": bool(allow_checkpoint_superset),
        },
        "inputs": {
            "retrieval": _input_entry(retrieval_path, input_hashes["retrieval"]),
            "checkpoint": {
                **_input_entry(checkpoint_path, input_hashes["checkpoint"]),
                "run_signature": checkpoint_signature,
            },
            "control": {
                **_input_entry(control_path, input_hashes["control"]),
                "run_signature": control_signature,
            },
            "mask": _input_entry(mask_path, input_hashes["mask"]),
        },
        "semantic_inputs": semantic_inputs,
        "store": store_identity,
        "output": {
            "path": str(out_path), "sha256": _sha256(out_path),
            "run_signature": replay_signature, "rows": len(output),
        },
        "counts": {
            "rows": len(output), "target": len(target),
            "checkpoint_attempted": len(checkpoint_attempted),
            "attempted": len(attempted), "pending": counts["pending"],
            "ignored_checkpoint_attempted": len(ignored_checkpoint_attempted),
            "accepted": counts["accepted"], "rejected": counts["rejected"],
            "outside_mask": counts["outside_mask"],
        },
        "ignored_checkpoint_attempted_ids": ignored_checkpoint_attempted,
        "transitions": dict(sorted(transitions.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "accepted_ids": accepted_ids,
        "rejected_ids": rejected_ids,
        "pending_ids": pending_ids,
        "records": records,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(audit_path, audit)
    return audit


def _scope_attempted(attempted: set[int], target: set[int], *,
                     allow_checkpoint_superset: bool) -> tuple[set[int], list[int]]:
    unexpected = attempted - target
    if unexpected and not allow_checkpoint_superset:
        raise ValueError(
            f"checkpoint attempted IDs outside mask: {sorted(unexpected)[:10]}"
        )
    # Out-of-mask responses are never replayed. Their presence is recorded in
    # the audit only; every corresponding output row comes from the control.
    return attempted & target, sorted(unexpected)


def _saved_samples(row: dict, qid: int) -> list[str]:
    trace = row.get("selection_trace") or {}
    attempts = trace.get("attempts") or []
    if not isinstance(attempts, list) or not attempts:
        raise ValueError(f"id={qid}: completed trace has no attempts")
    ordered = sorted(attempts, key=lambda value: int(value.get("index", 0)))
    samples = []
    for expected, attempt in enumerate(ordered, 1):
        if int(attempt.get("index", 0)) != expected:
            raise ValueError(f"id={qid}: attempt indices are not contiguous")
        if attempt.get("raw_truncated"):
            raise ValueError(f"id={qid}: attempt {expected} raw response is truncated")
        raw = str(attempt.get("raw_response") or "")
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if str(attempt.get("raw_sha256") or "") != digest:
            raise ValueError(f"id={qid}: attempt {expected} raw response hash mismatch")
        samples.append(GenerationSample(
            raw,
            finish_reason=str(attempt.get("generation_finish_reason") or "unknown"),
            token_count=attempt.get("generation_tokens"),
            max_tokens=attempt.get("generation_max_tokens"),
        ))
    if int(trace.get("samples_received", len(samples))) != len(samples):
        raise ValueError(f"id={qid}: samples_received does not match attempts")
    return samples


def _semantic_identity() -> dict:
    package = Path(__file__).resolve().parents[1]
    paths = {
        "codegen/atomic_slots.py": package / "codegen" / "atomic_slots.py",
        "codegen/executor.py": package / "codegen" / "executor.py",
        "codegen/generate.py": package / "codegen" / "generate.py",
        "codegen/selection_v2.py": package / "codegen" / "selection_v2.py",
        "codegen/selection_v2_replay.py": package / "codegen" / "selection_v2_replay.py",
        "codegen/semantic.py": package / "codegen" / "semantic.py",
        "codegen/units.py": package / "codegen" / "units.py",
        "extraction/build_store.py": package / "extraction" / "build_store.py",
        "extraction/report_parser.py": package / "extraction" / "report_parser.py",
        "extraction/unit_policy.py": package / "extraction" / "unit_policy.py",
        "retrieval/shortlist.py": package / "retrieval" / "shortlist.py",
        "utils/viet_text.py": package / "utils" / "viet_text.py",
    }
    files = []
    for name, path in sorted(paths.items()):
        files.append({"name": name, "sha256": _sha256(path),
                      "bytes": path.stat().st_size})
    return {"manifest_sha256": _json_sha256(files), "files": files}


def _store_identity(store_dir: Path, retrieval: dict[int, dict]) -> dict:
    tickers = set()
    for rec in retrieval.values():
        for candidate in rec.get("candidates") or []:
            ticker = str(candidate.get("ticker") or "").strip().upper()
            if not ticker:
                ticker = str(candidate.get("report_id") or "").split("_", 1)[0].upper()
            if ticker:
                tickers.add(ticker)
    paths = [store_dir / "reports.parquet"]
    paths.extend(store_dir / "tables" / f"{ticker}.parquet"
                 for ticker in sorted(tickers))
    files = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"required replay store file missing: {path}")
        files.append({"path": path.relative_to(store_dir).as_posix(),
                      "sha256": _sha256(path), "bytes": path.stat().st_size})
    return {"path": str(store_dir), "manifest_sha256": _json_sha256(files),
            "files": files}


def _read_unique(path: Path, label: str) -> dict[int, dict]:
    rows = read_jsonl(path)
    out = {}
    for row in rows:
        qid = int(row["id"])
        if qid in out:
            raise ValueError(f"{label} has duplicate id={qid}")
        out[qid] = row
    if not out:
        raise ValueError(f"{label} is empty")
    return out


def _validate_same_universe(a: dict[int, dict], b: dict[int, dict], label: str) -> None:
    if set(a) != set(b):
        raise ValueError(f"{label} ID universes differ")
    mismatches = []
    for qid in a:
        qa = unicodedata.normalize("NFC", str(a[qid].get("question") or "")).strip()
        qb = unicodedata.normalize("NFC", str(b[qid].get("question") or "")).strip()
        if not qa or qa != qb:
            mismatches.append(qid)
    if mismatches:
        raise ValueError(f"{label} question mismatch: ids={mismatches[:10]}")


def _read_mask(path: Path) -> set[int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    values = raw.get("ids") if isinstance(raw, dict) else raw
    if not isinstance(values, list) or not values:
        raise ValueError("mask must contain a non-empty ids list")
    ids = [int(value) for value in values]
    if len(ids) != len(set(ids)):
        raise ValueError("mask has duplicate IDs")
    if isinstance(raw, dict) and raw.get("count") is not None \
            and int(raw["count"]) != len(ids):
        raise ValueError("mask count does not match ids")
    return set(ids)


def _one_signature(rows: dict[int, dict], label: str) -> str:
    signatures = {str(row.get("run_signature") or "").strip()
                  for row in rows.values()}
    if "" in signatures or len(signatures) != 1:
        raise ValueError(f"{label} must have exactly one non-empty run_signature")
    return next(iter(signatures))


def _signature(input_hashes: dict, semantic_sha: str, store_sha: str,
               checkpoint_signature: str, control_signature: str, *,
               k: int, top_n: int, rescue_no_candidates: bool,
               rescue_table_k: int, rescue_min_score: float,
               allow_checkpoint_superset: bool = False) -> str:
    return _json_sha256({
        "policy": POLICY, "compiler_policy": POLICY_VERSION,
        "inputs": input_hashes, "semantic_manifest_sha256": semantic_sha,
        "store_manifest_sha256": store_sha,
        "checkpoint_run_signature": checkpoint_signature,
        "control_run_signature": control_signature,
        "k": int(k), "top_n": int(top_n),
        "rescue_no_candidates": bool(rescue_no_candidates),
        "rescue_table_k": int(rescue_table_k),
        "rescue_min_score": float(rescue_min_score),
        "allow_checkpoint_superset": bool(allow_checkpoint_superset),
    })


def _input_entry(path: Path, digest: str) -> dict:
    return {"path": str(path), "sha256": digest}


def _json_sha256(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_path(a: Path, b: Path) -> bool:
    return a.resolve(strict=False) == b.resolve(strict=False)
