"""Fail-closed structural QA for clean Selection-v2 checkpoints.

Selection-v2 emits ``selection_trace.schema_version == 2`` and
``selection_trace.mode == 'select_v2'``.  This validator intentionally checks
that producer contract rather than the unrelated Selection-v1 trace schema.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

TRACE_SCHEMA_VERSION = 2
TRACE_MODE = "select_v2"
TRACE_OUTCOMES = {"accepted", "rejected", "no_samples", "no_candidates"}


def _read_jsonl(path: Path, label: str) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"{label} line {line_number} is invalid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise SystemExit(f"{label} line {line_number} is not an object")
            rows.append(row)
    if not rows:
        raise SystemExit(f"{label} is empty: {path}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_error(qid, message: str, result: dict) -> SystemExit:
    summary = {
        "id": qid,
        "status": result.get("status"),
        "source": result.get("source"),
        "llm_attempt_status": result.get("llm_attempt_status"),
        "trace_schema": (result.get("selection_trace") or {}).get("schema_version")
        if isinstance(result.get("selection_trace"), dict) else None,
        "trace_mode": (result.get("selection_trace") or {}).get("mode")
        if isinstance(result.get("selection_trace"), dict) else None,
        "trace_outcome": (result.get("selection_trace") or {}).get("outcome")
        if isinstance(result.get("selection_trace"), dict) else None,
    }
    return SystemExit(
        f"codegen validation failed for id={qid}: {message}; "
        f"record_summary={json.dumps(summary, ensure_ascii=False, sort_keys=True)}"
    )


def validate_codegen(
    retrieval_path: Path,
    codegen_path: Path,
    *,
    expected_count: int = 0,
    require_complete_llm: bool = False,
) -> dict:
    retrieval = _read_jsonl(retrieval_path, "retrieval")
    results = _read_jsonl(codegen_path, "codegen")
    expected = retrieval[:expected_count] if expected_count else retrieval
    if len(results) != len(expected):
        raise SystemExit(
            f"codegen row count={len(results)}, expected={len(expected)}"
        )
    expected_ids = [row.get("id") for row in expected]
    result_ids = [row.get("id") for row in results]
    if len(set(result_ids)) != len(result_ids):
        raise SystemExit("codegen contains duplicate IDs")
    if result_ids != expected_ids:
        raise SystemExit(
            "codegen IDs/order do not match clean retrieval; "
            f"expected_prefix={expected_ids[:5]}, actual_prefix={result_ids[:5]}"
        )

    signatures = set()
    source_counts: collections.Counter[str] = collections.Counter()
    status_counts: collections.Counter[str] = collections.Counter()
    outcome_counts: collections.Counter[str] = collections.Counter()
    rejection_counts: collections.Counter[str] = collections.Counter()
    completed = 0
    accepted = 0

    try:
        from tqdm.auto import tqdm
        iterator = tqdm(
            zip(expected, results), total=len(results), desc="validate codegen",
            unit="record", dynamic_ncols=True,
        )
    except ImportError:
        iterator = zip(expected, results)

    for retrieval_row, result in iterator:
        qid = result.get("id")
        if str(result.get("question", "")).strip() != str(
            retrieval_row.get("question", "")
        ).strip():
            raise _record_error(qid, "question text mismatch", result)
        try:
            answer = float(result.get("answer"))
        except (TypeError, ValueError) as exc:
            raise _record_error(qid, "answer is not numeric", result) from exc
        if not math.isfinite(answer):
            raise _record_error(qid, f"answer is not finite: {answer!r}", result)
        if not isinstance(result.get("pandas_query"), str):
            raise _record_error(qid, "missing pandas_query string", result)
        if not isinstance(result.get("used_vars"), list):
            raise _record_error(qid, "missing used_vars list", result)
        signature = str(result.get("run_signature") or "")
        if not signature:
            raise _record_error(qid, "missing run_signature", result)
        signatures.add(signature)
        source_counts[str(result.get("source") or "missing")] += 1
        status_counts[str(result.get("status") or "missing")] += 1

        is_completed = result.get("llm_attempt_status") == "completed"
        completed += int(is_completed)
        trace = result.get("selection_trace")
        if is_completed:
            if not isinstance(trace, dict):
                raise _record_error(qid, "completed attempt has no trace object", result)
            if trace.get("schema_version") != TRACE_SCHEMA_VERSION:
                raise _record_error(
                    qid,
                    f"trace schema must be {TRACE_SCHEMA_VERSION}, got "
                    f"{trace.get('schema_version')!r}",
                    result,
                )
            if trace.get("mode") != TRACE_MODE:
                raise _record_error(
                    qid,
                    f"trace mode must be {TRACE_MODE!r}, got {trace.get('mode')!r}",
                    result,
                )
            outcome = str(trace.get("outcome") or "")
            if outcome not in TRACE_OUTCOMES:
                raise _record_error(qid, f"invalid trace outcome {outcome!r}", result)
            if not isinstance(trace.get("attempts"), list):
                raise _record_error(qid, "trace attempts must be a list", result)
            outcome_counts[outcome] += 1
            accepted += int(outcome == "accepted")
            rejection_counts.update(trace.get("rejection_counts") or {})
        if require_complete_llm and not is_completed:
            raise _record_error(
                qid,
                "LLM checkpoint is incomplete; rerun generation with exact-signature resume",
                result,
            )

    if len(signatures) != 1:
        raise SystemExit(f"mixed run signatures: {sorted(signatures)}")
    return {
        "validator_profile": "clean-codegen-select-v2-v2",
        "retrieval_records": len(retrieval),
        "validated_records": len(results),
        "llm_completed": completed,
        "llm_accepted": accepted,
        "source_counts": dict(sorted(source_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "selection_outcomes": dict(sorted(outcome_counts.items())),
        "selection_rejections": dict(sorted(rejection_counts.items())),
        "run_signature": next(iter(signatures)),
        "codegen_sha256": _sha256(codegen_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--codegen", required=True)
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument("--require-complete-llm", action="store_true")
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    payload = Path(args.payload)
    report = validate_codegen(
        payload / "retrieval.jsonl",
        Path(args.codegen),
        expected_count=args.expected_count,
        require_complete_llm=args.require_complete_llm,
    )
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(f"audit report -> {target}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
