"""Fail-closed structural QA for clean Selection-v2 checkpoints."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path


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
        raise SystemExit("codegen IDs/order do not match clean retrieval")

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
            raise SystemExit(f"question text mismatch for id={qid}")
        try:
            answer = float(result.get("answer"))
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"non-numeric answer for id={qid}") from exc
        if not math.isfinite(answer):
            raise SystemExit(f"non-finite answer for id={qid}: {answer!r}")
        if not isinstance(result.get("pandas_query"), str):
            raise SystemExit(f"missing pandas_query string for id={qid}")
        if not isinstance(result.get("used_vars"), list):
            raise SystemExit(f"missing used_vars list for id={qid}")
        signature = str(result.get("run_signature") or "")
        if not signature:
            raise SystemExit(f"missing run_signature for id={qid}")
        signatures.add(signature)
        source_counts[str(result.get("source") or "missing")] += 1
        status_counts[str(result.get("status") or "missing")] += 1

        is_completed = result.get("llm_attempt_status") == "completed"
        completed += int(is_completed)
        trace = result.get("selection_trace")
        if is_completed:
            if not isinstance(trace, dict) or trace.get("schema_version") != 1:
                raise SystemExit(f"invalid Selection-v2 trace for id={qid}")
            outcome = str(trace.get("outcome") or "missing")
            outcome_counts[outcome] += 1
            accepted += int(outcome == "accepted")
            rejection_counts.update(trace.get("rejection_counts") or {})
        if require_complete_llm and not is_completed:
            raise SystemExit(
                f"LLM checkpoint is incomplete at id={qid}; rerun the full "
                "generation cell with resume enabled"
            )

    if len(signatures) != 1:
        raise SystemExit(f"mixed run signatures: {sorted(signatures)}")
    return {
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
