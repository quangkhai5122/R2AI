"""Recompute integrity and behavior metrics for a clean B1 NF4 Kaggle run.

This audit deliberately uses no answer labels. It measures provenance,
completeness, execution coverage, fallback behavior, and reproducibility; it
does not estimate competition accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


ANALYSIS_VERSION = 1


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def counter_dict(values: Iterable[Any]) -> dict[str, int]:
    counts = Counter(str(value) for value in values)
    return dict(sorted(counts.items()))


def rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def same_number(left: Any, right: Any) -> bool:
    return (
        finite_number(left)
        and finite_number(right)
        and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)
    )


def unique_index(
    rows: list[dict[str, Any]], label: str
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        qid = row.get("id")
        if not isinstance(qid, int):
            raise ValueError(f"{label}: non-integer id {qid!r}")
        if qid in indexed:
            raise ValueError(f"{label}: duplicate id {qid}")
        indexed[qid] = row
    return indexed


def nested(row: dict[str, Any], field: str) -> Any:
    value: Any = row
    for part in field.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def cross_tab(rows: Iterable[dict[str, Any]], *fields: str) -> dict[str, int]:
    return counter_dict(
        " | ".join(str(nested(row, field)) for field in fields) for row in rows
    )


def segment_profile(
    retrieval_rows: list[dict[str, Any]],
    codegen: dict[int, dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], str],
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for retrieval in retrieval_rows:
        buckets[str(key_fn(retrieval))].append(codegen[int(retrieval["id"])])
    profile: dict[str, dict[str, Any]] = {}
    for key, rows in sorted(buckets.items()):
        total = len(rows)
        ok = sum(row.get("status") == "ok" for row in rows)
        accepted = sum(nested(row, "selection_trace.outcome") == "accepted" for row in rows)
        no_candidates = sum(
            nested(row, "selection_trace.outcome") == "no_candidates" for row in rows
        )
        llm_used = sum(row.get("source") == "llm_select_v2" for row in rows)
        profile[key] = {
            "records": total,
            "ok": ok,
            "ok_rate": rate(ok, total),
            "accepted": accepted,
            "accepted_rate": rate(accepted, total),
            "no_candidates": no_candidates,
            "no_candidates_rate": rate(no_candidates, total),
            "llm_used": llm_used,
            "llm_used_rate": rate(llm_used, total),
        }
    return profile


def audit(run_dir: Path, retrieval_path: Path, b0_path: Path) -> dict[str, Any]:
    codegen_path = run_dir / "codegen_results_nf4.jsonl"
    smoke_path = run_dir / "codegen_smoke_nf4.jsonl"
    kaggle_audit_path = run_dir / "codegen_audit_nf4.json"
    local_audit_path = run_dir / "local_audit.json"
    handoff_path = run_dir / "submission_manifest_nf4.json"
    results_path = run_dir / "submission_clean_nf4" / "results.json"
    zip_path = run_dir / "submission_clean_nf4" / "submission.zip"
    runtime_paths = {
        name: run_dir / f"runtime_{name}_nf4.json"
        for name in ("preflight", "smoke", "full")
    }
    required = [
        codegen_path,
        smoke_path,
        kaggle_audit_path,
        local_audit_path,
        handoff_path,
        results_path,
        zip_path,
        retrieval_path,
        b0_path,
        *runtime_paths.values(),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required artifacts: {missing}")

    codegen_rows = read_jsonl(codegen_path)
    smoke_rows = read_jsonl(smoke_path)
    retrieval_rows = read_jsonl(retrieval_path)
    b0_rows = read_jsonl(b0_path)
    submission_rows = read_json(results_path)
    if not isinstance(submission_rows, list):
        raise ValueError(f"{results_path}: expected a JSON array")

    codegen = unique_index(codegen_rows, "codegen")
    smoke = unique_index(smoke_rows, "smoke")
    retrieval = unique_index(retrieval_rows, "retrieval")
    b0 = unique_index(b0_rows, "B0")
    submission = unique_index(submission_rows, "submission")
    common_ids = sorted(set(codegen) & set(retrieval))

    codegen_hash = sha256(codegen_path)
    submission_hash = sha256(zip_path)
    kaggle_audit = read_json(kaggle_audit_path)
    local_audit = read_json(local_audit_path)
    handoff = read_json(handoff_path)
    runtimes = {name: read_json(path) for name, path in runtime_paths.items()}

    with zipfile.ZipFile(zip_path) as archive:
        archive_names = archive.namelist()
        archive_rows = json.loads(archive.read("results.json"))
    archive_index = unique_index(archive_rows, "submission archive")
    unsafe_members = [
        name
        for name in archive_names
        if Path(name).is_absolute() or ".." in Path(name).parts
    ]

    question_mismatches = [
        qid
        for qid in common_ids
        if str(codegen[qid].get("question")) != str(retrieval[qid].get("question"))
    ]
    submission_answer_mismatches = [
        qid
        for qid in sorted(set(submission) & set(codegen))
        if not same_number(submission[qid].get("answer"), codegen[qid].get("answer"))
    ]
    archive_answer_mismatches = [
        qid
        for qid in sorted(set(archive_index) & set(codegen))
        if not same_number(archive_index[qid].get("answer"), codegen[qid].get("answer"))
    ]
    signatures = sorted({str(row.get("run_signature") or "") for row in codegen_rows})

    accepted_rows = [
        row for row in codegen_rows if nested(row, "selection_trace.outcome") == "accepted"
    ]
    rejected_rows = [
        row for row in codegen_rows if nested(row, "selection_trace.outcome") == "rejected"
    ]
    no_candidate_rows = [
        row
        for row in codegen_rows
        if nested(row, "selection_trace.outcome") == "no_candidates"
    ]

    attempt_stages: list[str] = []
    attempt_reasons: list[str] = []
    finish_reasons: list[str] = []
    rejection_counts: Counter[str] = Counter()
    grounding_families: Counter[str] = Counter()
    candidate_counts: list[int] = []
    total_samples = 0
    total_evaluated = 0
    generation_truncated = 0
    semantic_incomplete = 0
    for row in codegen_rows:
        trace = row.get("selection_trace") or {}
        shortlist = trace.get("shortlist") or {}
        candidate_counts.append(int(trace.get("candidate_count") or 0))
        total_samples += int(trace.get("samples_received") or 0)
        total_evaluated += int(trace.get("attempts_evaluated") or 0)
        semantic_incomplete += int(not bool(shortlist.get("semantic_fact_complete")))
        for reason, count in (trace.get("rejection_counts") or {}).items():
            rejection_counts[str(reason)] += int(count)
        for reason, count in (
            shortlist.get("metric_grounding_rejections") or {}
        ).items():
            grounding_families[str(reason).split(":", 1)[0]] += int(count)
        for attempt in trace.get("attempts") or []:
            attempt_stages.append(str(attempt.get("stage")))
            attempt_reasons.append(str(attempt.get("reason_code") or "none"))
            finish_reasons.append(str(attempt.get("generation_finish_reason")))
            generation_truncated += int(bool(attempt.get("generation_truncated")))

    transitions: list[str] = []
    both_ok_changes = 0
    for qid in sorted(set(b0) & set(codegen)):
        before = b0[qid]
        after = codegen[qid]
        transitions.append(f"{before.get('status')} -> {after.get('status')}")
        if before.get("status") == after.get("status") == "ok":
            both_ok_changes += int(
                not same_number(before.get("answer"), after.get("answer"))
            )

    smoke_comparison = []
    for qid in sorted(set(smoke) & set(codegen)):
        smoke_comparison.append(
            {
                "id": qid,
                "answer_equal": same_number(
                    smoke[qid].get("answer"), codegen[qid].get("answer")
                ),
                "smoke_source": smoke[qid].get("source"),
                "full_source": codegen[qid].get("source"),
                "smoke_outcome": nested(smoke[qid], "selection_trace.outcome"),
                "full_outcome": nested(codegen[qid], "selection_trace.outcome"),
            }
        )

    b0_ok = sum(row.get("status") == "ok" for row in b0_rows)
    b1_ok = sum(row.get("status") == "ok" for row in codegen_rows)
    checks = {
        "id_sets_equal": set(codegen) == set(retrieval) == set(b0) == set(submission),
        "question_alignment": not question_mismatches,
        "submission_answers_match_codegen": not submission_answer_mismatches,
        "archive_answers_match_codegen": not archive_answer_mismatches,
        "all_answers_finite": all(finite_number(row.get("answer")) for row in codegen_rows),
        "all_llm_attempts_completed": all(
            row.get("llm_attempt_status") == "completed" for row in codegen_rows
        ),
        "single_nonempty_run_signature": len(signatures) == 1 and bool(signatures[0]),
        "codegen_hash_matches_audit": codegen_hash == kaggle_audit.get("codegen_sha256"),
        "codegen_hash_matches_handoff": codegen_hash == handoff.get("codegen_sha256"),
        "submission_hash_matches_handoff": (
            submission_hash == handoff.get("submission_sha256")
        ),
        "local_audit_matches_kaggle_audit": local_audit == kaggle_audit,
        "archive_is_path_safe": not unsafe_members,
        "archive_member_count_matches_handoff": (
            len(archive_names) == handoff.get("archive_members")
        ),
        "runtime_model_consistent": len(
            {str(report.get("model")) for report in runtimes.values()}
        )
        == 1,
        "runtime_profile_consistent": len(
            {str(report.get("runtime_profile")) for report in runtimes.values()}
        )
        == 1,
        "runtime_payload_hash_consistent": (
            runtimes["smoke"].get("payload_manifest_sha256")
            == runtimes["full"].get("payload_manifest_sha256")
        ),
    }

    return {
        "analysis_version": ANALYSIS_VERSION,
        "scope": {
            "run_dir": str(run_dir.resolve()),
            "retrieval": str(retrieval_path.resolve()),
            "b0": str(b0_path.resolve()),
            "labels_used": False,
            "accuracy_claim_supported": False,
        },
        "provenance": {
            "model": runtimes["full"].get("model"),
            "model_revision_observed": runtimes["preflight"].get(
                "model_revision_observed"
            ),
            "runtime_profile": runtimes["full"].get("runtime_profile"),
            "quantization": runtimes["full"].get("quantization"),
            "packages": runtimes["full"].get("packages"),
            "cuda": runtimes["full"].get("cuda"),
            "gpus": runtimes["full"].get("gpus"),
            "payload_manifest_sha256": runtimes["full"].get(
                "payload_manifest_sha256"
            ),
            "run_signature": signatures[0] if len(signatures) == 1 else signatures,
            "codegen_sha256": codegen_hash,
            "submission_sha256": submission_hash,
        },
        "integrity_checks": checks,
        "integrity_details": {
            "codegen_records": len(codegen_rows),
            "retrieval_records": len(retrieval_rows),
            "b0_records": len(b0_rows),
            "submission_records": len(submission_rows),
            "archive_records": len(archive_rows),
            "archive_members": len(archive_names),
            "question_mismatch_ids": question_mismatches,
            "submission_answer_mismatch_ids": submission_answer_mismatches,
            "archive_answer_mismatch_ids": archive_answer_mismatches,
            "unsafe_archive_members": unsafe_members,
        },
        "behavior": {
            "status_counts": counter_dict(row.get("status") for row in codegen_rows),
            "source_counts": counter_dict(row.get("source") for row in codegen_rows),
            "selection_outcomes": counter_dict(
                nested(row, "selection_trace.outcome") for row in codegen_rows
            ),
            "outcome_by_source": cross_tab(
                codegen_rows, "selection_trace.outcome", "source"
            ),
            "outcome_by_status": cross_tab(
                codegen_rows, "selection_trace.outcome", "status"
            ),
            "accepted_by_final_source": counter_dict(
                row.get("source") for row in accepted_rows
            ),
            "rejected_by_final_source": counter_dict(
                row.get("source") for row in rejected_rows
            ),
            "no_candidates_by_final_status": counter_dict(
                row.get("status") for row in no_candidate_rows
            ),
            "arbitration_reasons": counter_dict(
                (row.get("arbitration") or {}).get("reason", "none")
                for row in codegen_rows
            ),
            "ok_rate": rate(b1_ok, len(codegen_rows)),
            "selection_accept_rate": rate(len(accepted_rows), len(codegen_rows)),
            "no_candidates_rate": rate(len(no_candidate_rows), len(codegen_rows)),
            "llm_final_use_rate": rate(
                sum(row.get("source") == "llm_select_v2" for row in codegen_rows),
                len(codegen_rows),
            ),
            "zero_answers": sum(
                finite_number(row.get("answer")) and float(row["answer"]) == 0.0
                for row in codegen_rows
            ),
            "failed_zero_answers": sum(
                row.get("status") == "failed"
                and finite_number(row.get("answer"))
                and float(row["answer"]) == 0.0
                for row in codegen_rows
            ),
        },
        "generation": {
            "samples_received_total": total_samples,
            "attempts_evaluated_total": total_evaluated,
            "attempt_stage_counts": counter_dict(attempt_stages),
            "attempt_reason_counts": counter_dict(attempt_reasons),
            "finish_reason_counts": counter_dict(finish_reasons),
            "generation_truncated_attempts": generation_truncated,
            "selection_rejections_recomputed": dict(sorted(rejection_counts.items())),
            "selection_rejections_stored": kaggle_audit.get("selection_rejections"),
        },
        "shortlist": {
            "candidate_count_distribution": counter_dict(candidate_counts),
            "metric_grounding_rejection_families": dict(
                sorted(grounding_families.items())
            ),
            "semantic_incomplete_records": semantic_incomplete,
        },
        "b0_comparison": {
            "b0_status_counts": counter_dict(row.get("status") for row in b0_rows),
            "b1_status_counts": counter_dict(row.get("status") for row in codegen_rows),
            "status_transitions": counter_dict(transitions),
            "both_ok_answer_changes": both_ok_changes,
            "ok_coverage_delta_records": b1_ok - b0_ok,
            "ok_coverage_delta_rate": round(
                rate(b1_ok, len(codegen_rows)) - rate(b0_ok, len(b0_rows)), 6
            ),
        },
        "smoke_vs_full": {
            "overlap_records": len(smoke_comparison),
            "answer_matches": sum(row["answer_equal"] for row in smoke_comparison),
            "source_matches": sum(
                row["smoke_source"] == row["full_source"] for row in smoke_comparison
            ),
            "outcome_matches": sum(
                row["smoke_outcome"] == row["full_outcome"]
                for row in smoke_comparison
            ),
            "records": smoke_comparison,
        },
        "segments": {
            "output_type": segment_profile(
                retrieval_rows,
                codegen,
                lambda row: str(
                    (row.get("route") or {}).get("output_type", "missing")
                ),
            ),
            "plan_op": segment_profile(
                retrieval_rows,
                codegen,
                lambda row: str(
                    ((row.get("route") or {}).get("plan") or {}).get(
                        "op", "missing"
                    )
                ),
            ),
            "canonical_coverage": segment_profile(
                retrieval_rows,
                codegen,
                lambda row: (
                    "has_metric_key"
                    if (row.get("route") or {}).get("metric_keys")
                    else "canonical_miss"
                ),
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=Path("artifacts/clean_v1/b1_nf4")
    )
    parser.add_argument(
        "--retrieval", type=Path, default=Path("artifacts/clean_v1/retrieval.jsonl")
    )
    parser.add_argument(
        "--b0", type=Path, default=Path("artifacts/clean_v1/b0_results.jsonl")
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit(args.run_dir, args.retrieval, args.b0)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
        print(f"audit -> {args.out}")
    print(rendered)


if __name__ == "__main__":
    main()
