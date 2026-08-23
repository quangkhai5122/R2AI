"""Run frozen B0 and G3B retrieval/submission metrics for G3C stages."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vifinqa.g3c.common import read_json, sha256_file, write_json
from vifinqa.g3c.freeze import load_candidate_freeze
from vifinqa.g3c.paired import build_paired_diagnostics
from vifinqa.g3c.validate import validate_gpu_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dev", "promotion"), required=True)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--gpu-results", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--questions")
    parser.add_argument("--candidate-freeze")
    parser.add_argument("--store-dir", default="artifacts/store")
    parser.add_argument("--corpus", default="data/g3b_v1/g3b_corpus.jsonl")
    parser.add_argument("--baseline-evaluation")
    parser.add_argument("--baseline-submission")
    args = parser.parse_args()

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "evaluation_index.json"
    if index_path.exists():
        existing = read_json(index_path)
        if existing.get("mode") != args.mode:
            raise SystemExit("existing evaluation index mode mismatch")
        print(json.dumps(existing, ensure_ascii=False, indent=2))
        return

    import_report = output / "gpu_import_validation.json"
    validation = validate_gpu_results(
        payload_dir=args.payload,
        result_dir=args.gpu_results,
        output_path=import_report,
        require_scientific=True,
        candidate_freeze_path=args.candidate_freeze,
    )
    if validation["mode"] != args.mode:
        raise SystemExit("requested mode/result mode mismatch")
    gpu = read_json(
        Path(args.gpu_results) / "g3c_gpu_result_manifest.json"
    )
    questions = args.questions or (
        "data/g3b_v1/g3b_dev_questions.jsonl"
        if args.mode == "dev"
        else "data/g3b_v1/g3b_promotion_questions.jsonl"
    )
    if args.mode == "promotion":
        if not args.candidate_freeze:
            raise SystemExit("promotion evaluation requires --candidate-freeze")
        g3c_freeze = load_candidate_freeze(args.candidate_freeze)
        stages = [g3c_freeze["selected_stage"]]
    else:
        stages = list(gpu["stages_written"])

    index = {
        "schema_version": "g3c_local_evaluation_index_v1",
        "mode": args.mode,
        "gpu_import_validation_sha256": sha256_file(import_report),
        "stages": {},
    }
    for stage in stages:
        stage_dir = output / stage.lower()
        stage_dir.mkdir(parents=True, exist_ok=True)
        retrieval = (
            Path(args.gpu_results)
            / gpu["stage_artifacts"][stage]["path"]
        )
        codegen = stage_dir / "b0_codegen.jsonl"
        submission = stage_dir / "submission"
        evaluation = stage_dir / "g3b_evaluation.json"
        _run([
            sys.executable, str(ROOT / "scripts/60_run_clean_b0_v2.py"),
            "--retrieval", str(retrieval),
            "--store-dir", args.store_dir,
            "--out", str(codegen),
            "--checkpoint-every", "10",
        ])
        _run([
            sys.executable, str(ROOT / "scripts/74_g3b_build_submission.py"),
            "--retrieval", str(retrieval),
            "--codegen", str(codegen),
            "--questions", questions,
            "--store-dir", args.store_dir,
            "--out-dir", str(submission),
            "--sub-k", "5",
        ])
        evaluation_command = [
            sys.executable, str(ROOT / "scripts/72_g3b_evaluate.py"),
            "--policy-mode", args.mode,
            "--evidence-mode", "end_to_end",
            "--submission", str(submission),
            "--out", str(evaluation),
        ]
        submission_freeze = None
        promotion_marker = None
        if args.mode == "promotion":
            submission_freeze = stage_dir / "g3b_submission_freeze.json"
            _run([
                sys.executable,
                str(ROOT / "scripts/71_g3b_freeze_candidate.py"),
                "--candidate-name",
                f"{g3c_freeze['candidate_name']}-exact-submission",
                "--submission", str(submission),
                "--out", str(submission_freeze),
            ])
            evaluation_command.extend([
                "--candidate-freeze", str(submission_freeze)
            ])
            promotion_marker = output / "PROMOTION_EVALUATION_OPENED.json"
            if promotion_marker.exists():
                raise SystemExit(
                    "promotion evaluation marker already exists; refusing "
                    "to open the locked evaluator twice"
                )
            write_json(promotion_marker, {
                "schema_version": "g3c_promotion_open_marker_v1",
                "candidate_fingerprint": g3c_freeze[
                    "candidate_fingerprint"
                ],
                "selected_stage": stage,
                "submission_freeze_sha256": sha256_file(
                    submission_freeze
                ),
            })
        _run(evaluation_command)
        index["stages"][stage] = {
            "retrieval": str(retrieval.resolve()),
            "retrieval_sha256": sha256_file(retrieval),
            "codegen": str(codegen.resolve()),
            "codegen_sha256": sha256_file(codegen),
            "submission": str(submission.resolve()),
            "evaluation": str(evaluation.resolve()),
            "evaluation_sha256": sha256_file(evaluation),
            "g3b_submission_freeze": (
                str(submission_freeze.resolve())
                if submission_freeze is not None else None
            ),
            "promotion_open_marker": (
                str(promotion_marker.resolve())
                if promotion_marker is not None else None
            ),
        }

    if args.mode == "dev":
        baseline_evaluation = Path(
            index["stages"]["R0"]["evaluation"]
        )
        baseline_submission = Path(
            index["stages"]["R0"]["submission"]
        )
    else:
        if not args.baseline_evaluation or not args.baseline_submission:
            raise SystemExit(
                "promotion paired diagnostics require baseline paths"
            )
        baseline_evaluation = Path(args.baseline_evaluation)
        baseline_submission = Path(args.baseline_submission)
    for stage in stages:
        if stage == "R0":
            continue
        paired = output / stage.lower() / "paired_vs_r0.json"
        build_paired_diagnostics(
            baseline_evaluation_path=baseline_evaluation,
            candidate_evaluation_path=index["stages"][stage]["evaluation"],
            baseline_submission_dir=baseline_submission,
            candidate_submission_dir=index["stages"][stage]["submission"],
            corpus_path=args.corpus,
            policy_mode=args.mode,
            output_path=paired,
        )
        index["stages"][stage]["paired_diagnostics"] = str(
            paired.resolve()
        )
        index["stages"][stage]["paired_diagnostics_sha256"] = (
            sha256_file(paired)
        )
    write_json(index_path, index)
    print(json.dumps({
        "mode": args.mode,
        "stages": list(index["stages"]),
        "index": str(index_path),
    }, ensure_ascii=False, indent=2))


def _run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
