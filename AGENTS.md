# R2AI clean-canonical branch handoff

These instructions apply to the entire repository.

## Current source of truth

This checkout is the clean-canonical-baseline-v1 workstream. Root README.md,
RUNBOOK.md, and CLAUDE.md contain substantial legacy schema-8/P2.x history and
must not override the branch-specific files below:

- docs/clean_canonical_baseline_v1/README.md
- docs/clean_canonical_baseline_v1/RUNBOOK.md
- docs/clean_canonical_baseline_v1/B1_14B_NF4_RUN_ANALYSIS.md
- experiments/clean_canonical_baseline_v1/registry.json
- experiments/clean_canonical_baseline_v1/runs/b1_14b_nf4_2026-08-21.json

## Completed clean run

B1 completed on Kaggle with Qwen/Qwen2.5-Coder-14B-Instruct, Hugging Face, and
bitsandbytes runtime NF4. It is not a 7B or AWQ run. Preserve
artifacts/clean_v1/b1_nf4 as immutable run evidence.

Do not infer answer accuracy from status=ok, LLM acceptance, or the integrity
audit. No answer labels were used. Keep verified findings, interpretations, and
pending scientific claims separate.

## Active files

- config: configs/clean_canonical_baseline_v1/b1_select_v2_14b_nf4.json
- notebook: kaggle/vifinqa-clean-canonical-b1-14b-nf4.ipynb
- NF4 launcher: kaggle/kaggle_clean_codegen_nf4.py
- payload builder: scripts/59_make_clean_payload_v5.py
- output audit: scripts/63_audit_b1_nf4_run.py

Files under history directories are provenance only and are not runnable
instructions.
The schema-8 P2.x runner kaggle/kaggle_codegen.py and notebooks named
kaggle/vifinqa-codegen*.ipynb remain legacy paths and may mention AWQ. They are
not clean runtime entrypoints.


## Active G3C retrieval experiment

G3A/G3B and the G3 evaluator are complete and frozen. G3C implementation/local
verification completed on 2026-08-23. The first Kaggle attempt stopped in cell
2 before Qwen inference because Kaggle strips the upload-only
dataset-metadata.json sidecar. The transport-only v2 fix and the full real-Qwen
dev evaluation completed on 2026-08-24. R4 passed the frozen gate and the
one-shot locked+hard promotion evaluation. G3C is complete for retrieval;
answer/execution accuracy did not improve.

Use:

- docs/g3c_qwen_retrieval/G3C_RUNBOOK.md
- docs/g3c_qwen_retrieval/G3C_SESSION_IMPLEMENTATION_2026-08-23.md
- docs/g3c_qwen_retrieval/G3C_SESSION_KAGGLE_PAYLOAD_V2_FIX_2026-08-24.md
- docs/g3c_qwen_retrieval/G3C_SESSION_DEV_QWEN_RESULTS_2026-08-24.md
- docs/g3c_qwen_retrieval/G3C_SESSION_PROMOTION_RESULTS_2026-08-24.md
- configs/g3c_qwen_retrieval_v1.json
- experiments/g3c_qwen_retrieval_v1/dev_protocol_freeze_v2.json
- artifacts/g3c_v1/dev_payload_v2
- artifacts/g3c_v1/dev_qwen_results
- artifacts/g3c_v1/dev_local_eval/g3c_candidate_freeze.json
- artifacts/g3c_v1/promotion_payload
- artifacts/g3c_v1/promotion_qwen_results
- artifacts/g3c_v1/promotion_local_eval/PROMOTION_EVALUATION_OPENED.json
- artifacts/g3c_v1/promotion_local_eval/r4/g3b_evaluation.json
- kaggle/vifinqa-g3c-dev-qwen-retrieval.ipynb
- scripts/77_g3c_validate_gpu_results.py
- scripts/78_g3c_evaluate_stages.py
- scripts/79_g3c_select_freeze.py

The current protocol fingerprint is
af86c8ffc276cc0a92ceeb3cc0ddc3a7eeaa7b6a1e4430dc2cade8ac4c9621c5.
The frozen R4 candidate fingerprint is
1cb02ad5e436d95efe18db0f7d8bbcb2e47cb6f65291892b4a78002d32f8336a.
Do not rebuild, bypass, or tune either after seeing Qwen evidence. The fake
three-question R0-R4 run validates engineering contracts only and is never
scientific evidence. The promotion payload/evaluator has already consumed its
single permitted run. Never delete the open marker or rerun promotion.

Preserve experiments/g3c_qwen_retrieval_v1/dev_protocol_freeze.json and
artifacts/g3c_v1/dev_payload as immutable v1 failure provenance. They are not
active runtime inputs.


## Public-bias guard

Do not tune thresholds, add ID lists, or implement per-question fixes from the
1,012 official records. The source-derived G3 tune/locked benchmark is frozen.
The next engineering milestone is a post-freeze official 1,012-record
crash/schema/finiteness run for R4, without leaderboard tuning. The next
scientific milestone is G3D typed planning/cell grounding with R4 fixed and a
new unopened source-derived holdout. The opened G3B promotion set is
regression-only and must not select G3D.

The five private submissions must represent distinct preregistered hypotheses,
not five small score-driven edits.

## Engineering conventions

Keep datasets, retrieval, generation, validation, submission, configs, and
artifacts separate. Preserve deterministic seeds and hashes. Any new candidate
must use a new output directory and run signature; never overwrite B0 or B1.

Use pytest with -p no:cacheprovider and an artifacts-local --basetemp because the
repository-root .pytest_cache may have host ACL issues.
