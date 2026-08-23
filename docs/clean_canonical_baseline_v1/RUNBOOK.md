# Clean canonical baseline v1 runbook

## 1. Audit the completed B1 run

From the repository root:

    python scripts/63_audit_b1_nf4_run.py --out artifacts/clean_v1/b1_nf4/analysis_recomputed.json

Required result: every integrity_checks value is true. This audit uses no answer
labels and must not be described as an accuracy evaluation.

The frozen artifact directory is artifacts/clean_v1/b1_nf4. Do not overwrite it
when developing B2 or rebuilding a payload.

## 2. Re-run local clean gates

    python -m pytest -p no:cacheprovider --basetemp artifacts/pytest_clean_v1 tests/test_canonical_metrics.py tests/test_clean_profile.py tests/test_clean_retrieval.py tests/test_clean_payload_builder.py tests/test_clean_nf4_runner.py tests/test_clean_b1_14b_nf4_notebook.py -q
    python -m compileall -q vifinqa scripts kaggle

Use an artifact-local basetemp because the repository-root pytest cache may have
host ACL problems.

## 3. Rebuild clean retrieval and B0 only when source changes

    python scripts/57_clean_retrieve_v2.py
    python scripts/60_run_clean_b0_v2.py
    python scripts/61_build_clean_submission.py --codegen artifacts/clean_v1/b0_results.jsonl --out-dir artifacts/clean_v1/submission_b0

Every route must keep clean_profile=clean, a retrieval_config_sha256, and a
metric_variants lexical fallback when metric_keys is empty.

## 4. Build the next schema-9 payload

    python scripts/59_make_clean_payload_v5.py --dry-run
    python scripts/59_make_clean_payload_v5.py --dataset-id lequangkhai5122005/vifinqa-clean-canonical-v1

The builder has no target-dir flag. Verify these manifest fields before upload:

- schema_version=9;
- runtime_profile=hf-bitsandbytes-nf4-v1;
- validation_profile=clean-codegen-select-v2-v2;
- default_model=Qwen/Qwen2.5-Coder-14B-Instruct;
- public_id_masks=false;
- official_derived_gold=false.

Upload a new version of the existing dataset:

    kaggle datasets version -p artifacts/clean_v1/kaggle_payload -m "schema9: Qwen 14B runtime NF4 canonical" --dir-mode zip

Do not use datasets create for the existing slug and do not append a trailing
dot to the command.

## 5. Run B1 on Kaggle

Import kaggle/vifinqa-clean-canonical-b1-14b-nf4.ipynb and attach exactly one
current schema-9 dataset. Enable a GPU and Internet unless the model is attached
as a Kaggle input.

Run all cells. The active path is base Qwen 14B plus runtime bitsandbytes NF4.
Do not install gptqmodel or autoawq and do not use an AWQ/GPTQ model ID.

Required handoff files:

- runtime_preflight_nf4.json;
- runtime_smoke_nf4.json;
- runtime_full_nf4.json;
- run_config_nf4.json;
- codegen_results_nf4.jsonl;
- codegen_audit_nf4.json;
- submission_manifest_nf4.json;
- submission_clean_nf4/submission.zip;
- the executed notebook or complete text log when available.

Only download/use submission.zip after the complete-LLM validator and archive
replay pass.

## 6. Resume rules

The full cell may resume only against the same output path and exact
run_signature. The signature binds model, payload, mode, sampling, seed, input
limit, quantization, batch grouping, checkpoint grouping, and limit.

Smoke and full intentionally have different signatures. Never copy smoke rows
into a full checkpoint.

## 7. Gate before any next private candidate

G3A/G3B evaluation is complete and frozen. Verify it before starting or
interpreting any G3C treatment:

    python scripts/69_g3b_build.py validate
    python scripts/73_freeze_g3_evaluation.py validate

Expected final freeze fingerprint:

    242f5b288350ba7b5728dd00bf262c38a69463cb86efd021663fb4f21ed8a877

Rules for every next candidate:

1. Select retrieval treatments and thresholds using G3B `primary_tune` only.
2. Freeze the exact candidate/config/prediction artifacts before promotion.
3. Evaluate `primary_locked + hard` once through promotion mode.
4. Use the 1,012 official records only for crash/invariant regression after the
   candidate is frozen.
5. Do not add per-ID fixes, masks, or public-derived overlays.
6. Pre-register source/config/model/payload/OOD/submission hashes before using a
   private slot.

G3C retrieval is now implemented and has its own operational source of truth:

    docs/g3c_qwen_retrieval/G3C_RUNBOOK.md

The full Qwen dev run and one-shot locked+hard promotion completed on
2026-08-24. R4 replicated positive retrieval gains and is frozen as P-B;
answer/execution accuracy were unchanged. Never rerun the consumed promotion
evaluator. Keep R4 fixed for the official crash/invariant audit and build a new
unopened source-derived holdout before selecting G3D planner changes.
