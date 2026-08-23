# G3C Qwen Retrieval Runbook

Status date: 2026-08-24

Current status: G3C dev and one-shot promotion completed on 2026-08-24. R4
passed the dev gate and improved every retrieval metric on the frozen
`primary_locked + hard` promotion set. Answer/execution accuracy was unchanged.
The promotion evaluator is consumed and must never be rerun.

The first Kaggle attempt stopped in cell 2 before any Qwen model load or
retrieval inference. Kaggle consumed dataset-metadata.json as an upload control
file and did not mount it under /kaggle/input, while payload schema v1 required
it as a core runtime file. Schema v2 classifies it as a hash-bound upload
sidecar: it is validated when present locally and may be absent after Kaggle
mounting. All 247 core files remain mandatory with exact size and SHA-256.

This is the operational source of truth for G3C. The clean B1 runbook remains
the source of truth for B1 and must not be used as a G3C runtime recipe.

Measured dev analysis and the exact claim boundary are recorded in:

    docs/g3c_qwen_retrieval/G3C_SESSION_DEV_QWEN_RESULTS_2026-08-24.md

Measured promotion analysis and the next-stage decision are recorded in:

    docs/g3c_qwen_retrieval/G3C_SESSION_PROMOTION_RESULTS_2026-08-24.md

## 1. Scientific boundary

G3C is a retrieval-only ablation. The deterministic B0 answer stack,
planner/compiler, Selection-v2, arbitration, G3A/G3B data, evaluator, and
promotion policy remain frozen.

The mandatory rules are:

1. Use only G3B primary_tune to compare R0 through R4 and select a stage.
2. Never place G3B gold, evaluator data, review records, corpus records, or the
   1,012 official questions in the dev GPU payload.
3. Do not use IDs, family labels, per-question fixes, or promotion outcomes to
   form leaf queries or tune thresholds.
4. Freeze exactly one passing dev stage before building a promotion payload.
5. Run G3C on primary_locked plus hard exactly once.
6. Record answer and execution metrics, but do not optimize them during G3C.
7. A fake-backend or limited run is engineering evidence only.

The frozen G3 evaluation fingerprint is:

    242f5b288350ba7b5728dd00bf262c38a69463cb86efd021663fb4f21ed8a877

The frozen G3C dev protocol is:

    config SHA-256:
    a691497b9581136de2dc20adedca06868a0e791aee5c8fb9b5c3f657f5b9ded1

    behavior tree SHA-256:
    f1f99c3189393ccd092784b3cd756b6989fe928335f1693255c7da831506e0b5

    protocol fingerprint:
    af86c8ffc276cc0a92ceeb3cc0ddc3a7eeaa7b6a1e4430dc2cade8ac4c9621c5

Any change to a bound retrieval, answer, evaluation, notebook, or runtime file
causes protocol validation and payload building to fail. That failure is a
guard, not a cleanup task.

## 2. Stage ladder

- R0: byte-preserved frozen canonical/BM25 retrieval control.
- R0L: label-blind atomic-leaf lexical retrieval plus the same hard report
  guards. This separates the effect of better leaf query formation from Qwen.
- R1: R0 plus per-leaf lexical and normalized Qwen3 dense rankings, fused by
  reciprocal-rank fusion.
- R2: Qwen3 reranking of a bounded R1 union per leaf.
- R3: R2 with a two-table per-leaf quota, deduplicated at table level while
  preserving global relevance order.
- R4: bounded row lexical prefilter followed by Qwen3 row reranking, then
  actual table reordering with the R3 coverage constraint.

All non-R0 candidates must remain within exact ticker, report year, report ID,
and scope guards derived without labels.

## 3. Pre-registered dev gate

A non-R0 stage is eligible only if every check passes against dev R0:

- Leaf Recall@5 delta is at least +0.02.
- FullPlanCoverage delta is at least +0.03.
- DOCS F2 regression is no worse than -0.01.
- TABLES F2 regression is no worse than -0.01.
- hard-constraint violations equal zero.
- evaluation integrity passes.

If multiple stages pass, select by:

1. larger FullPlanCoverage delta;
2. larger Leaf Recall@5 delta;
3. larger TABLES F2;
4. lower cumulative stage runtime;
5. lexical stage name as the final deterministic tie break.

If no stage passes, stop. Do not weaken a gate after seeing dev results and do
not build a promotion payload.

## 4. Runtime contract

The model/runtime lock is:

- Qwen/Qwen3-Embedding-4B at revision
  5cf2132abc99cad020ac570b19d031efec650f2b;
- Qwen/Qwen3-Reranker-4B at revision
  22e683669bc0f0bd69640a1354a6d0aebcfeede5;
- matching tokenizer revisions;
- full 2560-dimensional normalized embeddings with official last-token
  pooling;
- official CausalLM yes/no reranker scoring, not sequence classification;
- Transformers 4.53.3, Accelerate 1.8.1, Safetensors 0.5.3, TQDM 4.67.1;
- FP16, SDPA, no quantization;
- sequential model loading;
- embedding batch size 4 and reranker batch size 2;
- table reranker limit 768 tokens and row reranker limit 384 tokens;
- use_cache disabled for the reranker forward pass.

Use a Kaggle GPU with at least 16 GB memory. A T4 is the preferred conservative
target for this contract; a P100 can be tried without changing the config.
Enable Internet so the exact Hugging Face revisions can be downloaded.

Do not change batch size after an OOM and call it the same experiment. Save the
failure log. A changed batch/runtime contract is a new protocol and must be
frozen before its first real run.

## 5. Local and Kaggle responsibility split

Local performs:

- G3/G3B freeze validation;
- label-blind leaf audits;
- protocol freeze validation;
- stripped payload build and validation;
- imported GPU-result validation;
- deterministic B0 codegen and offline submission construction;
- G3B retrieval, submission, answer, execution, family, OOD, and paired
  evaluation;
- dev selection and candidate freeze;
- the one-shot promotion evaluator.

Kaggle GPU performs only:

- passage/query embedding;
- dense ranking;
- bounded table and row reranker scoring;
- R0 through R4 retrieval artifact construction for dev;
- R0 plus the single selected stage for promotion;
- cache/result manifest creation and self-validation.

The Kaggle payload contains id and question only for question rows. It contains
the store, R0 retrieval, runtime code, config, and protocol freeze. It does not
contain G3B gold, evaluator files, the G3B corpus, manual reviews, or official
public questions.

## 6. Artifact layout

Inputs and frozen contracts:

    configs/g3c_qwen_retrieval_v1.json
    experiments/g3_evaluation_v1/g3_evaluation_freeze.json
    experiments/g3c_qwen_retrieval_v1/dev_protocol_freeze_v2.json

Dev:

    artifacts/g3c_v1/dev_payload_v2
    artifacts/g3c_v1/dev_qwen_results
    artifacts/g3c_v1/dev_qwen_import_validation.json
    artifacts/g3c_v1/dev_local_eval

Promotion, created only after a passing dev freeze:

    artifacts/g3c_v1/promotion_payload
    artifacts/g3c_v1/promotion_qwen_results
    artifacts/g3c_v1/promotion_qwen_import_validation.json
    artifacts/g3c_v1/promotion_local_eval

Never overwrite B0, B1, G3B baseline, or an earlier Qwen result directory.

Preserved failed pre-inference transport evidence:

    experiments/g3c_qwen_retrieval_v1/dev_protocol_freeze.json
    artifacts/g3c_v1/dev_payload

The preserved v1 paths are not runnable instructions.

## 7. Current local preflight

Run from the repository root in the correct Python environment:

    python scripts/69_g3b_build.py validate
    python scripts/73_freeze_g3_evaluation.py validate
    python scripts/80_g3c_freeze_protocol.py validate
    python scripts/76_g3c_build_payload.py validate --payload artifacts/g3c_v1/dev_payload_v2
    python -m pytest tests/test_g3c.py -q -p no:cacheprovider --basetemp artifacts/pytest_g3c_preflight

Expected current dev payload:

    schema: g3c_gpu_payload_v2
    mode: dev
    questions: 54
    core files: 247
    upload sidecars: 1
    total upload files including manifest: 249
    size: 95,873,569 bytes (91.43 MiB)
    question fields: id, question
    payload fingerprint:
    5584010ab665510b592662cfc42327da19e1148c87fdef8b5cfe4155cdcbd9a4

Do not rebuild protocol freeze v2 before the first Qwen run unless the
experiment is intentionally redesigned. If behavior files drift, stop and
document a new protocol instead of overwriting this one.

## 8. Upload the dev payload

The active payload is ready at artifacts/g3c_v1/dev_payload_v2 and already
contains:

    dataset id:
    lequangkhai5122005/vifinqa-g3c-qwen-retrieval-dev-v1

The dataset slug already exists with the failed v1 payload. Upload v2 as a new
dataset version:

    kaggle datasets version -p artifacts/g3c_v1/dev_payload_v2 -m "G3C payload schema v2 transport fix protocol af86c8ff payload 5584010a" --dir-mode zip

Do not append a trailing dot. Keep the dataset private unless the competition
rules and source licenses explicitly permit publication.

## 9. Run the dev notebook on Kaggle

Import:

    kaggle/vifinqa-g3c-dev-qwen-retrieval.ipynb

Detach or refresh the old v1 input and attach the newest dataset version. Verify
that cell 1 prints payload fingerprint 5584010a..., enable GPU and Internet,
restart the Kaggle session, then Run all without editing model IDs, revisions,
instructions, thresholds, stage settings, or output arguments.

The notebook checks:

- exactly one G3C payload manifest is attached;
- mode is dev;
- Transformers is exactly 4.53.3;
- CUDA is available;
- schema is g3c_gpu_payload_v2;
- all core payload hashes and the exact core file set pass;
- dataset-metadata.json is accepted only as the declared upload sidecar and may
  be absent from the Kaggle mount;
- protocol/config fingerprints match.

Progress bars are shown for leaf/table preparation, embedding batches, dense
ranking, table reranker pairs, and bounded row reranking.

Expected downloadable outputs:

    /kaggle/working/g3c_dev_run/
    /kaggle/working/g3c_dev_results.zip
    /kaggle/working/g3c_dev_run_import_validation.json

The result manifest must say:

    backend: qwen
    scientific_evidence_valid: true
    smoke_limit: 0
    stages_written: R0, R0L, R1, R2, R3, R4
    question_count: 54

## 10. Import and validate dev results locally

Extract g3c_dev_results.zip so that the manifest is directly at:

    artifacts/g3c_v1/dev_qwen_results/g3c_gpu_result_manifest.json

Then run:

    python scripts/77_g3c_validate_gpu_results.py --payload artifacts/g3c_v1/dev_payload_v2 --results artifacts/g3c_v1/dev_qwen_results --out artifacts/g3c_v1/dev_qwen_import_validation.json

This rejects limited/fake output, wrong mode, changed protocol, wrong model
revisions, missing stages, extra files, hash drift, duplicate candidates,
depth overflow, and hard-guard violations.

## 11. Run all dev retrieval and submission metrics

    python scripts/78_g3c_evaluate_stages.py --mode dev --payload artifacts/g3c_v1/dev_payload_v2 --gpu-results artifacts/g3c_v1/dev_qwen_results --out-dir artifacts/g3c_v1/dev_local_eval

This runs the same frozen B0 answer path for every stage, builds an offline
submission, and evaluates:

- DOCS P/R/F2/MRR@5;
- TABLES P/R/F2/MRR@5;
- Leaf Recall@5 and FullPlanCoverage;
- answer and execution accuracy;
- family and OOD diagnostic views;
- exact paired gains/losses and false positives versus R0;
- candidate provenance, hard violations, runtime, cache and artifact hashes.

The evaluation index is:

    artifacts/g3c_v1/dev_local_eval/evaluation_index.json

If that complete index already exists, the command validates mode and returns
it instead of silently rebuilding a second interpretation.

## 12. Apply the dev gate and freeze one candidate

    python scripts/79_g3c_select_freeze.py --gpu-result-manifest artifacts/g3c_v1/dev_qwen_results/g3c_gpu_result_manifest.json --evaluation-index artifacts/g3c_v1/dev_local_eval/evaluation_index.json --selection-out artifacts/g3c_v1/dev_local_eval/g3c_dev_selection.json --freeze-out artifacts/g3c_v1/dev_local_eval/g3c_candidate_freeze.json

Exit code 0 means one stage passed and was frozen. Exit code 2 means no stage
passed. Review every stage evaluation and paired_vs_r0 report before accepting
the mechanical selection.

The candidate freeze binds the selected stage, config, protocol, model and
tokenizer revisions, instruction hash, dev payload/run/result hashes, selected
retrieval/evaluation hashes, deltas, and the one-run promotion policy.

Completed result on 2026-08-24:

    selected stage: R4
    candidate fingerprint: 1cb02ad5e436d95efe18db0f7d8bbcb2e47cb6f65291892b4a78002d32f8336a
    R4 delta: DOCS F2 +0.085190, TABLES F2 +0.086929
    R4 delta: Leaf Recall@5 +0.120370, FullPlanCoverage +0.203704
    answer/execution delta: 0.000000

## 13. Build and upload promotion only after a passing freeze

No promotion payload should exist before this point.

    python scripts/76_g3c_build_payload.py build --mode promotion --candidate-freeze artifacts/g3c_v1/dev_local_eval/g3c_candidate_freeze.json --out-dir artifacts/g3c_v1/promotion_payload
    python scripts/76_g3c_build_payload.py validate --payload artifacts/g3c_v1/promotion_payload

First dataset creation on Windows with Kaggle CLI 2.2.4:

    cd artifacts/g3c_v1/promotion_payload
    python -m kaggle datasets create -p . --dir-mode zip
    cd ../../..

Using a multi-component relative path for `-p` can make Kaggle CLI 2.2.4 build
an invalid temporary manifest filename. Running from the immutable payload
directory with `-p .` avoids that client-side path bug.

Existing slug:

    cd artifacts/g3c_v1/promotion_payload
    python -m kaggle datasets version -p . -m "G3C one-shot promotion candidate" --dir-mode zip
    cd ../../..

Import and Run all:

    kaggle/vifinqa-g3c-promotion-qwen-retrieval.ipynb

The promotion payload and notebook bind exactly R0 plus one selected stage.
They cannot run the full ablation ladder.

Completed payload/upload evidence on 2026-08-24:

    selected stage: R4
    question count: 55
    payload fingerprint: 05a98f22ba6b69eccd91f854be1ddc78b0192c050ab7bdfdfdb639af735f97be
    dataset: lequangkhai5122005/vifinqa-g3c-qwen-retrieval-promotion-v1

## 14. Import and open the one-shot promotion evaluator

Status: completed successfully on 2026-08-24. The commands below are preserved
for provenance only. Do not run them again.

Extract the Kaggle archive to:

    artifacts/g3c_v1/promotion_qwen_results

Validate first:

    python scripts/77_g3c_validate_gpu_results.py --payload artifacts/g3c_v1/promotion_payload --results artifacts/g3c_v1/promotion_qwen_results --out artifacts/g3c_v1/promotion_qwen_import_validation.json --candidate-freeze artifacts/g3c_v1/dev_local_eval/g3c_candidate_freeze.json

Then open promotion exactly once:

    python scripts/78_g3c_evaluate_stages.py --mode promotion --payload artifacts/g3c_v1/promotion_payload --gpu-results artifacts/g3c_v1/promotion_qwen_results --candidate-freeze artifacts/g3c_v1/dev_local_eval/g3c_candidate_freeze.json --baseline-evaluation artifacts/g3b_v1/b0_promotion_evaluation.json --baseline-submission artifacts/g3b_v1/b0_promotion_submission --out-dir artifacts/g3c_v1/promotion_local_eval

Before the locked evaluator runs, the script freezes the exact offline
submission and writes:

    artifacts/g3c_v1/promotion_local_eval/PROMOTION_EVALUATION_OPENED.json

If the process fails after that marker is written, do not delete the marker and
rerun. Preserve the marker, logs, submission freeze, and partial output for a
methodological review. Deleting it would turn a one-shot test into repeated
promotion probing.

Completed evidence:

    promotion run signature: 718bbdd81f765257d67fc985999207ae341a197e00d93b5853acfdd8f2aec038
    strict import: passed, 55 R0 + 55 R4 records, zero hard violations
    marker: artifacts/g3c_v1/promotion_local_eval/PROMOTION_EVALUATION_OPENED.json
    evaluation SHA-256: 4cec1fbe61170711a862fc12f925263ef869d5fa9399b4ee2ec411abfa9ff66c
    R4 promotion delta: DOCS F2 +0.101784, TABLES F2 +0.093116
    R4 promotion delta: Leaf +0.133333, FullPlan +0.181819
    answer/execution delta: 0.000000

## 15. Resume and failure rules

- A completed GPU directory can be reused only with the same run signature.
  The signature binds payload, config, backend, mode, stage and limit.
- A different request against that directory is rejected before any overwrite.
- An interrupted run without a final manifest may reuse content-addressed
  vector/score caches. Cache keys bind model, content, instruction, config and
  protocol.
- If an imported hash fails, redownload the archive; do not hand-edit output.
- If model download fails, verify Internet and the exact pinned revisions.
- If GPU memory fails, preserve the log. Do not edit the frozen payload in
  place.
- If dev gate fails, report the failure and stop before promotion.
- Fake backend requires a positive limit and always records
  scientific_evidence_valid=false.
- Never upload any folder named OFFLINE_EVAL_DO_NOT_UPLOAD.zip as a competition
  submission.

## 16. Completed versus pending evidence

Completed evidence:

- G3B and G3 evaluation freezes validate.
- label-blind dev leaf audit: 54 questions, zero missing exact-report leaves,
  zero invariant errors;
- structural-only promotion leaf audit: 55 questions, zero missing
  exact-report leaves, zero invariant errors; this is not performance evidence;
- protocol and payload validation;
- packaged three-question fake R0-R4 smoke;
- six stage artifacts, zero hard violations;
- B0 codegen/submission handoff smoke;
- 378 repository tests and notebook code-cell compilation;
- real Qwen dev result import: 54 questions x 6 stages, exact hashes, zero hard
  violations;
- all dev retrieval/submission/answer evaluations;
- pre-registered gate: R4 selected and candidate frozen;
- promotion payload: 55 questions, R0 plus R4 only, validated and uploaded;
- promotion GPU import: exact candidate/protocol/payload binding, zero hard
  violations;
- one-shot promotion evaluation: completed with integrity passed;
- G3C exit criterion: met for retrieval, not for answer improvement.

Pending:

- any official 1,012-record post-freeze crash/invariant run.

R4 improves retrieval over R0 on both the 54-question dev set and the frozen
55-question same-corpus/different-question promotion set. This is bounded
generalization evidence within G3B, not proof about the private distribution.
Do not describe R4 as improving answer accuracy, and do not reuse the opened
promotion set to select or tune G3D.
