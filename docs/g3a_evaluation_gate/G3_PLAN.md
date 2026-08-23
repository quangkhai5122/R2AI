# G3 implementation plan

## G3A - evaluation and promotion gate

Status: complete. G3A v1 remains the immutable 144-question regression layer.

The source-derived extension is stored separately in
`data/g3a_extension_v1`; it does not mutate `data/g3a_v1`. The public
question set is used only for exact ID/text exclusion.

## G3B - typed, compositional, and OOD evaluation corpus

Status: complete and frozen on 2026-08-23.

G3B adds 109 source-derived questions across ranking, count, CAGR,
percentage-point change, nested arithmetic, note tables, scope ambiguity,
prior-period ambiguity, and non-money outputs. Gold programs reuse Selection-v2
`facts/bindings/root`; no third IR or production formula engine was added.

The gate provides:

- dev on `primary_tune` only;
- promotion on `primary_locked + hard` only after candidate freeze;
- oracle-evidence and end-to-end modes;
- competition-shaped DOCS/TABLES/answer/execution metrics;
- Leaf Recall@K, FullPlanCoverage, and typed reasoning diagnostics;
- LOTO, LOYO, LORO, LOMO, composition, and scope/period stress views.

The leave-one-* and stress views overlap and are diagnostic slices, not
independent replications. The complete contract is frozen at
`experiments/g3_evaluation_v1/g3_evaluation_freeze.json`.

## G3C - per-leaf retrieval and grounding

Status: complete on 2026-08-24. R4 passed the frozen dev gate and its one-shot
`primary_locked + hard` promotion evaluation.

The frozen ladder is:

    R0 frozen control
    R0L leaf-aware lexical control
    R1 Qwen3 dense union
    R2 Qwen3 table reranker
    R3 per-leaf quota
    R4 bounded row reranker and table reorder

Hard ticker/report/year/scope guards remain outside neural scoring. Query and
passage formation are label-blind, numeric values are excluded from neural
passages, and code generation/arbitration remain frozen.

The v2 dev protocol and 54-question stripped payload are frozen. Payload v1 is
preserved as a pre-inference Kaggle transport failure and is not an active
runtime input. R4 improved dev DOCS F2 by 0.085190, TABLES F2 by 0.086929,
Leaf Recall@5 by 0.120370, and FullPlanCoverage by 0.203704 versus R0. Answer
and execution accuracy were unchanged. On promotion, R4 improved DOCS F2 by
0.101784, TABLES F2 by 0.093116, Leaf Recall@5 by 0.133333, and
FullPlanCoverage by 0.181819. Answer/execution again remained unchanged. The
promotion evaluator is consumed and G3C must not be tuned from its residuals.

Operational runbook:
../g3c_qwen_retrieval/G3C_RUNBOOK.md

Implementation evidence:
../g3c_qwen_retrieval/G3C_SESSION_IMPLEMENTATION_2026-08-23.md

Dev Qwen result and candidate-freeze evidence:
../g3c_qwen_retrieval/G3C_SESSION_DEV_QWEN_RESULTS_2026-08-24.md

One-shot promotion evidence:
../g3c_qwen_retrieval/G3C_SESSION_PROMOTION_RESULTS_2026-08-24.md

## G3D - typed 14B planner

Use an eligible model at or below 15B to emit multiple typed plans. Compile and
replay every plan. Accept an LLM plan only through calibrated consensus or a
pre-registered override rule. Preserve the deterministic core as fallback.

The original G3B promotion set was opened by G3C and is now regression-only.
Before G3D development, construct and freeze a new source-derived
same-corpus/different-question tune/locked/hard split, disjoint by source fact
group and question text, with a manually reviewed hard subset. R4 remains fixed
under G3D and retrieval metrics are non-regression constraints.

## G3E - diverse specialist

Train either a source-derived financial planner adapter or a cell/leaf reranker.
Use program-first questions from the corpus plus external financial QA mapped
to the same intermediate representation. Do not train on official public
question text.

## G3F - calibrated ensemble and five private submissions

Build out-of-fold gates using only pre-answer features. Candidate portfolio:

1. P-A: typed canonical deterministic core.
2. P-B: P-A plus per-leaf hybrid retrieval.
3. P-C: P-B plus typed 14B planner.
4. P-D: genuinely diverse specialist.
5. P-E: calibrated ensemble.

The five submissions are model/system families, not five leaderboard patches.
Each must have a frozen G3A report, config hash, run signature, and exact
disagreement audit against B0.
