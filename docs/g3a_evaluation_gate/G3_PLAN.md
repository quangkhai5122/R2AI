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

Add guarded rescue only when a required leaf is missing or confidence is low:

- leaf-specific query decomposition and candidate quotas;
- BM25 plus dense/cross-encoder union;
- cell/row reranking;
- hard negatives for wrong year, scope, component, period, and unit.

Keep code generation and arbitration frozen during the retrieval ablation.

## G3D - typed 14B planner

Use an eligible model at or below 15B to emit multiple typed plans. Compile and
replay every plan. Accept an LLM plan only through calibrated consensus or a
pre-registered override rule. Preserve the deterministic core as fallback.

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
