# G3 implementation plan

## G3A - evaluation and promotion gate

Status: implemented in v1.

Purpose:

- approximate private evaluation with new questions over the same corpus;
- score document retrieval, table retrieval, answer, and execution together;
- prevent public-question tuning and cross-split fact leakage;
- require evidence-reviewed hard gold and immutable provenance;
- promote treatments under unknown private weights.

Before final candidate freeze, expand G3A to v1.1 with ranking, count, CAGR,
nested arithmetic, note-table facts, and ambiguous-scope examples. Keep v1
unchanged as a regression layer.

## G3B - typed canonical deterministic core

Port generalized formula, operand-role, period, unit, sign, and scope semantics
into one typed operator registry. Do not port public id masks, gold-derived
overrides, or a second uncontrolled formula engine.

Required ablation:

1. G3A B0 frozen baseline.
2. Typed operator registry only.
3. Registry plus deterministic replay/arbitration.

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
