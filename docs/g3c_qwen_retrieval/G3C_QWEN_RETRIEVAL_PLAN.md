# G3C Plan — Qwen3 Embedding 4B + Reranker 4B

## Starting point

G3 evaluation contract is frozen at:

`242f5b288350ba7b5728dd00bf262c38a69463cb86efd021663fb4f21ed8a877`

Current G3B promotion baseline (55 locked+hard):
- DOCS F2: `0.898216`
- TABLES F2: `0.613054`
- Leaf Recall@5: `0.786364`
- FullPlanCoverage: `0.654545`
- scope_delta FullPlanCoverage: `0.0`
- ranking_argmax FullPlanCoverage: `0.2`

Oracle typed programs execute correctly on all G3B records, so G3C is a **retrieval/grounding-only** treatment.

## Hard boundary

During G3C, freeze:
- G3A/G3B data, views, evaluator and promotion policy;
- router semantics unless only exposing existing atomic leaves;
- B0/B1 planner, compiler, Selection-v2, arbitration and answer logic;
- official 1,012 questions outside model/config selection.

Use `primary_tune` only for development. Before evaluating `primary_locked + hard`, freeze the exact G3C candidate manifest.

## Models

Use:
- `Qwen/Qwen3-Embedding-4B`
- `Qwen/Qwen3-Reranker-4B`

Requirements:
- pin immutable Hugging Face commit revisions released before `2026-06-01`;
- record model SHA/revision, tokenizer revision, license and runtime packages;
- do not use floating `main`;
- keep the CPU clean lock unchanged; create a separate G3C GPU/runtime fingerprint.

Initial implementation should use FP16 and load embedding/reranker sequentially. Quantization is a separate ablation only if required by hardware.

## Retrieval architecture

Keep hard financial constraints outside neural scoring:
`ticker/entity → report scope/type → year/period → candidate retrieval`.

Neural similarity must never override an explicit wrong ticker, year, scope or report type.

### Passage serialization

**Table passage**
- ticker, report_id, year, scope/doc_type;
- table position/page/unit;
- table title/header;
- compact representative row labels.

**Row/cell passage**
- ticker, year/period, scope;
- report/table identity;
- row code + normalized row label;
- column name/period;
- unit metadata.

Do **not** include numeric cell values in neural relevance text for the first ablation.

### Query serialization

Retrieve per atomic leaf, not only from the full question.

Leaf query contains:
- original Vietnamese leaf/metric phrase;
- canonical metric + aliases/components when available;
- ticker, year/period, scope and qualifier semantics.

Use one frozen English retrieval instruction for Qwen3; do not tune instruction wording on promotion data.

## Controlled ablations

### R0 — Frozen control
Current canonical/BM25 retrieval.

### R1 — Dense union
`R0 ∪ Qwen3-Embedding-4B`

- use normalized embeddings;
- start with full 2560-d representation;
- cache candidate embeddings with content/model/config SHA;
- dense retrieval is a recall source, not a replacement for lexical/canonical retrieval.

Measure whether Leaf Recall@5 / FullPlanCoverage improve before adding reranking.

### R2 — Qwen3 reranker
Rerank only the R1 union with `Qwen3-Reranker-4B`.

- use the official Transformers-style relevance score;
- freeze one finance-specific English instruction;
- rerank a bounded top-N pool per leaf;
- preserve hard ticker/year/scope/period guards.

Primary goal: improve TABLES precision/F2 and top-rank leaf quality without losing recall.

### R3 — Per-leaf quota
Guarantee candidate capacity for every required atomic leaf.

- allocate top-k candidates per leaf;
- deduplicate at table level only after per-leaf retrieval;
- retain provenance: strict lexical, dense rescue, reranker score, selected leaf.

Primary metric: `FullPlanCoverage`.

### R4 — Row/cell reranking
After table pruning, apply the same guarded reranker to row/cell passages.

Primary goal: reduce shortlist/no-candidate and wrong year/scope/period grounding failures.

## Evaluation

For every R0–R4 report:
- DOCS P/R/F2/MRR@5;
- TABLES P/R/F2/MRR@5;
- Leaf Recall@5;
- FullPlanCoverage;
- per-family and OOD-view breakdowns;
- candidate count / retrieval source / rerank provenance;
- runtime, GPU peak memory and cache hashes.

Also report exact paired changes vs R0:
- gained/lost gold leaves;
- gained/lost full plans;
- table false positives removed/introduced;
- scope/year/report-type violations (must remain zero after hard guards).

Recommended dev promotion gate (pre-register before first run):
- material positive delta in both Leaf Recall@5 and FullPlanCoverage;
- no >1 pp regression in DOCS F2;
- no >1 pp regression in TABLES F2;
- no integrity or hard-constraint violation.

Do not optimize Answer Accuracy during G3C; record it only as a secondary invariant because reasoning is frozen.

## Candidate freeze and promotion

1. Run R0–R4 on `primary_tune`.
2. Select one retrieval configuration only from dev evidence.
3. Freeze:
   - model/tokenizer revisions;
   - instructions;
   - passage/query serializers;
   - embedding dimension;
   - top-N / per-leaf quota;
   - all thresholds and fusion rules;
   - cache hashes and runtime fingerprint.
4. Create a machine-verifiable candidate fingerprint.
5. Evaluate exactly once on `primary_locked + hard`.
6. Run the official 1,012 questions only for crash/schema/invariant regression, never for threshold selection.

## Expected implementation boundary

Prefer new modules such as:

- `vifinqa/g3c/embedding.py`
- `vifinqa/g3c/reranker.py`
- `vifinqa/g3c/serialize.py`
- `vifinqa/g3c/retrieval.py`
- `vifinqa/g3c/evaluate.py`

Do not mutate the frozen clean retriever in-place; wrap/reuse it as R0.

## Exit criterion

G3C is complete when one frozen retrieval candidate shows reproducible improvement over R0 on the G3 dev gate and survives one promotion evaluation without material DOCS/TABLES regression.

Only then propagate the chosen retrieval path into the later B1/typed-planner experiment.

## Dev outcome - 2026-08-24

The full frozen Qwen dev run completed and strict validation passed. R4 won the
pre-registered gate with deltas versus R0 of `+0.085190` DOCS F2,
`+0.086929` TABLES F2, `+0.120370` Leaf Recall@5, and `+0.203704`
FullPlanCoverage. Answer and execution accuracy remained unchanged at 9/54.

Candidate fingerprint:

    1cb02ad5e436d95efe18db0f7d8bbcb2e47cb6f65291892b4a78002d32f8336a

The R0+R4 promotion payload was frozen and evaluated exactly once. R4 improved
promotion DOCS F2 by `+0.101784`, TABLES F2 by `+0.093116`, Leaf Recall@5 by
`+0.133333`, and FullPlanCoverage by `+0.181819`; answer/execution remained
unchanged. The G3C retrieval exit criterion is met. See
`G3C_SESSION_PROMOTION_RESULTS_2026-08-24.md` for the bounded claim and next
decision.

## P-B closeout and official engineering implementation - 2026-08-24

P-B/R4 is now formally closed and registered. The 1,012-question official
execution protocol was frozen only after a label-blind workload audit and an
exact-equivalence review. The implementation uses both T4s without changing
frozen model calls: complete embedding batches are assigned across GPUs and
complete questions are assigned across four reranker shards.

The workload is split across three Kaggle runs because a conservative
Promotion-based projection places a monolithic two-T4 run at 12.280 hours.
All three official phases completed in 12.281 sequential hours and passed the
exact canaries and strict local finalizer. The frozen official R4 SHA-256 is
`1281af9a737fd235e61b275c4ffebe34624d4790be6c043c81ca88d8552132d5`.

Fourteen official questions have an atomic leaf with no exact report, a case
for which the original frozen runner has no R4 output. They are preregistered
as explicit R0 passthroughs with no fallback search. Exact frozen R4 applies to
the other 998 questions. This is an engineering totalization, not a tuned
retrieval change.

See `G3C_OFFICIAL_1012_RUNBOOK.md` and
`G3C_SESSION_PB_CLOSEOUT_OFFICIAL_1012_2026-08-24.md`. The measured completion
and B1-fixed public diagnostic are recorded in
`G3C_SESSION_OFFICIAL_RESULTS_PUBLIC_DIAGNOSTIC_2026-08-24.md`.

## Implementation review and controlled amendments - 2026-08-23

Status at the 2026-08-23 implementation freeze: implemented and locally
verified; real Qwen dev evidence was pending. The 2026-08-24 outcome above
supersedes that historical status without changing the frozen treatment.

The plan's main hypothesis, hard financial guards, label boundary, staged
ablation, frozen answer path and tune-before-promotion policy were retained.
The following changes were required before the first GPU run.

### Added R0L control

R0L uses the new atomic-leaf decomposition with lexical retrieval only. Without
it, R1 would mix leaf-query improvements with the Qwen embedding treatment and
could not attribute a gain to the neural model.

The implemented ladder is:

    R0 -> R0L -> R1 -> R2 -> R3 -> R4

### Added a separate label-blind decomposer

The production route does not expose complete leaves for several G3B
compositions. G3C therefore wraps the existing route instead of mutating it.
The decomposer accepts question, route, canonical metric registry and store
metadata only. Family labels, gold, review files and ID lists are outside its
API.

### Corrected report-period and ranking hazards

- Multi-year non-prior leaves bind to their own report years. Prior-period
  leaves bind to the later container report.
- Per-leaf quota selection preserves global relevance order.
- R4 neural-scores the full 24-row lexical prefilter and keeps the best 12
  row results before table fusion.
- R4 changes candidate order rather than writing unused row metadata.

### Corrected Qwen runtime details

- Embedding uses AutoModel, official last-token pooling, L2 normalization and
  the full 2560 dimensions.
- Reranking uses AutoModelForCausalLM and final-token yes/no softmax.
- Models load sequentially in FP16 with SDPA and immutable revisions.
- Reranker use_cache is disabled; table and row limits are 768 and 384 tokens.
- Cache keys include content, model, instruction, config and protocol.

The initial reranker batch size is 2 for a conservative 16 GB Kaggle target.
Quantization remains outside this ablation.

### Made the gate executable

A stage must satisfy all of:

- Leaf Recall@5 delta versus R0 at least +0.02;
- FullPlanCoverage delta versus R0 at least +0.03;
- DOCS F2 delta no worse than -0.01;
- TABLES F2 delta no worse than -0.01;
- zero hard-constraint violations;
- passing evaluator integrity.

Answer and execution accuracy are recorded but are not selection objectives.

### Added two freezes and two payloads

The dev protocol was frozen before a real Qwen run. A second candidate freeze
is created only if one stage passes the dev gate. Dev and promotion use
different stripped payloads and notebooks. Promotion binds exactly one selected
stage and is opened once.

Current protocol fingerprint after the pre-inference Kaggle transport fix:

    af86c8ffc276cc0a92ceeb3cc0ddc3a7eeaa7b6a1e4430dc2cade8ac4c9621c5

Current dev payload fingerprint:

    5584010ab665510b592662cfc42327da19e1148c87fdef8b5cfe4155cdcbd9a4

Payload schema v1 and protocol fingerprint 116f26a5... are preserved as failed
pre-inference transport provenance. No Qwen output or metric was observed
before the v2 fix, so the registered retrieval and promotion gates are
unchanged.
