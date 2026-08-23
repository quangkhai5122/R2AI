# G3C one-shot promotion result and next-stage decision - 2026-08-24

## Outcome

The hash-bound R4 candidate completed its single permitted evaluation on the
55-question `primary_locked + hard` promotion set. Strict GPU-result validation
passed before the evaluator was opened, the one-shot marker was written and
preserved, and the evaluation completed with integrity passed.

R4 improves every frozen retrieval/submission metric over R0 on the combined
promotion set. It also improves every retrieval point estimate separately on
`primary_locked` and on the manually reviewed hard set. Answer and execution
accuracy are unchanged.

G3C therefore meets its registered retrieval exit criterion: the selected
candidate improved on dev and survived one promotion evaluation without DOCS
or TABLES regression. This is same-corpus/different-question evidence, not a
claim about the unknown private-question distribution or an end-to-end answer
improvement.

The promotion evaluator has now been consumed. Do not delete its marker or run
it again.

## Bound provenance

- selected candidate: `g3c-qwen-retrieval-v1-r4`;
- candidate fingerprint:
  `1cb02ad5e436d95efe18db0f7d8bbcb2e47cb6f65291892b4a78002d32f8336a`;
- promotion payload fingerprint:
  `05a98f22ba6b69eccd91f854be1ddc78b0192c050ab7bdfdfdb639af735f97be`;
- promotion GPU run signature:
  `718bbdd81f765257d67fc985999207ae341a197e00d93b5853acfdd8f2aec038`;
- GPU result manifest SHA-256:
  `d36b26629c00a8c64f8cf4e53fae056a938f23eb274b6ebfcd2eefa8a9ce8df5`;
- strict import validation SHA-256:
  `26dbc0297a388948f838d2f00adc845b30f6a909775d2f9ce539faa02a7b892f`;
- R0 retrieval SHA-256:
  `25043d58c103c80e9de756f32f097c375d0f41fecfa4b6f6240828425bb471af`;
- R4 retrieval SHA-256:
  `3ee0d1694cf7a88705c41a9b03ad2ea3edeacd23ef3ad346ebd3c7e8343ea963`;
- R4 codegen SHA-256:
  `def007d1c983307f1582d64d71521bf55da34650c95f97c5e281f7e92520b08e`;
- exact-submission freeze SHA-256:
  `588a732b78bac370a15e4b51d29fee7d288d45b36c0b133cdd369f902ba72972`;
- R4 evaluation SHA-256:
  `4cec1fbe61170711a862fc12f925263ef869d5fa9399b4ee2ec411abfa9ff66c`;
- paired diagnostic SHA-256:
  `0ce4c4c9e1a3e2cf8af9ff42250438530e040e885cacebe54628c38121459a2b`;
- promotion-open marker SHA-256:
  `4ec0577b4a4329ff63a83b0f0481e99db9a966b910f80fc675d8f4895c364b9e`.

The imported R0 retrieval is byte-identical to the frozen G3B promotion
control. Both R0 and R4 contain exactly 55 records and 20 candidates per
record, with zero ticker/report/year/scope violations. The pinned model,
tokenizer, instruction, config and protocol fingerprints match the dev freeze.

The Kaggle workload used two Tesla T4 GPUs and took 3,407.707 seconds (56.80
minutes): 1,447.024 seconds for embeddings and 1,954.467 seconds for reranking.
Peak GPU memory was 8,897,808,896 bytes (8.29 GiB).

## Commands completed

Strict validation completed before any locked labels were opened:

    python scripts/77_g3c_validate_gpu_results.py --payload artifacts/g3c_v1/promotion_payload --results artifacts/g3c_v1/promotion_qwen_results --out artifacts/g3c_v1/promotion_qwen_import_validation.json --candidate-freeze artifacts/g3c_v1/dev_local_eval/g3c_candidate_freeze.json

The following command then consumed the one permitted promotion evaluation:

    python scripts/78_g3c_evaluate_stages.py --mode promotion --payload artifacts/g3c_v1/promotion_payload --gpu-results artifacts/g3c_v1/promotion_qwen_results --candidate-freeze artifacts/g3c_v1/dev_local_eval/g3c_candidate_freeze.json --baseline-evaluation artifacts/g3b_v1/b0_promotion_evaluation.json --baseline-submission artifacts/g3b_v1/b0_promotion_submission --out-dir artifacts/g3c_v1/promotion_local_eval

The preserved marker is:

    artifacts/g3c_v1/promotion_local_eval/PROMOTION_EVALUATION_OPENED.json

Post-evaluation verification passed without reopening the evaluator:

- active protocol freeze validation;
- immutable promotion payload validation;
- targeted G3C suite: 22 passed in 3.58 seconds;
- full repository suite: 378 passed in 78.01 seconds;
- Markdown/diff whitespace check: passed.

## Aggregate promotion metrics

P/R/F2 are macro metrics, MRR is MRR@5, Leaf is Leaf Recall@5, and Full is
FullPlanCoverage.

| Metric | R0 | R4 | Delta |
| --- | ---: | ---: | ---: |
| DOCS precision | 0.900000 | 1.000000 | +0.100000 |
| DOCS recall | 0.909091 | 1.000000 | +0.090909 |
| DOCS F2 | 0.898216 | 1.000000 | +0.101784 |
| DOCS MRR | 0.990909 | 1.000000 | +0.009091 |
| TABLES precision | 0.328658 | 0.383030 | +0.054372 |
| TABLES recall | 0.824242 | 0.936364 | +0.112122 |
| TABLES F2 | 0.613054 | 0.706170 | +0.093116 |
| TABLES MRR | 0.798182 | 0.836970 | +0.038788 |
| Leaf Recall@5 | 0.786364 | 0.919697 | +0.133333 |
| FullPlanCoverage | 0.654545 | 0.836364 | +0.181819 |
| Answer/Execution | 0.181818 | 0.181818 | 0.000000 |

R4 retrieves every required document for all 55 questions. Full-plan coverage
rises from 36/55 to 46/55.

## Dev-to-promotion replication

The effect sizes are stable across the tune and promotion sets:

| Delta versus R0 | Dev, 54 | Promotion, 55 |
| --- | ---: | ---: |
| DOCS F2 | +0.085190 | +0.101784 |
| TABLES F2 | +0.086929 | +0.093116 |
| Leaf Recall@5 | +0.120370 | +0.133333 |
| FullPlanCoverage | +0.203704 | +0.181819 |
| Answer/Execution | 0.000000 | 0.000000 |

This consistency is much stronger evidence than the dev gate alone. It also
isolates the causal boundary: the complete R4 retrieval treatment transfers,
whereas the frozen answer layer does not benefit enough to cross answer
tolerance.

## Locked and hard subsets

| Split | N | DOCS F2 delta | TABLES F2 delta | Leaf delta | Full delta | Answer delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| primary_locked | 33 | +0.121802 | +0.133873 | +0.181818 | +0.242424 | 0 |
| hard | 22 | +0.071757 | +0.031981 | +0.060606 | +0.090909 | 0 |

All 22 hard records were manually reviewed and approved by source-cell recheck
plus family recomputation. The hard-set direction is favorable, but its effect
is smaller: R4 gains three hard full plans and loses one, for a net gain of two.

An exploratory post-hoc paired bootstrap with seed 3304 and 100,000 resamples
gave positive combined-set 95% percentile intervals for DOCS F2, TABLES F2,
Leaf and Full. A family-cluster bootstrap also kept all four intervals positive.
On the 22-question hard subset alone, only DOCS F2 had an interval fully above
zero; TABLES, Leaf and Full crossed zero because the subset is small and has
mixed question-level gains/losses. This uncertainty analysis was not part of
the registered gate and does not authorize hard-set tuning.

## Paired and family diagnostics

R4 gains 12 full-plan questions and loses two. It improves Leaf recall on 15
questions and reduces it on three. It removes 131 R0 false-positive tables and
introduces 116, a net reduction of 15 but substantial two-way shortlist churn.

The largest promotion improvements are:

| Family | TABLES F2 delta | Leaf delta | Full delta |
| --- | ---: | ---: | ---: |
| scope_delta | +0.302 | +0.300 | +0.600 |
| ranking_argmax | +0.294 | +0.333 | +0.200 |
| cagr | +0.175 | +0.200 | +0.200 |
| simple_average | +0.165 | +0.300 | +0.400 |
| debt_assets_ratio | +0.077 | +0.100 | +0.200 |

`ranking_argmin` is the only family with a negative TABLES/Leaf point delta
(`-0.059` and `-0.067`), although Full improves by `+0.200`. `count_positive`,
`note_lookup` and `percentage_point_change` are unchanged in table/leaf/full
coverage. Unlike dev, `ranking_argmax` improves strongly on promotion; this
family instability is another reason not to patch individual residuals.

The scope/period stress view improves DOCS F2 `0.853 -> 1.000`, TABLES F2
`0.592 -> 0.685`, Leaf `0.787 -> 0.938`, and Full `0.650 -> 0.850`. Every LOYO
block improves. The weakest time block is `from_2022`, where Full changes only
from `0.778` to `0.833`.

Nine questions still lack a full R4 plan: five `primary_locked` and four hard.
The residuals cover CAGR, count-positive, nested margin, three ranking
questions and two hard scope-delta questions. The two lost full plans are one
locked count-positive and one hard ranking-argmax record. These are frozen
diagnostics, not a repair list.

## Answer bottleneck

The answer result exactly replicates the dev finding:

- R0 and R4 both have 30 `status=ok` and 25 `status=failed` rows;
- both have 20 `rule_composite`, 10 `rule`, and 25 `none` sources;
- typed-plan coverage remains zero;
- the same ten questions are correct: all five `note_lookup` and all five
  `simple_average` questions;
- no other family has a correct answer;
- R4 changes nine numeric answers, six move closer to gold and three move
  farther away, but none becomes correct.

This is not an argument against R4. Private scoring includes retrieval, so R4
is a distinct and useful P-B hypothesis. It is evidence that table retrieval is
no longer the only dominant bottleneck. Correct row/cell grounding, units,
operand roles and formula compilation must now be addressed separately.

## Decision and next direction

1. Mark G3C complete and preserve R4 as the immutable P-B retrieval candidate.
   Do not rerun or retune it from promotion residuals.
2. Run R4 on the 1,012 official questions only as a post-freeze crash/schema/
   finiteness audit. Freeze the exact official artifact, but do not use public
   scores or question-level disagreements to alter R4.
3. Start G3D as a separate typed planner plus row/cell-grounding hypothesis,
   with R4 fixed underneath it. Primary objectives are typed coverage,
   operator/operand/output-type/AST correctness, execution and answer accuracy;
   retrieval metrics remain non-regression constraints.
4. Do not reuse the now-open G3B `primary_locked + hard` set as a blind G3D
   promotion test. Keep all 109 G3B questions as an immutable regression suite
   and construct a new source-derived, same-corpus/different-question G3D
   tune/locked/hard split before implementation or threshold selection.
5. Make the new split disjoint by source fact group and question text, retain
   official-question exclusion, manually review the hard portion, freeze all
   hashes, and preregister one G3D promotion run.

R4 is now eligible to occupy one of the five private-submission hypotheses, but
it should not consume a slot until its official 1,012-record artifact passes
the post-freeze engineering audit. A later R4+G3D candidate must remain a
separate hypothesis rather than a patched replacement selected from this
opened holdout.
