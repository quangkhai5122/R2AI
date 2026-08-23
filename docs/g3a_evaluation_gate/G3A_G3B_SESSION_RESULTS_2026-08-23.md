# G3A-G3B session results - 2026-08-23

## Outcome

The session plan is complete. G3A v1 remains byte-for-byte unchanged, the G3A
extension and G3B typed/compositional/OOD corpus are reviewed and immutable,
both evaluator modes have been exercised, B0 has been measured on dev and on a
pre-frozen promotion candidate, and the complete evaluation contract is sealed
before G3C.

Final evaluation-freeze fingerprint:

`242f5b288350ba7b5728dd00bf262c38a69463cb86efd021663fb4f21ed8a877`

## Review of the proposed plan

The proposed direction was correct and was retained, with four clarifications:

1. LOTO, LOYO, LORO, LOMO, composition, and scope/period stress are overlapping
   views of the same 109 records. They are useful breakdowns but are not
   independent test replications.
2. Typed gold reuses Selection-v2 `facts/bindings/root` exactly. A third IR
   would create semantic drift and was not introduced.
3. Promotion requires a machine-verifiable candidate freeze. Merely writing
   that a candidate was frozen would not prevent post-result mutation.
4. Oracle evidence bypasses retrieval. Its DOCS/TABLES metrics are marked
   non-interpretable; only end-to-end mode measures retrieval.

A further conservative choice is that missing typed output receives zero on the
full denominator. Metrics conditional on producing typed output are retained
only as secondary diagnostics, so a low-coverage planner cannot appear strong
by abstaining on hard cases.

## Implemented artifacts

- `configs/g3b_evaluation_v1.json`: deterministic allocation, review, view, and
  leaf-K policy.
- `data/g3a_extension_v1/`: immutable extension manifest, questions, and gold.
- `data/g3b_v1/`: corpus, dev/promotion question files, oracle predictions,
  review queue/ledger, and six OOD view manifests.
- `vifinqa/g3b/`: source extraction, generation, split/view construction,
  strict validation, dual-mode evaluation, and freeze manifests.
- `scripts/69_g3b_build.py` through `scripts/74_g3b_build_submission.py`:
  build, review, candidate freeze, evaluate, final freeze, and offline packaging.
- `experiments/g3_evaluation_v1/g3_evaluation_freeze.json`: final contract
  freeze.
- `tests/test_g3b_*.py`: split/review/IR/view/evaluator/freeze regression tests.

No production retrieval, planner, compiler, arbitration, model, threshold, B0
config, or B1 config was modified.

## Corpus and gold assurance

| Item | Result |
|---|---:|
| Total questions | 109 |
| Primary tune | 54 |
| Primary locked | 33 |
| Hard | 22 |
| Program families | 11 |
| Required reviews | 72 |
| Approved reviews | 72 |
| Pending reviews | 0 |
| Public exact ID/text overlap | 0 |
| Cross-split fact overlap | 0 |

The families are ranking argmax/argmin, positive count, CAGR,
percentage-point change, nested margin average, simple average, note lookup,
scope delta, prior-period lookup, and debt/assets ratio. Outputs cover number,
percent, percentage points, ratio, count, and year.

Review subjects bind question, answer, exact evidence, leaf specs, typed
program, relevant documents, and relevant tables. Each required record was
rechecked against the source store and independently recomputed by family.
This is a strong automated/agent evidence audit, not a substitute for a second
human financial-domain review.

Frozen fingerprints:

- G3A v1 tree:
  `bf896aa2216f5e40f97f68dfbbadae07a152bae9abcd12c1d77bec26c1992d24`
- G3A extension:
  `77642f8472a547e6a31317dceb38d60793f22d79bf8184c072d147fd92220742`
- G3B corpus:
  `3f649850864fe2a95f0bc0f15de721905cf4ae3cf7579d6a8752fb4fdbea011b`

## Baseline results

| Metric | Dev: tune (54) | Promotion: locked+hard (55) |
|---|---:|---:|
| DOCS precision macro | 0.882716 | 0.900000 |
| DOCS recall macro | 0.922840 | 0.909091 |
| DOCS F2 macro | 0.903025 | 0.898216 |
| DOCS MRR@5 | 0.972222 | 0.990909 |
| TABLES precision macro | 0.331041 | 0.328658 |
| TABLES recall macro | 0.817901 | 0.824242 |
| TABLES F2 macro | 0.612058 | 0.613054 |
| TABLES MRR@5 | 0.789198 | 0.798182 |
| Leaf Recall@5 | 0.808642 | 0.786364 |
| FullPlanCoverage | 0.685185 | 0.654545 |
| Answer Accuracy | 0.166667 | 0.181818 |
| Execution Accuracy | 0.166667 | 0.181818 |
| Typed output coverage | 0.000000 | 0.000000 |

The promotion submission was sealed before evaluation. Candidate-freeze
fingerprint:

`c1342d5e803d5448aab2b34ca8e0090dd247646c237a8eebf7d28427238bddee`

The dev and promotion vectors are close. There is no evidence here of a
tune-only uplift that collapses on locked/hard. The stable weakness is instead
structural:

- retrieval is useful but incomplete: Leaf Recall@5 is about 0.79 and complete
  leaf coverage about 0.65 on promotion;
- B0 emits no typed program on this corpus, so typed coverage and every typed
  diagnostic are zero;
- note lookup and simple average reach 1.0 answer accuracy, while all other
  compositional/typed families are 0.0;
- scope delta is the clearest retrieval stress: FullPlanCoverage is 0.0 in both
  dev and promotion;
- promotion ranking-argmax is also retrieval-limited: Leaf Recall@5 is 0.4 and
  FullPlanCoverage is 0.2.

The oracle Selection-v2 program executes correctly on all 54 dev and all 55
promotion records, with 1.0 AST match, typed execution, answer, and execution.
This does not prove a planner can infer those programs. It shows that, given
the reviewed evidence and typed gold, the semantic contract/compiler can
represent and execute the required reasoning.

## Interpretation and next strategy

G3C should remain a retrieval-only ablation as planned. The immediate target is
not answer accuracy; it is improved Leaf Recall@5 and FullPlanCoverage without
regressing DOCS/TABLES competition metrics. This isolates retrieval value before
a typed planner is introduced.

Recommended preregistered sequence:

1. frozen current canonical/BM25 control;
2. add Qwen3-Embedding-4B candidate union;
3. add Qwen3-Reranker-4B;
4. add per-leaf candidate quota;
5. add row/cell reranking.

Select treatment/config using `primary_tune` only. Freeze the exact candidate
manifest before opening `primary_locked + hard`. Keep planner, compiler,
arbitration, model, thresholds, and public 1,012-question behavior outside the
selection loop. After choosing the retrieval treatment, run official questions
only for crash/invariant regression, not for threshold selection.

The first G3C success criterion should be a material improvement in both Leaf
Recall@5 and FullPlanCoverage, no integrity failure, and no unacceptable
regression in DOCS/TABLES P/R/F2/MRR@5. Answer accuracy is still recorded, but
a retrieval-only treatment is not required to solve the typed-reasoning gap.

## Acceptance and verification

All ten session acceptance criteria passed:

- G3A v1 tree hash unchanged;
- public questions used only for exact ID/text exclusion;
- Selection-v2 semantic surface reused;
- requested families and output types covered;
- all OOD views have explicit leakage assertions and frozen hashes;
- dev/promotion policy enforced;
- oracle/e2e modes implemented;
- retrieval plus typed metrics implemented;
- production files/configs untouched;
- final freeze created and revalidated.

Verification evidence:

- targeted G3B tests: 11 passed;
- full repository tests: 356 passed;
- strict G3B validation: valid, 109 questions, zero pending reviews;
- compile checks passed;
- both B0 reports passed submission-integrity checks.

For exact commands and artifact locations, see
`docs/g3a_evaluation_gate/G3B_RUNBOOK.md`.
