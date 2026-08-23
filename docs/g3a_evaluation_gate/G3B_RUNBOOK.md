# G3B evaluation runbook

Status: completed and frozen on 2026-08-23.

This runbook records the commands already executed by the Codex agent. The user
does not need to rerun them. They remain here for reproducibility and future
candidate sessions.

## Frozen evaluation contract

- G3A v1 tree SHA-256:
  `bf896aa2216f5e40f97f68dfbbadae07a152bae9abcd12c1d77bec26c1992d24`
- G3A v1 bundle fingerprint:
  `956596dff2db953f715f4cfed5a62209438897cce1dc4d37e596d3168b8213b2`
- G3A extension fingerprint:
  `77642f8472a547e6a31317dceb38d60793f22d79bf8184c072d147fd92220742`
- G3B corpus fingerprint:
  `3f649850864fe2a95f0bc0f15de721905cf4ae3cf7579d6a8752fb4fdbea011b`
- Final G3 evaluation freeze:
  `242f5b288350ba7b5728dd00bf262c38a69463cb86efd021663fb4f21ed8a877`

Validate the frozen contract before any G3C result is interpreted:

    python scripts/69_g3b_build.py validate
    python scripts/73_freeze_g3_evaluation.py validate

The final freeze is stored at
`experiments/g3_evaluation_v1/g3_evaluation_freeze.json`.

## Corpus construction and review

The executed authoring sequence was:

    python scripts/69_g3b_build.py build
    python scripts/70_g3b_review.py
    python scripts/69_g3b_build.py build
    python scripts/69_g3b_build.py validate

The first build created the review queue. The independent audit then re-read
every required source cell and recomputed each answer using a family-specific
formula. The second build bound the approved ledger into every review subject
and regenerated the manifests.

Final counts:

- 109 questions: 54 `primary_tune`, 33 `primary_locked`, 22 `hard`;
- 11 families; 10 records per family except `simple_average` with 9;
- 72/72 required reviews approved; zero pending;
- 39 number, 20 percent, 20 year, 10 count, 10 ratio, and
  10 percentage-point outputs;
- no exact public question ID/text overlap and no fact overlap across the three
  primary splits.

The 72 reviews are source-cell and independent-formula audits performed by the
Codex agent. They are not a second human/domain review; the ledger records
`human_domain_review=false`.

## Evaluation policy

`dev` exposes only `primary_tune`. `promotion` exposes
`primary_locked + hard` and refuses to run without a candidate-freeze
manifest binding the corpus, evaluator config, and exact prediction artifacts.

The LOTO, LOYO, LORO, LOMO, composition, and scope/period documents are
overlapping diagnostic views. They must not be reported as independent
replications or averaged as though their samples were disjoint.

Oracle-evidence mode inserts gold evidence. Its DOCS/TABLES values are therefore
bypassed placeholders and must not be interpreted as retrieval performance.
End-to-end mode is the only mode that measures submitted retrieval.

Missing typed output scores zero over the complete evaluation denominator.
Conditional `given_typed` fields are secondary diagnostics.

## Baseline commands already executed

Oracle control:

    python scripts/72_g3b_evaluate.py --policy-mode dev --evidence-mode oracle_evidence --typed-predictions data/g3b_v1/g3b_oracle_predictions.jsonl --out artifacts/g3b_v1/oracle_dev_evaluation.json
    python scripts/71_g3b_freeze_candidate.py --candidate-name oracle-selection-v2-control --typed-predictions data/g3b_v1/g3b_oracle_predictions.jsonl --out artifacts/g3b_v1/oracle_promotion_freeze.json
    python scripts/72_g3b_evaluate.py --policy-mode promotion --evidence-mode oracle_evidence --typed-predictions data/g3b_v1/g3b_oracle_predictions.jsonl --candidate-freeze artifacts/g3b_v1/oracle_promotion_freeze.json --out artifacts/g3b_v1/oracle_promotion_evaluation.json

B0 dev:

    python scripts/57_clean_retrieve_v2.py --questions data/g3b_v1/g3b_dev_questions.jsonl --out artifacts/g3b_v1/b0_dev_retrieval.jsonl
    python scripts/60_run_clean_b0_v2.py --retrieval artifacts/g3b_v1/b0_dev_retrieval.jsonl --out artifacts/g3b_v1/b0_dev_codegen.jsonl --no-resume
    python scripts/74_g3b_build_submission.py --retrieval artifacts/g3b_v1/b0_dev_retrieval.jsonl --codegen artifacts/g3b_v1/b0_dev_codegen.jsonl --questions data/g3b_v1/g3b_dev_questions.jsonl --out-dir artifacts/g3b_v1/b0_dev_submission --sub-k 5
    python scripts/72_g3b_evaluate.py --policy-mode dev --evidence-mode end_to_end --submission artifacts/g3b_v1/b0_dev_submission --out artifacts/g3b_v1/b0_dev_evaluation.json

B0 promotion was retrieved, generated, and packaged first. Only then was the
exact submission frozen and the locked/hard report opened:

    python scripts/71_g3b_freeze_candidate.py --candidate-name clean-b0-frozen-production --submission artifacts/g3b_v1/b0_promotion_submission --out artifacts/g3b_v1/b0_promotion_freeze.json
    python scripts/72_g3b_evaluate.py --policy-mode promotion --evidence-mode end_to_end --submission artifacts/g3b_v1/b0_promotion_submission --candidate-freeze artifacts/g3b_v1/b0_promotion_freeze.json --out artifacts/g3b_v1/b0_promotion_evaluation.json

All G3B submission directories contain an offline-only marker and use synthetic
IDs. Do not upload any G3A/G3B evaluation zip to the competition.

## Baseline diagnostics

| Metric | Dev (54) | Promotion (55) |
|---|---:|---:|
| DOCS F2 macro | 0.903025 | 0.898216 |
| DOCS MRR@5 | 0.972222 | 0.990909 |
| TABLES F2 macro | 0.612058 | 0.613054 |
| TABLES MRR@5 | 0.789198 | 0.798182 |
| Leaf Recall@5 | 0.808642 | 0.786364 |
| FullPlanCoverage | 0.685185 | 0.654545 |
| Answer Accuracy | 0.166667 | 0.181818 |
| Execution Accuracy | 0.166667 | 0.181818 |
| Typed output coverage | 0.000000 | 0.000000 |

The Selection-v2 oracle control is 1.0 on typed AST, typed execution, answer,
and execution in both policies. That establishes internal gold/compiler
consistency only.

## Test command

    python -m pytest tests -q -p no:cacheprovider --basetemp artifacts/pytest_g3b_full_20260823
    python -m compileall -q vifinqa scripts

Observed result: 356 tests passed. The repository-root `.pytest_cache` has a
host ACL problem, so cache is intentionally disabled and basetemp remains under
`artifacts`.

## Boundary for G3C

Production retrieval, planner, compiler, arbitration, model, threshold, and
B0/B1 configs were not changed in G3B. G3C may now begin, but each retrieval
treatment must be selected on dev and frozen before promotion evaluation.
Keep planner/compiler/arbitration unchanged throughout the retrieval ablation.
