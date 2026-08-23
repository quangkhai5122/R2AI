# G3A evaluation gate v1

G3A is an offline, same-corpus/new-question gate for ViFinQA. It evaluates
the complete competition output vector instead of optimizing a guessed private
leaderboard weight.

## What is locked

- 144 source-derived questions over the official document corpus.
- primary_tune: 80 questions.
- primary_locked: 48 questions.
- hard: 16 manually evidence-reviewed questions.
- 254 selected source cells.
- No question id or exact normalized question text overlaps the 1,012 public
  questions.
- No fact leaf overlaps tune, locked, and hard splits.
- The public question file is read only to compute exclusion hashes. Public
  wording is never a generation template.
- Hard reviews bind to a SHA-256 subject over question, answer, evidence,
  program, relevant documents, and relevant tables. A changed gold record
  invalidates its review automatically.

Strict bundle fingerprint:

956596dff2db953f715f4cfed5a62209438897cce1dc4d37e596d3168b8213b2

## Metric contract

Every evaluation report contains:

- DOCS macro precision, recall, F2, and MRR@5.
- TABLES macro precision, recall, F2, and MRR@5.
- Answer accuracy with per-record absolute tolerance.
- Execution accuracy by replaying the submitted pandas expression against the
  submitted CSV evidence.
- Integrity checks for missing/extra/duplicate ids, question mismatch,
  non-finite answers, and duplicate retrieval items.
- Breakdowns by split, set, operator, and difficulty.

Private weights are unknown. Raw metrics are authoritative. Balanced,
answer-heavy, and retrieval-heavy scores are sensitivity scenarios only.

## Rebuild and validate

Run from the repository root in the project environment:

    python scripts/65_g3a_build.py build
    python scripts/65_g3a_build.py validate

The second command fails if a bundle hash changes or any hard record lacks an
approved, matching review.

## Run a candidate

    python scripts/57_clean_retrieve.py --questions data/g3a_v1/g3a_questions.jsonl --out artifacts/g3a_v1/candidate_retrieval.jsonl
    python scripts/60_run_clean_b0.py --retrieval artifacts/g3a_v1/candidate_retrieval.jsonl --out artifacts/g3a_v1/candidate_codegen.jsonl
    python scripts/68_g3a_build_submission.py --retrieval artifacts/g3a_v1/candidate_retrieval.jsonl --codegen artifacts/g3a_v1/candidate_codegen.jsonl --questions data/g3a_v1/g3a_questions.jsonl --out-dir artifacts/g3a_v1/candidate_submission --sub-k 5
    python scripts/66_g3a_evaluate.py --submission artifacts/g3a_v1/candidate_submission --out artifacts/g3a_v1/candidate_evaluation.json
    python scripts/67_g3a_compare.py --baseline artifacts/g3a_v1/b0_evaluation.json --candidate artifacts/g3a_v1/candidate_evaluation.json --out artifacts/g3a_v1/candidate_promotion.json

The offline submission builder always uses offline_eval=True. The output is
named OFFLINE_EVAL_DO_NOT_UPLOAD.zip, and the directory contains a
DO_NOT_UPLOAD.txt marker.

## Replayed B0 baseline

The checked local replay over all 144 questions produced:

| Metric | B0 |
|---|---:|
| DOCS precision macro | 0.996528 |
| DOCS recall macro | 0.993056 |
| DOCS F2 macro | 0.992670 |
| DOCS MRR@5 | 0.996528 |
| TABLES precision macro | 0.303935 |
| TABLES recall macro | 0.947917 |
| TABLES F2 macro | 0.648848 |
| TABLES MRR@5 | 0.888310 |
| Answer accuracy | 0.520833 |
| Execution accuracy | 0.520833 |

Answer accuracy by operator is diagnostic:

- lookup: 1.000000
- difference: 0.923077
- average: 0.894737
- growth percent: 0.000000
- net margin: 0.000000
- debt to assets: 0.000000

This confirms that the gate separates retrieval health from missing typed
formula semantics.

## Promotion policy

A candidate is blocked when:

- submission integrity fails;
- any main task exceeds its configured regression allowance;
- any configured private-weight scenario regresses;
- hard-set answer accuracy regresses; or
- no main task has a material gain.

Thresholds are configuration, not competition facts. Change them only before a
treatment is evaluated, then commit and freeze the config hash.

## Scope limitations

G3A v1 covers six program families and eight high-confidence VAS line codes. The 16 hard records were manually evidence-reviewed by the Codex agent and independently recomputed, but have not yet received a second human/domain review.
It is a reliable first gate, not a complete model of private questions. It does
not yet represent ranking, count, CAGR, nested formulas, note-table semantics,
non-monetary quantities, or ambiguous entity aliases. G3A v1.1 should add those
families with independently reviewed gold before the five private candidates
are frozen.
