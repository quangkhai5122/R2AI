# Leaderboard Checkpoints

## Execution 0.2806 - canonical direct24 + semantic4

- Date: 2026-08-13
- Branch: `improve_baseline_kien`
- Source commit: `f59797907c1f343d50ff1fbe3f8f24559568b371`
- Retrieval: `artifacts/retrieval_p2_canonical_qualified_hybrid_w010.jsonl`
- Codegen: `artifacts/codegen_result6_canonical_direct24_semantic4_w010.jsonl`
- Submission: `artifacts/submission_p2_canonical_result6_direct24_semantic4_w010/submission.zip`
- Codegen SHA-256: `74c295c5b145ab79ba543b6f505eb72a458ba88eeb984886861b23f28189330a`
- Submission SHA-256: `5999f798565fcc7e9e5545f50a6701737ef846b75bc7c181a78b7bb3e6704176`

Leaderboard metrics:

```json
{
  "TABLES_F2MACRO": 0.4761,
  "DOCS_F2MACRO": 0.8918,
  "TABLES_PRECISION": 0.316,
  "TABLES_RECALL": 0.6629,
  "TABLES_MRR5": 0.6359,
  "DOCS_PRECISION": 0.9487,
  "DOCS_RECALL": 0.8883,
  "DOCS_MRR5": 0.9654,
  "ANSWER_ACCURACY": 0.2806,
  "EXECUTION_ACCURACY": 0.2806
}
```

The submission keeps the audited `0.2648` core and canonical retrieval, then
overrides 24 direct/high-confidence metrics and four semantic metric cases.
Changed question IDs:

```text
4, 61, 72, 101, 109, 123, 176, 201, 255, 265, 278, 289, 307, 310,
337, 665, 676, 698, 699, 718, 719, 739, 743, 746, 762, 778, 795, 800
```

Compared with the preceding `0.2648` submission, this checkpoint gains 16
correct executions out of 1,012 questions. All 1,012 pandas expressions replay
locally and the source test suite passes (`173 passed`). Generated artifacts
remain ignored by Git; the hashes above identify the exact local files.

## Execution 0.2846 - canonical metric v2 fill-only

- Date: 2026-08-18
- Baseline: checkpoint 0.2806
- Submission: `artifacts/candidates/submission_02806_metric_v2b_fill_k15/submission.zip`
- Equivalent later submission: `artifacts/candidates/submission_02806_metric_v2h_fill_k15/submission.zip`
- v2b SHA-256: `30e56e5b14e463411a73cf52150a41956149f8aab63c3ad08b9b61aa92c804b7`
- v2h SHA-256: `035549e2ee3da7c0f6e4ac66dbdbe00bf6d6b8be46a9d5cac50104735db6c8d8`

Leaderboard:

```json
{
  "ANSWER_ACCURACY": 0.2846,
  "EXECUTION_ACCURACY": 0.2846
}
```

Both submissions are grader-equivalent: all 1,012 `id`, `question`, `answer`,
`pandas_query`, `evidence`, `relevant_docs`, and `relevant_tables` fields are
identical. They change exactly IDs `757, 830, 847` from the 0.2806 checkpoint.
The score moves from 142/506 to 144/506 correct answers, so the three changes
produce exactly two net new correct answers and one neutral change. Aggregate
leaderboard feedback cannot identify which ID is neutral.

Canonical dictionary v2 subsequently reached 945/1,012 recognized routes, but
the fail-closed merge admitted no additional answer beyond the same three IDs.
Further dictionary-only expansion is therefore at diminishing returns; the
remaining opportunity is nested selector/typed formula IR and exact-cell
resolution, not more broad aliases.


## Local Candidate: Typed IR Fill V5 (Not Leaderboard-Submitted)

- Date: 2026-08-20
- Retrieval: `artifacts/candidates/retrieval_metric_v2h_full.jsonl`
- Deterministic codegen: `artifacts/candidates/codegen_typed_ir_integrated_v5_full_k15.jsonl`
- Merged codegen: `artifacts/candidates/codegen_02806_typed_ir_fill_v5_k15.jsonl`
- Submission: `artifacts/candidates/submission_02806_typed_ir_fill_v5_k15/submission.zip`
- Submission SHA-256: `e9e6bf5564263062ed4d17979206181079d34a8d4f9c1cc3a2b8df24dce51bc6`

The production fill path now tries direct ranking IR, nested selector IR, and
flat typed IR only after the existing deterministic solvers fail. It replays
the generated expression and requires semantic dataframe grounding before
accepting a result. The full batch produced 28 direct IR answers and 0 nested
IR answers after hardening; the conservative fill-only gate accepted 19 IDs:

```text
822, 832, 860, 866, 874, 876, 879, 883, 900, 906, 925, 928, 929,
933, 953, 974, 985, 989, 999
```

The candidate keeps the `.2846` checkpoint for all other rows and was built
with `sub-k=5`; all 1,012 expressions replay locally and the ZIP contains
1,012 result entries plus 1,553 evidence CSV files. It has not been submitted
to the leaderboard, so no answer-accuracy claim is attached to it.


## Execution 0.2905 - typed IR fill v5

- Date: 2026-08-21
- Baseline: local typed-IR fill v5 candidate over the 0.2846 checkpoint
- Retrieval: `artifacts/candidates/retrieval_metric_v2h_full.jsonl`
- Codegen: `artifacts/candidates/codegen_02806_typed_ir_fill_v5_k15.jsonl`
- Submission: `artifacts/candidates/submission_02806_typed_ir_fill_v5_k15_final/submission.zip`
- Submission SHA-256: `12b7a8d5cebd000c68a127d10adaac40de8b71072223e3b393a3e0f6d4078878`

Leaderboard metrics:

```json
{
  "TABLES_F2MACRO": 0.4668,
  "DOCS_F2MACRO": 0.8948,
  "TABLES_PRECISION": 0.3029,
  "TABLES_RECALL": 0.6552,
  "TABLES_MRR5": 0.6108,
  "DOCS_PRECISION": 0.9493,
  "DOCS_RECALL": 0.892,
  "DOCS_MRR5": 0.9644,
  "ANSWER_ACCURACY": 0.2905,
  "EXECUTION_ACCURACY": 0.2905
}
```

The score is 147/506 correct, +3 over 0.2846. Answer/EXEC improved, but
TABLES_F2MACRO fell by 0.0093 versus the 0.2806 retrieval checkpoint, so the
next A/B must keep this codegen while restoring the previous retrieval pool.


## Local Candidates: exact-cell verifier and retrieval A/B

- Date: 2026-08-21
- Base answer control: `0.2905` typed-IR fill v5.
- Exact-cell verifier codegen: `artifacts/candidates/codegen_02905_exact_cell_verify_v8_k15.jsonl`
- Exact-cell policy accepted only ID `53`, changing `8.45` to `378.56` from the
  `IJC` 2021 separate note table `10. Bất động sản đầu tư`, cell
  `Số cuối năm × Giá trị còn lại`. No ID allowlist is used.

Unsubmitted ZIPs:

| Candidate | Retrieval | Answer changes vs `.2905` | ZIP SHA-256 |
|---|---|---:|---|
| `submission_02905_typed_ir_oldretr_k5` | `.2806` checkpoint pool | none | `559cb44656481fa9e2d36d01e1dd21ab749a98c41f0fe49edab4a89ea3ca7573` |
| `submission_02905_exact_cell_v8_k15` | v2h | ID `53` only | `d2d84d9e8f01ff9815a50bc9c65f521269a2fb86964c8103f8debbab2327cbcd` |
| `submission_02905_exact_cell_oldretr_k5` | `.2806` checkpoint pool | ID `53` only | `fff0c707b4f8a1e850ca410654e842a73f17089342cf26c2c26216cb523f1b25` |

All three candidates pass strict replay with 1,012 entries and 1,553 CSV
files. The old-retrieval A/B changes 727 table lists and 196 doc lists while
keeping answer/query/evidence identical; the combined candidate therefore
isolates the two intended changes. No leaderboard score is claimed yet.
