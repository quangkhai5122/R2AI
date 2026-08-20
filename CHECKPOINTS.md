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

## Candidate - canonical metric dictionary v2 (leaderboard pending)

- Branch: `improve_baseline_kien`
- Retrieval: `artifacts/retrieval_p2_canonical_v2_qualified_hybrid_w010.jsonl`
- Rule codegen: `artifacts/codegen_rule_canonical_v2_w010_k15.jsonl`
- Kaggle output: `kaggle/results-7/codegen_canonical_v2_sel7b.jsonl`
- Audited blend: `artifacts/codegen_result7_canonical_v2_audited6_w010.jsonl`
- Submission: `artifacts/submission_p2_canonical_v2_result7_audited6_w010/submission.zip`
- Kaggle payload: `artifacts/kaggle-payload-canonical-v2-w010.zip`
- Kaggle notebook: `r2ai-qwen2-5-coder-7b-canonical-v2-w010.ipynb`
- Retrieval SHA-256: `c7fa88ad54294eb03881569126e24033e2ac73c8f111ae867110c1c44b03c28c`
- Kaggle-output SHA-256: `f2e03eb7aee290cf09d20f8aaa1ac5a04dbae7d15e090c5585ae8e95f0de2f96`
- Audited-blend SHA-256: `ee1c189c5968f15c2abe8e98677296fc07fc745c87cf3ccf26cf65d7f2534314`
- Submission SHA-256: `5fb8fb48a4bfa5e539baebf4e108bc8c7e8364966c43a9db25ff67a1fd0a7147`
- Payload SHA-256: `e0f681a0de9978da6a7081dbd9b423633945ddf96441e2a353b6aa4924b7cb41`

V2 registers 139 metrics and links 791/1,012 official questions (78.16%), up
from 534/1,012 (52.77%). It adds structured qualifiers and exact row identity.
On the 40-question synthetic validation set, rule execution rises from 0.700
for canonical v1 to 0.725 for v2. Kaggle inference produced 868 executable
rows, including 120 rows that failed in the 0.2806 checkpoint. A semantic audit
found that confidence-only selection admitted many false positives on nested
questions, so the earlier `safe20` artifact is deprecated and must not be
submitted. The final conservative blend preserves the complete 0.2806
checkpoint and adds only six table-verified answers: `589, 591, 635, 757, 830,
838`. All 1,012 submission expressions replay locally; the source suite passes
(`183 passed`).

This is intentionally recorded as a candidate, not a leaderboard checkpoint.
Replace this note with measured metrics only after the official submission has
completed.
