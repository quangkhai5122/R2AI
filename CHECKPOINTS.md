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
- Safe blend: `artifacts/codegen_result6_canonical_v2_safe20_w010.jsonl`
- Submission: `artifacts/submission_p2_canonical_v2_safe20_w010/submission.zip`
- Kaggle payload: `artifacts/kaggle-payload-canonical-v2-w010.zip`
- Kaggle notebook: `r2ai-qwen2-5-coder-7b-canonical-v2-w010.ipynb`
- Retrieval SHA-256: `c7fa88ad54294eb03881569126e24033e2ac73c8f111ae867110c1c44b03c28c`
- Safe-blend codegen SHA-256: `88379883fe7933e14b948d7e6d02c9bf56f3c390e7c8b9061195bdf60462500b`
- Submission SHA-256: `28a0ccd226461ba5e6cb22158c1504c902cb8fdfa8485b304bbaf7327953a091`
- Payload SHA-256: `e0f681a0de9978da6a7081dbd9b423633945ddf96441e2a353b6aa4924b7cb41`

V2 registers 139 metrics and links 791/1,012 official questions (78.16%), up
from 534/1,012 (52.77%). It adds structured qualifiers and exact row identity.
On the 40-question synthetic validation set, rule execution rises from 0.700
for canonical v1 to 0.725 for v2. The official safe blend preserves every
successful 0.2806 checkpoint row and fills only 20 failed rows whose v2 rule
confidence is at least 90 with no `AMBIGUOUS` or `UNIT-WARN` marker. All 1,012
submission expressions replay locally; the source suite passes (`182 passed`).

This is intentionally recorded as a candidate, not a leaderboard checkpoint.
Replace this note with measured metrics only after the Kaggle v2 inference and
official submission have completed.
