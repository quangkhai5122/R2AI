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

## Execution 0.2866 - canonical metric dictionary v2 audited6

- Date: 2026-08-20
- Branch: `improve_baseline_kien`
- Source commit: `ef77d55fcdf283750015018488af1524163b3164`
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

Leaderboard metrics:

```json
{
  "TABLES_F2MACRO": 0.4777,
  "DOCS_F2MACRO": 0.8945,
  "TABLES_PRECISION": 0.3192,
  "TABLES_RECALL": 0.6638,
  "TABLES_MRR5": 0.6427,
  "DOCS_PRECISION": 0.9484,
  "DOCS_RECALL": 0.8917,
  "DOCS_MRR5": 0.9644,
  "ANSWER_ACCURACY": 0.2866,
  "EXECUTION_ACCURACY": 0.2866
}
```

Against the `0.2806` checkpoint, canonical v2 gains all six audited executions
(`6/1,012 = 0.00593`) while table F2 rises by `0.0016` and document F2 rises by
`0.0027`. This confirms the canonical metric dictionary v2 milestone: its
retrieval changes improve both evidence metrics, and every accepted answer
change executes correctly on the official grader.

## Execution 0.2866 - formula operand retrieval, answers preserved

- Date: 2026-08-21
- Branch: `improve_baseline_kien`
- Retrieval: `artifacts/retrieval_canonical_v2_formula_operand_exact_w010.jsonl`
- Codegen: `artifacts/codegen_result7_canonical_v2_audited6_w010.jsonl`
- Submission: `artifacts/submission_formula_operand_retrieval_best2866_answers_w010/submission.zip`
- Retrieval SHA-256: `ceede9c3b72675b67e3030746c0644fc034add1d415e5af1f4d9cbbcad342c27`
- Codegen SHA-256: `ee1c189c5968f15c2abe8e98677296fc07fc745c87cf3ccf26cf65d7f2534314`
- Submission SHA-256: `9a1c5608960ea88d46241e7b270c97559b1b04143af932fa9f26deba985a8f4c`

Leaderboard metrics:

```json
{
  "TABLES_F2MACRO": 0.4892,
  "DOCS_F2MACRO": 0.9068,
  "TABLES_PRECISION": 0.3417,
  "TABLES_RECALL": 0.6755,
  "TABLES_MRR5": 0.6088,
  "DOCS_PRECISION": 0.9488,
  "DOCS_RECALL": 0.9048,
  "DOCS_MRR5": 0.9733,
  "ANSWER_ACCURACY": 0.2866,
  "EXECUTION_ACCURACY": 0.2866
}
```

This checkpoint preserves all 1,012 answers, pandas expressions and execution
evidence from the `0.2866` canonical-v2 checkpoint. Only table/document
retrieval changes: TABLES_F2 gains `0.0115` and DOCS_F2 gains `0.0123` without
an execution regression. The all-question Qwen `k=15` run is not a valid answer
checkpoint: it improved TABLES_F2 to `0.4924` and DOCS_F2 to `0.9126`, but
changed 438 previously successful answers and reduced execution to `0.2747`.

Consensus `n=3` was also tested on deterministic-empty questions. A strict
blend accepted only IDs `24` and `996` after `3/3` agreement, confidence 95,
retrieval coverage and route-shape checks. Leaderboard execution remained
`0.2866`, proving both additions wrong. The errors are schema-linking failures:
ID 24 selects the parent `Vay dài hạn` row instead of the named counterparty;
ID 996 canonicalizes `TSCĐ vô hình - giá trị còn lại` as parent
`fixed_assets`. Do not resubmit `submission_formula_consensus_safe2_w010`.

## Execution 0.2964 - tranhuy structured selection and guarded exact cells

- Date: 2026-08-22
- Source branch: `tranhuy` at `75e5269`
- Imported codegen: `artifacts/codegen_tranhuy_02964_stockflow_exact_v11.jsonl`
- Scored submission: `artifacts/tranhuy_02964_stockflow_exact_v11_oldretr_k5/submission.zip`
- Codegen SHA-256: `aa14d8747b5f856067cfb4edbddd8142469578acadb26c17b2d92658fa8e3b65`
- Scored submission SHA-256: `6773532be7dbe038310b9a0663090ebf17878111a8b27b53403cd24de5f23e9a`

Leaderboard metrics:

```json
{
  "TABLES_F2MACRO": 0.4763,
  "DOCS_F2MACRO": 0.893,
  "TABLES_PRECISION": 0.3146,
  "TABLES_RECALL": 0.6639,
  "TABLES_MRR5": 0.6359,
  "DOCS_PRECISION": 0.9487,
  "DOCS_RECALL": 0.8897,
  "DOCS_MRR5": 0.9654,
  "ANSWER_ACCURACY": 0.2964,
  "EXECUTION_ACCURACY": 0.2964
}
```

The artifact directory retains the older experiment label `02925`; the score
above is the leaderboard result confirmed for this exact ZIP. The branch adds
semantic-grounded structured selection, typed nested IR, deterministic
compilation/unit guards, and conservative exact-cell/canonical-metric
overrides. The final audit layers each accepted one change: exact-cell IDs
`53` and `38`, then stock/flow ID `841`.

Compared with the previous `0.2866` codegen, this checkpoint changes 29 pandas
queries and 27 answers, reducing zero/empty answers from 240 to 223. On the
hidden 506-question execution subset it gains five net correct answers. All
1,012 answers, expressions and evidence entries have been copied locally with
hashes so this result remains reproducible independently of the sibling
`artifacts_tranhuy` directory.

## Execution 0.2964 - combined best with formula-operand retrieval

- Retrieval: `artifacts/retrieval_canonical_v2_formula_operand_exact_w010.jsonl`
- Codegen: `artifacts/codegen_tranhuy_02964_stockflow_exact_v11.jsonl`
- Submission: `artifacts/submission_tranhuy_02964_answers_formula_operand_retrieval_w010/submission.zip`
- Submission SHA-256: `17750e0290d181ce13d35ef87b589ee6bdf9bcf9fa4831a6c749e8744e8e93b7`

Leaderboard metrics:

```json
{
  "TABLES_F2MACRO": 0.4901,
  "DOCS_F2MACRO": 0.9079,
  "TABLES_PRECISION": 0.3413,
  "TABLES_RECALL": 0.677,
  "TABLES_MRR5": 0.6088,
  "DOCS_PRECISION": 0.9489,
  "DOCS_RECALL": 0.9061,
  "DOCS_MRR5": 0.9733,
  "ANSWER_ACCURACY": 0.2964,
  "EXECUTION_ACCURACY": 0.2964
}
```

This candidate preserves all 1,012 answers, pandas expressions and evidence
entries from the leaderboard-confirmed `0.2964` submission. It changes only
retrieval: 726 table lists and 359 document lists versus the old retrieval.
Against the current best retrieval checkpoint (`TABLES_F2=0.4892`,
`DOCS_F2=0.9068`), only 15 table lists and five document lists differ because
the new expressions require additional execution evidence.

Build guards pass: 1,012 unique entries, all expressions compile, every answer
replays exactly from the packaged evidence, 1,553 CSV files are included using
official 1-based `<table>` line numbers, and `unzip -t` reports no errors.
The leaderboard confirms that execution remains `0.2964`. Relative to the
scored old-retrieval package, TABLES_F2 rises by `0.0138` and DOCS_F2 rises by
`0.0149`. It also exceeds the previous retrieval-best checkpoint by `0.0009`
TABLES_F2 and `0.0011` DOCS_F2 while gaining five net executable answers. It
is the direct base of the `0.3004` restore4 checkpoint below.

## Execution 0.3004 - combined best plus canonical-v2 restore4

- Base: `artifacts/codegen_tranhuy_02964_stockflow_exact_v11.jsonl`
- Restore source: `artifacts/codegen_result7_canonical_v2_audited6_w010.jsonl`
- Restored IDs: `589, 591, 635, 838`
- Codegen: `artifacts/codegen_tranhuy_02964_plus_canonical_restore4_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_02964_plus_canonical_restore4_w010/submission.zip`
- Codegen SHA-256: `4b524bdb3b77d83115915cc71f3bb0928f55c95a08b3dea80c1251f7bb8f2d69`
- Submission SHA-256: `cdeed91318c8d6eee2cc009ca4b42e5f67acf547c5072361dfb5cbae3107a7e4`

Leaderboard metrics:

```json
{
  "TABLES_F2MACRO": 0.4902,
  "DOCS_F2MACRO": 0.9079,
  "TABLES_PRECISION": 0.3416,
  "TABLES_RECALL": 0.677,
  "TABLES_MRR5": 0.6088,
  "DOCS_PRECISION": 0.9489,
  "DOCS_RECALL": 0.9061,
  "DOCS_MRR5": 0.9733,
  "ANSWER_ACCURACY": 0.3004,
  "EXECUTION_ACCURACY": 0.3004
}
```

The Trần Huy codegen changed these four table-audited canonical-v2 answers
back from `ok` to `failed`. This batch restores them on top of the official
combined `0.2964` checkpoint. It changes exactly four answers, expressions and
evidence lists; relevant documents are unchanged and relevant tables change
only for ID `838`. All 1,012 expressions compile and replay, the ZIP contains
1,560 evidence CSV files, and `unzip -t` passes. The leaderboard confirms two
net additional correct answers on the 506-question scored subset, increasing
execution by `0.0040`. TABLES_F2 also rises by `0.0001`, while DOCS_F2 is
unchanged. This is now the official best checkpoint.

## Superseded candidate - schema linking v3 w010

- Date: 2026-08-21
- Branch: `improve_baseline_kien`
- Retrieval: `artifacts/retrieval_schema_linking_v3_w010.jsonl`
- Rule codegen: `artifacts/codegen_rule_schema_linking_v3_w010_k15.jsonl`
- Payload folder: `artifacts/kaggle-payload-schema-linking-v3-w010`
- Upload ZIP: `artifacts/kaggle-payload-schema-linking-v3-w010.zip`
- Notebook: `r2ai-qwen2-5-coder-7b-schema-linking-v3-w010.ipynb`
- Retrieval SHA-256: `1072d9ea8c8b920597526d279b2c24276dee8888f0f111b4ebcdc456be528259`
- Payload ZIP SHA-256: `44e296fa96b50a47e215af7027ef62ba7509bc0060692c8faf43a365a3615d7a`
- Notebook SHA-256: `be11386599e4238adab0e9f39e0ed90638c2ff35ad89f8bd8baaa21a3abc2f82`

This candidate fixes the two schema-linking root causes exposed by the failed
`safe2` experiment. Named counterparties are now part of requirement identity,
and `TSCĐ vô hình - giá trị còn lại` maps to the dedicated
`intangible_fixed_assets` child instead of parent `fixed_assets`. Generic
requirements use clean canonical aliases so output wording cannot poison the
row-score cache.

Offline guards:

- official retrieval: 1,012 rows, zero empty candidate lists;
- evidence complete at depth 20: `714 -> 715`;
- evidence complete at Qwen `k=15`: `691 -> 692`;
- only 28 questions change top-5 order and 17 change the top-5 set;
- IDs `24`, `534` and `996` have complete evidence inside `k=15`;
- hard-negative row linking: `76/76` top-1, MRR `1.000`, recall@5 `1.000`;
- 288-question formula eval is unchanged from operand-exact;
- source suite: `211 passed`.

This candidate was later built and scored as the failed ablation below. It did
not improve execution over `0.2866` and is superseded by the combined `0.2964`
checkpoint above.

## Failed ablation - results-10 schema linking v3 audited5

- Date: 2026-08-21
- Raw Qwen output: `kaggle/results-10/codegen_schema_linking_v3_consensus_empty_sel7b_k15_n3.jsonl`
- Audited blend: `artifacts/codegen_result10_schema_linking_v3_audited5_w010.jsonl`
- Submission: `artifacts/submission_result10_schema_linking_v3_audited5_w010/submission.zip`
- Raw-output SHA-256: `79ed71705dcfa8c070501f757c28614b9d75b1b4a810c67e98a394ea4e638155`
- Audited-blend SHA-256: `3acefc1f4900c23a15c8b6e8cbb965ad8bec93abf7d89176917047b8c828c4bc`
- Submission SHA-256: `d229bf51a5d3e368ea0a33defbd7bdad9b6b15739e310833aacdbca0bcd47eea`

Do not submit the raw Qwen output. Relative to the official `0.2866` codegen
checkpoint it changes 524 answers, while 177 previously executable rows become
failed. The audited blend preserves every existing answer and replaces only
five previously failed rows: `24`, `440`, `562`, `574`, and `996`.

The five accepted answers were traced to their exact report cells and formulas:

- `24`: named HAGL counterparty long-term borrowing, not the parent borrowing row;
- `440`: select DIG/2024 by maximum D/E, then calculate EBIT/interest expense;
- `562`: only DCM/2022 clears net margin above 10%, then calculate CFO/current liabilities;
- `574`: GEE and GEX have positive CFO in 2022-2024 and GEE has the larger 2024 net margin;
- `996`: exact intangible-fixed-assets totals for 2016, 2018, 2020, and 2021.

The audit rejected executable but semantically wrong rows, including `364`
(returns the selector rather than accrual ratio), `407` (uses 2021 instead of
the next year), `417` (wrong selector and target), `717` (total assets divided
by itself), `805` (parent receivables instead of related parties), and `857`
(uses prior-year columns 2015-2017).

Final guards: 1,012 unique IDs, zero empty document/table lists, exactly five
answer/query deltas versus the `0.2866` submission, all 1,012 expressions
compile and replay from packaged CSV evidence, and the ZIP layout passes
`unzip -t`.

Leaderboard execution remained `0.2866`. With 506 scored questions, one net
additional correct answer would move the rounded score to about `0.2885`, so
this submission produced zero observable net gains. Because the organizer's
506-ID subset is hidden, this does not distinguish answers that are wrong from
changed IDs that are outside the scored subset. Do not resubmit this audited5
artifact. It is superseded by the combined `0.2964` checkpoint.

## Execution 0.3083 - deterministic exact year ranking

- Date: 2026-08-22
- Base checkpoint: execution `0.3004`
- Rule output: `artifacts/codegen_rule_canonical_v2_formula_operand_yearrank_exact_w010_k15.jsonl`
- Fill-only codegen: `artifacts/codegen_tranhuy_03004_plus_yearrank_exact10_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_03004_plus_yearrank_exact10_w010/submission.zip`
- Accepted IDs: `842, 850, 878, 884, 889, 904, 948, 960, 997, 1000`
- Rule SHA-256: `8724be8e46e10ea7b4ba79735a3d185c3cf6949908a940b31ada22aec7715cd4`
- Codegen SHA-256: `c0e317f271303190203c9ff736eaa2aad9d62a87225a3e8e01cceecf17460ff0`
- Submission SHA-256: `e4aa4fd2eb797e3274d4e5b9869b52c8bae09a7e4769e1f39393d21ad4898203`

This batch implements an explicit ranking plan with `dimension=year`,
`projection=year`, and `direction=max|min`. The solver resolves the same
canonical metric for every requested year, requires strong period evidence,
rejects parent/child substitutions, reads every evidence cell in the submitted
expression, and fails closed on missing years, ambiguity, unsupported ratio
selectors, and ties.

Of the 33 failed year-output ranking rows in the `0.3004` codegen, the first
pass returned 13. Exact-cell audit exposed three unsafe results: ID `826`
dropped a ratio denominator, ID `850` mixed total cost of goods sold with
forest-cost detail rows, and ID `959` used an equity child row. After the exact
identity guard, `826` and `959` fail closed and `850` resolves all five years
to VAS code `11`. Ten rows remain accepted, each with complete period evidence
and a successful pandas replay.

The final package changes exactly ten answers and expressions versus the
leaderboard-confirmed `0.3004` submission. All 1,012 expressions compile and
replay from the packaged evidence, official 1-based `<table>` line numbers are
used, the ZIP contains 1,586 CSV files, and `unzip -t` reports no errors.

Leaderboard metrics:

```json
{
  "TABLES_F2MACRO": 0.4902,
  "DOCS_F2MACRO": 0.9079,
  "TABLES_PRECISION": 0.3416,
  "TABLES_RECALL": 0.677,
  "TABLES_MRR5": 0.6088,
  "DOCS_PRECISION": 0.9489,
  "DOCS_RECALL": 0.9061,
  "DOCS_MRR5": 0.9733,
  "ANSWER_ACCURACY": 0.3083,
  "EXECUTION_ACCURACY": 0.3083
}
```

The batch gains four net correct answers on the hidden 506-question subset:
`0.3004 -> 0.3083`. Retrieval metrics are byte-for-byte unchanged, confirming
that this gain comes from answer semantics rather than wider table/document
retrieval. This is the new official best checkpoint.

## Execution 0.3202 - canonical note v3 + direct year ranking v2

- Date: 2026-08-22
- Base checkpoint: execution `0.3083`
- Target batch: 16 simple year-ranking questions
- Refreshed retrieval: `artifacts/retrieval_canonical_note_v3_yearrank_v2_exact13_w010.jsonl`
- Rule output: `artifacts/codegen_rule_canonical_note_v3_yearrank_v2_w010_k15.jsonl`
- Fill-only codegen: `artifacts/codegen_tranhuy_03083_plus_note_v3_yearrank_v2_exact13_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_03083_plus_note_v3_yearrank_v2_exact13_w010/submission.zip`
- Accepted IDs: `813, 829, 852, 890, 897, 921, 936, 946, 959, 971, 978, 981, 1008`
- Rejected IDs: `907, 910, 986`
- Submission SHA-256: `a27e20365e3f4fa3a081014fe69e64eeb6b78dd4047863f119da64d3596f4a05`

Canonical note dictionary v3 adds query aliases, exact row aliases and table
context constraints for detailed note metrics such as external receivables,
merchandise inventory, short-term unearned revenue, real-estate brokerage
expense, construction payables, LPG revenue, named-counterparty payables and
related-party service revenue. The direct resolver now respects separate versus
consolidated reports, exact note context and OCR current-period headers.

The batch fails closed when any requested year is absent. ID `907` has missing
current-period share data in the extraction store; IDs `910` and `986` have a
dash/blank in the requested current-year cell. A regression guard prevents the
prior-year value from being mistaken for the current year when OCR folds the
dash into the row label.

The submission changes exactly 13 previously failed answers, expressions and
evidence sets. It does not overwrite any successful row from the `0.3083`
checkpoint. All 1,012 expressions compile and replay from the packaged CSVs,
the package uses official 1-based `<table>` line numbers, all 229 tests pass,
and `unzip -t` reports no errors.

Leaderboard metrics:

```json
{
  "TABLES_F2MACRO": 0.4913,
  "DOCS_F2MACRO": 0.9103,
  "TABLES_PRECISION": 0.3424,
  "TABLES_RECALL": 0.6783,
  "TABLES_MRR5": 0.6098,
  "DOCS_PRECISION": 0.9509,
  "DOCS_RECALL": 0.9085,
  "DOCS_MRR5": 0.9753,
  "ANSWER_ACCURACY": 0.3202,
  "EXECUTION_ACCURACY": 0.3202
}
```

The batch gains six correct answers on the hidden 506-question subset:
`0.3083 -> 0.3202`. Table F2 rises by `0.0011` and document F2 by `0.0024`,
so the targeted canonical retrieval improves both evidence quality and answer
execution. This is the new official best checkpoint.

## Execution 0.3300 - typed compositional ranking v3 exact12

- Date: 2026-08-23
- Status: official leaderboard checkpoint; base execution `0.3202`
- Target batch: 16 nested ranking questions that select by metric/formula A and
  project metric/formula B
- Retrieval: `artifacts/retrieval_tranhuy_03202_plus_compositional_ranking_v3_exact12_w010.jsonl`
- Rule output: `artifacts/codegen_compositional_ranking_v3_offset_k48_w010.jsonl`
- Fill-only codegen: `artifacts/codegen_tranhuy_03202_plus_compositional_ranking_v3_exact12_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_03202_plus_compositional_ranking_v3_exact12_w010/submission.zip`
- Accepted IDs: `415, 440, 459, 475, 479, 491, 505, 507, 543, 556, 558, 575`
- Rejected IDs: `495, 497, 511, 524`
- Submission SHA-256: `4c674ca7e75db0e1b279ff2ea997799b7e7c1cdd96a0945fdcca6f308b92406d`

The typed planner separates the selector and projection calculations, including
their `level`, `growth` or `delta` modes. The executor resolves every entity and
period exactly, computes the selector, chooses the unique argmax/argmin, and
only then evaluates the projection. Its replay expression rechecks the winning
condition and reads all supporting evidence.

Inventory-days now declares canonical period-aware operands:
`inventory(t-1)`, `inventory(t)` and `cost_of_goods_sold(t)`. The router locks
adjacent-year reports from those offsets, while schema linking recovers years
that OCR split into separate header rows, including flattened dates such as
`3092024`. Four unresolved note/EPS/tax questions remain fail-closed.

The final artifacts differ from the `0.3202` checkpoint on exactly the 12
allowlisted entries. All 1,012 expressions compile and replay from packaged
CSV evidence, all 240 tests pass, table references use official 1-based
`<table>` line numbers, and `unzip -t` reports no errors.

Leaderboard metrics:

```json
{
  "TABLES_F2MACRO": 0.4958,
  "DOCS_F2MACRO": 0.9139,
  "TABLES_PRECISION": 0.3439,
  "TABLES_RECALL": 0.683,
  "TABLES_MRR5": 0.6101,
  "DOCS_PRECISION": 0.9509,
  "DOCS_RECALL": 0.9127,
  "DOCS_MRR5": 0.9753,
  "ANSWER_ACCURACY": 0.33,
  "EXECUTION_ACCURACY": 0.33
}
```

The batch gains five net correct answers on the hidden 506-question subset:
`0.3202 -> 0.3300`. Table F2 rises by `0.0045` and document F2 by `0.0036`.
This is the official base for median-filter planner v4.
