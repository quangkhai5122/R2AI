# Clean canonical baseline progress - 2026-08-21

## Provenance

- branch: clean-canonical-baseline-v1;
- implementation base: main at 0ce20aa72e636c33af659709b38d98da89e98c77;
- completed-run source alignment verified post hoc against
  a5a540b0ddeee6041b10abfc872e3d04617a0120;
- remote payload schema: 9;
- remote runtime profile: hf-bitsandbytes-nf4-v1;
- validator profile: clean-codegen-select-v2-v2.

The Git commit was not embedded in the remote artifacts. Packaged source hashes
matched the checkout in nine pre-edit provenance tests.

## G0-G2 status

G0 quarantine, G1 clean provenance/retrieval, and G2 canonical
finance/Selection-v2 integration are complete. Historical public-derived
artifacts remain ineligible as clean seeds.

## Completed B1 run

The effective model was Qwen/Qwen2.5-Coder-14B-Instruct, not 7B and not AWQ.
The model revision observed by Kaggle was
aedcc2d42b622764e023cf882b6652e646b95671. It ran on two Tesla T4 GPUs with
bitsandbytes NF4.

- records: 1,012;
- LLM completed: 1,012;
- status: 763 ok, 249 failed;
- source: 53 LLM, 353 rule, 357 rule_composite, 249 none;
- Selection v2: 263 accepted, 203 rejected, 546 no_candidates;
- codegen SHA-256:
  a8c2b93279daa7099ce0fdcead123cf9df134687fefeb48b484dd66267f9371c;
- submission SHA-256:
  c98f1859e41a924458abfc7f5b2f2673e028136e7a73b2cb04c6cb84467cb75c.

All independent integrity and replay checks passed. No labels were used, so this
does not establish accuracy.

## B0 comparison

B0 has 743 ok and 269 failed records. B1 converts 20 failed records to ok,
never converts ok to failed, and changes 33 answers among the 743 records that
remain ok. These 53 LLM decisions require OOD labels before their direction can
be judged.

## Drift corrected in this session

- canonical config and registry now identify B1 14B NF4;
- active launcher, payload builder, and notebook default to base Qwen 14B;
- AWQ/GPTQ is historical only;
- actual runtime packages, model revision, hashes, and observed counts are saved
  in an immutable run record;
- a reproducible output audit was added;
- active docs no longer send agents to a 7B/AWQ notebook;
- the notebook now writes a full run_config and enforces complete dependency
  version specifications.

The completed run's old payload declared a 7B default. That field is preserved
in the run record as historical provenance rather than edited retroactively.

## Next milestone

G3 is the next required milestone: build and freeze a source-derived OOD
evaluation split independent of the 1,012 official questions. After G3, test one
guarded B2 evidence/shortlist-rescue ablation while holding model, compiler, and
arbitration fixed.

Do not use aggregate official traces to tune thresholds or add per-ID fixes.
