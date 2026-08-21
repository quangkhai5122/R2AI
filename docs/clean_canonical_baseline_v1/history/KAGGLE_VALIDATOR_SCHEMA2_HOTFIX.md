# Kaggle Selection-v2 validator hotfix

## Root cause

The smoke generation itself reached the validation step. The failure at
`validate codegen: 0/3` was caused by a validator/producer contract mismatch:

- `vifinqa.codegen.selection_v2` emits `selection_trace.schema_version = 2`
  and `selection_trace.mode = "select_v2"`.
- The first validator incorrectly required trace schema 1, which belongs to
  the older Selection-v1 trace path.

As a result, a valid completed Selection-v2 record was rejected immediately at
the first row.

## Fix

- Corrected source: `kaggle/validate_clean_codegen_v2.py`
- Payload builder: `scripts/59_make_clean_payload_v5.py`
- Packaged destination remains `code/validate_clean_codegen.py`, so the
  existing `vifinqa-clean-canonical-v1-nf4-v2.ipynb` notebook needs no logic
  change.
- The payload manifest now contains:

```json
"validation_profile": "clean-codegen-select-v2-v2"
```

The corrected validator requires trace schema 2, mode `select_v2`, a valid
outcome, a list-valued attempts trace, a single run signature, finite answers,
exact ID/question alignment, and completed attempts when requested. Error
messages now include a bounded record summary.

## Rebuild and upload

```powershell
python scripts\59_make_clean_payload_v5.py --dataset-id lequangkhai5122005/vifinqa-clean-canonical-v1
kaggle datasets version -p artifacts\clean_v1\kaggle_payload -m "schema9: fix Selection-v2 validator trace schema=2" --dir-mode zip
```

Wait until the new dataset version finishes processing. In Kaggle, refresh or
remove/re-add the notebook Input so it points to the latest dataset version,
then restart the session. Existing `/kaggle/input` mounts do not change inside
an already-running session.

After the payload-discovery cell, confirm:

```python
print(MANIFEST.get("validation_profile"))
assert MANIFEST.get("validation_profile") == "clean-codegen-select-v2-v2"
```

If this prints `None`, the notebook is still mounted to the old payload.

## Verification

- Direct validator contract tests: passed.
- Producer/validator contract-alignment test: passed.
- End-to-end test creating three real Selection-v2 output records and
  validating them with `--require-complete-llm`: passed.
- Built payload hash/source alignment test: passed.
- Full repository suite: 343 passed.

The CUDA/model execution remains Kaggle-only, but the reported failure was
after generation and is reproduced by the old validator contract locally.
