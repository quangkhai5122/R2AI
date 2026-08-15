# RUNBOOK — P2.4 tune gold hoàn tất

Ngày hoàn tất: 2026-08-11.

## Trạng thái chuẩn

- Tune: **100/100** record `verified`.
- Exact evidence: 578 references, 502 exact cells, 397 tables, 235 reports.
- Typed AST và deterministic pandas replay: PASS 100/100.
- Complex semantic audit: PASS 21/21.
- Complex evidence trỏ vào cột metadata `CHỈ TIÊU/MS/TM`: 0.
- Locked: chưa mở (`locked_opened=false`).
- Bundle fingerprint:
  `311f17edcc8540d52b407c7ab84637f3052108bcb997adaf0fcf8fc04cb436d1`.
- Canonical gold SHA256:
  `8a3725276c4baafbb3ecbeaa3ea8f3bfcc488f16406e3994bd09b3a1a255d331`.

Artifact chính thức:

- `artifacts/devset_p24/p24_tune_authoring.final.v2.jsonl`
- `artifacts/devset_p24/p24_tune_gold.final.jsonl`
- `artifacts/devset_p24/p24_tune_gold.final.audit.json`
- `artifacts/devset_p24/p24_tune_gold.semantic_audit.final.json`
- `notebooks/P2_4_TUNE_GOLD_AUDIT_FINAL.ipynb`

Không dùng predecessor `p24_tune_gold.verified.jsonl`; artifact đó đã được đánh dấu
`DO_NOT_USE` sau khi audit semantic phát hiện OGC PAT chọn cột số thứ tự và HPX inventory
chọn cột `TM`. Bản final đã sửa resolver, dựng lại toàn bộ gold và replay lại 100/100.

## Dựng lại từ authoring parts

Các output đều exclusive-create; nếu file đã tồn tại, dùng tên output mới để audit một lần
chạy mới, không xóa/ghi đè artifact chuẩn.

```powershell
python scripts/40_p24_generate_complex_specs_final.py
python scripts/41_p24_combine_tune_specs_final.py
python scripts/35_p24_fold_medians.py `
  --input artifacts/devset_p24/p24_tune_authoring.corrected.jsonl `
  --output artifacts/devset_p24/p24_tune_authoring.corrected.folded.jsonl
python scripts/36_p24_retain_folded_evidence.py `
  --input artifacts/devset_p24/p24_tune_authoring.corrected.folded.jsonl `
  --output artifacts/devset_p24/p24_tune_authoring.final.v2.jsonl
python scripts/17_p24_author_gold_ext.py `
  --specs artifacts/devset_p24/p24_tune_authoring.final.v2.jsonl `
  --output artifacts/devset_p24/p24_tune_gold.final.jsonl
```

Median trong bốn câu được constant-fold từ chính exact evidence để giữ typed AST dưới
node-budget 256. Input chỉ dùng để tính median vẫn được giữ trong AST bằng provenance term
trọng số 0; strict validator yêu cầu tập AST refs bằng chính xác tập evidence IDs.

## Gate bắt buộc trước khi tune P2.2

```powershell
python scripts/37_p24_validate_tune_gold_forensic.py `
  --gold artifacts/devset_p24/p24_tune_gold.final.jsonl

python scripts/38_p24_audit_tune_gold.py `
  --gold artifacts/devset_p24/p24_tune_gold.final.jsonl `
  --specs artifacts/devset_p24/p24_tune_authoring.final.v2.jsonl `
  --output artifacts/devset_p24/p24_tune_gold.final.audit.<new>.json

python scripts/43_p24_complete_semantic_audit.py `
  --output artifacts/devset_p24/p24_tune_gold.semantic_audit.<new>.json

python -m jupyter nbconvert --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=300 `
  notebooks/P2_4_TUNE_GOLD_AUDIT_FINAL.ipynb
```

Expected:

- forensic validator: `count=100`, `complete=true`, gold hash đúng như trên;
- semantic audit: `count=21`, `detailed_check_count=21`,
  `metadata_value_columns=[]`, hai duplicate invariant đều `true`;
- notebook cell cuối: `status=PASS`, `records=100`,
  `complex_semantic_checks=21`, `locked_opened=false`.

Không dùng `scripts/14_p24_devset.py validate-gold` cho gold này. Serializer chuẩn chủ ý
bỏ một số numeric raw-grid cells bị OCR nhận nhầm là note code, trong khi P2.4 giữ exact
raw-grid evidence. `scripts/37` áp dụng nguyên vẹn identity/schema/evidence/AST/hash/replay
checks và chỉ thay loader bằng forensic loader read-only. Thay đổi này không đi vào
retrieval, codegen hoặc submission pipeline.

## Dùng tune gold để đánh giá P2.2

Mọi candidate codegen phải đủ 1.012 ID và có đúng một non-empty run signature. Chạy
evaluator với file gold final, luôn tạo output mới:

```powershell
python scripts/14_p24_devset.py evaluate --split tune `
  --gold artifacts/devset_p24/p24_tune_gold.final.jsonl `
  --codegen <complete_codegen.jsonl> `
  --output artifacts/devset_p24/<candidate>_tune_eval.json
```

Tune chỉ đại diện 16/21 strata; `represented_population_mass=0.988142`. Báo cả metric
theo stratum, output type, source, `missing_strata` và represented mass. Không dùng một
con số aggregate duy nhất để chọn kiến trúc.

## Locked protocol

Locked 50 câu tiếp tục giữ kín cho đến khi:

1. P2.2 architecture, prompt, thresholds và repair policy đã đóng băng;
2. run signature cuối đã cố định;
3. không còn quyết định nào được điều chỉnh bằng locked result.

Trước thời điểm đó không đọc/copy/fill hash/seal/evaluate bất kỳ locked question/template
nào. Tune và locked phải được báo cáo riêng; locked chỉ được đánh giá một lần sau freeze.
