# P2.4 — hướng dẫn gán nhãn dev set

Dev set gồm **100 câu tune** và **50 câu locked**, lấy mẫu cố định theo
`operation × độ phức tạp số fact`. Không dùng bất kỳ câu locked nào để sửa rule,
prompt, shortlist, threshold hoặc chọn checkpoint.

## 1. Dựng và kiểm tra split

```powershell
$p24Manifest = "artifacts\devset_p24\p24_manifest.json"
if (Test-Path -LiteralPath $p24Manifest) {
  python scripts/14_p24_devset.py validate-bundle
} else {
  python scripts/14_p24_devset.py build
  python scripts/14_p24_devset.py validate-bundle
}
```

Kết quả ở `artifacts/devset_p24/`. Không sửa các file `*_questions.jsonl`,
`*.template.jsonl` hoặc `p24_manifest.json`; mọi thay đổi đều làm hash guard fail.
Lệnh `build` là one-shot và từ chối ghi đè nếu bất kỳ file bundle cố định nào đã tồn tại;
muốn thử một split khác phải dùng `--out-dir` mới, không dựng lại trên bundle chuẩn.
Tạo working copy mới; không ghi trực tiếp vào template:

```powershell
$tuneDraft = "artifacts\devset_p24\p24_tune_gold.draft.jsonl"
if (Test-Path -LiteralPath $tuneDraft) { throw "Tune draft already exists" }
Copy-Item -LiteralPath `
  "artifacts\devset_p24\p24_tune_gold.template.jsonl" `
  -Destination $tuneDraft
```

## 2. Một record được coi là gold khi nào

Đặt `label_status="verified"` chỉ sau khi đủ cả bốn lớp sau:

1. **Evidence chính xác:** mỗi `E1`, `E2`, ... trỏ đúng `report_id`, `table_pos`,
   `row`, `col`, `label`, `code`, `col_name`, `value`, `unit_scale`. `variable`
   phải khớp với dataframe trong replay.
2. **Output chính xác:** kiểm tra `type`, `value`, đơn vị, scale và số chữ số làm
   tròn. Phân biệt `%`, điểm phần trăm, tỷ số `lần`, `count` và `year`.
3. **AST có kiểu:** root là `{"kind":"op",...}`; leaf dữ liệu tham chiếu
   `{"kind":"evidence","evidence_id":"E1"}`. Mọi evidence phải được dùng và
   không được tham chiếu evidence ngoài danh sách.
4. **Replay:** `pandas_query` là đúng một biểu thức `eval`, `used_vars` khớp
   chính xác evidence, kết quả bằng `output.value` trong tolerance. Dùng lệnh
   `fill-hashes` để điền canonical `evidence_sha256` và `ast_sha256`; không tự
   tính/copy hash bằng tay.

Ví dụ AST cho `rank(growth(A_2024, A_2023), growth(B_2024, B_2023))`:

```json
{"kind":"op","op":"ranking_max","args":[
  {"kind":"op","op":"growth_pct","args":[
    {"kind":"evidence","evidence_id":"E1"},
    {"kind":"evidence","evidence_id":"E2"}]},
  {"kind":"op","op":"growth_pct","args":[
    {"kind":"evidence","evidence_id":"E3"},
    {"kind":"evidence","evidence_id":"E4"}]}
]}
```

Không suy diễn từ output của Qwen. Mở báo cáo/bảng gốc, kiểm tra ô và công thức
độc lập; nếu câu mơ hồ, để `draft` và mô tả trong `annotator_notes`.

## 3. Kiểm tra tune và chống leakage

```powershell
# Có thể kiểm identity/schema khi file vẫn còn record trống.
python scripts/14_p24_devset.py validate-gold --split tune `
  --gold artifacts/devset_p24/p24_tune_gold.draft.jsonl --allow-template

# Sau khi đủ 100 record verified, ghi file mới có canonical hashes.
python scripts/14_p24_devset.py fill-hashes --split tune `
  --input artifacts/devset_p24/p24_tune_gold.draft.jsonl `
  --output artifacts/devset_p24/p24_tune_gold.jsonl

python scripts/14_p24_devset.py validate-gold --split tune `
  --gold artifacts/devset_p24/p24_tune_gold.jsonl

python scripts/14_p24_devset.py check-tune-input `
  --input artifacts/devset_p24/p24_tune_gold.jsonl
```

`check-tune-input` phải được chạy trước mọi script tuning; nó từ chối ID, text của
locked set và cả ID ngoài tune set.

Đánh giá một codegen hoàn chỉnh (bắt buộc đủ 1.012 ID và một run signature):

```powershell
python scripts/14_p24_devset.py evaluate --split tune `
  --gold artifacts/devset_p24/p24_tune_gold.jsonl `
  --codegen <complete_codegen.jsonl> `
  --output artifacts/devset_p24/<candidate>_tune_eval.json
```

Report có ANSWER, EXEC, executable rate, coverage, breakdown theo stratum/output/source,
population-weighted aggregate và hash provenance. `fill-hashes`/`evaluate` dùng exclusive
create: nếu output đã tồn tại, lệnh dừng thay vì ghi đè.

Tune hiện chỉ đại diện **16/21 strata**, thiếu 5 strata tương ứng 12/1.012 câu;
`represented_population_mass=0.988142`. Vì vậy phải báo cả
`represented_population_mass`, `missing_strata` và
`complete_population_coverage`; weighted tune chỉ là ước lượng có điều kiện trên các
strata được đại diện, không phải metric population-wide.

Một reviewer thứ hai kiểm độc lập ít nhất 20% tune và toàn bộ record mơ hồ. Ghi agreement
theo evidence cell, output type và answer; không chỉ so answer cuối.

## 4. Khóa locked set

Chỉ mở/gán locked sau khi đã đóng băng code, config và run signature cuối. Tạo draft/gold
locked tương tự tune, strict-validate rồi tạo seal:

```powershell
$lockedDraft = "artifacts\devset_p24\p24_locked_gold.draft.jsonl"
$lockedGold = "artifacts\devset_p24\p24_locked_gold.jsonl"
$lockedSeal = "artifacts\devset_p24\p24_locked_gold.seal.json"
if ((Test-Path -LiteralPath $lockedDraft) -or `
    (Test-Path -LiteralPath $lockedGold) -or `
    (Test-Path -LiteralPath $lockedSeal)) {
  throw "Locked working/seal file already exists"
}
Copy-Item -LiteralPath `
  "artifacts\devset_p24\p24_locked_gold.template.jsonl" `
  -Destination $lockedDraft

# Sau khi hoàn tất và kiểm độc lập đủ 50 record:
python scripts/14_p24_devset.py fill-hashes --split locked `
  --input $lockedDraft --output $lockedGold
python scripts/14_p24_devset.py validate-gold --split locked `
  --gold $lockedGold

python scripts/14_p24_devset.py seal-locked `
  --gold $lockedGold --seal $lockedSeal

python scripts/14_p24_devset.py verify-locked `
  --gold $lockedGold --seal $lockedSeal

python scripts/14_p24_devset.py evaluate --split locked `
  --gold $lockedGold `
  --codegen <complete_codegen.jsonl> `
  --seal $lockedSeal `
  --output artifacts/devset_p24/<candidate>_locked_eval.json
```

Sau khi seal, bất kỳ thay đổi nào ở manifest, câu hỏi hoặc gold locked đều phải
làm `verify-locked` thất bại. Báo cáo tune và locked riêng; không chọn phương án
dựa trên locked rồi gọi đó là đánh giá giữ kín. Seal chỉ chứng minh integrity, không
chứng minh locked chưa từng bị xem; nên tách người/không gian gán locked khỏi người tune.
`seal-locked` là one-shot và từ chối ghi đè seal đã tồn tại; nếu verify thất bại thì
không tạo seal mới để hợp thức hóa artifact đã đổi.
Vì sampler giữ chỗ cho strata hiếm, báo cả breakdown và population-weighted metric;
không diễn giải raw locked accuracy như một mẫu ngẫu nhiên không chệch của 1.012 câu.
