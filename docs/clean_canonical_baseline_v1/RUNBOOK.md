# Runbook

## 1. Freeze và test local

```powershell
D:\miniconda3\python.exe -m pytest -p no:cacheprovider --basetemp artifacts/pytest_clean_v1 tests/test_canonical_metrics.py tests/test_clean_profile.py tests/test_clean_retrieval.py tests/test_clean_payload_builder.py -q
D:\miniconda3\python.exe -m compileall -q vifinqa scripts kaggle
```

## 2. Tạo clean retrieval

```powershell
D:\miniconda3\python.exe scripts/57_clean_retrieve.py
```

Mỗi route phải có `clean_profile=clean`, `metric_keys`, `metric_qualifiers` và cùng một `retrieval_config_sha256`.

## 3. Chạy B0

```powershell
D:\miniconda3\python.exe scripts/60_run_clean_b0.py
D:\miniconda3\python.exe scripts/61_build_clean_submission.py --codegen artifacts/clean_v1/b0_results.jsonl --out-dir artifacts/clean_v1/submission_b0
```

B0 là artifact so sánh bắt buộc, không phải candidate tối ưu theo public.

## 4. Build payload schema 9

```powershell
D:\miniconda3\python.exe scripts/59_make_clean_payload.py --dry-run
D:\miniconda3\python.exe scripts/59_make_clean_payload.py --dataset-id <kaggle-user>/vifinqa-clean-canonical-v1
```

Builder không có tham số `--target-dir`. Manifest phải ghi `public_id_masks=false` và `official_derived_gold=false`.

## 5. Chạy B1 trên Kaggle

Mở `kaggle/vifinqa-clean-canonical-v1.ipynb`, attach đúng Dataset schema 9 và chạy toàn bộ. Notebook tạo `codegen_results.jsonl` và `submission/submission.zip` trong `/kaggle/working`.

## 6. Gate trước private

1. Hash source/config/payload/model revision/submission ZIP.
2. Chạy OOD source-derived set đã khóa; không chọn theo official/public set.
3. Chỉ sau khi candidate freeze mới chạy P2.4 regression để phát hiện crash/regression lớn.
4. Không sửa per-ID sau khi xem regression/public score.
5. Không upload hoặc chạy GPU từ automation nếu chưa có phê duyệt vận hành.
