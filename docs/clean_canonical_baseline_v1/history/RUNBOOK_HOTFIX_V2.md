# Runbook hotfix v2

## Nguyên nhân lỗi dòng 5

`metric_keys=[]` không có nghĩa artifact không clean. Nó chỉ cho biết canonical registry chưa nhận diện metric như `chi phi phat`; clean retrieval vẫn giữ `metric_variants` để lexical fallback hoạt động. Validator cũ đã kiểm tra quá chặt.

## Lệnh thay thế

```powershell
python scripts/57_clean_retrieve_v2.py
python scripts/60_run_clean_b0_v2.py
```

Hai lệnh đều có tqdm. Retrieval ghi prefix vào `retrieval.jsonl.partial` và chỉ thay file chính khi hoàn tất. B0 checkpoint mỗi 25 câu và mặc định resume các record có cùng run signature.

Smoke nhanh:

```powershell
python scripts/57_clean_retrieve_v2.py --limit 10 --out artifacts/clean_v1/smoke_retrieval_v2.jsonl
python scripts/60_run_clean_b0_v2.py --limit 10 --retrieval artifacts/clean_v1/smoke_retrieval_v2.jsonl --out artifacts/clean_v1/smoke_b0_v2.jsonl
```

Canonical misses được hiển thị để audit coverage ontology, nhưng không làm dừng pipeline nếu lexical fallback vẫn tồn tại.
