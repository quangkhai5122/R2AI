# RUNBOOK — sổ tay lệnh ViFinQA

> **File này là nguồn sự thật duy nhất về LỆNH CHẠY.** Mỗi khi pipeline đổi,
> ghi đè trực tiếp vào đây (đừng tạo file mới) để không bị lạc phiên bản.
>
> Cập nhật lần cuối: **2026-08-04 (P2.0)** — thêm **`--llm-mode select`**:
> model chỉ chọn dòng trong shortlist bằng JSON, ta sinh pandas. Xoá 3 lớp lỗi
> đã audit ở lượt #12 (cột năm 35%, đơn vị 15%, regex 90% → **0%**).

---

## 0. Trạng thái hiện tại (đọc trước khi chạy bất cứ thứ gì)

| Thành phần | Phiên bản/giá trị đang dùng |
|---|---|
| `vifinqa` package | 0.2.0 |
| Payload schema | 2 (có `payload-manifest.json`, SHA-256) |
| `TABLE_POS_MODE` | `line` (BTC xác nhận: vị trí = **số dòng** của `<table>`) |
| `SUBMISSION_K` | 5 |
| Retrieval artifact chuẩn | `artifacts/retrieval.jsonl` (phải là bản **P1**) |
| Codegen artifact chuẩn | `artifacts/codegen_results.jsonl` |
| Điểm tốt nhất đã nộp | #8: TABLES_F2 .4241 / DOCS_F2 .8628 / ANSWER .1285 / EXEC .1285 |
| Backend LLM Kaggle | `hf` (transformers). **vLLM không chạy trên T4** |

**Kiểm tra nhanh trạng thái trước mỗi phiên làm việc:**

```powershell
cd D:\Python_Project\Hackathon\R2AI_2026
.venv\Scripts\activate
python -c "import json;d=[json.loads(l) for l in open('artifacts/retrieval.jsonl',encoding='utf-8')];print('retrieval:',len(d),'| co plan P1:', 'plan' in d[0]['route'])"
python -m pytest tests -q
```

`co plan P1: False` ⇒ retrieval là bản CŨ, phải chạy lại §2 trước khi làm gì khác.

---

## 0bis. ĐỔI GÌ THÌ PHẢI CHẠY LẠI GÌ (tra bảng này trước khi chạy)

| Sửa file ở... | 01 store | 02 retrieve | 03 rule | 04+upload payload | chạy Kaggle |
|---|:--:|:--:|:--:|:--:|:--:|
| `extraction/`, `utils/viet_num.py` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `router/`, `retrieval/` | – | ✅ | ✅ | ✅ | ✅ |
| `codegen/rule_*.py`, `units.py` | – | – | ✅ | ✅ | ✅ |
| `codegen/prompts.py`, `selection.py`, `generate.py`, `llm_client.py` | – | – | – | ✅ | ✅ |
| `submission/build.py`, `scripts/05` | – | – | – | – | – |
| chỉ `tests/`, `*.md` | – | – | – | – | – |

Payload luôn phải dựng lại khi **bất kỳ** file nào trong `vifinqa/` hoặc
`kaggle/kaggle_codegen.py` đổi, vì manifest băm SHA-256 toàn bộ code.

### Riêng bản P2.0 (`--llm-mode select`) — đang ở trạng thái này

Chỉ đụng `codegen/{selection,prompts,generate}.py` + `kaggle_codegen.py`.
Đã kiểm chứng: rule baseline trước/sau **giống hệt 150/150 bản ghi**.

```powershell
# BỎ QUA 01, 02, 03 — store, retrieval.jsonl, codegen_results.jsonl vẫn dùng được
python scripts/04_make_kaggle_payload.py --dataset-id <user1>/vifinqa-payload
kaggle datasets version -p artifacts\kaggle_payload --dir-mode zip -m "P2.0 selection mode"

# acc #2
$env:KAGGLE_CONFIG_DIR = "D:\kaggle_acc2"
python scripts/04_make_kaggle_payload.py --dataset-id <user2>/vifinqa-payload
kaggle datasets version -p artifacts\kaggle_payload --dir-mode zip -m "P2.0 selection mode"
Remove-Item Env:\KAGGLE_CONFIG_DIR
```

Rồi chạy notebook theo §7bis với `--llm-mode select`.

## 1. Cài đặt (một lần)

```powershell
cd D:\Python_Project\Hackathon\R2AI_2026
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install pytest            # để chạy test
pip install kaggle            # để up payload bằng CLI
```

---

## 2. Pipeline local (CPU) — chạy theo đúng thứ tự này

### 2.1 Build store (chỉ chạy lại khi đổi code `extraction/`)

```powershell
python scripts/01_build_store.py
```

- ~30 phút cho 1.973 báo cáo. Ra `artifacts/store/`.
- Smoke: `python scripts/01_build_store.py --tickers VNM,VJC,ACB`
- **Bắt buộc chạy lại nếu** sửa `html_tables.py`, `report_parser.py`, `build_store.py`,
  hoặc `viet_num.py`.

### 2.2 Retrieval (chạy lại khi đổi `router/` hoặc `retrieval/`)

```powershell
python scripts/02_retrieve.py --out artifacts/retrieval.jsonl
```

- ~5 phút. **Đây là bước hay bị quên nhất** — mọi thay đổi router/metric/plan
  đều nằm trong file này.
- Smoke: thêm `--limit 150`.
- `legacy` vẫn là mặc định để không âm thầm thay ranking của checkpoint.

**Advanced retrieval v1 (opt-in, chưa dùng để thay checkpoint 0.2806):**

```powershell
# RRF lexical/schema ablation - chạy được trước khi có dense index
python scripts/02_retrieve.py --retrieval-mode rrf `
  --out artifacts/retrieval_rrf_lexical.jsonl

# Một lần: tạo BGE-M3 row-label cache trong artifacts/store/label_index
python scripts/10_build_label_index.py --device cuda

# Hybrid BM25 + lexical/schema + BGE-M3 bằng Reciprocal Rank Fusion
python scripts/02_retrieve.py --retrieval-mode rrf --use-dense --dense-required `
  --out artifacts/retrieval_rrf_bge_m3.jsonl
```

Mỗi output có sidecar `<retrieval>.meta.json`; candidate RRF ghi `channel_ranks`,
`fusion_score` và `dense_score` để audit. Không ghi đè `artifacts/retrieval.jsonl`
cho đến khi đã khôi phục artifact 0.2806 và hoàn tất retrieval-only regression.


**A/B web từ checkpoint 0.2806 đã khôi phục:**

```powershell
# Rerank-only: khóa route và đúng pool top-20 của checkpoint; chưa dùng dense.
.venv\Scripts\python.exe scripts/02_retrieve.py `
  --retrieval-mode rrf `
  --route-source artifacts/artifacts/retrieval_p2_canonical_qualified_hybrid_w010.jsonl `
  --freeze-candidate-pool `
  --store-dir artifacts/artifacts/store `
  --out artifacts/candidates/retrieval_02806_rrf_frozenpool_full.jsonl

# Giữ nguyên codegen 0.2806, chỉ thay relevant_tables/docs.
.venv\Scripts\python.exe scripts/05_build_submission.py `
  --retrieval artifacts/candidates/retrieval_02806_rrf_frozenpool_full.jsonl `
  --codegen artifacts/artifacts/codegen_result6_canonical_direct24_semantic4_w010.jsonl `
  --store-dir artifacts/artifacts/store `
  --out-dir artifacts/candidates/submission_02806_rrf_frozenpool_v1
```

File nộp là
`artifacts/candidates/submission_02806_rrf_frozenpool_v1/submission.zip`.
Manifest SHA/provenance nằm cạnh ZIP. Với lượt này ANSWER/EXEC phải giữ `.2806`;
chỉ đọc thay đổi TABLES/DOCS để quyết định giữ hay loại RRF.


**Canonical Metric v2 — batch final hiện tại:**

- Version: `canonical_metric_v2_2026_08_18f`.
- Coverage v1 hoặc v2: **945/1.012**; unresolved: **67**.
- 67 câu còn lại chủ yếu là nested selector, entity-specific note hoặc scenario;
  không tự gán profile direct để tránh false-positive.
- Full batch đã chạy với `k=15`, sau đó merge fill-only trên checkpoint 0.2806.
- Candidate cuối chỉ thay các ID `757, 830, 847`; 1.009 record lõi giữ nguyên.

File nộp:
`artifacts/candidates/submission_02806_metric_v2h_fill_k15/submission.zip`

SHA-256:
`035549e2ee3da7c0f6e4ac66dbdbe00bf6d6b8be46a9d5cac50104735db6c8d8`

Manifest:
`artifacts/candidates/submission_02806_metric_v2h_fill_k15/candidate-manifest.json`


Leaderboard result: `ANSWER=EXEC=.2846`, equivalent to 144/506 correct and
`+2` correct over checkpoint `.2806`. `v2b` and `v2h` are grader-identical;
do not spend another submission slot comparing them. Of IDs `757,830,847`,
two add a correct answer and one is neutral, but aggregate score does not reveal
which one. Dictionary-only work is now secondary to nested typed formula/IR.

**Leaderboard result for the submitted typed-IR ZIP:** `ANSWER=EXEC=.2905`
(147/506 correct, +3 over `.2846`). Retrieval metrics were `TABLES_F2=.4668`,
`DOCS_F2=.8948`; answer gains came from the 19 typed-IR fills, while table F2
regressed versus the frozen retrieval checkpoint. Keep this result as the
answer control and test retrieval separately.

**Typed IR fill candidate (local, not submitted):**

- Full deterministic run: `artifacts/candidates/codegen_typed_ir_integrated_v5_full_k15.jsonl`
- Fill-only merge: `artifacts/candidates/codegen_02806_typed_ir_fill_v5_k15.jsonl`
- Submission ZIP: `artifacts/candidates/submission_02806_typed_ir_fill_v5_k15/submission.zip`
- Accepted IDs: `822, 832, 860, 866, 874, 876, 879, 883, 900, 906, 925, 928, 929, 933, 953, 974, 985, 989, 999`
- ZIP SHA-256: `e9e6bf5564263062ed4d17979206181079d34a8d4f9c1cc3a2b8df24dce51bc6`

Run the candidate with `--typed-ir-fill` only as a source batch. For a
submission, always merge fill-only against the frozen `.2846` codegen and run
the strict build/replay step. The direct planner is deliberately fail-closed
for selector/target and ratio questions; the current candidate has no nested
IR rows after the final guard pass.


**Canonical metric v2 batch fill candidate:**

```powershell
# Audit profile coverage
.venv\Scripts\python.exe scripts/15_metric_v2_audit.py `
  --codegen artifacts/candidates/codegen_metric_v2b_full_k15.jsonl `
  --out artifacts/candidates/metric_v2_audit_after_batch.json

# Candidate đã tạo; không cần chạy lại nếu chỉ muốn nộp
```

File nộp:
`artifacts/candidates/submission_02806_metric_v2b_fill_k15/submission.zip`

Candidate này giữ nguyên checkpoint và chỉ fill 3 `source=none` bằng resolver v2:
IDs `757, 830, 847`. ZIP SHA-256:
`30e56e5b14e463411a73cf52150a41956149f8aab63c3ad08b9b61aa92c804b7`.
Manifest nằm cạnh ZIP. `ANSWER/EXEC` kỳ vọng tăng hoặc giữ; `TABLES/DOCS` dùng
retrieval checkpoint, chỉ thêm evidence tables cho 3 câu đổi answer.

**Next A/B candidates after `.2905`:**

- Exact-cell verifier only, v2h retrieval: `artifacts/candidates/submission_02905_exact_cell_v8_k15/submission.zip`
- Retrieval recovery only, old `.2806` pool: `artifacts/candidates/submission_02905_typed_ir_oldretr_k5/submission.zip`
- Combined recommendation: `artifacts/candidates/submission_02905_exact_cell_oldretr_k5/submission.zip`
- Combined ZIP SHA-256: `fff0c707b4f8a1e850ca410654e842a73f17089342cf26c2c26216cb523f1b25`

The combined candidate keeps the `.2905` answers, applies the exact-cell
verifier correction for ID `53`, and restores the `.2806` retrieval pool. It
is unsubmitted and should be treated as an A/B, not as a confirmed score.

### 2.3 Rule baseline (không cần GPU)

```powershell
python scripts/03_rule_baseline.py --retrieval artifacts/retrieval.jsonl --out artifacts/codegen_results.jsonl
```

Kỳ vọng hiện tại: có cả `rule` (lookup) lẫn `rule_composite`
(growth/difference/ratio/ranking). Trên 150 câu đầu: `rule 123 / rule_composite 5 / none 22`.

Nếu thấy `rule_composite 0` trên toàn bộ 1.012 câu ⇒ retrieval là bản cũ
(thiếu `plan`), quay lại §2.2.

### 2.4 Build submission

```powershell
python scripts/05_build_submission.py --retrieval artifacts/retrieval.jsonl --codegen artifacts/codegen_results.jsonl --out-dir artifacts/submission
```

Nộp file: `artifacts\submission\submission.zip`

Log phải thấy: `expression-form check: all 1012 queries eval-compilable`.
Nếu thấy `[WARN] ... NOT single expressions` → xem §6.3.

**Biến thể (mỗi lần nộp chỉ đổi MỘT biến):**

```powershell
# ablation k=7
python scripts/05_build_submission.py --retrieval artifacts/retrieval.jsonl --codegen artifacts/codegen_results.jsonl --out-dir artifacts/submission_k7 --sub-k 7

# chỉ nộp các id trong file câu hỏi chính thức (nếu BTC phát bộ riêng)
python scripts/05_build_submission.py ... --questions <duong_dan>\questions.jsonl
```

**KHÔNG dùng `--expand-docs`** — đã kiểm chứng làm sập DOCS_F2 (.84 → .61).

---

## 3. Bộ eval offline (đo trước khi tốn GPU / lượt nộp)

> ⛔ **KHÔNG BAO GIỜ NỘP BỘ EVAL.** Câu hỏi tự sinh, id riêng ⇒ leaderboard 0.0
> và mất một lượt nộp (đã dính lần #9). Luôn dùng `--offline-eval`.

```powershell
python scripts/09_gen_eval_suite.py --per-class 60
python scripts/02_retrieve.py --questions artifacts/eval/eval_questions.jsonl --out artifacts/eval/eval_retrieval.jsonl
python scripts/03_rule_baseline.py --retrieval artifacts/eval/eval_retrieval.jsonl --out artifacts/eval/eval_codegen.jsonl
python scripts/05_build_submission.py --retrieval artifacts/eval/eval_retrieval.jsonl --codegen artifacts/eval/eval_codegen.jsonl --out-dir artifacts/eval/eval_submission --offline-eval
python scripts/07_evaluate.py --submission artifacts/eval/eval_submission --gold artifacts/eval/eval_gold.json --by-class
```

**Baseline rule-only hiện tại (300 câu) — mốc so sánh cho mọi thay đổi sau này:**

| class | n | answer | exec | F2 |
|---|---:|---:|---:|---:|
| lookup | 60 | 0.783 | 0.783 | 0.556 |
| ratio_pct | 60 | 0.400 | 0.400 | 0.306 |
| growth_pct | 60 | 0.383 | 0.383 | 0.350 |
| difference | 60 | 0.283 | 0.283 | 0.329 |
| ranking | 60 | 0.133 | 0.133 | 0.332 |
| **tổng** | **300** | **0.3967** | **0.3967** | **0.374** |

(trước rule composite: 0.157 tổng, 4 lớp phức hợp đều 0.000)

`ranking` thấp là do phải đúng **cả 4** công ty trong một câu; suy ra độ chính xác
mỗi fact ≈ 0.6.

Nếu lỡ tạo `submission.zip` trong thư mục eval: xoá tay
`artifacts\eval\eval_submission\submission.zip`.

---

## 4. Kaggle — payload

### 4.1 Khi nào phải rebuild payload

Rebuild nếu **bất kỳ** thứ nào sau đây đổi: code trong `vifinqa/`,
`kaggle/kaggle_codegen.py`, `artifacts/store/`, `artifacts/retrieval.jsonl`.

### 4.2 Dựng payload

```powershell
python scripts/04_make_kaggle_payload.py --dataset-id <user1>/vifinqa-payload
python scripts/04_make_kaggle_payload.py --dry-run        # kiểm tra trước, không ghi đè
```

Ra `artifacts\kaggle_payload\` (~100 MB) + `payload-manifest.json`.

### 4.3 Up lên acc #1

```powershell
kaggle datasets version -p artifacts\kaggle_payload --dir-mode zip -m "P1 metric+shortlist+plan"
```

(Lần đầu tiên với một acc: đổi `version` → `create`.)

### 4.4 Up lên acc #2

```powershell
# một lần: tải kaggle.json của acc #2 vào D:\kaggle_acc2\kaggle.json
$env:KAGGLE_CONFIG_DIR = "D:\kaggle_acc2"
python scripts/04_make_kaggle_payload.py --dataset-id <user2>/vifinqa-payload   # PHẢI chạy lại, id nằm trong metadata
kaggle datasets create -p artifacts\kaggle_payload --dir-mode zip               # lần đầu
kaggle datasets version -p artifacts\kaggle_payload --dir-mode zip -m "refresh" # các lần sau
Remove-Item Env:\KAGGLE_CONFIG_DIR
```

**Bẫy:** `--dataset-id` được ghi vào `dataset-metadata.json`; nếu không chạy lại
`04_...` trước khi đổi acc, CLI sẽ đẩy nhầm acc. Kiểm tra:

```powershell
type artifacts\kaggle_payload\dataset-metadata.json
```

---

## 5. Kaggle — chạy notebook

### 5.1 Codegen (Qwen)

Notebook: `kaggle/vifinqa-codegen.ipynb` → File → Import Notebook.
Settings: **GPU T4 x2**, **Internet On**, Add Input = dataset payload (chỉ MỘT bản).

Log đúng phải có: `payload verified: schema=2` → `run signature: ...` →
`baseline written (...)` → `LLM queue: ...` → `[chunk 1/N] ...`

Cấu hình đang khuyến nghị (7B, 4-bit):

```
!python /kaggle/working/code/kaggle_codegen.py --payload $PAYLOAD --backend hf \
    --model Qwen/Qwen2.5-Coder-7B-Instruct --load-4bit \
    --out /kaggle/working/codegen_results.jsonl \
    --n 1 --k 4 --max-tokens 256 --batch-size 4 \
    --checkpoint-every 32 --time-budget-min 400
```

Biến thể:

| Mục đích | Thêm/đổi cờ |
|---|---|
| Bật semantic matching | `--use-dense` (cần `pip install -q sentence-transformers` + `store/label_index/`) |
| Model 14B | `--model Qwen/Qwen2.5-Coder-14B-Instruct --load-4bit --batch-size 2` |
| Bỏ qua câu rule đã chắc | `--rule-first` |
| Chạy tiếp phiên trước | đặt file cũ vào `/kaggle/working/codegen_results.jsonl`, giữ NGUYÊN mọi cờ (khác cờ ⇒ `run_signature` khác ⇒ không resume) |
| OOM | `--batch-size 2`, `--k 3`, `--max-tokens 192` |

Tải `codegen_results.jsonl` từ tab Output → về local:

```powershell
python scripts/05_build_submission.py --retrieval artifacts/retrieval.jsonl --codegen <duong_dan>\codegen_results.jsonl --out-dir artifacts/submission_qwen
```

### 5.2 Embed label index (encode-only, ~15 phút)

Notebook: `kaggle/vifinqa-embed.ipynb`. GPU T4 x1 là đủ.
Tải `label_index/` → chép vào `artifacts\store\label_index\` → rebuild payload (§4).

---

## 6. Sửa lỗi thường gặp

### 6.1 Leaderboard trả 0.0 toàn bộ
Đã nộp nhầm bộ eval offline. Chỉ nộp file tên đúng `submission.zip` được sinh từ
`artifacts/retrieval.jsonl` (câu hỏi thật).

### 6.2 EXEC thấp hơn ANSWER
Có `pandas_query` nhiều dòng → grader `eval` báo SyntaxError. Sửa không cần GPU:

```powershell
python scripts/08_repair_expressions.py --submission artifacts/submission --out-dir artifacts/submission_fixed
python scripts/08_repair_expressions.py --codegen artifacts/codegen_results.jsonl   # sửa luôn ở gốc
```

### 6.3 `[WARN] ... NOT single expressions` khi build
Chạy `08_repair_expressions.py --codegen ...` rồi build lại.

### 6.4 Kaggle: `Engine core initialization failed`
vLLM V1 không chạy trên T4. Dùng `--backend hf` (mặc định notebook).

### 6.5 Kaggle: payload hash mismatch / schema
Payload cũ hoặc attach nhiều bản. Gỡ hết input, up lại theo §4, attach đúng 1 bản.

### 6.6 Windows in ra ký tự lỗi
Mọi script đã gọi `setup_stdout()` (UTF-8). Nếu vẫn lỗi: `chcp 65001`.

---

## 7. Nhật ký nộp bài (cập nhật sau mỗi lần nộp)

| # | Cấu hình | TABLES_F2 | DOCS_F2 | ANSWER | EXEC |
|---|---|---:|---:|---:|---:|
| 3 | pos=line, k=10, rule | .3641 | .8399 | .085 | .085 |
| 4 | + expand-docs ❌ | .3641 | .6066 | .085 | .085 |
| 5 | pos=line, k=5, rule | .4092 | .8093 | .085 | .085 |
| 6 | + Qwen 7B (query nhiều dòng ❌) | .4092 | .8093 | .1047 | .0613 |
| 8 | **P1 rule-only** | **.4241** | **.8628** | **.1285** | **.1285** |
| 9 | nhầm nộp bộ eval ❌ | 0 | 0 | 0 | 0 |
| 10 | **P1.5 rule composite** | **.4337** | **.8777** | **.1542** | **.1542** |
| 11 | P1.6 hợp nhất scoring | .4337 | .8777 | .1522 | .1522 |
| 12 | Qwen 7B `--llm-target empty` | .4334 | .8774 | **.1561** | .1561 |
| _ | _(điền lần nộp tiếp theo)_ | | | | |

**#12: 175 đáp án LLM mới → chỉ ~+2 câu đúng (≈2%).** Nguyên nhân đã audit:
35% query không lọc cột năm, 15% quên chia ANSWER_SCALE, 90% thiếu `regex=False`.
Chi tiết + hướng sửa: `CLAUDE.md` mục "CHẨN ĐOÁN LƯỢT QWEN #12".

**#11 trung tính** (−0.002 = 1 câu/506): bỏ 24 đáp án nhưng trong đó chỉ ~1 câu
đúng. Eval dự báo lookup +0.10 nhưng thực tế ≈ 0 ⇒ **lần thứ hai eval dự báo sai**.
Rule đã tới điểm lợi tức giảm dần: P1.5 +0.026, P1.6 −0.002.

## 7bis. Chạy Qwen song song 2 acc (giai đoạn hiện tại)

Điều kiện: đã chạy §2.2 + §2.3 với code P1.6 và rebuild payload (§4).

| | Acc #1 | Acc #2 |
|---|---|---|
| Notebook | `vifinqa-codegen.ipynb` | `vifinqa-codegen.ipynb` |
| Model | `Qwen/Qwen2.5-Coder-7B-Instruct --load-4bit` | `Qwen/Qwen2.5-Coder-14B-Instruct --load-4bit --batch-size 2` |
| Out | `codegen_qwen7b.jsonl` | `codegen_qwen14b.jsonl` |
| Khác nhau đúng 1 biến | kích thước model | |

**`--llm-target` quyết định GPU tiêu vào đâu** (đo trên artifact hiện tại):

| target | số câu | ước tính 7B | ghi chú |
|---|---:|---:|---|
| `all` | 1012 | ~5.0h | phủ hết, nhưng phần lớn rule đã trả lời đúng |
| `weak` | 787 | ~3.9h | rỗng + rule confidence < 78 |
| **`empty`** | **569** | **~2.8h** | **ROI cao nhất**: chỉ câu rule bó tay |

Arbitration giữ đáp án rule khi bất đồng mà rule tự tin, nên chạy `all` phần lớn
là tiêu GPU vào việc xác nhận lại thứ đã đúng.

### `--llm-mode select` — DÙNG CÁI NÀY từ nay

Sau lượt #12 (model tự viết pandas): 35% query không lọc cột năm, 15% quên chia
`ANSWER_SCALE`, 90% thiếu `regex=False` → 175 đáp án mới chỉ ra ~2 câu đúng.

Ở `--llm-mode select`, model **chỉ xuất JSON** `{"op":..., "operands":[...]}`
chọn dòng trong shortlist; **ta** sinh biểu thức pandas với cột và đơn vị đúng.
Mô phỏng end-to-end: query thiếu `regex=False` **0%**, query không lọc cột **0%**.
Output ~30 token thay vì 256 → nhanh hơn nhiều, chạy được `--llm-target all`.

```
# ACC #1 — 7B, selection mode, TOÀN BỘ câu (arbitration mới có việc để làm)
!python /kaggle/working/code/kaggle_codegen.py --payload $PAYLOAD --backend hf \
    --model Qwen/Qwen2.5-Coder-7B-Instruct --load-4bit \
    --llm-mode select --llm-target all \
    --out /kaggle/working/codegen_sel7b.jsonl \
    --n 1 --k 4 --max-tokens 96 --batch-size 8 \
    --checkpoint-every 32 --time-budget-min 400

# ACC #2 — 14B, khác đúng MỘT biến (kích thước model)
!python /kaggle/working/code/kaggle_codegen.py --payload $PAYLOAD --backend hf \
    --model Qwen/Qwen2.5-Coder-14B-Instruct --load-4bit \
    --llm-mode select --llm-target all \
    --out /kaggle/working/codegen_sel14b.jsonl \
    --n 1 --k 4 --max-tokens 96 --batch-size 4 \
    --checkpoint-every 32 --time-budget-min 400
```

`--max-tokens 96` là đủ cho một object JSON; `--batch-size` tăng được vì output ngắn.

Chế độ cũ `--llm-mode code` vẫn giữ để đối chứng, nhưng không khuyến nghị.

Nếu phiên còn dư thời gian sau khi xong `empty`: chạy lại **cùng lệnh** đổi
`--llm-target weak` — resume sẽ giữ nguyên phần đã làm và chỉ bổ sung phần yếu.

Về local, mỗi file build một submission riêng rồi so:

```powershell
python scripts/05_build_submission.py --retrieval artifacts/retrieval.jsonl --codegen <path>\codegen_qwen7b.jsonl  --out-dir artifacts/submission_q7
python scripts/05_build_submission.py --retrieval artifacts/retrieval.jsonl --codegen <path>\codegen_qwen14b.jsonl --out-dir artifacts/submission_q14
```

Kiểm nhanh Qwen có thực sự thêm giá trị (trước khi tốn lượt nộp):

```powershell
python -c "import json;from collections import Counter;r=[json.loads(l) for l in open(r'<path>\codegen_qwen7b.jsonl',encoding='utf-8')];print(Counter(x['source'] for x in r));print('arbitration:',Counter((x.get('arbitration') or {}).get('reason','n/a') for x in r))"
```

Đọc kết quả: `rule and llm agree` nhiều = tín hiệu tốt; `disagree; rule weak -> llm`
là phần LLM thực sự đóng góp; `source=none` còn nhiều = LLM cũng bó tay.

Ở lượt #12 toàn bộ 183 bản ghi đều là `rule produced nothing` — vì
`--llm-target empty` khiến LLM chỉ thấy câu rule bó tay nên **arbitration chưa
từng chạy**. Với `--llm-target all` sẽ thấy đủ 4 nhóm lý do.

## 8. Hiệu chuẩn eval offline ↔ leaderboard (QUAN TRỌNG)

| | eval offline | leaderboard |
|---|---:|---:|
| trước P1.5 | 0.157 | 0.1285 |
| sau P1.5 | 0.3967 | 0.1542 |
| mức tăng | **+153%** | **+20%** |

**Bộ eval phóng đại mức cải thiện khoảng 7 lần.** Hai lý do: (a) câu synthetic
dùng đúng 6 mã VAS mà rule engine biết; (b) phân bố lớp khác thực tế
(eval chia đều 20% mỗi lớp; bộ thi thật: lookup 46%, ranking 21%).

⇒ Dùng eval để biết **hướng** (tăng hay giảm), KHÔNG dùng để dự đoán **mức**.
Chỉ leaderboard mới là trọng tài về độ lớn.

## 9. Bản đồ khoảng trống hiện tại (đo trên artifact #10)

467/1012 câu có đáp án (46%); độ chính xác trên phần đã trả lời ≈ 33%.

| op | số câu | % rỗng | ghi chú |
|---|---:|---:|---|
| lookup | 468 | 25% | 118 câu — lỗ hổng dễ lấp nhất |
| **ranking** | **214** | **95%** | 203 câu — lớn nhất, nhưng là **composite lồng nhau** |
| difference | 147 | 66% | 97 câu |
| average | 70 | 91% | 64 câu |
| growth_pct | 50 | 68% | 34 câu |
| ratio | 38 | 37% | 14 câu |

**Vì sao `ranking` gần như rỗng hoàn toàn:** câu thật lồng 3 tầng, ví dụ
*"Trong nhóm HPG, HSG, MSR và NKG, doanh nghiệp có mức tăng lớn nhất từ 2023
sang 2024 của tỷ lệ X"* = rank(growth(ratio)). Rule engine hiện chỉ giải một
tầng. Bộ eval của tôi chỉ có ranking một tầng nên không lộ ra điều này.
