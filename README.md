# ViFinQA — Giai đoạn 0: Hạ tầng & Baseline Pipeline

Triển khai chiến lược trong `ViFinQA_Claude_Strategy.md`:
**structured lookup → BM25 trong báo cáo đã khoá → Text-to-Pandas (Qwen2.5-Coder-7B trên Kaggle) → semantic guard/self-debug → submission.zip**

Phần **CPU chạy local** (máy bạn), phần **GPU chạy Kaggle** (baseline 7B NF4). Hai bên trao đổi qua payload có manifest SHA-256 và `codegen_results.jsonl`.

```
Local (CPU)                                Kaggle (GPU T4 x2)
───────────────────────────                ─────────────────────────────
01_build_store.py    parse 1.973 báo cáo   
02_retrieve.py       router + BM25         
03_rule_baseline.py  baseline KHÔNG cần GPU
04_make_kaggle_payload.py ──► upload ────► vifinqa-codegen.ipynb
                                           (Qwen2.5-Coder-7B + NF4/HF
                                            + semantic guard + self-debug)
05_build_submission.py ◄─── download ◄──── codegen_results.jsonl
        │
        ▼
  submission.zip (results.json + data/*.csv)
```

## 0. Cài đặt local

```powershell
cd D:\Python_Project\Hackathon\R2AI_2026
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Chạy pipeline local (CPU)

```powershell
# 1) Parse toàn bộ corpus -> artifacts/store/ (~15-30 phút, chạy 1 lần)
python scripts/01_build_store.py
#    smoke test trước nếu muốn: python scripts/01_build_store.py --tickers VNM,VJC,ACB

# 2) Retrieval -> artifacts/retrieval.jsonl
python scripts/02_retrieve.py

# 3) Baseline rule-based (không cần GPU) -> artifacts/codegen_results.jsonl
python scripts/03_rule_baseline.py

# 4) Đóng gói submission -> artifacts/submission/submission.zip
python scripts/05_build_submission.py
```

Đến đây bạn đã có một bài nộp hợp lệ end-to-end (nộp sớm để hiệu chuẩn — xem mục 4). LLM trên Kaggle sẽ thay thế/ghi đè bước 3.

## 2. Triển khai LLM trên Kaggle (chi tiết từng bước)

### 2.1. Chuẩn bị tài khoản (1 lần)
1. Tạo tài khoản [kaggle.com](https://www.kaggle.com) → **Settings → Phone verification** (bắt buộc để bật GPU & Internet trong notebook).
2. (Khuyên dùng CLI) **Settings → API → Create New Token** → tải `kaggle.json` → đặt vào `C:\Users\<user>\.kaggle\kaggle.json` → `pip install kaggle`.
3. Quota GPU miễn phí: ~**30 giờ/tuần**, mỗi phiên tối đa ~**9 giờ** (xem panel Settings của notebook). T4 x2 (2×16GB) là cấu hình dùng ở đây.

### 2.2. Upload payload
```powershell
python scripts/04_make_kaggle_payload.py --dry-run --dataset-id lequangkhai5122005/vifinqa-payload
python scripts/04_make_kaggle_payload.py --dataset-id lequangkhai5122005/vifinqa-payload
kaggle datasets version -p artifacts\kaggle_payload --dir-mode zip -m "P0 crash-safe runner, manifest schema v2"
```
Lệnh `version` ở trên dành cho dataset đã có ID `lequangkhai5122005/vifinqa-payload`.
Chỉ dùng `kaggle datasets create` khi tạo dataset lần đầu. Không dùng CLI thì mở
dataset hiện có trên Kaggle và tạo **New Version** từ cả thư mục `artifacts/kaggle_payload/`.

Khi cập nhật code/store/retrieval: rebuild payload rồi chạy
`kaggle datasets version -p artifacts\kaggle_payload --dir-mode zip -m "refresh payload"`.
Builder giữ lại dataset ID cũ và sinh `payload-manifest.json`; notebook từ chối payload cũ hoặc lệch hash.

### 2.3. Tạo & chạy notebook
1. **Kaggle → Code → New Notebook** → **File → Import Notebook** → chọn `kaggle/vifinqa-codegen.ipynb` trong repo này.
2. Panel phải: **Accelerator = GPU T4 x2**, **Internet = On**, **Add Input** → chọn dataset `vifinqa-payload` của bạn.
3. Chạy lần lượt các cell (notebook yêu cầu đúng một payload, kiểm manifest rồi copy nguyên trạng code sang `/kaggle/working`):
   - cài `transformers accelerate bitsandbytes` (~1–2 phút)
   - **smoke test 12 câu** — kiểm tra output trước
   - **full run** (`--n 1 --temperature 0 --k 4 --max-tokens 256`; runner ghi baseline trước và checkpoint theo chunk)
   - trước khi để full run tiếp tục, log phải có lần lượt `payload verified: schema=2`,
     `run signature: ...`, `baseline written (...)`, `LLM queue: ...`, rồi
     `[chunk 1/...]`. Nếu vẫn thấy `LLM round 1: 1012 prompts`, bạn đang chạy
     notebook hoặc payload cũ: dừng phiên, gỡ input cũ và attach version mới.
4. **Backend là transformers, KHÔNG phải vLLM.** vLLM bản mới (V1 engine) không khởi động được trên T4 (Turing/SM75) — lỗi `Engine core initialization failed` — và engine V0 hỗ trợ T4 đã bị xoá khỏi vLLM. Notebook dùng `--backend hf`: chậm hơn nhưng chạy chắc. Muốn thử vLLM thì pin `vllm==0.7.3` (`--backend vllm`, xem cell cuối notebook).
5. Model mặc định `Qwen/Qwen2.5-Coder-7B-Instruct` + 4-bit NF4. Giữ nguyên 7B
   cho lượt baseline có kiểm soát; chỉ thử 14B sau khi 7B đã có score và còn ngân sách GPU.
6. Tải `codegen_results.jsonl` từ tab **Output** về máy.

Chạy nền không cần giữ tab: **Save Version → Save & Run All (Commit)** — kết quả nằm trong tab Output của version.

### 2.4. Về local, đóng gói nộp
```powershell
python scripts/05_build_submission.py --codegen <thư_mục_tải_về>\codegen_results.jsonl
```

### 2.5. Troubleshooting Kaggle
| Lỗi | Cách xử lý |
|---|---|
| `Engine core initialization failed` (vLLM) | V1 engine không chạy trên T4 → dùng `--backend hf` (mặc định notebook) |
| `unexpected keyword argument 'swap_space'` | đã fix: `VllmBatchClient` tự drop kwarg vLLM không nhận |
| CUDA OOM | HF runner tự giảm batch `4→2→1`; nếu batch 1 vẫn OOM, giảm `--max-input-tokens`, `--k` hoặc `--max-tokens` |
| Tải model chậm / hết disk | dùng 7B thay 14B; xoá `/root/.cache/huggingface` giữa các lần |
| Hết phiên | checkpoint luôn đủ 1.012 dòng; chạy lại cùng `--out`, payload và cấu hình để resume. Phiên Kaggle mới phải copy checkpoint cũ thành `/kaggle/working/codegen_results.jsonl` trước; lệch `run_signature` sẽ không tái dùng LLM answer cũ |
| Log `LLM round 1: 1012 prompts` | notebook/payload cũ, chưa có chunk runner → upload dataset version mới, import lại notebook hiện hành và chỉ attach đúng một payload |
| Kết quả toàn `source=rule` | LLM chưa sinh được code chạy được → tăng `--max-tokens`, xem log |
| Muốn model khác | `--model Qwen/Qwen2.5-Coder-14B-Instruct --load-4bit`, `microsoft/phi-4` (verifier giai đoạn 1) |

## 3. Validation tự tạo (khuyến nghị mạnh — không có train/dev chính thức)

```powershell
python scripts/06_gen_validation.py --n 300
python scripts/02_retrieve.py --questions artifacts/validation/val_questions.jsonl --out artifacts/val_retrieval.jsonl
python scripts/03_rule_baseline.py --retrieval artifacts/val_retrieval.jsonl --out artifacts/val_codegen.jsonl
python scripts/05_build_submission.py --retrieval artifacts/val_retrieval.jsonl --codegen artifacts/val_codegen.jsonl --out-dir artifacts/val_submission
python scripts/07_evaluate.py --submission artifacts/val_submission --gold artifacts/validation/val_gold.json
```
In ra P/R/F2 macro, Answer Acc, Exec Acc. Bộ synthetic hiện dùng cùng parser/store để tạo gold nên chỉ là smoke test kỹ thuật, không đủ để chọn `--sub-k` hoặc model.

## 4. Hiệu chuẩn leaderboard (kết quả 5 lần nộp đầu)

**Đã xác nhận:**

1. **Vị trí bảng = SỐ DÒNG (1-based) nơi `<table>` bắt đầu trong file OCR .txt** — BTC đã XÁC NHẬN CHÍNH THỨC qua trao đổi với đội (và leaderboard verify: TABLES_F2 0.0 → 0.364). `--pos-mode line` giờ là **mặc định** — không cần truyền flag nữa. (Sinh lại bộ validation nếu tạo trước thay đổi này: `python scripts/06_gen_validation.py`.)
2. **`--expand-docs` làm GIẢM điểm** (DOCS_F2 0.84 → 0.61: precision 0.91→0.28 đổi lấy recall +5.5%) — không dùng. Gold trung bình chỉ ~1.15 doc/câu.
3. **`--sub-k 5` đã thắng k=10 trên leaderboard:** TABLES_F2 macro 0.3641→0.4092, precision 0.176→0.2621, recall 0.684→0.5852; DOCS_F2 giảm 0.8399→0.8093. k=5 là cutoff tốt nhất đã thử, chưa chứng minh tối ưu so với k=7. F2 được tính từng câu rồi macro-average, không suy lại từ aggregate P/R.
4. Warning `gold=506 pred=1012` vô hại — BTC chấm 506 câu trong số 1012; nộp đủ vẫn phủ hết gold.
5. TABLES_RECALL trong-doc ≈ 0.82 (0.684/0.834): ~18% bảng gold (đa phần thuyết minh) chưa vào top-10 → mục tiêu của Giai đoạn 1 (BGE-M3 + reranker trong scope đã khoá).

**Trạng thái các trục điểm:** DOCS (0.84 F2) tốt; TABLES đang tối ưu k + ranking; ANSWER/EXEC (0.085) là trần của rule-baseline — **chạy Kaggle LLM là đòn bẩy lớn nhất hiện tại**.

`--k` của codegen là số bảng đưa vào prompt (baseline Kaggle hiện dùng 4 để giảm
prefill/OOM). `--sub-k` khi build submission là số bảng nộp cho grader (giữ 5 theo
submission #5). Đây là hai biến khác nhau.

Lệnh nộp chuẩn hiện hành (pos-mode line là mặc định):

```powershell
python scripts/05_build_submission.py --codegen <codegen_results.jsonl>
```

> Bối cảnh dự án cho các phiên làm việc AI sau này được duy trì trong `CLAUDE.md` (gốc repo) — cập nhật file đó khi có phát hiện mới từ leaderboard/BTC.

## 5. Cấu trúc code

```
vifinqa/
  config.py                đường dẫn + hằng số (TABLE_POS_BASE, SUBMISSION_K...)
  utils/viet_num.py        parse số VN: "1.234.567", "(1.839)"→âm, "12,5"
  utils/viet_text.py       bỏ dấu, chuẩn hoá, fuzzy match
  extraction/html_tables.py   tách <table> + rowspan/colspan -> grid
  extraction/report_parser.py đơn vị bảng (triệu/tỷ, explicit/sticky), TableRec
  extraction/build_store.py   dual store parquet theo ticker + Store reader
  router/entities.py       ticker (regex+fuzzy theo code_stock.csv), năm,
                           "công ty mẹ"→separate, đơn vị câu hỏi, metric phrase
  router/router.py         khoá report_ids; growth→thêm năm trước; fallback năm kề
  retrieval/bm25.py        BM25 tự cài (không phụ thuộc lib)
  retrieval/serialize.py   bảng -> doc BM25; bảng -> "tidy CSV" (long-format)
  retrieval/retrieve.py    BM25 trong scope đã khoá + label-match boost
  codegen/prompts.py       PoT prompt (schema tidy + đơn vị + ANSWER_SCALE)
  codegen/executor.py      sandbox exec, ép scalar float, timeout (POSIX)
  codegen/rule_codegen.py  fallback tất định: fuzzy label + cột năm -> 1 dòng pandas
  codegen/llm_client.py    HF batch có OOM backoff / vLLM/OpenAI-compatible tuỳ chọn
  codegen/semantic.py      AST dataflow + semantic/output sanity guard
  codegen/generate.py      baseline flush -> LLM chunks -> checkpoint/self-debug -> rule
  submission/build.py      strict evidence/replay guard + results.json + submission.zip
  validation/              sinh câu hỏi synthetic có đáp án + chấm P/R/F2/Acc
scripts/01..07             CLI từng bước
kaggle/                    kaggle_codegen.py + notebook import sẵn
```

**Định dạng "tidy CSV"** (file trong `data/` của bài nộp): mỗi ô số của bảng gốc là 1 dòng `row,label,code,col,col_name,value,unit_scale` — dtype thuần nhất nên `pd.read_csv` của BTC tái lập đúng DataFrame mà `pandas_query` dùng (số đã parse, ngoặc đơn đã thành số âm, kèm mã số VAS và hệ số đơn vị). Đề bài cho phép nộp dữ liệu đã chuẩn hoá.

## 6. Lộ trình tiếp theo (theo strategy)

- Giai đoạn 1: BGE-M3 + Qwen3-Reranker-4B trong scope đã khoá (thêm vào `retrieval/`), tăng `--n` 8–16, thêm verifier Phi-4/GenSelect (`codegen/`), dò k theo F2.
- Giai đoạn 2: synthetic data → QLoRA Qwen2.5-Coder-14B (train trên Kaggle/Colab), ensemble đa codegen, gia cố câu thuyết minh.
