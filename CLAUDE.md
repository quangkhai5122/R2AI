# ViFinQA Competition — Project Memory

Cuộc thi: Financial Table Retrieval & Text-to-Pandas trên BCTC 100 công ty niêm yết VN
(1.973 báo cáo OCR, 2015–2025). Chiến lược gốc: `ViFinQA_Claude_Strategy.md`.
Luật thi: `instructions/*.md`. Hướng dẫn vận hành: `README.md`.
Thay đổi P0/Kaggle mới nhất: `SESSION_2026-08-03_CODEX_P0_KAGGLE_RECOVERY.md`.

## SỰ THẬT ĐÃ XÁC NHẬN (không cần kiểm chứng lại)

1. **Vị trí bảng trong `relevant_tables` = SỐ DÒNG (1-based) nơi `<table>` bắt đầu
   trong file OCR .txt** — BTC xác nhận trực tiếp qua trao đổi với đội, và đã
   verify trên leaderboard (TABLES_F2: 0.0 với thứ-tự-bảng → 0.364 với line-number).
   Code: store có cột `line_no`; nội bộ vẫn key bằng `table_pos` (thứ tự 0-based);
   map sang line khi build submission (`config.TABLE_POS_MODE = "line"` là mặc định).
2. **`report_id` format đúng** = tên thư mục báo cáo, vd
   `VNM_financial_statements_2023_consolidated` (DOCS_F2 0.84, MRR5 0.95).
3. **Test set = 506 câu** trong số 1.012 câu public (HF). Nộp đủ 1.012 vẫn được
   chấm bình thường; warning `gold=506 pred=1012` vô hại. BTC KHÔNG phát dữ liệu
   riêng — bản HuggingFace trong `data/ViFinQA/` là nguồn chuẩn.
4. **Suy ra từ leaderboard:** gold trung bình ~2.6 bảng/câu, ~1.15 doc/câu, chủ
   yếu trong cùng 1 báo cáo.
5. **`--expand-docs` ĐÃ THỬ VÀ THẤT BẠI** (DOCS_F2 0.84→0.61; precision sập
   0.91→0.28 đổi lấy +5.5% recall). Đừng thử lại.
6. **Grader chấp nhận format hiện tại**: tidy CSV evidence + pandas_query dạng
   expression (EXEC == ANSWER chứng tỏ query chạy được trên hệ của BTC).
7. **vLLM KHÔNG CHẠY ĐƯỢC trên Kaggle T4** (Turing/SM75). Đã thử 2 lần:
   `swap_space` bị bỏ khỏi EngineArgs (đã fix bằng retry-loop drop kwarg), rồi
   `RuntimeError: Engine core initialization failed` — V1 engine không hỗ trợ
   SM75, mà V0 engine (chạy được T4) đã bị xoá khỏi vLLM nên `VLLM_USE_V1=0`
   vô hiệu. **Giải pháp: `HfBatchClient` (transformers batched generate),
   `--backend hf` — mặc định của notebook.** Chỉ thử lại vLLM khi pin
   `vllm==0.7.3` và còn dư thời gian.
8. **`SUBMISSION_K=5` ĐÃ ĐƯỢC LEADERBOARD KIỂM CHỨNG** so với k=10:
   TABLES_F2 0.3641→0.4092 (+0.0451; +12,4% tương đối), precision
   ~0.176→0.2621, recall ~0.684→0.5852 và MRR5 gần như không đổi
   (0.588→0.5882). DOCS_F2 giảm nhẹ 0.8399→0.8093. Kết quả này ủng hộ
   k=5 thay k=10, nhưng CHƯA chứng minh k=5 tối ưu so với k=7. Không suy
   F2 macro bằng cách cắm macro precision/recall vào công thức F2.

## LỊCH SỬ NỘP BÀI

| # | Cấu hình | TABLES_F2 | DOCS_F2 | ANSWER | Ghi chú |
|---|---|---|---|---|---|
| 1 | rule-only, pos=order base0, k=10 | 0.0 | 0.8399 | 0.085 | |
| 2 | như #1, base1 | 0.0 | 0.8399 | 0.085 | giống hệt #1 → order sai |
| 3 | pos=line, k=10 | **0.3641** | 0.8399 | 0.085 | line ĐÚNG (P .176/R .684/MRR5 .588) |
| 4 | #3 + expand-docs | 0.3641 | 0.6066 | 0.085 | expand-docs FAIL |
| 5 | pos=line, k=5, rule-only | **0.4092** | 0.8093 | 0.085 | EXEC .085; TABLE P .2621/R .5852/MRR5 .5882; DOC P .9168/R .7999/MRR5 .9457 |

## TRẠNG THÁI & VIỆC TIẾP THEO

- ANSWER/EXEC 0.085 là trần của rule-baseline. Submission #5 vẫn là rule-only:
  artifact có 188 `source=rule`, 824 `source=none`, 0 `source=llm`; baseline
  Qwen trên Kaggle CHƯA có điểm leaderboard.
- **Việc giá trị nhất: chạy lại Kaggle codegen bằng payload schema v2 hiện hành.**
  Notebook `kaggle/vifinqa-codegen.ipynb` dùng backend `hf`, mặc định
  `Qwen/Qwen2.5-Coder-7B-Instruct --load-4bit` (bitsandbytes NF4). Payload cũ
  từng gây log `LLM round 1: 1012 prompts`, không có baseline/chunk và CUDA OOM;
  nguyên nhân là notebook cũ không cập nhật `generate.py`. Notebook hiện tại
  KHÔNG hot-patch: nó kiểm SHA-256 rồi copy đúng nguyên trạng code trong payload.
- `SUBMISSION_K=5` đã được kiểm chứng: TABLES_F2=0.4092, tốt hơn k=10
  (0.3641), dù DOCS_F2 giảm 0.8399→0.8093. Có thể build ablation k=7 từ
  cùng retrieval/codegen, không cần chạy lại LLM. Lượt Qwen đầu tiên phải giữ
  `sub-k=5`, `expand-docs=False` để so trực tiếp với #5.
- Trong-doc table recall ≈ 0.82 → ~18% bảng gold (đa phần thuyết minh) chưa vào
  top-k. Giai đoạn 1 theo strategy: BGE-M3 + Qwen3-Reranker trong scope đã khoá.
- Payload Kaggle làm việc trong không gian `table_pos` (order) — map sang line
  diễn ra ở local lúc build submission → codegen_results.jsonl cũ/mới đều tương thích.
- P0 data snapshot đã dựng riêng ở `artifacts/store_p0` +
  `artifacts/retrieval_p0.jsonl` và full replay sạch. Nó đổi top-5 ở 181/1.012
  câu so với control #5, nên CHƯA trộn vào lượt Qwen control; xem session log.

## KIẾN TRÚC (tóm tắt)

```
data/ViFinQA/                nguồn HF: financial_statements/, questions/, code_stock.csv
vifinqa/                     package chính (xem README §5 cho map chi tiết)
  config.py                  TABLE_POS_MODE="line", SUBMISSION_K=5, đường dẫn
  extraction/                parse <table> + số VN + đơn vị + store parquet (line_no ở đây)
  router/ retrieval/         structured lookup (ticker/năm/doc_type) + BM25 in-scope
  codegen/                   HF batch + semantic guard/self-debug + rule fallback
  submission/build.py        map table_pos→line_no, build zip
  validation/                synthetic val (gold cũng dùng line-number)
scripts/01..07               pipeline CLI theo thứ tự
kaggle/                      notebook verify đúng một payload + kaggle_codegen.py
artifacts/                   store/, retrieval.jsonl, submission/ (gitignored)
```

## HIỆU NĂNG KAGGLE (đo trên dữ liệu thật, 2026-08)

- Prompt dài: k=5 → ~3.070 token, k=4 → ~2.630, k=3 → ~2.040 (median).
- `--rule-first` chỉ bỏ qua **~10%** câu (rule conf≥90) → vẫn còn ~900 câu cho LLM.
- 7B NF4 + batch 4 + n=2 + max_tokens 384 + k=5 = **quá nặng cho 1 phiên 9h**
  (user chạy 9h chưa xong). Cấu hình khuyến nghị: `--n 1 --k 4 --max-tokens 256
  --batch-size 4 --temperature 0`. Prefill chiếm phần lớn → giảm `k` là đòn
  bẩy mạnh nhất; greedy giúp resume ổn định hơn khi chỉ lấy một sample.
- `run_codegen` giờ **crash-safe**: ghi rule-baseline đầy đủ ngay phút thứ ~2,
  rồi LLM ghi đè dần theo chunk (`--checkpoint-every 32`), `--time-budget-min`
  dừng sạch trước giới hạn phiên, chạy lại = **resume** (bỏ qua câu đã có
  source `llm*`). File out LUÔN đủ 1.012 dòng và nộp được ở mọi thời điểm.
- HF runner tự giảm batch khi CUDA OOM (`4→2→1`) và checkpoint round 1 trước
  self-debug. Resume chỉ nhận kết quả có cùng `run_signature`; phiên Kaggle mới
  phải đưa checkpoint cũ về đúng đường dẫn `/kaggle/working/codegen_results.jsonl`.

## GOTCHAS

- Đổi code extraction ⇒ PHẢI rebuild store (`scripts/01_build_store.py`, ~30 phút full).
- `artifacts/validation/val_gold.json` sinh trước 2026-08-02 dùng vị trí order —
  chạy lại `scripts/06_gen_validation.py` nếu cần đánh giá offline.
- vLLM API trôi giữa version: `VllmBatchClient` tự drop kwarg không hỗ trợ.
  T4 lỗi V1/compute-capability → pin `vllm==0.7.3`.
- Notebook yêu cầu đúng một payload schema v2 và kiểm SHA-256 mọi file runtime;
  attach nhiều phiên bản payload hoặc payload chưa upload lại sẽ dừng ngay.
- Windows console: mọi script đã gọi `setup_stdout()` (UTF-8).
- Corpus TiniX license CC BY-NC 4.0 — ghi nguồn, phi thương mại.
- Mỗi lần nộp chỉ đổi MỘT biến (trừ khi các metric nhóm độc lập: k ảnh hưởng
  TABLES, codegen ảnh hưởng ANSWER/EXEC — có thể gộp).
