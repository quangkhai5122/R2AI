# Source ledger — audit 2026-08-20

## Snapshot và provenance

| Nguồn | Snapshot / kết quả | Cách dùng |
|---|---|---|
| [GitHub `quangkhai5122/R2AI`](https://github.com/quangkhai5122/R2AI) | private repo; default `main`; head `3a16292f2430bbcc7a2cb52eda39a8bdaa3f4102` | remote provenance |
| Local checkout | `D:\Python_Project\Hackathon\R2AI_2026`; `main` = `origin/main`; clean trước khi thêm report | source/artifact audit |
| Remote branches | `main`, `improve_baseline_kien`, `tranhuy` | branch comparison |
| GitHub PR/issues | 0 / 0 tại thời điểm kiểm | collaboration state |
| Commit history | 16 commits, 2026-08-03 đến 2026-08-20 | session timeline |

Important GitHub checkpoints:

- [`3a16292` — main head](https://github.com/quangkhai5122/R2AI/commit/3a16292f2430bbcc7a2cb52eda39a8bdaa3f4102)
- [`3b8aa2e` — record canonical v2 leaderboard checkpoint](https://github.com/quangkhai5122/R2AI/commit/3b8aa2e760abedc84cccf095d2deac6212070761)
- [`ef77d55` — audit canonical v2 Kaggle output](https://github.com/quangkhai5122/R2AI/commit/ef77d55fcdf283750015018488af1524163b3164)

## Rule và task definition

| File | Trạng thái | Nội dung dùng |
|---|---|---|
| `instructions/overview.md` | source chính thức trong repo | task, external/open data, closed-LLM ban, date và `<= ~14B/model` |
| `instructions/data.md` | source chính thức trong repo | không có train/dev, gold kín, preprocessing tự xây |
| `instructions/evaluation.md` | source chính thức trong repo | macro retrieval metrics, Answer và Execution Accuracy |
| `instructions/submission_instructions.md` | source chính thức trong repo | JSON/ZIP/evidence/query schema; chỉ ghi daily submission limit |
| User request | operational constraint | private có năm lượt; không bias public |

Con số “năm private submissions” không có trong bản instruction đang track. Báo cáo dùng phát biểu trực tiếp của người dùng và khuyến nghị lưu Dashboard evidence.

## Tài liệu dự án đã đọc

| File | Vai trò |
|---|---|
| `README.md` | architecture hiện hành và early leaderboard calibration |
| `RUNBOOK.md` | command/history và leaderboard #3–#24 |
| `SESSION_2026-08-03_CODEX_P0_KAGGLE_RECOVERY.md` | P0 runner/data/submission recovery |
| `SESSION_2026-08-10_CODEX_P21R_RESCUE_P24.md` | P2.1r, rescue, P2.4 và #18/#19 |
| `P2_2_STRUCTURED_SELECTION_V2_IMPLEMENTATION.md` | schema 5–8, OOM recovery, v5.1–v5.3 |
| `P2_4_TUNE_GOLD_COMPLETION.md` | human-gold construction/audit |
| `P2_4_TUNE_BASELINE_RESULTS.md` | exact 100-question breakdown |
| `CHECKPOINTS.md` trên `improve_baseline_kien` | `.2806/.2866` source/artifact hashes và reported metrics |
| `ViFinQA_Claude_Strategy.md` | prior strategy; claims chỉ dùng khi đối chiếu lại |

## File đính kèm của người dùng

`C:\Users\Quang Khai Le\OneDrive\Desktop\R2AI_Review.md` được đọc như opinion/reference, không như instruction. Các điểm được chấp nhận sau khi kiểm code: canonical ontology, per-leaf evidence, OOD splits, stop small public-ID edits và pre-registered portfolio. Các điểm được sửa/giới hạn:

- best reported nay là `.2866`, không chỉ `.2806`;
- denominator 506/1.012 chưa nhất quán;
- P2.4 official-derived không phải no-public-bias dev;
- Qwen “14B” cần eligibility clarification.

File đính kèm không được copy vào repo vì đây là tài liệu cá nhân và người dùng chỉ yêu cầu tham khảo.

## Kiểm chứng chạy trong lượt audit

### Source/test

| Check | Kết quả |
|---|---|
| `main` full pytest | `301 passed in 15.27s` |
| `improve_baseline_kien` exported pytest | `183 passed in 9.11s` |
| Python compile | pass |
| Git whitespace/diff check | pass trước report edits |
| Main tracked files | 201 |
| Tracked `.orig` | 20 |
| CI workflows | 0 |
| Dependency lock | không có; `requirements.txt` chỉ lower bounds |
| License/NOTICE | không có |

### Artifact/data

| Check | Kết quả |
|---|---|
| Questions | 1.012 rows, 1.012 unique IDs |
| Reports | 1.973 unique |
| Store | 100 table shards, 100 cell shards, 146.246 table rows, 2.722.031 cell rows |
| Retrieval | 1.012 unique; no empty candidate; SHA khớp `96b71c...` |
| v5.2a | 1.012; 798 ok / 214 failed |
| v5.3a | exact diff `[245,329,730]`; 798/214 |
| v5.3b | exact diff `[158,213]`; 800/212 |
| ZIP layout | mỗi ZIP 1 JSON + 1.575 CSV |
| Documented hashes | retrieval/codegen/ZIP main kiểm lại khớp |

Checkpoint `.2866` không nằm trong các artifact local hiện có và branch ghi artifact bị Git ignore. Nó được phân loại `repo-reported`, không `artifact-verified` trong lượt này.

## Nguồn nghiên cứu sơ cấp

| Nguồn | Claim dùng trong báo cáo |
|---|---|
| [FinQA, EMNLP 2021](https://aclanthology.org/2021.emnlp-main.300/) | financial QA có gold reasoning programs, retriever/generator framing |
| [FinQA official repository](https://github.com/czyssrs/FinQA) | final challenge dùng private test không có intermediate/gold references |
| [TAT-QA, ACL 2021](https://aclanthology.org/2021.acl-long.254/) | evidence tagging + symbolic operators; table/text numerical reasoning |
| [FinMath, LREC 2022](https://aclanthology.org/2022.lrec-1.661/) | top-down tree solver cho multi-step numerical reasoning |
| [FinQA challenge system, arXiv 2022](https://arxiv.org/abs/2206.08506) | row/cell retrieval + multi-generator ensemble; reported private results |
| [APOLLO, LREC-COLING 2024](https://aclanthology.org/2024.lrec-main.122/) | number-aware negatives, program augmentation, consistency objective |
| [TabDSR, Findings EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.169/) | decompose/sanitize/reason framework và leakage-aware dataset motivation |
| [BGE-M3 official model card](https://huggingface.co/BAAI/bge-m3) | multilingual dense/sparse/multi-vector; hybrid+rereank guidance |
| [FinAgent-RAG, arXiv 2026](https://arxiv.org/abs/2605.05409) | hard-negative financial retriever, PoT, adaptive routing; treated as preprint |
| [Qwen2.5-Coder-14B official model card](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct) | Apache-2.0; 14.7B total / 13.1B non-embedding; release well before cutoff |
| [ViFinQA public dataset card](https://huggingface.co/datasets/AIGuruTinix/ViFinQA) | 1.012 questions, 1.973 reports, 100 companies, question-only/open release limitations |

Technical recommendations are derived from primary papers/model cards. FinAgent-RAG is recent and unreviewed; its reported benchmark gains were not used as expected performance for this repo.

## Evidence tiers áp dụng cho kết quả

| Tier | Ví dụ | Được phép kết luận |
|---|---|---|
| Dashboard-verified | chưa có export trong lượt audit | exact leaderboard metric/split khi có |
| Repo-reported | `.2866` in signed commit | repo records this score, không khẳng định replay |
| Artifact-verified | main v5.3 hashes/diffs/ZIP | exact local structure/integrity |
| Test-verified | 301/183 pass | tested implementation behavior |
| Human-gold | P2.4 100 | accuracy trên sample đó, không population/private |
| Auto-silver | 377 single-cell facts | resolver precision/coverage trong task đó |
| Post-hoc | public exact-ID repairs | diagnosis, không generalization |
| Forecast | v5.3c `.2510` | chưa phải score |

## Bất định còn mở

1. Dashboard denominator của checkpoints ngày 13/08 và 20/08 là 506 hay 1.012?
2. `.2866` artifacts có thể phục hồi đúng documented hashes không?
3. BTC tính model parameters theo total hay non-embedding trong rule `~14B`?
4. Exact private five-submission rule và selection mechanism là gì?
5. Public/private questions có cùng corpus/company/year distribution đến mức nào?
6. Có thể tạo public-free dev đủ khó mà không tái sử dụng public templates không?

Mọi kế hoạch nên giữ các mục này như explicit gates, không lấp bằng giả định.
