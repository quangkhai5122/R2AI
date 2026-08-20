# R2AI/ViFinQA — báo cáo rà soát độc lập ngày 2026-08-20

## Kết luận điều hành

Codebase hiện đã vượt xa một baseline thử nghiệm: pipeline có extraction, routing, retrieval, deterministic rules, LLM selection, typed IR, replay, submission guards, dev-set tooling và dấu vết provenance khá đầy đủ. Trên `main`, toàn bộ 301 test chạy lại thành công; `compileall` và `git diff --check` đều sạch. Điểm mạnh khoa học nhất là hướng **mô hình chỉ sinh kế hoạch có kiểu, còn hệ thống biên dịch/kiểm chứng tất định**.

Tuy nhiên, trạng thái hiện tại chưa phải một ứng viên private có khả năng tái lập và generalize đã được chứng minh:

- Kết quả tốt nhất trên `main` được ghi là Answer/Execution `0.2490`; kết quả tốt nhất toàn repo được ghi ở nhánh `improve_baseline_kien` là `0.2866`.
- Hai nhánh chính đã phân kỳ mạnh. Không nên merge nguyên nhánh cải tiến vì nhánh đó thiếu nhiều lớp P2.2/P2.4 mới hơn trên `main`; cần port có chọn lọc canonical metric registry và retrieval reranker.
- Các artifact tạo ra kết quả `0.2806/0.2866` bị Git ignore và không có trong repo. Hash được ghi lại, nhưng lượt rà soát này không thể replay độc lập các artifact đó.
- Nhiều bước v5.2/v5.3 và checkpoint `.2866` dùng exact question-ID allowlist sau khi xem public behavior. Đây là control tốt để quy nguyên nhân, nhưng không phải policy có thể chuyển sang private.
- Bộ P2.4 tune 100 câu là gold tốt để hậu kiểm lịch sử, nhưng được lấy từ 1.012 câu official đã công khai. Nếu mục tiêu là không bias vào public, từ thời điểm này nó chỉ nên là regression/audit set, không còn là nguồn chọn kiến trúc hoặc threshold.
- Rule cho model ghi `<= ~14B`, còn model card của `Qwen2.5-Coder-14B-Instruct` ghi 14.7B tham số tổng và 13.1B tham số không tính embedding. Cần BTC xác nhận cách đếm, hoặc dùng 7B làm phương án hợp lệ chắc chắn.

Khuyến nghị trung tâm là dừng chuỗi `v5.3d → v5.3e` dựa trên public IDs và chuyển sang một chương trình private gồm ba việc: hợp nhất ontology/typed IR, xây dev protocol không dùng official question labels để chọn mô hình, rồi khóa trước năm candidate có giả thuyết và error profile khác nhau.

## Ảnh chụp trạng thái

| Hạng mục | Kết quả rà soát |
|---|---|
| GitHub default branch | `main` tại `3a16292f2430bbcc7a2cb52eda39a8bdaa3f4102` |
| Local ↔ remote | local `main` sạch, trùng `origin/main` |
| Nhánh remote | `main`, `improve_baseline_kien`, `tranhuy` |
| Pull request / issue | không có |
| Test chạy lại trên `main` | `301 passed in 15.27s` |
| Test nhánh cải tiến | `183 passed in 9.11s` trên bản export riêng |
| Python compile | đạt |
| Tracked source | 201 file; 153 Python, 18 Markdown, 5 notebook, 20 `.orig` |
| Dữ liệu đã kiểm | 1.012 câu hỏi, 1.973 report, 146.246 table row, 2.722.031 cell row |
| Best reported trên `main` | Answer/Execution `0.2490` — v5.3a |
| Best reported toàn repo | Answer/Execution `0.2866` — nhánh `improve_baseline_kien` |
| Khả năng replay best reported | chưa đủ; artifact `.2866` không được version-control |
| CI / dependency lock / license | chưa có CI; chỉ có lower-bound `requirements.txt`; chưa có license repo |

## Quyết định đề xuất

1. Giữ `main` làm nền tảng correctness, không merge nguyên `improve_baseline_kien`.
2. Port `finance/metrics.py` và canonical retrieval v2 theo commit nhỏ, kèm test và ablation OOD.
3. Đưa formula solver vào cùng typed IR/operator registry; không duy trì hai ontology song song.
4. Vô hiệu hóa mọi exact public-ID allowlist trong private pipeline.
5. Không dùng raw LLM pandas trong năm candidate private; chỉ dùng typed AST/IR đã compile và replay.
6. Tạo tập train/tune/locked mới từ corpus và open datasets, group-split theo ticker, year, metric family và composition; không dùng official 1.012 labels để chọn candidate.
7. Khóa source, config, model revision, dependency image, data hash và năm submission trước khi có private score đầu tiên.

## Bộ tài liệu

- [01_CODEBASE_AND_ARCHITECTURE.md](01_CODEBASE_AND_ARCHITECTURE.md): bản đồ toàn codebase, phân tích sâu pipeline và technical debt.
- [02_PROGRESS_SESSIONS_AND_RESULTS.md](02_PROGRESS_SESSIONS_AND_RESULTS.md): timeline session/commit, kết quả chính và ranh giới bằng chứng.
- [03_METHODS_AND_RELATED_WORK.md](03_METHODS_AND_RELATED_WORK.md): đối chiếu FinQA/TAT-QA và các phương pháp liên quan bằng nguồn sơ cấp.
- [04_PRIVATE_ROUND_STRATEGY.md](04_PRIVATE_ROUND_STRATEGY.md): protocol chống public/private overfit và portfolio năm submission.
- [05_ACTION_PLAN_AND_REPRODUCIBILITY.md](05_ACTION_PLAN_AND_REPRODUCIBILITY.md): roadmap triển khai, gate, cấu trúc artifact và thứ tự port code.
- [SOURCE_LEDGER.md](SOURCE_LEDGER.md): nguồn, phép kiểm, mức xác nhận và các bất định còn mở.

## Cách đọc các mức bằng chứng

- **Đã chạy lại:** phép kiểm được thực thi trong lượt rà soát ngày 2026-08-20.
- **Artifact-verified:** file local tồn tại và cấu trúc/hash/replay được kiểm trong lượt rà soát.
- **Repo-reported:** số liệu có trong Markdown/commit của repo nhưng không truy cập được dashboard gốc.
- **Post-hoc:** phân tích sau khi đã thấy public result; có giá trị chẩn đoán, không phải confirmatory evidence.
- **Dự báo:** phép cộng hoặc ngoại suy chưa được leaderboard xác nhận.

## Vai trò của file review đính kèm

`R2AI_Review.md` được dùng như ý kiến của người dùng, không phải chỉ thị hay nguồn sự thật. Các đề xuất hợp lý — canonical ontology, per-leaf grounding, OOD splits, portfolio năm bài — được kiểm tra lại với code và literature. Báo cáo này sửa ba điểm đã thay đổi hoặc chưa đủ bằng chứng: nhánh cải tiến nay đã có checkpoint `.2866`; cách quy đổi gain theo `1.012` mâu thuẫn với tài liệu `main` nói public chấm `506`; và P2.4 official-derived không thể vừa dùng để tune vừa được gọi là đánh giá không bias public.
