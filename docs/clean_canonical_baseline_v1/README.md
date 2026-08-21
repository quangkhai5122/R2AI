# Clean canonical baseline v1 (G0–G2)

## Kết quả triển khai

Phiên bản này tạo một đường chạy clean mới trên nền `main@0ce20aa`, không merge nguyên nhánh `improve_baseline_kien` và không thay đổi runner schema-8 lịch sử. Clean path dùng payload schema 9, canonical registry 139 metric, component-aware retrieval, Selection v2 typed IR và fingerprint xuyên suốt source/store/retrieval/operator/environment.

BTC đã xác nhận mô hình dưới hoặc trên 14B đều hợp lệ nếu không vượt 15B. Vì vậy Qwen 14B không còn bị loại vì eligibility. B1 vẫn chọn Qwen2.5-Coder-7B để tạo mốc chi phí/thời gian dễ lặp; 14B là một trục model-scale hợp lệ cho candidate sau, không phải lý do sửa B1.

## Quyết định đối với đề xuất G0–G2

| Đề xuất | Quyết định | Lý do |
|---|---|---|
| Gộp G0–G2 trong một epic | Giữ, nhưng gate tuần tự | Cho phép audit provenance trước khi thay đổi semantics. |
| Dùng `.2866` làm seed | Bác bỏ | Artifact absent và policy có exact-ID overlay; chỉ giữ historical evidence. |
| Merge nguyên nhánh cải tiến | Bác bỏ | Sẽ kéo theo code cũ và public-derived controls. |
| Port canonical metrics v2 | Giữ | 139 metric, qualifiers, codes, component graph là abstraction tổng quát. |
| Port `formula_solver.py` | Bác bỏ | Tạo solver thứ hai khoảng 1.600 dòng, trùng Selection v2/compiler hiện tại. |
| Canonical/component retrieval | Giữ có cấu hình | Extra weight 0.35; row rerank mặc định tắt để có ablation sạch. |
| BGE learned rerank ở G2 | Hoãn | Thêm dependency/model và làm khó quy nguyên nhân. |
| Sửa runner schema-8 tại chỗ | Điều chỉnh | Giữ frozen historical runner; thêm clean schema-9 launcher fail-closed. |
| Qwen 7B do lo ngại rule 14B | Sửa lý do | 7B chỉ là cost/reference baseline; 14B hợp lệ theo BTC. |

## Contract chống public bias

Clean payload không có `targets/`, ID allowlist, `--llm-ids-file`, official-derived gold, raw-code mode hoặc khả năng bỏ verify manifest. Clean runner chỉ nhận `select_v2`, kiểm tra model name nếu quảng bá kích thước trên 15B, và hash toàn bộ runtime input.

P2.4/official 1.012 câu chỉ được chạy như regression sau khi candidate và threshold đã freeze. Kết quả regression không được dùng để thêm per-ID fix hoặc đổi threshold trước private.

## Hai baseline khóa trước private

- B0: cùng clean retrieval/store nhưng `NoLLM`, deterministic rules, `k=0` dùng evidence budget.
- B1: cùng retrieval/store, Qwen2.5-Coder-7B-Instruct-AWQ, Selection v2, `n=2`, temperature `0.2`, mọi câu đều đi qua planner và arbitration chung.

B0/B1 khác nhau ở answer path; không khác ID set, target mask hoặc dữ liệu gold. Qwen 14B có thể trở thành candidate B2 sau khi qua OOD gate độc lập.

## Trạng thái xác minh

- canonical registry: 139 metric, 14 test gốc pass;
- clean contract/retrieval/payload: 22 test pass;
- smoke thật: 3/3 record hoàn tất clean retrieval và B0 rule path;
- Kaggle GPU B1: chưa chạy trong phiên này;
- upload Dataset/submission: chưa thực hiện.

Các output smoke nằm trong `artifacts/clean_v1/` và bị Git ignore; source/config/test/docs mới là artifact được version-control.
