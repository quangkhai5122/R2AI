# Khung năm lượt private

Năm lượt không được dùng như năm bước hill-climb theo public/private score. Trước khi có private feedback, mỗi artifact phải được pre-register bằng source hash, config hash, model revision, payload hash, OOD report và submission ZIP hash.

| Lượt | Candidate | Khác biệt cấp chiến lược | Điều kiện được chiếm slot |
|---|---|---|---|
| S1 | B0 deterministic | không LLM; ưu tiên precision/replay | full build sạch, zero crash, OOD deterministic report |
| S2 | B1 Selection v2 7B | typed semantic planner nhỏ | tăng OOD semantic coverage mà không tăng crash/unit errors quá gate |
| S3 | Selection v2 14B | model-scale/diversity, cùng compiler | disagreement có ích với S2 trên OOD locked set; model ≤15B |
| S4 | row-aware retrieval | thay đổi evidence acquisition, giữ answer compiler | cải thiện retrieval recall trên source-derived OOD và không làm precision sụt quá gate |
| S5 | architecture-diverse candidate | ví dụ deterministic formula-family planner hoặc model family khác ≤15B | error correlation thấp với S2/S3; không phải overlay vài ID |

Nếu một candidate không qua gate, không tự động thay bằng một chỉnh sửa nhỏ trên candidate tốt nhất. Slot đó nên được giữ cho một giả thuyết độc lập hoặc bỏ trống. Private score, nếu được trả về từng lượt, chỉ dùng để quan sát; không dùng để sửa threshold/ID list cho lượt kế tiếp trừ khi rule chính thức mô tả private như một development phase.

B0/B1 trong phiên bản này mới là hai artifact đã định nghĩa đầy đủ. S3–S5 là portfolio hypotheses, chưa phải implementation đã xác minh.
