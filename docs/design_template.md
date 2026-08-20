# Multi-Agent Research System Design Document

### 🎓 Thông Tin Học Viên
- **Họ và tên:** Nguyễn Đăng Nam
- **Mã học viên:** 2A202601307
- **Khóa học:** VinUni AI20k - Khóa 3 Track 3

---

## 1. Problem (Vấn Đề Kỹ Thuật)

Xây dựng hệ thống nghiên cứu sâu tự động (**Autonomous Deep Research System**) có khả năng tiếp nhận câu hỏi mở đa chiều từ người dùng, tự động truy xuất tài liệu từ kho tri thức/Web, phân tích phản biện và tổng hợp báo cáo chuyên sâu có trích dẫn nguồn chuẩn học thuật.

## 2. Why Multi-Agent? (Vì sao Single-Agent Chưa Đủ?)

- **Cognitive Overload (Quá tải nhận thức):** Single-Agent phải gánh vác cùng lúc việc tìm kiếm, đọc hiểu, phân tích mâu thuẫn và hành văn trong 1 prompt duy nhất, dẫn đến câu trả lời hời hợt và dễ bỏ sót thông tin.
- **Cascading Hallucinations:** Thiếu cơ chế kiểm chứng chéo độc lập khiến thông tin sai lệch từ một nguồn dễ bị khuếch đại thành kết luận sai.
- **Separation of Concerns:** Multi-Agent phân tách rõ ràng trách nhiệm: Researcher (Thu thập), Analyst (Phản biện & Trade-offs), Writer (Tổng hợp & Trích dẫn) và Supervisor (Điều phối tập trung).

## 3. Agent Roles & Failure Modes

| Agent | Responsibility | Input | Output | Failure mode & Cách khắc phục |
|---|---|---|---|---|
| **Supervisor** | Điều phối tập trung, kiểm tra trạng thái State, chọn Worker tiếp theo | `ResearchState` | `next_route: str` | Lặp vô hạn ➔ Rào chắn `max_iterations = 6` ép dừng chu trình. |
| **Researcher** | Tối ưu truy vấn, tìm kiếm trên Corpus/Web, lọc tài liệu chất lượng | `request.query` | `sources`, `research_notes` | Không tìm thấy tài liệu ➔ Tự động fallback sang kho 30 topics hoặc mock data. |
| **Analyst** | Kiểm chứng chéo, phát hiện mâu thuẫn giữa các nguồn, rút trích trade-offs | `research_notes` | `analysis_notes` | Bỏ sót mâu thuẫn ➔ Prompt ép buộc so sánh 2 chiều và đúc kết trade-offs. |
| **Writer** | Soạn thảo báo cáo Markdown chuẩn, gán trích dẫn `[1]`, `[2]` | `analysis_notes`, `sources` | `final_answer` | Bịa trích dẫn (Hallucinated citations) ➔ Thuật toán kiểm tra regex index nguồn hợp lệ. |

## 4. Shared State Schema (ResearchState)

- `request: ResearchQuery`: Lưu trữ câu hỏi nghiên cứu, cấu hình `max_sources` và `audience`.
- `sources: list[SourceDocument]`: Danh sách tài liệu, snippet, URL nguồn thu thập được từ SearchClient.
- `research_notes: str | None`: Tóm tắt các phát hiện thô của Researcher để Analyst dễ dàng tiếp thu.
- `analysis_notes: str | None`: Luận điểm phản biện & trade-offs của Analyst.
- `final_answer: str | None`: Báo cáo Markdown hoàn chỉnh của Writer có trích dẫn `[1]`, `[2]`.
- `route_history: list[str]`: Nhật ký các bước điều phối của Supervisor (`['researcher', 'analyst', 'writer', 'done']`).
- `agent_results: list[AgentResult]`: Chi tiết kết quả, số lượng token, chi phí USD của từng lần gọi agent.
- `iteration: int`: Biến đếm số lượt điều phối qua lại, phục vụ guardrail ngắt `max_iterations`.

## 5. Routing Policy & Graph Flow

```text
               ┌─────────────┐
               │    START    │
               └──────┬──────┘
                      ▼
            ┌───────────────────┐
      ┌────►│    SUPERVISOR     │◄────┐
      │     └─────────┬─────────┘     │
      │               │ (Conditional) │
      │   ┌───────────┼───────────┐   │
      │   ▼           ▼           ▼   │
      │ ┌──────────┐┌──────────┐┌───┴─┴────┐
      │ │Researcher││ Analyst  ││  Writer  │
      │ └────┬─────┘└────┬─────┘└───┬──────┘
      │      │           │          │
      └──────┴───────────┴──────────┘
                      │ (done)
                      ▼
                 ┌─────────┐
                 │   END   │
                 └─────────┘
```

## 6. Safety Guardrails & Operations

- **Max iterations:** Giới hạn tối đa 6 vòng lặp điều phối để triệt tiêu nguy cơ kẹt vòng lặp vô hạn.
- **Timeout:** 60 giây cho mỗi HTTP request.
- **Retry:** Thư viện `tenacity` retry tự động 3 lần với exponential backoff khi gặp lỗi mạng/rate-limit.
- **Fallback:** Tự động chuyển đổi giữa Tavily Web Search ➔ Offline 30-Topics Corpus ➔ Mock Docs.
- **Validation:** Pydantic schema validation cho toàn bộ Input/Output dữ liệu.

## 7. Benchmark Plan & Evaluation Metrics

- **Truy vấn đo lường:** 30 chủ đề kỹ thuật chuyên sâu từ kho `ai_agent_offline_research_corpus_v2`.
- **5 Chỉ số đo lường:**
  1. *Latency:* Thời gian chạy (wall-clock seconds).
  2. *Cost:* Tổng chi phí token USD.
  3. *Quality Score:* Chấm điểm 0-10 theo Rubric (Độ dài, Cấu trúc, Kỷ luật trích dẫn, Chiều sâu).
  4. *Citation Coverage:* Tỷ lệ nguồn tham khảo được trích dẫn thực tế trong bài viết.
  5. *Failure Rate:* Tỷ lệ lỗi (0%).
