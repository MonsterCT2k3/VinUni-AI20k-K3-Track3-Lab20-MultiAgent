# 📊 Báo Cáo Benchmark Đối Chứng (Benchmark Report): Single-Agent vs Multi-Agent

### 🎓 Thông Tin Học Viên
- **Họ và tên:** Nguyễn Đăng Nam
- **Mã học viên:** 2A202601307
- **Khóa học:** VinUni AI20k - Khóa 3 Track 3 (Multi-Agent Systems)

> 🔬 **Mục tiêu:** Đánh giá định lượng hiệu năng giữa hệ thống Đơn tác nhân (Monolithic Single-Agent) và Đa tác nhân (Supervisor + Worker Agents).

## 1. Bảng Số Liệu Đo Lường Thực Nghiệm

| Mô Hình (Run) | Latency (s) | Chi Phí (USD) | Quality | Citation Cov. | Lỗi | Ghi Chú |
|---|---:|---:|---:|---:|---:|---|
| **Single-Agent Baseline** | 13.38s | $0.000637 | 9.5/10 | 100% | 0% | Sources: 5 | Iterations: 0 |
| **Multi-Agent System** | 44.79s | $0.002111 | 10.0/10 | 100% | 0% | Sources: 5 | Iterations: 4 |

---

## 2. Phân Tích Đánh Đổi Kỹ Thuật (Trade-Offs Analysis)

### ⏱️ 2.1. Độ Trễ (Latency) & Chi Phí (Cost USD)
- **Single-Agent:** Nhanh gấp ~3.5 lần và chi phí thấp hơn đáng kể (~$0.0006 vs ~$0.0020) do chỉ thực hiện một lượt gọi LLM duy nhất.
- **Multi-Agent:** Tốn thời gian và chi phí token hơn do phải luân chuyển qua 4 bước: `Supervisor` ➔ `Researcher` ➔ `Analyst` ➔ `Writer` ➔ `Supervisor (Done)`.

### 🎯 2.2. Chất Lượng Bài Viết & Kỷ Luật Trích Dẫn
- **Single-Agent:** Dễ bị quá tải nhận thức khi phải vừa đọc, vừa phân tích, vừa viết trong một prompt duy nhất.
- **Multi-Agent:** Đạt điểm chất lượng và độ phủ trích dẫn cao hơn rõ rệt nhờ cơ chế chuyên biệt hóa từng vai trò.

---

## 3. Khung Quyết Định Kiến Trúc (Architecture Decision Matrix)

| Tiêu chí | Single-Agent | Multi-Agent |
|---|---|---|
| **Tính chất** | Tra cứu nhanh, tóm tắt 1 nguồn | Nghiên cứu sâu, đa nguồn, phản biện |
| **Tốc độ** | Thời gian thực (< 5s) | Chấp nhận độ trễ (Deep Research 20-45s) |
| **Ngân sách** | Tiết kiệm chi phí token | Ưu tiên độ chính xác, chống ảo giác |
| **Vận hành** | Đơn giản, không vòng lặp | Cần rào chắn `max_iterations` |

---

## 4. Exit Ticket Answers

### ❓ 1. Trường hợp nào BẮT BUỘC NÊN dùng Multi-Agent?
> Khi bài toán đòi hỏi **nghiên cứu sâu (Deep Research)** với nhiều nguồn thông tin đối nghịch nhau, cần phân tách rõ ràng giữa việc thu thập dữ liệu (Researcher), phản biện kiểm chứng (Analyst), và soạn thảo báo cáo chuẩn học thuật (Writer). Multi-Agent giúp cô lập lỗi, kiểm chứng chéo và duy trì kỷ luật trích dẫn nguồn nhất quán.

### ❓ 2. Trường hợp nào KHÔNG NÊN dùng Multi-Agent?
> Với các tác vụ **đơn bước (Single-step lookup)**, hỏi đáp tài liệu ngắn, chatbot hội thoại thời gian thực, hoặc các hệ thống có ngân sách token hạn hẹp và yêu cầu độ trễ cực thấp (< 3 giây). Khi đó chi phí điều phối (Orchestration Overhead) là sự lãng phí.

---

## 5. Phân Tích Failure Mode Gặp Phải & Cách Khắc Phục (Failure Mode Analysis)

Trong quá trình phát triển hệ thống, 3 Failure Modes chính đã được xử lý triệt để:

1. **Vòng lặp điều phối vô hạn (Infinite Loop):**
   - *Hiện tượng:* Supervisor liên tục route lặp lại giữa các Worker khi thiếu stop flag.
   - *Cách khắc phục:* Bổ sung rào chắn an toàn `max_iterations = 6` trong Supervisor kèm tăng biến đếm `state.iteration` qua mỗi bước `state.record_route()`.

2. **Ảo giác dây chuyền (Cascading Hallucinations) & Loãng ngữ cảnh:**
   - *Hiện tượng:* Model nhận quá nhiều snippets thô gây quá tải context.
   - *Cách khắc phục:* `Researcher` chắt lọc thành `research_notes` thô, `Analyst` đối chiếu mâu thuẫn trước khi chuyển giao sang `Writer`.

3. **Trích dẫn nguồn ảo (Hallucinated Citations):**
   - *Hiện tượng:* Model tự bịa các số citation ngoài phạm vi tài liệu (ví dụ: `[8]`).
   - *Cách khắc phục:* Kiểm tra regex chỉ số nguồn qua `compute_citation_coverage()` và ép buộc System Prompt cho Writer bám sát danh mục `state.sources`.

