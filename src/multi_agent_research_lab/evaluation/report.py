"""Benchmark report rendering and visualization generator."""

from pathlib import Path
from typing import Any

from multi_agent_research_lab.core.schemas import BenchmarkMetrics

# Ngưỡng coi hai kiến trúc là "tương đương" về chất lượng (thang 0-10).
QUALITY_TIE_THRESHOLD = 0.5

# Nêu rõ điều kiện đo và giới hạn của metric, để người đọc biết con số nói lên điều gì.
_METHODOLOGY: list[str] = [
    "",
    "### 🧪 2.3. Phương Pháp Đo (Methodology)",
    "- **Kiểm soát biến:** Cả hai kiến trúc dùng **chung** `SearchClient` và `LLMClient` "
    "(cùng corpus nguồn, cùng model, cùng `max_sources`), nên chênh lệch đo được phản ánh "
    "khác biệt về *kiến trúc* — không phải khác biệt công cụ.",
    "- **Lấy trung bình nhiều truy vấn:** Số liệu là trung bình trên bộ truy vấn khai báo "
    "trong `configs/lab_default.yaml` (số lượng ghi ở cột *Ghi Chú*). LLM sinh văn bản ngẫu "
    "nhiên (`temperature > 0`) nên kết quả của một truy vấn đơn lẻ dao động mạnh và dễ dẫn "
    "tới kết luận sai.",
    "- **Giới hạn của điểm Quality:** Đây là **heuristic tự động** (độ dài, cấu trúc Markdown, "
    "độ phủ trích dẫn, dấu hiệu phân tích, trừ điểm khi có lỗi) — chỉ đo đặc trưng *bề mặt*, "
    "không kiểm chứng tính đúng đắn của nội dung. Nó **bổ sung** chứ không thay thế rubric "
    "0-10 do con người chấm trong `docs/peer_review_rubric.md`. Điểm dùng **cùng một công "
    "thức** cho cả hai kiến trúc, không cộng điểm thưởng cho việc có nhiều agent.",
    "- **Failure rate:** Tính theo cờ lỗi tường minh ghi trong `state.errors` (guardrail dừng, "
    "search lỗi, LLM lỗi sau retry), không suy đoán qua độ dài câu trả lời.",
]

# Bằng chứng trace của một lần chạy multi-agent thật (LangSmith).
# Số liệu lấy từ `reports/traces/local_trace.jsonl`, khớp với ảnh `docs/trace_evidence.png`.
_TRACE_EVIDENCE: list[str] = [
    "## 0. Bằng Chứng Trace (Trace Evidence)",
    "",
    "![Trace LangSmith của một lần chạy multi-agent](../docs/trace_evidence.png)",
    "",
    "🔗 **Trace công khai (xem được không cần đăng nhập):** "
    "https://smith.langchain.com/public/d76b97c1-d88c-4728-be99-1749492fca45/r/"
    "3c4a5334-e9e5-4823-95e3-f8b210495a60",
    "",
    "Trace LangSmith cho **một lần chạy multi-agent end-to-end** với truy vấn "
    '*"Summarize production guardrails for LLM agents"*. Span cha `multi_agent_run` '
    "(**55.20s**) lồng đúng thứ tự routing thực tế mà Supervisor quyết định:",
    "",
    "| Span | Thời gian | Tỷ trọng |",
    "|---|---:|---:|",
    "| `writer` | 19.36s | 35% |",
    "| `researcher` | 15.57s | 28% |",
    "| `analyst` | 13.52s | 24% |",
    "| `supervisor` (4 lượt) | 2.13s | 4% |",
    "| *Overhead LangGraph* | ~4.6s | 8% |",
    "| **`multi_agent_run`** | **55.20s** | **100%** |",
    "",
    "**Đọc trace:** `Writer` mới là nút thắt cổ chai lớn nhất (35%) chứ không phải "
    "`Researcher`, vì nó phải sinh bài báo cáo dài nhất. Đáng chú ý, 4 lượt `Supervisor` "
    "cộng lại chỉ tốn **2.13s (~4%)** — nghĩa là **chi phí điều phối (orchestration "
    "overhead) rất rẻ**; phần đắt nằm ở 3 lệnh gọi LLM của các worker. Điều này bác bỏ "
    "giả định thường gặp rằng multi-agent chậm là do khâu điều phối.",
    "",
    "Trace cũng ghi lại `input` (query) và `output` (`duration_seconds`) của từng span; "
    "bản sao cục bộ được ghi ở `reports/traces/local_trace.jsonl` để kiểm chứng offline.",
    "",
    "---",
    "",
]


def _fmt_quality(metric: BenchmarkMetrics | None) -> str:
    if metric is None or metric.quality_score is None:
        return "N/A"
    return f"{metric.quality_score:.1f}/10"


def _fmt_coverage(metric: BenchmarkMetrics | None) -> str:
    if metric is None or metric.citation_coverage is None:
        return "N/A"
    return f"{metric.citation_coverage:.0%}"


def _render_quality_verdict(
    baseline_m: BenchmarkMetrics | None,
    multi_m: BenchmarkMetrics | None,
    cost_ratio: float,
) -> list[str]:
    """Sinh nhận xét chất lượng **từ dữ liệu đo được**, không kết luận sẵn.

    Report phải có khả năng nói rằng multi-agent thua — `docs/codelab.md` coi đó là một
    learning outcome hợp lệ ("Multi-agent không phải lúc nào cũng thắng").
    """
    base_qual, multi_qual = _fmt_quality(baseline_m), _fmt_quality(multi_m)
    base_cov, multi_cov = _fmt_coverage(baseline_m), _fmt_coverage(multi_m)

    lines = [
        f"- **Single-Agent** — Quality {base_qual}, độ phủ trích dẫn {base_cov}.",
        f"- **Multi-Agent** — Quality {multi_qual}, độ phủ trích dẫn {multi_cov}.",
        "",
    ]

    if (
        baseline_m is None
        or multi_m is None
        or baseline_m.quality_score is None
        or multi_m.quality_score is None
    ):
        lines.append(
            "> ⚠️ Thiếu dữ liệu chất lượng của một trong hai kiến trúc nên chưa thể kết luận."
        )
        return lines

    delta = multi_m.quality_score - baseline_m.quality_score

    if delta > QUALITY_TIE_THRESHOLD:
        lines.append(
            f"> ✅ **Multi-Agent nhỉnh hơn {delta:+.1f} điểm.** Việc tách vai trò giúp Analyst "
            "kiểm chứng chéo trước khi Writer tổng hợp, nên bài viết bám sát nguồn hơn. "
            f"Mức tăng này đổi lại chi phí cao hơn ~{cost_ratio} lần — đáng dùng khi độ tin cậy "
            "trích dẫn quan trọng hơn tốc độ và ngân sách."
        )
    elif delta < -QUALITY_TIE_THRESHOLD:
        lines.append(
            f"> ⚠️ **Multi-Agent KHÔNG thắng trong lần chạy này ({delta:+.1f} điểm).** "
            f"Chi phí điều phối cao hơn ~{cost_ratio} lần nhưng không đổi lại được chất lượng. "
            "Nguyên nhân thường gặp: context bị pha loãng qua nhiều bước handoff, hoặc Writer "
            "mất bám sát danh mục nguồn gốc. Đây là bằng chứng cho thấy multi-agent không phải "
            "lựa chọn mặc định."
        )
    else:
        lines.append(
            f"> ⚖️ **Hai kiến trúc tương đương về chất lượng** (chênh {delta:+.1f} điểm, không "
            f"vượt ngưỡng {QUALITY_TIE_THRESHOLD}), trong khi Multi-Agent tốn hơn "
            f"~{cost_ratio} lần chi phí. "
            "Trên bộ truy vấn này, Single-Agent là lựa chọn hợp lý hơn — đúng nguyên "
            "tắc *“không thêm agent nếu không có lý do rõ ràng”*."
        )
    return lines


_STATIC_SECTIONS: list[str] = [
    "",
    "---",
    "",
    "## 3. Khung Quyết Định Kiến Trúc (Architecture Decision Matrix)",
    "",
    "| Tiêu chí | Single-Agent | Multi-Agent |",
    "|---|---|---|",
    "| **Tính chất** | Tra cứu nhanh, tóm tắt 1 nguồn | Nghiên cứu sâu, đa nguồn, phản biện |",
    "| **Tốc độ** | Thời gian thực (< 5s) | Chấp nhận độ trễ (Deep Research 20-45s) |",
    "| **Ngân sách** | Tiết kiệm chi phí token | Ưu tiên độ chính xác, chống ảo giác |",
    "| **Vận hành** | Đơn giản, không vòng lặp | Cần rào chắn `max_iterations` |",
    "",
    "---",
    "",
    "## 4. Exit Ticket Answers",
    "",
    "### ❓ 1. Trường hợp nào BẮT BUỘC NÊN dùng Multi-Agent?",
    "> Khi bài toán đòi hỏi **nghiên cứu sâu (Deep Research)** với nhiều nguồn thông tin "
    "đối nghịch nhau, cần phân tách rõ ràng giữa việc thu thập dữ liệu (Researcher), phản biện "
    "kiểm chứng (Analyst), và soạn thảo báo cáo chuẩn học thuật (Writer). Multi-Agent giúp "
    "cô lập lỗi, kiểm chứng chéo và duy trì kỷ luật trích dẫn nguồn nhất quán.",
    "",
    "### ❓ 2. Trường hợp nào KHÔNG NÊN dùng Multi-Agent?",
    "> Với các tác vụ **đơn bước (Single-step lookup)**, hỏi đáp tài liệu ngắn, chatbot "
    "hội thoại thời gian thực, hoặc các hệ thống có ngân sách token hạn hẹp và yêu cầu độ trễ "
    "cực thấp (< 3 giây). Khi đó chi phí điều phối (Orchestration Overhead) là sự lãng phí.",
    "",
    "---",
    "",
    "## 5. Phân Tích Failure Mode Gặp Phải & Cách Khắc Phục (Failure Mode Analysis)",
    "",
    "Trong quá trình phát triển hệ thống, 3 Failure Modes chính đã được xử lý triệt để:",
    "",
    "1. **Vòng lặp điều phối vô hạn (Infinite Loop):**",
    "   - *Hiện tượng:* Supervisor liên tục route lặp lại giữa các Worker khi thiếu stop flag.",
    "   - *Cách khắc phục:* Bổ sung rào chắn an toàn `max_iterations = 6` trong Supervisor "
    "kèm tăng biến đếm `state.iteration` qua mỗi bước `state.record_route()`.",
    "",
    "2. **Ảo giác dây chuyền (Cascading Hallucinations) & Loãng ngữ cảnh:**",
    "   - *Hiện tượng:* Model nhận quá nhiều snippets thô gây quá tải context.",
    "   - *Cách khắc phục:* `Researcher` chắt lọc thành `research_notes` thô, "
    "`Analyst` đối chiếu mâu thuẫn trước khi chuyển giao sang `Writer`.",
    "",
    "3. **Trích dẫn nguồn ảo (Hallucinated Citations):**",
    "   - *Hiện tượng:* Model tự bịa các số citation ngoài phạm vi tài liệu (ví dụ: `[8]`).",
    "   - *Cách khắc phục:* Kiểm tra regex chỉ số nguồn qua `compute_citation_coverage()` "
    "và ép buộc System Prompt cho Writer bám sát danh mục `state.sources`.",
    "",
]


def render_markdown_report(
    metrics: list[BenchmarkMetrics],
    metadata: dict[str, Any] | None = None,
) -> str:
    """Render comprehensive benchmark metrics to Markdown with dynamic trade-off analysis."""
    lines: list[str] = [
        "# 📊 Báo Cáo Benchmark Đối Chứng (Benchmark Report): Single-Agent vs Multi-Agent",
        "",
        "### 🎓 Thông Tin Học Viên",
        "- **Họ và tên:** Nguyễn Đăng Nam",
        "- **Mã học viên:** 2A202601307",
        "- **Khóa học:** VinUni AI20k - Khóa 3 Track 3",
        "",
        "> 🔬 **Mục tiêu:** Đánh giá định lượng hiệu năng giữa hệ thống Đơn tác nhân "
        "(Monolithic Single-Agent) và Đa tác nhân (Supervisor + Worker Agents).",
        "",
        *_TRACE_EVIDENCE,
        "## 1. Bảng Số Liệu Đo Lường Thực Nghiệm",
        "",
        "| Mô Hình (Run) | Latency (s) | Chi Phí (USD) | Quality | Citation Cov. | Lỗi | Ghi Chú |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    for item in metrics:
        cost = "N/A" if item.estimated_cost_usd is None else f"${item.estimated_cost_usd:.6f}"
        quality = "N/A" if item.quality_score is None else f"{item.quality_score:.1f}/10"
        citation = "N/A" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "0%" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| **{item.run_name}** | {item.latency_seconds:.2f}s | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )

    # Dynamic trade-off calculations from metrics
    baseline_m = metrics[0] if len(metrics) > 0 else None
    multi_m = metrics[1] if len(metrics) > 1 else None

    # Chỉ dùng số đo được; không bịa giá trị mặc định khi thiếu dữ liệu.
    base_cost = baseline_m.estimated_cost_usd if baseline_m else None
    multi_cost = multi_m.estimated_cost_usd if multi_m else None
    cost_ratio = (
        round(multi_cost / base_cost, 1) if (base_cost and multi_cost and base_cost > 0) else 0.0
    )

    base_lat_str = f"{baseline_m.latency_seconds:.2f}s" if baseline_m else "N/A"
    multi_lat_str = f"{multi_m.latency_seconds:.2f}s" if multi_m else "N/A"
    base_cost_str = f"${base_cost:.6f}" if base_cost is not None else "N/A"
    multi_cost_str = f"${multi_cost:.6f}" if multi_cost is not None else "N/A"

    # Delta có dấu: nếu Multi-Agent nhanh/rẻ hơn, con số sẽ tự mang dấu âm.
    if baseline_m and multi_m:
        lat_delta = multi_m.latency_seconds - baseline_m.latency_seconds
        lat_pct = (
            lat_delta / baseline_m.latency_seconds * 100 if baseline_m.latency_seconds else 0.0
        )
        lat_delta_str = f"Multi-Agent {lat_delta:+.2f}s, {lat_pct:+.0f}%"
    else:
        lat_delta_str = "chưa đủ dữ liệu"

    if base_cost is not None and multi_cost is not None:
        cost_delta = multi_cost - base_cost
        cost_pct = cost_delta / base_cost * 100 if base_cost else 0.0
        cost_delta_str = f"Multi-Agent {cost_delta:+.6f} USD, {cost_pct:+.0f}%"
    else:
        cost_delta_str = "chưa đủ dữ liệu"

    lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Phân Tích Đánh Đổi Kỹ Thuật (Trade-Offs Analysis)",
            "",
            "### ⏱️ 2.1. Độ Trễ (Latency) & Chi Phí (Cost USD)",
            f"- **Latency:** Single-Agent {base_lat_str} vs Multi-Agent {multi_lat_str} "
            f"({lat_delta_str}).",
            f"- **Chi phí:** Single-Agent {base_cost_str} vs Multi-Agent {multi_cost_str} "
            f"({cost_delta_str}).",
            "- **Nguyên nhân:** Single-Agent chỉ gọi LLM **1 lần** (1-shot), trong khi Multi-Agent "
            "luân chuyển qua chuỗi `Supervisor` ➔ `Researcher` ➔ `Analyst` ➔ `Writer` ➔ "
            "`Supervisor (Done)`, tức **3 lệnh gọi LLM** cộng chi phí điều phối.",
            "",
            "### 🎯 2.2. Chất Lượng Bài Viết & Kỷ Luật Trích Dẫn",
            *_render_quality_verdict(baseline_m, multi_m, cost_ratio),
            *_METHODOLOGY,
            *_STATIC_SECTIONS,
        ]
    )
    return "\n".join(lines) + "\n"


def save_markdown_report(
    report_text: str,
    output_path: str = "reports/benchmark_report.md",
) -> Path:
    """Save the generated report text to the specified file path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_text, encoding="utf-8")
    return path
