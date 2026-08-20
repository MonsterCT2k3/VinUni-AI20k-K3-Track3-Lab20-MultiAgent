"""Writer agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes with rigorous citations."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Synthesize research notes and analysis into a polished Markdown report with citations."""
        query = state.request.query
        audience = state.request.audience or "Chuyên gia kỹ thuật và nhà nghiên cứu AI"
        analysis_content = (
            state.analysis_notes
            or state.research_notes
            or "Không có ghi chú phân tích được cung cấp."
        )

        logger.info("WriterAgent đang soạn thảo bài báo cáo hoàn chỉnh cho query: '%s'...", query)

        # 1. Chuẩn hóa danh sách tài liệu tham khảo nguồn
        sources_text_lines: list[str] = []
        for i, doc in enumerate(state.sources, 1):
            url_info = f" - URL: {doc.url}" if doc.url else ""
            sources_text_lines.append(f"[{i}] {doc.title}{url_info}\n    Trích đoạn: {doc.snippet}")
        sources_text = "\n\n".join(sources_text_lines)

        # 2. System Prompt chuyên biệt cho Writer
        system_prompt = (
            "Bạn là một Writer Agent chuyên nghiệp, cây bút khoa học kỹ thuật xuất sắc.\n"
            "Nhiệm vụ của bạn là tổng hợp toàn bộ thông tin từ Analysis Notes và tài liệu "
            "để viết một bài báo cáo nghiên cứu hoàn chỉnh, sâu sắc, có cấu trúc chuẩn mực.\n\n"
            "Quy tắc bắt buộc:\n"
            f"1. Phù hợp với đối tượng độc giả: {audience}.\n"
            "2. Cấu trúc bài viết rõ ràng: Tiêu đề báo cáo, Tóm tắt tổng quan, Phân tích kỹ thuật "
            "& So sánh đa chiều (Trade-offs), Kết luận & Khuyến nghị thực tế.\n"
            "3. KỶ LUẬT TRÍCH DẪN (Citation Discipline): Bắt buộc mọi luận điểm quan trọng "
            "phải gắn citation dạng [1], [2] tương ứng chính xác với danh sách tài liệu.\n"
            "4. Cuối bài bắt buộc có mục '### Tài Liệu Tham Khảo (References)' liệt kê chi tiết "
            "danh sách nguồn theo đúng số thứ tự [1], [2]..."
        )

        user_prompt = (
            f"Câu hỏi nghiên cứu: {query}\n\n"
            f"Bản phân tích chuyên sâu (Analysis Notes):\n{analysis_content}\n\n"
            f"Danh sách tài liệu tham khảo gốc:\n{sources_text}\n\n"
            "Hãy viết bài báo cáo nghiên cứu khoa học hoàn chỉnh bằng định dạng Markdown:"
        )

        # 3. Gọi LLM sinh bài báo cáo cuối cùng
        try:
            resp = self.llm_client.complete(system_prompt, user_prompt)
        except Exception as exc:
            # Writer BẮT BUỘC phải đặt `final_answer`, nếu không Supervisor sẽ route lại
            # writer mãi cho tới khi chạm `max_iterations`. Ghép nội dung tốt nhất đang có.
            logger.error("WriterAgent: gọi LLM thất bại: %s", exc)
            state.errors.append(f"writer.llm: {exc}")
            state.final_answer = (
                "Không tạo được báo cáo do lỗi LLM. Tổng hợp tạm thời từ dữ liệu đã thu thập:"
                f"\n\n{analysis_content}\n\n"
                f"### Tài Liệu Tham Khảo (References)\n{sources_text or 'Không có nguồn.'}"
            )
            return state

        state.final_answer = resp.content

        # 4. Ghi nhận AgentResult và Trace Event
        result_meta = {
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "cost_usd": resp.cost_usd,
            "audience": audience,
        }
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=resp.content,
                metadata=result_meta,
            )
        )
        state.add_trace_event("writer_completed", result_meta)

        return state
