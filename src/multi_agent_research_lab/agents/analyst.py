"""Analyst agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights and critical analysis."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Analyze research notes, extract key claims, compare trade-offs, and store analysis."""
        query = state.request.query
        research_notes = state.research_notes or "Không có research notes được ghi nhận."

        logger.info("AnalystAgent đang phân tích và đối chiếu luận điểm cho query: '%s'...", query)

        # 1. System Prompt đóng vai trò Chuyên gia phản biện (Critical Analyst)
        system_prompt = (
            "Bạn là một Analyst Agent chuyên sâu về tư duy phản biện, đối chiếu sự thật và "
            "tổng hợp luận điểm khoa học.\n"
            "Nhiệm vụ của bạn là phân tích bản Research Notes và danh sách tài liệu để tạo ra "
            "Analysis Notes có cấu trúc rõ ràng.\n\n"
            "Quy tắc bắt buộc:\n"
            "1. Rút trích các luận điểm cốt lõi (Key Findings / Claims).\n"
            "2. Chỉ ra điểm đồng thuận (Consensus) và so sánh ưu / nhược điểm, sự đánh đổi "
            "(Trade-offs, Contradictions) giữa các quan điểm.\n"
            "3. Đánh giá độ tin cậy của bằng chứng (Evidence Strength) và chỉ ra các giới hạn.\n"
            "4. Giữ nguyên tính khách quan và gắn nhãn nguồn tham khảo tương ứng."
        )

        user_prompt = (
            f"Câu hỏi nghiên cứu: {query}\n\n"
            f"Bản ghi chép nghiên cứu thô (Research Notes):\n{research_notes}\n\n"
            f"Số lượng tài liệu nguồn tham chiếu: {len(state.sources)}\n\n"
            "Hãy thực hiện phân tích chuyên sâu và tạo Analysis Notes:"
        )

        # 2. Gọi LLM sinh bản phân tích
        resp = self.llm_client.complete(system_prompt, user_prompt)
        state.analysis_notes = resp.content

        # 3. Ghi nhận AgentResult và Trace Event
        result_meta = {
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "cost_usd": resp.cost_usd,
        }
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=resp.content,
                metadata=result_meta,
            )
        )
        state.add_trace_event("analyst_completed", result_meta)

        return state
