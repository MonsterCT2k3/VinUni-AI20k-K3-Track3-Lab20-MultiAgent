"""Critic agent implementation for fact-checking and hallucination review."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety/citation review agent."""

    name = "critic"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer, check citation fidelity, and append evaluation notes."""
        if not state.final_answer:
            return state

        query = state.request.query
        logger.info("CriticAgent đang thẩm định bài viết cuối cùng cho query: '%s'...", query)

        system_prompt = (
            "Bạn là một Critic Agent chuyên trách thẩm định chất lượng khoa học và tính xác thực.\n"
            "Nhiệm vụ: Đánh giá bài báo cáo cuối cùng đối chiếu với tài liệu nguồn:\n"
            "1. Kiểm tra xem có ảo tưởng thông tin (Hallucination) không.\n"
            "2. Kiểm tra độ chuẩn xác của các trích dẫn [1], [2]...\n"
            "3. Chấm điểm độ tin cậy từ 1 đến 10 và nhận xét ngắn gọn."
        )
        user_prompt = (
            f"Câu hỏi: {query}\n\n"
            f"Bài báo cáo:\n{state.final_answer}\n\n"
            f"Số lượng nguồn tham khảo: {len(state.sources)}\n\n"
            "Hãy đưa ra đánh giá thẩm định:"
        )

        resp = self.llm_client.complete(system_prompt, user_prompt)

        result_meta = {
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "cost_usd": resp.cost_usd,
        }
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=resp.content,
                metadata=result_meta,
            )
        )
        state.add_trace_event("critic_completed", result_meta)
        return state
