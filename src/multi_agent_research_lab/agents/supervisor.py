"""Supervisor / router implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        """Inspect current state, enforce guardrails, and determine the next route."""
        # 1. Guardrail: Chặn lặp vô hạn khi chạm max_iterations
        if state.iteration >= self.settings.max_iterations:
            logger.warning(
                "Đạt giới hạn vòng lặp tối đa (iteration=%d >= max_iterations=%d). Dừng workflow.",
                state.iteration,
                self.settings.max_iterations,
            )
            next_route = "done"
            # Ghi cờ lỗi tường minh: benchmark phải đếm đây là một lần chạy THẤT BẠI,
            # thay vì suy đoán qua độ dài `final_answer` (thông báo lỗi cũng dài > 50 ký tự).
            state.errors.append(
                f"guardrail_stop: đạt max_iterations={self.settings.max_iterations}"
            )
            if not state.final_answer:
                state.final_answer = (
                    "Quá trình nghiên cứu dừng lại do đạt giới hạn số lượt điều phối tối đa."
                )

        # 2. Bước 1: Nếu chưa có nguồn tài liệu -> route tới researcher
        elif not state.sources:
            next_route = "researcher"

        # 3. Bước 2: Nếu đã có nguồn nhưng chưa có phân tích -> route tới analyst
        elif not state.analysis_notes:
            next_route = "analyst"

        # 4. Bước 3: Nếu đã có phân tích nhưng chưa có bài viết cuối -> route tới writer
        elif not state.final_answer:
            next_route = "writer"

        # 5. Bước 4: Đã có final_answer -> kết thúc
        else:
            next_route = "done"

        # Ghi nhận lộ trình và tăng biến đếm iteration
        state.record_route(next_route)
        state.add_trace_event(
            "supervisor_route",
            {
                "next_route": next_route,
                "iteration": state.iteration,
                "has_sources": bool(state.sources),
                "has_analysis": bool(state.analysis_notes),
                "has_final_answer": bool(state.final_answer),
            },
        )
        return state
