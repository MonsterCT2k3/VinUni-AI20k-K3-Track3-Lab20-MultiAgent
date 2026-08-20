"""Critic agent: kiểm định xác định (deterministic) chất lượng bài viết cuối."""

import logging
import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

CITATION_PATTERN = re.compile(r"\[(?:Source\s*|Nguồn\s*)?(\d+)\]", re.IGNORECASE)

MIN_ANSWER_WORDS = 50


class CriticAgent(BaseAgent):
    """Validator xác định cho `final_answer` — **không gọi LLM**.

    Cố ý không dùng LLM: mọi thứ agent này kiểm tra (chỉ số trích dẫn nằm ngoài danh mục
    nguồn, bài viết rỗng/quá ngắn) đều là kiểm tra xác định. Dùng regex vừa miễn phí, vừa
    chạy trong mili-giây, và **đáng tin hơn** một LLM tự chấm điểm chính mình.

    Phát hiện được ghi vào `state.errors` nên chúng tự động phản ánh vào `failure_rate` và
    điểm phạt trong `compute_quality_score`.
    """

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Kiểm tra tính toàn vẹn của `final_answer` và ghi nhận các phát hiện."""
        findings: list[str] = []
        answer = state.final_answer or ""

        # 1. Bài viết phải tồn tại và đủ dài để coi là một câu trả lời thực chất.
        if not answer.strip():
            findings.append("final_answer rỗng")
        elif len(answer.split()) < MIN_ANSWER_WORDS:
            findings.append(f"final_answer quá ngắn (<{MIN_ANSWER_WORDS} từ)")

        # 2. Chống trích dẫn ảo: mọi [n] phải nằm trong danh mục nguồn đã thu thập.
        cited = {int(m) for m in CITATION_PATTERN.findall(answer)}
        max_index = len(state.sources)
        out_of_range = sorted(i for i in cited if i < 1 or i > max_index)
        if out_of_range:
            findings.append(f"trích dẫn ngoài dải nguồn (chỉ có {max_index} nguồn): {out_of_range}")
        if state.sources and not cited:
            findings.append("có nguồn tham khảo nhưng bài viết không trích dẫn nguồn nào")

        valid_cited = cited - set(out_of_range)
        coverage = len(valid_cited) / max_index if max_index else 0.0

        result_meta = {
            "findings": findings,
            "cited_indices": sorted(valid_cited),
            "citation_coverage": round(coverage, 3),
            "cost_usd": 0.0,  # validator xác định, không tốn token
        }
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content="; ".join(findings) if findings else "Không phát hiện vấn đề.",
                metadata=result_meta,
            )
        )
        state.add_trace_event("critic_completed", result_meta)

        if findings:
            logger.warning("CriticAgent phát hiện %d vấn đề: %s", len(findings), findings)
            state.errors.extend(f"critic: {f}" for f in findings)

        return state
