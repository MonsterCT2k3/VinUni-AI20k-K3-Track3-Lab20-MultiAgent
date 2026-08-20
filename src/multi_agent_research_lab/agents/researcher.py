"""Researcher agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        search_client: SearchClient | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.search_client = search_client or SearchClient()
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Search for relevant documents, store sources, and summarize research notes."""
        query = state.request.query
        max_sources = state.request.max_sources

        logger.info("ResearcherAgent đang tìm kiếm tài liệu cho query: '%s'...", query)

        # 1. Thu thập tài liệu nguồn
        try:
            sources = self.search_client.search(query=query, max_results=max_sources)
        except Exception as exc:
            # Search hỏng không được làm sập cả workflow: ghi lỗi rồi chạy tiếp với 0 nguồn.
            logger.error("ResearcherAgent: search thất bại: %s", exc)
            state.errors.append(f"researcher.search: {exc}")
            sources = []
        state.sources = sources

        # 2. Xây dựng nội dung tài liệu để tóm tắt
        sources_text_lines: list[str] = []
        for i, doc in enumerate(sources, 1):
            sources_text_lines.append(
                f"[Nguồn {i}] Tiêu đề: {doc.title}\n"
                f"    URL: {doc.url or 'N/A'}\n"
                f"    Trích đoạn: {doc.snippet}"
            )
        sources_text = "\n\n".join(sources_text_lines)

        # 3. Prompting LLM để trích xuất Research Notes thô
        system_prompt = (
            "Bạn là một Researcher Agent chuyên trách thu thập và cấu trúc hóa dữ liệu nghiên cứu "
            "ban đầu.\n"
            "Nhiệm vụ của bạn là đọc kỹ các tài liệu nguồn và trích xuất ra bản Research Notes "
            "khách quan, có cấu trúc rõ ràng.\n\n"
            "Quy tắc bắt buộc:\n"
            "1. Liệt kê các dữ kiện quan trọng (facts), định nghĩa, số liệu thực nghiệm.\n"
            "2. Ghi rõ nguồn gốc trích dẫn theo từng [Nguồn 1], [Nguồn 2]...\n"
            "3. Không đưa ra kết luận chủ quan hay suy diễn; giữ nguyên dữ kiện để chuyển cho "
            "Analyst phân tích tiếp theo."
        )
        user_prompt = (
            f"Câu hỏi nghiên cứu: {query}\n\n"
            f"Danh sách tài liệu đã thu thập được:\n{sources_text}\n\n"
            "Hãy tổng hợp Research Notes:"
        )

        try:
            resp = self.llm_client.complete(system_prompt, user_prompt)
        except Exception as exc:
            # Fallback: ghép thẳng trích đoạn nguồn thành notes thô để Analyst vẫn có đầu vào.
            logger.error("ResearcherAgent: gọi LLM thất bại: %s", exc)
            state.errors.append(f"researcher.llm: {exc}")
            state.research_notes = (
                "Không tóm tắt được bằng LLM. Trích đoạn nguồn thô:\n\n" + sources_text
                if sources
                else "Không thu thập được tài liệu nguồn nào."
            )
            return state

        state.research_notes = resp.content

        # 4. Ghi nhận AgentResult và Trace Event
        result_meta = {
            "source_count": len(sources),
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "cost_usd": resp.cost_usd,
        }
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=resp.content,
                metadata=result_meta,
            )
        )
        state.add_trace_event("researcher_completed", result_meta)

        return state
