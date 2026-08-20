"""Unit tests for individual worker agents."""

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse
from multi_agent_research_lab.services.search_client import SearchClient


class MockSearch(SearchClient):
    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        return [
            SourceDocument(title="Doc 1", url="https://example.com/1", snippet="Snippet 1"),
            SourceDocument(title="Doc 2", url="https://example.com/2", snippet="Snippet 2"),
        ]


class MockLLM(LLMClient):
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        return LLMResponse(content="Mock generated text [1]", input_tokens=50, output_tokens=20)


def test_researcher_agent_populates_sources_and_notes() -> None:
    state = ResearchState(request=ResearchQuery(query="Test query", max_sources=2))
    agent = ResearcherAgent(search_client=MockSearch(), llm_client=MockLLM())

    updated = agent.run(state)

    assert len(updated.sources) == 2
    assert updated.research_notes == "Mock generated text [1]"
    assert len(updated.agent_results) == 1
    assert updated.agent_results[0].agent == AgentName.RESEARCHER
    assert updated.trace[-1]["name"] == "researcher_completed"


def test_analyst_agent_populates_analysis_notes() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Test query"),
        sources=[SourceDocument(title="Doc 1", snippet="Snippet 1")],
        research_notes="Raw findings",
    )
    agent = AnalystAgent(llm_client=MockLLM())

    updated = agent.run(state)

    assert updated.analysis_notes == "Mock generated text [1]"
    assert len(updated.agent_results) == 1
    assert updated.agent_results[0].agent == AgentName.ANALYST
    assert updated.trace[-1]["name"] == "analyst_completed"


def test_writer_agent_populates_final_answer() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Test query"),
        sources=[SourceDocument(title="Doc 1", url="https://example.com/1", snippet="Snippet 1")],
        analysis_notes="Analyzed trade-offs",
    )
    agent = WriterAgent(llm_client=MockLLM())

    updated = agent.run(state)

    assert updated.final_answer == "Mock generated text [1]"
    assert len(updated.agent_results) == 1
    assert updated.agent_results[0].agent == AgentName.WRITER
    assert updated.trace[-1]["name"] == "writer_completed"


class _DeadLLM(LLMClient):
    """LLM luôn lỗi, mô phỏng provider chết sau khi đã retry hết."""

    def __init__(self) -> None:  # không gọi super(): tránh cần API key
        pass

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        raise RuntimeError("API down 503")


class _DeadSearch(SearchClient):
    def __init__(self) -> None:
        pass

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        raise RuntimeError("search backend unreachable")


def test_researcher_survives_search_failure() -> None:
    state = ResearchState(request=ResearchQuery(query="Test query validation"))
    agent = ResearcherAgent(search_client=_DeadSearch(), llm_client=MockLLM())

    updated = agent.run(state)

    assert updated.sources == []
    assert any("researcher.search" in e for e in updated.errors)


def test_researcher_falls_back_when_llm_fails() -> None:
    state = ResearchState(request=ResearchQuery(query="Test query validation"))
    agent = ResearcherAgent(search_client=MockSearch(), llm_client=_DeadLLM())

    updated = agent.run(state)

    assert updated.research_notes is not None
    assert any("researcher.llm" in e for e in updated.errors)


def test_analyst_falls_back_to_research_notes_when_llm_fails() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Test query validation"),
        research_notes="Ghi chép nghiên cứu thô",
    )
    updated = AnalystAgent(llm_client=_DeadLLM()).run(state)

    assert updated.analysis_notes is not None
    assert "Ghi chép nghiên cứu thô" in updated.analysis_notes
    assert any("analyst.llm" in e for e in updated.errors)


def test_writer_always_sets_final_answer_even_on_llm_failure() -> None:
    """Writer phải luôn đặt `final_answer`, nếu không Supervisor sẽ route lại mãi."""
    state = ResearchState(
        request=ResearchQuery(query="Test query validation"),
        sources=[SourceDocument(title="Doc 1", url="https://example.com/1", snippet="S1")],
        analysis_notes="Phân tích đã có",
    )
    updated = WriterAgent(llm_client=_DeadLLM()).run(state)

    assert updated.final_answer
    assert "Phân tích đã có" in updated.final_answer
    assert any("writer.llm" in e for e in updated.errors)
