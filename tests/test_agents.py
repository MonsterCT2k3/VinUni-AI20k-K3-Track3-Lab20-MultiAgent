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
