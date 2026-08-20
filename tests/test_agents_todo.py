"""Unit tests for SupervisorAgent routing policy."""

from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_to_researcher_when_no_sources() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    supervisor = SupervisorAgent()
    updated = supervisor.run(state)
    assert updated.route_history[-1] == "researcher"
    assert updated.iteration == 1


def test_supervisor_routes_to_analyst_when_has_sources() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        sources=[SourceDocument(title="Doc 1", snippet="Snippet 1")],
    )
    supervisor = SupervisorAgent()
    updated = supervisor.run(state)
    assert updated.route_history[-1] == "analyst"


def test_supervisor_routes_to_writer_when_has_analysis() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        sources=[SourceDocument(title="Doc 1", snippet="Snippet 1")],
        analysis_notes="Key insights and trade-offs.",
    )
    supervisor = SupervisorAgent()
    updated = supervisor.run(state)
    assert updated.route_history[-1] == "writer"


def test_supervisor_routes_to_done_when_has_final_answer() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        sources=[SourceDocument(title="Doc 1", snippet="Snippet 1")],
        analysis_notes="Key insights.",
        final_answer="Final research report [1].",
    )
    supervisor = SupervisorAgent()
    updated = supervisor.run(state)
    assert updated.route_history[-1] == "done"


def test_supervisor_enforces_max_iterations_guardrail() -> None:
    settings = Settings(max_iterations=3)
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        iteration=3,
    )
    supervisor = SupervisorAgent(settings=settings)
    updated = supervisor.run(state)
    assert updated.route_history[-1] == "done"
    assert updated.final_answer is not None
    # Guardrail phải ghi cờ lỗi tường minh để benchmark đếm được đây là một lần chạy hỏng.
    assert any("guardrail_stop" in err for err in updated.errors)
