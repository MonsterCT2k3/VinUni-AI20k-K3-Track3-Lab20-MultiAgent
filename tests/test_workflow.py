"""Unit tests for MultiAgentWorkflow and LangGraph orchestration."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


class DummyResearcher(BaseAgent):
    name = "researcher"

    def run(self, state: ResearchState) -> ResearchState:
        state.sources = [SourceDocument(title="Test Doc", snippet="Test Snippet")]
        state.research_notes = "Dummy research notes."
        return state


class DummyAnalyst(BaseAgent):
    name = "analyst"

    def run(self, state: ResearchState) -> ResearchState:
        state.analysis_notes = "Dummy analysis notes."
        return state


class DummyWriter(BaseAgent):
    name = "writer"

    def run(self, state: ResearchState) -> ResearchState:
        state.final_answer = "Dummy final report with citations [1]."
        return state


def test_workflow_build_compiles() -> None:
    workflow = MultiAgentWorkflow()
    compiled = workflow.build()
    assert compiled is not None


def test_workflow_runs_end_to_end_with_dummy_agents() -> None:
    settings = Settings(max_iterations=6)
    workflow = MultiAgentWorkflow(
        settings=settings,
        supervisor=SupervisorAgent(settings=settings),
        researcher=DummyResearcher(),  # type: ignore[arg-type]
        analyst=DummyAnalyst(),  # type: ignore[arg-type]
        writer=DummyWriter(),  # type: ignore[arg-type]
    )

    initial_state = ResearchState(request=ResearchQuery(query="Test query for workflow"))
    final_state = workflow.run(initial_state)

    assert final_state.final_answer == "Dummy final report with citations [1]."
    assert len(final_state.sources) == 1
    assert final_state.analysis_notes == "Dummy analysis notes."
    assert "researcher" in final_state.route_history
    assert "analyst" in final_state.route_history
    assert "writer" in final_state.route_history
    # Writer chảy thẳng qua Critic rồi tới END, nên không cần thêm một vòng Supervisor "done".
    assert final_state.route_history == ["researcher", "analyst", "writer"]
    assert [r.agent.value for r in final_state.agent_results][-1] == "critic"
