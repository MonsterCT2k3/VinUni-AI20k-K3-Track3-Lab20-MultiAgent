"""LangGraph workflow implementation for the Multi-Agent Research System."""

from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState


class MultiAgentWorkflow:
    """Builds, compiles, and runs the multi-agent graph with Supervisor orchestration."""

    def __init__(
        self,
        settings: Settings | None = None,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
        critic: CriticAgent | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.supervisor = supervisor or SupervisorAgent(settings=self.settings)
        self.researcher = researcher or ResearcherAgent()
        self.analyst = analyst or AnalystAgent()
        self.writer = writer or WriterAgent()
        self.critic = critic or CriticAgent()
        self._compiled_graph: CompiledStateGraph[Any, Any, Any, Any] | None = None

    def _supervisor_node(self, state: ResearchState) -> ResearchState:
        """Supervisor node: inspects state and decides next route."""
        return self.supervisor.run(state)

    def _researcher_node(self, state: ResearchState) -> ResearchState:
        """Researcher node: searches and collects source documents."""
        return self.researcher.run(state)

    def _analyst_node(self, state: ResearchState) -> ResearchState:
        """Analyst node: analyzes and structures research insights."""
        return self.analyst.run(state)

    def _writer_node(self, state: ResearchState) -> ResearchState:
        """Writer node: synthesizes final report with citations."""
        return self.writer.run(state)

    def _route_decision(self, state: ResearchState | dict[str, Any]) -> str:
        """Read last routing decision made by Supervisor."""
        if isinstance(state, ResearchState):
            history = state.route_history
        else:
            history = state.get("route_history", [])
        return history[-1] if history else "done"

    def build(self) -> CompiledStateGraph[Any, Any, Any, Any]:
        """Create and compile the LangGraph StateGraph workflow."""
        builder = StateGraph(ResearchState)

        # 1. Đăng ký các Nodes
        builder.add_node("supervisor", self._supervisor_node)
        builder.add_node("researcher", self._researcher_node)
        builder.add_node("analyst", self._analyst_node)
        builder.add_node("writer", self._writer_node)

        # 2. Thiết lập điểm bắt đầu
        builder.set_entry_point("supervisor")

        # 3. Thiết lập Conditional Edges từ Supervisor
        builder.add_conditional_edges(
            "supervisor",
            self._route_decision,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "done": END,
            },
        )

        # 4. Handoff Edges: Mọi Worker sau khi làm xong đều quay về Supervisor
        builder.add_edge("researcher", "supervisor")
        builder.add_edge("analyst", "supervisor")
        builder.add_edge("writer", "supervisor")

        self._compiled_graph = builder.compile()
        return self._compiled_graph

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the workflow graph and return the final state."""
        if self._compiled_graph is None:
            self._compiled_graph = self.build()

        result = self._compiled_graph.invoke(state)

        if isinstance(result, ResearchState):
            return result
        return ResearchState.model_validate(result)
