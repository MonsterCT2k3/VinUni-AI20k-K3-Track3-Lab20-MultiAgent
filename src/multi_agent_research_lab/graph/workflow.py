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
from multi_agent_research_lab.observability.tracing import trace_span


class MultiAgentWorkflow:
    """Builds, compiles, and runs the multi-agent graph with Supervisor orchestration.

    Hình dạng đồ thị::

        supervisor --route--> researcher --> supervisor
                          \\--> analyst    --> supervisor
                          \\--> writer     --> critic --> END
                          \\--> done (END)

    Supervisor là trung tâm định tuyến: mọi worker giao quyền lại cho nó, và nó quyết định
    bước kế tiếp bằng cách soi `ResearchState`. Điều kiện dừng được bảo đảm ở **hai tầng**:
    Supervisor trả về "done" khi đã có `final_answer` hoặc chạm `max_iterations`, và về mặt
    cấu trúc `writer` luôn chảy qua `critic` rồi tới END nên đồ thị không thể quay lại vòng
    lặp sau khi đã có câu trả lời.
    """

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

    def _node(self, agent: Any, span_name: str) -> Any:
        """Bọc một agent thành node LangGraph, đo thời gian chạy qua `trace_span`.

        Nhờ vậy trace trả lời được câu hỏi "agent nào tốn bao nhiêu" ở mức từng bước, chứ
        không chỉ tổng thời gian cả run.
        """

        def _run(state: ResearchState) -> ResearchState:
            with trace_span(span_name, {"iteration": state.iteration}) as span:
                result: ResearchState = agent.run(state)
            result.add_trace_event(
                f"{span_name}_span", {"duration_seconds": span["duration_seconds"]}
            )
            return result

        return _run

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
        builder.add_node("supervisor", self._node(self.supervisor, "supervisor"))
        builder.add_node("researcher", self._node(self.researcher, "researcher"))
        builder.add_node("analyst", self._node(self.analyst, "analyst"))
        builder.add_node("writer", self._node(self.writer, "writer"))
        builder.add_node("critic", self._node(self.critic, "critic"))

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

        # 4. Handoff Edges: Researcher/Analyst xong thì quay về Supervisor để định tuyến tiếp
        builder.add_edge("researcher", "supervisor")
        builder.add_edge("analyst", "supervisor")

        # 5. Writer luôn đi qua Critic để thẩm định trước khi kết thúc.
        #    Đây cũng là stop condition mang tính cấu trúc: một khi đã có `final_answer`,
        #    đồ thị không thể quay lại vòng lặp điều phối nữa.
        builder.add_edge("writer", "critic")
        builder.add_edge("critic", END)

        self._compiled_graph = builder.compile()
        return self._compiled_graph

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the workflow graph and return the final state.

        Toàn bộ lần chạy được bọc trong một span cha `multi_agent_run`, nên mọi span của
        từng node lồng bên dưới thành một cây trace duy nhất thay vì các span rời rạc.
        """
        if self._compiled_graph is None:
            self._compiled_graph = self.build()

        with trace_span("multi_agent_run", {"query": state.request.query}):
            result = self._compiled_graph.invoke(state)

        if isinstance(result, ResearchState):
            return result
        return ResearchState.model_validate(result)
