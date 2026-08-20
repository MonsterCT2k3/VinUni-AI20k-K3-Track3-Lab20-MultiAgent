"""Unit tests for the deterministic CriticAgent validator."""

from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState

_LONG_BODY = " ".join(["word"] * 60)


def _state(final_answer: str, source_count: int = 5) -> ResearchState:
    return ResearchState(
        request=ResearchQuery(query="Test query validation"),
        sources=[SourceDocument(title=f"D{i}", snippet="S") for i in range(1, source_count + 1)],
        final_answer=final_answer,
    )


def test_critic_accepts_a_well_cited_answer() -> None:
    state = CriticAgent().run(_state(f"{_LONG_BODY} [1][2][3]"))

    assert state.errors == []
    assert state.agent_results[-1].agent == AgentName.CRITIC
    assert state.agent_results[-1].metadata["cited_indices"] == [1, 2, 3]


def test_critic_flags_hallucinated_citations() -> None:
    """Writer bịa [8][9] khi chỉ có 5 nguồn -> phải bị bắt."""
    state = CriticAgent().run(_state(f"{_LONG_BODY} [1] [8] [9]"))

    assert any("ngoài dải nguồn" in e for e in state.errors)
    assert state.agent_results[-1].metadata["cited_indices"] == [1]


def test_critic_flags_answer_with_sources_but_no_citations() -> None:
    state = CriticAgent().run(_state(_LONG_BODY))
    assert any("không trích dẫn nguồn nào" in e for e in state.errors)


def test_critic_flags_too_short_answer() -> None:
    state = CriticAgent().run(_state("Ngắn quá [1]"))
    assert any("quá ngắn" in e for e in state.errors)


def test_critic_flags_empty_answer() -> None:
    state = CriticAgent().run(_state(""))
    assert any("rỗng" in e for e in state.errors)


def test_critic_costs_nothing() -> None:
    """Critic là validator xác định, không được tốn thêm lệnh gọi LLM."""
    state = CriticAgent().run(_state(f"{_LONG_BODY} [1][2][3][4][5]"))
    assert state.agent_results[-1].metadata["cost_usd"] == 0.0
