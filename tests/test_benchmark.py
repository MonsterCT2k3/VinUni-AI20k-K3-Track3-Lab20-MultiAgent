"""Unit tests for benchmark metrics calculation."""

from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    ResearchQuery,
    SourceDocument,
)
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    compute_citation_coverage,
    compute_estimated_cost,
    compute_quality_score,
    run_benchmark,
)


def test_compute_citation_coverage_numeric() -> None:
    sources = [
        SourceDocument(title="Doc 1", snippet="Snippet 1"),
        SourceDocument(title="Doc 2", snippet="Snippet 2"),
        SourceDocument(title="Doc 3", snippet="Snippet 3"),
    ]
    report = "Theo nghiên cứu [1] và [3], hệ thống đa tác nhân hoạt động hiệu quả."
    coverage = compute_citation_coverage(report, sources)
    # Cited 2 out of 3 -> ~0.667
    assert coverage == 0.667


def test_compute_citation_coverage_prefixed() -> None:
    sources = [
        SourceDocument(title="Doc 1", snippet="Snippet 1"),
        SourceDocument(title="Doc 2", snippet="Snippet 2"),
    ]
    report = "Dữ liệu từ [Nguồn 1] và [Source 2] xác nhận kết quả."
    coverage = compute_citation_coverage(report, sources)
    assert coverage == 1.0


def test_compute_citation_coverage_empty() -> None:
    assert compute_citation_coverage(None, []) == 0.0
    assert compute_citation_coverage("Some text", []) == 0.0


def test_compute_estimated_cost_from_agent_results() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Test query validation"),
        agent_results=[
            AgentResult(agent=AgentName.RESEARCHER, content="", metadata={"cost_usd": 0.0005}),
            AgentResult(agent=AgentName.ANALYST, content="", metadata={"cost_usd": 0.0008}),
            AgentResult(agent=AgentName.WRITER, content="", metadata={"cost_usd": 0.0012}),
        ],
    )
    total_cost = compute_estimated_cost(state)
    assert total_cost == 0.0025


def test_compute_quality_score() -> None:
    body = "Nội dung bài viết dài hơn 250 từ. " * 30
    state = ResearchState(
        request=ResearchQuery(query="Test query validation"),
        final_answer=f"# Tiêu đề\n## Tổng quan\n## Phân tích\n{body}",
        analysis_notes="Some analysis notes",
    )
    score = compute_quality_score(state, citation_coverage=1.0)
    assert score >= 8.0


def test_run_benchmark() -> None:
    def mock_runner(query: str) -> ResearchState:
        body = "Nội dung [1] đầy đủ " * 30
        return ResearchState(
            request=ResearchQuery(query=query),
            sources=[SourceDocument(title="D1", snippet="S1")],
            final_answer=f"# Báo cáo\n## So sánh\n{body}",
            agent_results=[
                AgentResult(agent=AgentName.WRITER, content="", metadata={"cost_usd": 0.001})
            ],
        )

    state, metrics = run_benchmark("mock_run", "Test query validation", mock_runner)
    assert metrics.run_name == "mock_run"
    assert metrics.citation_coverage == 1.0
    assert metrics.estimated_cost_usd == 0.001
    assert metrics.quality_score is not None and metrics.quality_score > 0
    assert metrics.failure_rate == 0.0
