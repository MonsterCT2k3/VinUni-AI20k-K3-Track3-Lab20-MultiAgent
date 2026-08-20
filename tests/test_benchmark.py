"""Unit tests for benchmark metrics calculation."""

from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    BenchmarkMetrics,
    ResearchQuery,
    SourceDocument,
)
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    DEFAULT_BENCHMARK_QUERIES,
    aggregate_metrics,
    compute_citation_coverage,
    compute_estimated_cost,
    compute_quality_score,
    load_benchmark_queries,
    run_benchmark,
    run_benchmark_suite,
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
    # Bài viết dài, đủ header, và có từ vựng phân tích (so sánh / ưu điểm / trade-off).
    body = "Phân tích so sánh ưu điểm và nhược điểm cùng trade-off của hai kiến trúc. " * 30
    state = ResearchState(
        request=ResearchQuery(query="Test query validation"),
        final_answer=f"# Tiêu đề\n## Tổng quan\n## Phân tích\n{body}",
        analysis_notes="Some analysis notes",
    )
    score = compute_quality_score(state, citation_coverage=1.0)
    assert score >= 8.0


def test_quality_score_is_architecture_neutral() -> None:
    """Cùng một `final_answer` phải nhận cùng điểm, bất kể single- hay multi-agent.

    Guard cho lỗi cũ: điểm thưởng +1.5 dựa trên `analysis_notes`/`agent_results` khiến
    baseline không bao giờ đạt điểm tối đa, làm phép so sánh mất ý nghĩa.
    """
    sources = [SourceDocument(title=f"Doc {i}", snippet="S") for i in range(1, 6)]
    request = ResearchQuery(query="Test query validation")
    answer = "# A\n## B\n### C\n" + ("word " * 300) + "[1][2][3][4][5]"

    single_agent = ResearchState(request=request, sources=sources, final_answer=answer)
    multi_agent = ResearchState(
        request=request,
        sources=sources,
        final_answer=answer,
        analysis_notes="Phân tích chuyên sâu",
        agent_results=[
            AgentResult(agent=AgentName.RESEARCHER, content=""),
            AgentResult(agent=AgentName.ANALYST, content=""),
            AgentResult(agent=AgentName.WRITER, content=""),
        ],
    )

    assert compute_quality_score(single_agent, 1.0) == compute_quality_score(multi_agent, 1.0)


def test_quality_score_penalizes_recorded_errors() -> None:
    request = ResearchQuery(query="Test query validation")
    answer = "# A\n## B\n### C\n" + ("so sánh ưu điểm trade-off " * 60)

    clean = ResearchState(request=request, final_answer=answer)
    with_errors = ResearchState(request=request, final_answer=answer, errors=["e1", "e2"])

    assert compute_quality_score(with_errors, 1.0) == compute_quality_score(clean, 1.0) - 2.0


def test_quality_score_zero_for_guardrail_abort() -> None:
    """Run bị guardrail hủy (không nguồn, không phân tích) không được tính là chất lượng."""
    state = ResearchState(
        request=ResearchQuery(query="Test query validation"),
        final_answer="Quá trình nghiên cứu dừng lại do đạt giới hạn số lượt điều phối tối đa.",
        errors=["guardrail_stop: đạt max_iterations=6"],
    )
    assert compute_quality_score(state, 0.0) == 0.0


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


def test_failure_rate_flags_guardrail_abort() -> None:
    """Run bị guardrail hủy phải bị đếm là thất bại, dù `final_answer` dài > 50 ký tự.

    Guard cho lỗi cũ: `failure = len(final_answer) > 50` chấm thông báo lỗi 71 ký tự của
    guardrail là "thành công", nên một lần chạy không nghiên cứu gì báo failure_rate = 0%.
    """

    def aborted_runner(query: str) -> ResearchState:
        return ResearchState(
            request=ResearchQuery(query=query),
            final_answer=(
                "Quá trình nghiên cứu dừng lại do đạt giới hạn số lượt điều phối tối đa."
            ),
            errors=["guardrail_stop: đạt max_iterations=6"],
        )

    _, metrics = run_benchmark("aborted", "Test query validation", aborted_runner)
    assert metrics.failure_rate == 1.0
    assert metrics.notes is not None and "guardrail_stop" in metrics.notes


def test_failure_rate_zero_on_successful_run() -> None:
    def good_runner(query: str) -> ResearchState:
        return ResearchState(
            request=ResearchQuery(query=query),
            sources=[SourceDocument(title="D1", snippet="S1")],
            final_answer="# Báo cáo\n## So sánh\n" + ("Nội dung [1] đầy đủ " * 30),
        )

    _, metrics = run_benchmark("ok", "Test query validation", good_runner)
    assert metrics.failure_rate == 0.0


def test_load_benchmark_queries_reads_config(tmp_path) -> None:
    cfg = tmp_path / "lab.yaml"
    cfg.write_text(
        'benchmark:\n  queries:\n    - "Query one here"\n    - "Query two here"\n',
        encoding="utf-8",
    )
    assert load_benchmark_queries(cfg) == ["Query one here", "Query two here"]


def test_load_benchmark_queries_falls_back_when_config_missing(tmp_path) -> None:
    """Thiếu/hỏng config không được làm benchmark ngừng chạy."""
    assert load_benchmark_queries(tmp_path / "khong-ton-tai.yaml") == list(
        DEFAULT_BENCHMARK_QUERIES
    )


def test_aggregate_metrics_averages_across_runs() -> None:
    runs = [
        BenchmarkMetrics(
            run_name="r",
            latency_seconds=10.0,
            estimated_cost_usd=0.001,
            quality_score=8.0,
            citation_coverage=1.0,
            failure_rate=0.0,
        ),
        BenchmarkMetrics(
            run_name="r",
            latency_seconds=20.0,
            estimated_cost_usd=0.003,
            quality_score=6.0,
            citation_coverage=0.5,
            failure_rate=1.0,
        ),
    ]
    agg = aggregate_metrics("r", runs)
    assert agg.latency_seconds == 15.0
    assert agg.estimated_cost_usd == 0.002
    assert agg.quality_score == 7.0
    assert agg.citation_coverage == 0.75
    assert agg.failure_rate == 0.5
    assert agg.notes is not None and agg.notes.startswith("Trung bình 2 truy vấn")


def test_run_benchmark_suite_survives_a_failing_query() -> None:
    """Một truy vấn lỗi chỉ tính là 1 lần thất bại, không làm sập cả bộ benchmark."""

    def flaky(query: str) -> ResearchState:
        if "bad" in query:
            raise RuntimeError("runner exploded")
        return ResearchState(
            request=ResearchQuery(query=query),
            sources=[SourceDocument(title="D1", snippet="S1")],
            final_answer="# Báo cáo\n## So sánh\n" + ("Nội dung [1] đầy đủ " * 30),
        )

    agg = run_benchmark_suite("mixed", flaky, ["good query one", "bad query two"])
    assert agg.failure_rate == 0.5
    assert agg.notes is not None and agg.notes.startswith("Trung bình 2 truy vấn")


def test_aggregate_metrics_keeps_failure_reasons() -> None:
    """failure_rate khác 0 phải kèm lý do, nếu không report không giải thích được lỗi gì."""
    runs = [
        BenchmarkMetrics(run_name="m", latency_seconds=50.0, failure_rate=0.0, notes="Sources: 5"),
        BenchmarkMetrics(
            run_name="m",
            latency_seconds=55.0,
            failure_rate=1.0,
            notes="Sources: 5 | Lỗi: critic: trích dẫn ngoài dải nguồn",
        ),
    ]
    agg = aggregate_metrics("m", runs)
    assert agg.failure_rate == 0.5
    assert agg.notes is not None
    assert "1/2 truy vấn lỗi" in agg.notes
    assert "trích dẫn ngoài dải nguồn" in agg.notes
