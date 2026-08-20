from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.evaluation.report import render_markdown_report


def _baseline(quality: float = 9.5, coverage: float = 1.0) -> BenchmarkMetrics:
    return BenchmarkMetrics(
        run_name="Single-Agent Baseline",
        latency_seconds=11.56,
        estimated_cost_usd=0.000620,
        quality_score=quality,
        citation_coverage=coverage,
        failure_rate=0.0,
    )


def _multi(quality: float, coverage: float) -> BenchmarkMetrics:
    return BenchmarkMetrics(
        run_name="Multi-Agent System",
        latency_seconds=42.77,
        estimated_cost_usd=0.002238,
        quality_score=quality,
        citation_coverage=coverage,
        failure_rate=0.0,
    )


def test_report_renders_markdown() -> None:
    report = render_markdown_report([BenchmarkMetrics(run_name="baseline", latency_seconds=1.23)])
    assert "Benchmark Report" in report
    assert "baseline" in report


def test_report_admits_when_multi_agent_loses() -> None:
    """Report phải nói được rằng multi-agent thua, thay vì luôn khen 'đạt điểm tối đa'."""
    report = render_markdown_report([_baseline(), _multi(quality=6.0, coverage=0.4)])

    assert "KHÔNG thắng" in report
    assert "-3.5 điểm" in report
    # Không được khẳng định thắng lợi khi dữ liệu nói ngược lại.
    assert "Đạt điểm tối đa" not in report
    assert "tuân thủ 100% kỷ luật trích dẫn" not in report


def test_report_calls_a_tie_a_tie() -> None:
    """Chênh lệch dưới ngưỡng phải được gọi là tương đương, không phải chiến thắng."""
    report = render_markdown_report([_baseline(), _multi(quality=10.0, coverage=1.0)])

    assert "tương đương về chất lượng" in report
    assert "KHÔNG thắng" not in report


def test_report_credits_multi_agent_when_it_actually_wins() -> None:
    report = render_markdown_report(
        [_baseline(quality=7.0, coverage=0.6), _multi(quality=10.0, coverage=1.0)]
    )

    assert "Multi-Agent nhỉnh hơn +3.0 điểm" in report
    assert "KHÔNG thắng" not in report


def test_report_reports_signed_deltas_not_fabricated_defaults() -> None:
    """Số liệu phải lấy từ metrics thật; thiếu dữ liệu thì ghi N/A chứ không bịa."""
    report = render_markdown_report([_baseline(), _multi(quality=10.0, coverage=1.0)])
    assert "+31.21s" in report
    assert "+270%" in report

    sparse = render_markdown_report([BenchmarkMetrics(run_name="baseline", latency_seconds=1.23)])
    assert "N/A" in sparse
    # Các hằng số bịa sẵn trong bản cũ không được xuất hiện nữa.
    assert "13.06s" not in sparse
    assert "0.000624" not in sparse
