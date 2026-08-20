"""Benchmark suite and evaluation metrics calculation for single-agent vs multi-agent."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics, SourceDocument
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def compute_citation_coverage(
    final_answer: str | None,
    sources: list[SourceDocument],
) -> float:
    """Calculate the ratio of source documents properly cited in the final report.

    Extracts citation numbers like [1], [2], [Source 1], [Nguồn 1] or matches against URLs.
    """
    if not final_answer or not sources:
        return 0.0

    total_sources = len(sources)
    if total_sources == 0:
        return 0.0

    # 1. Regex tìm các chỉ số trích dẫn dạng [1], [2], [Source 1], [Nguồn 1]
    bracket_citations = re.findall(r"\[(?:Source\s*|Nguồn\s*)?(\d+)\]", final_answer, re.IGNORECASE)
    cited_indices = {int(num) for num in bracket_citations}

    # Đếm số nguồn hợp lệ nằm trong dải 1 .. total_sources
    valid_cited_count = sum(1 for idx in cited_indices if 1 <= idx <= total_sources)

    # 2. Bổ sung đối chiếu theo URL nếu có
    for i, doc in enumerate(sources, 1):
        if i not in cited_indices and doc.url and doc.url in final_answer:
            valid_cited_count += 1

    coverage = valid_cited_count / total_sources
    return min(1.0, max(0.0, round(coverage, 3)))


def compute_estimated_cost(state: ResearchState) -> float:
    """Aggregate total USD cost from agent execution results and traces."""
    total_cost = 0.0

    if state.agent_results:
        for r in state.agent_results:
            total_cost += float(r.metadata.get("cost_usd", 0.0))

    elif state.trace:
        for event in state.trace:
            payload = event.get("payload", {})
            if isinstance(payload, dict) and "cost_usd" in payload:
                total_cost += float(payload.get("cost_usd", 0.0))

    return round(total_cost, 6)


def compute_quality_score(
    state: ResearchState,
    citation_coverage: float,
) -> float:
    """Evaluate report quality based on structure, depth, citations, and analysis."""
    if not state.final_answer or len(state.final_answer.strip()) < 50:
        return 0.0

    score = 0.0
    text = state.final_answer

    # Độ dài và tính đầy đủ (tối đa 4 điểm)
    words = len(text.split())
    if words >= 250:
        score += 4.0
    elif words >= 100:
        score += 2.5
    else:
        score += 1.0

    # Cấu trúc Markdown rõ ràng (tối đa 2 điểm)
    headers_count = len(re.findall(r"^#{1,4}\s+", text, re.MULTILINE))
    if headers_count >= 3:
        score += 2.0
    elif headers_count >= 1:
        score += 1.0

    # Kỷ luật trích dẫn nguồn (tối đa 2.5 điểm)
    score += round(citation_coverage * 2.5, 2)

    # Chiều sâu phân tích hoặc sự cộng tác đa vai trò (tối đa 1.5 điểm)
    if state.analysis_notes or len(state.agent_results) > 1:
        score += 1.5
    elif "so sánh" in text.lower() or "ưu điểm" in text.lower() or "trade-off" in text.lower():
        score += 1.0

    return min(10.0, max(0.0, round(score, 1)))


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Execute a runner, measure wall-clock latency, compute multi-dimensional metrics."""
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    coverage = compute_citation_coverage(state.final_answer, state.sources)
    cost = compute_estimated_cost(state)
    quality = compute_quality_score(state, coverage)
    failure = 0.0 if (state.final_answer and len(state.final_answer) > 50) else 1.0

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 2),
        estimated_cost_usd=cost,
        quality_score=quality,
        citation_coverage=coverage,
        failure_rate=failure,
        notes=f"Sources: {len(state.sources)} | Iterations: {state.iteration}",
    )
    return state, metrics
