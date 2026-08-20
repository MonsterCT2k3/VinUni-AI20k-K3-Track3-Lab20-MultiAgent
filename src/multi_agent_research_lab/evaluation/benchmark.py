"""Benchmark suite and evaluation metrics calculation for single-agent vs multi-agent."""

import logging
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter

import yaml

from multi_agent_research_lab.core.schemas import BenchmarkMetrics, SourceDocument
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]

CONFIG_PATH = Path("configs") / "lab_default.yaml"

# Dùng khi không đọc được file config (giữ benchmark luôn chạy được).
DEFAULT_BENCHMARK_QUERIES: tuple[str, ...] = (
    "Research GraphRAG state-of-the-art and write a 500-word summary",
    "Compare single-agent and multi-agent workflows for customer support",
    "Summarize production guardrails for LLM agents",
)


def load_benchmark_queries(config_path: Path | None = None) -> list[str]:
    """Đọc `benchmark.queries` từ `configs/lab_default.yaml`.

    Chạy nhiều truy vấn rồi lấy trung bình quan trọng vì LLM sinh văn bản ngẫu nhiên
    (`temperature > 0`): kết quả của một truy vấn đơn lẻ dao động mạnh và dễ dẫn tới
    kết luận sai về kiến trúc nào tốt hơn.
    """
    path = config_path or CONFIG_PATH
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        queries = data.get("benchmark", {}).get("queries") or []
        valid = [q for q in queries if isinstance(q, str) and q.strip()]
        if valid:
            return valid
        logger.warning("Không tìm thấy `benchmark.queries` hợp lệ trong %s.", path)
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Không đọc được config benchmark %s: %s", path, exc)
    return list(DEFAULT_BENCHMARK_QUERIES)


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


# Từ khóa thể hiện chiều sâu phân tích, chấm trực tiếp trên nội dung bài viết.
DEPTH_MARKERS: tuple[str, ...] = (
    "so sánh",
    "ưu điểm",
    "nhược điểm",
    "trade-off",
    "đánh đổi",
    "hạn chế",
    "mâu thuẫn",
)


def compute_quality_score(
    state: ResearchState,
    citation_coverage: float,
) -> float:
    """Chấm điểm heuristic (0-10) cho `final_answer`.

    Đây là *structural heuristic score*, chỉ đo các đặc trưng bề mặt (độ dài, cấu trúc
    Markdown, độ phủ trích dẫn, dấu hiệu phân tích) — nó **bổ sung** chứ **không thay thế**
    rubric 0-10 do con người chấm trong `docs/peer_review_rubric.md`.

    Nguyên tắc công bằng: mọi thành phần điểm chỉ đọc `final_answer` (nội dung được chấm)
    và `state.errors`. Hàm này **không** đọc `analysis_notes` hay số lượng `agent_results`,
    vì đó là đặc trưng *hình dạng pipeline* mà chỉ multi-agent mới có được — chấm theo chúng
    sẽ khiến single-agent baseline không bao giờ đạt điểm tối đa, làm phép so sánh mất ý nghĩa.
    """
    if not state.final_answer or len(state.final_answer.strip()) < 50:
        return 0.0

    text = state.final_answer

    # Độ dài và tính đầy đủ (tối đa 4 điểm) — thang liên tục, không có bậc nhảy ở mốc cố định
    words = len(text.split())
    score = min(words / 300, 1.0) * 4.0

    # Cấu trúc Markdown rõ ràng (tối đa 2 điểm)
    headers_count = len(re.findall(r"^#{1,4}\s+", text, re.MULTILINE))
    score += min(headers_count / 3, 1.0) * 2.0

    # Kỷ luật trích dẫn nguồn (tối đa 2.5 điểm)
    score += citation_coverage * 2.5

    # Chiều sâu phân tích (tối đa 1.5 điểm) — chấm trên NỘI DUNG, áp dụng chung cho mọi kiến trúc
    lowered = text.lower()
    depth_hits = sum(1 for marker in DEPTH_MARKERS if marker in lowered)
    score += min(depth_hits * 0.5, 1.5)

    # Phạt lỗi đã ghi nhận (tối đa -3 điểm)
    score -= min(len(state.errors), 3)

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
    # Thất bại = có lỗi được ghi nhận tường minh, HOẶC không sinh được câu trả lời thực chất.
    # Không suy đoán qua độ dài `final_answer`: thông báo lỗi của guardrail cũng dài > 50 ký tự
    # nên cách cũ chấm một lần chạy hỏng hoàn toàn là "thành công".
    has_answer = bool(state.final_answer and len(state.final_answer.strip()) > 50)
    failure = 1.0 if (state.errors or not has_answer) else 0.0

    notes = f"Sources: {len(state.sources)} | Iterations: {state.iteration}"
    if state.errors:
        # Nêu rõ lỗi để guardrail chứng minh được nó thực sự bắt được sự cố.
        notes += f" | Lỗi: {'; '.join(state.errors[:3])}"

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 2),
        estimated_cost_usd=cost,
        quality_score=quality,
        citation_coverage=coverage,
        failure_rate=failure,
        notes=notes,
    )
    return state, metrics


def aggregate_metrics(run_name: str, runs: Sequence[BenchmarkMetrics]) -> BenchmarkMetrics:
    """Gộp nhiều lần đo của cùng một kiến trúc thành một dòng số liệu trung bình.

    Các trường tuỳ chọn (`cost`, `quality`, `coverage`) chỉ lấy trung bình trên những lần
    chạy thực sự có giá trị, tránh việc một lần chạy lỗi kéo tụt trung bình thành `None`.
    """
    if not runs:
        raise ValueError("Cần ít nhất một lần chạy để tổng hợp số liệu.")

    n = len(runs)
    costs = [m.estimated_cost_usd for m in runs if m.estimated_cost_usd is not None]
    qualities = [m.quality_score for m in runs if m.quality_score is not None]
    coverages = [m.citation_coverage for m in runs if m.citation_coverage is not None]
    failures = [m.failure_rate for m in runs if m.failure_rate is not None]

    return BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(sum(m.latency_seconds for m in runs) / n, 2),
        estimated_cost_usd=round(sum(costs) / len(costs), 6) if costs else None,
        quality_score=round(sum(qualities) / len(qualities), 1) if qualities else None,
        citation_coverage=round(sum(coverages) / len(coverages), 3) if coverages else None,
        failure_rate=round(sum(failures) / len(failures), 3) if failures else None,
        notes=_summarize_notes(n, runs),
    )


def _summarize_notes(n: int, runs: Sequence[BenchmarkMetrics]) -> str:
    """Giữ lại lý do thất bại khi gộp số liệu.

    Nếu không nêu, một `failure_rate` khác 0 sẽ không giải thích được lỗi gì đã xảy ra —
    trong khi đó chính là thông tin rubric "Failure guard" cần.
    """
    notes = f"Trung bình {n} truy vấn"
    failed = [m for m in runs if (m.failure_rate or 0.0) > 0.0]
    if not failed:
        return notes

    reasons: list[str] = []
    for m in failed:
        _, _, detail = (m.notes or "").partition("Lỗi: ")
        for reason in filter(None, (r.strip() for r in detail.split(";"))):
            if reason not in reasons:
                reasons.append(reason)

    notes += f" | {len(failed)}/{n} truy vấn lỗi"
    if reasons:
        notes += f": {'; '.join(reasons[:3])}"
    return notes


def run_benchmark_suite(
    run_name: str,
    runner: Runner,
    queries: Sequence[str],
) -> BenchmarkMetrics:
    """Chạy `runner` qua toàn bộ `queries` rồi trả về một dòng số liệu trung bình.

    Một truy vấn lỗi không được làm hỏng cả bộ benchmark: lỗi được quy thành một lần chạy
    `failure_rate = 1.0` để vẫn phản ánh đúng vào tỉ lệ thất bại trung bình.
    """
    runs: list[BenchmarkMetrics] = []
    for query in queries:
        try:
            _, metrics = run_benchmark(run_name, query, runner)
        except Exception as exc:
            logger.error("Benchmark %r thất bại với truy vấn %r: %s", run_name, query, exc)
            metrics = BenchmarkMetrics(
                run_name=run_name,
                latency_seconds=0.0,
                failure_rate=1.0,
                notes=f"runner lỗi: {exc}",
            )
        runs.append(metrics)

    return aggregate_metrics(run_name, runs)
