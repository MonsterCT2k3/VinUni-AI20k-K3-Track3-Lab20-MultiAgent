"""Command-line entrypoint for the lab starter."""

from time import perf_counter
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report, save_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def run_single_agent_baseline(query: str) -> ResearchState:
    """Execute a single-agent baseline run (search + synthesize in one prompt)."""
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)

    search_client = SearchClient()
    llm_client = LLMClient()

    # 1. Thu thập tài liệu
    sources = search_client.search(request.query, max_results=request.max_sources)
    state.sources = sources

    # 2. Xây dựng ngữ cảnh tổng hợp
    context_lines: list[str] = []
    for i, src in enumerate(sources, 1):
        context_lines.append(
            f"[{i}] Tiêu đề: {src.title}\n"
            f"    URL: {src.url or 'N/A'}\n"
            f"    Nội dung: {src.snippet}"
        )
    context_str = "\n\n".join(context_lines)

    # 3. Monolithic Prompting
    system_prompt = (
        "Bạn là một trợ lý nghiên cứu độc lập (Single-Agent Baseline).\n"
        "Nhiệm vụ của bạn là đọc các tài liệu tham khảo được cung cấp dưới đây và viết một bài "
        "báo cáo nghiên cứu rõ ràng, súc tích (300-500 từ) trả lời câu hỏi của người dùng.\n\n"
        "Quy tắc bắt buộc:\n"
        "1. Trả lời đúng trọng tâm và cấu trúc mạch lạc (Tổng quan, So sánh, Kết luận).\n"
        "2. Bắt buộc gắn trích dẫn nguồn dạng [1], [2] tương ứng với tài liệu được cung cấp.\n"
        "3. Cuối bài phải có mục Tài Liệu Tham Khảo (References) liệt kê lại các nguồn."
    )
    user_prompt = (
        f"Câu hỏi nghiên cứu: {request.query}\n\n"
        f"Danh sách tài liệu tham khảo:\n{context_str}\n\n"
        "Hãy viết bài báo cáo hoàn chỉnh:"
    )

    started = perf_counter()
    resp = llm_client.complete(system_prompt, user_prompt)
    latency = perf_counter() - started

    state.final_answer = resp.content
    state.add_trace_event(
        "baseline_completion",
        {
            "latency_seconds": round(latency, 3),
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "cost_usd": resp.cost_usd,
            "source_count": len(sources),
        },
    )
    return state


def run_multi_agent_system(query: str) -> ResearchState:
    """Execute the full multi-agent workflow graph for a given query."""
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    workflow = MultiAgentWorkflow()
    return workflow.run(state)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a real single-agent baseline pipeline and display metrics."""
    _init()
    console.print(
        Panel(f"Đang thực thi Single-Agent Baseline: '{query}'...", style="cyan")
    )

    state = run_single_agent_baseline(query)

    # In kết quả bài viết
    console.print(
        Panel(
            state.final_answer or "Không có kết quả",
            title="Single-Agent Final Answer",
            style="green",
        )
    )

    # In bảng số liệu đo lường
    trace_info = state.trace[-1]["payload"] if state.trace else {}
    table = Table(
        title="Baseline Performance Metrics",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    table.add_row("Latency", f"{trace_info.get('latency_seconds', 0.0):.2f} s")
    table.add_row("Input Tokens", str(trace_info.get("input_tokens", "N/A")))
    table.add_row("Output Tokens", str(trace_info.get("output_tokens", "N/A")))
    cost = trace_info.get("cost_usd")
    table.add_row("Estimated Cost", f"${cost:.6f}" if cost is not None else "N/A")
    table.add_row("Sources Retrieved", str(trace_info.get("source_count", len(state.sources))))

    console.print(table)


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow and display live execution details."""
    _init()
    console.print(
        Panel(f"Đang kích hoạt Multi-Agent Research System: '{query}'...", style="bold blue")
    )

    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow()

    started = perf_counter()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    total_latency = perf_counter() - started

    # In Lộ trình điều hướng (Routing Flow)
    route_str = " ➔ ".join(result.route_history)
    flow_text = f"[bold cyan]Lộ trình ({len(result.route_history)} bước):[/bold cyan]\n{route_str}"
    console.print(
        Panel(
            flow_text,
            title="Supervisor Routing History",
            style="cyan",
        )
    )

    # In Kết quả Báo cáo Hoàn Chỉnh
    console.print(
        Panel(
            result.final_answer or "Không có kết quả cuối cùng.",
            title="Multi-Agent Final Answer (Writer)",
            style="bold green",
        )
    )

    # Tính toán tổng hợp chi phí và tokens từ agent_results
    total_in_tokens = sum(
        int(r.metadata.get("input_tokens", 0)) for r in result.agent_results
    )
    total_out_tokens = sum(
        int(r.metadata.get("output_tokens", 0)) for r in result.agent_results
    )
    total_cost = sum(
        float(r.metadata.get("cost_usd", 0.0)) for r in result.agent_results
    )

    # In bảng hiệu năng Multi-Agent
    table = Table(
        title="Multi-Agent System Performance Metrics",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    table.add_row("Total Latency", f"{total_latency:.2f} s")
    table.add_row("Iterations", str(result.iteration))
    table.add_row("Total Input Tokens", str(total_in_tokens))
    table.add_row("Total Output Tokens", str(total_out_tokens))
    table.add_row("Total Estimated Cost", f"${total_cost:.6f}")
    table.add_row("Sources Retrieved", str(len(result.sources)))
    table.add_row("Agents Executed", ", ".join(r.agent.value for r in result.agent_results))

    console.print(table)


@app.command("benchmark")
def benchmark(
    query: Annotated[
        str,
        typer.Option("--query", "-q", help="Research query to benchmark"),
    ] = "So sánh single-agent vs multi-agent trong bài toán nghiên cứu",
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Output markdown report path"),
    ] = "reports/benchmark_report.md",
) -> None:
    """Run head-to-head benchmark between Single-Agent Baseline and Multi-Agent Workflow."""
    _init()
    console.print(
        Panel(f"Bắt đầu Benchmark Đối Chứng cho query:\n'{query}'", style="bold yellow")
    )

    # 1. Chạy Single-Agent Baseline
    console.print("[dim]Đang chạy Single-Agent Baseline...[/dim]")
    _, baseline_m = run_benchmark("Single-Agent Baseline", query, run_single_agent_baseline)

    # 2. Chạy Multi-Agent System
    console.print("[dim]Đang chạy Multi-Agent System...[/dim]")
    _, multi_m = run_benchmark("Multi-Agent System", query, run_multi_agent_system)

    # 3. Xuất báo cáo Markdown
    report_md = render_markdown_report([baseline_m, multi_m])
    out_path = save_markdown_report(report_md, output_path=output)
    console.print(f"[bold green]Đã lưu báo cáo benchmark tại: {out_path}[/bold green]")

    # 4. In bảng tóm tắt so sánh
    table = Table(
        title="Head-to-Head Benchmark Summary",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Mô Hình", style="bold")
    table.add_column("Latency (s)", justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_column("Quality (0-10)", justify="right")
    table.add_column("Citation Cov.", justify="right")

    for m in [baseline_m, multi_m]:
        cost_str = f"${m.estimated_cost_usd:.6f}" if m.estimated_cost_usd is not None else "N/A"
        qual_str = f"{m.quality_score:.1f}" if m.quality_score is not None else "N/A"
        cov_str = f"{m.citation_coverage:.0%}" if m.citation_coverage is not None else "N/A"
        table.add_row(m.run_name, f"{m.latency_seconds:.2f}", cost_str, qual_str, cov_str)

    console.print(table)


if __name__ == "__main__":
    app()
