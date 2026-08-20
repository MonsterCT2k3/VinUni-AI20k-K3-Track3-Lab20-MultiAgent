"""Streamlit Interactive Web Application for Lab 20 - Multi-Agent Research System.

Provides live execution of Single-Agent Baseline and LangGraph Multi-Agent Workflow,
complete with real-time visual pipeline stages, live streaming logs, benchmark arena,
and a deep-dive Multi-Agent Architecture & Workflow tab.
"""

import os
from collections.abc import Callable
from datetime import datetime
from time import perf_counter
from typing import Any

import streamlit as st

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    compute_citation_coverage,
    compute_estimated_cost,
    compute_quality_score,
)
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

# Page Configuration
st.set_page_config(
    page_title="Multi-Agent Deep Research Lab",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .terminal-box {
        background-color: #0b0f19;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 14px;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 0.85rem;
        color: #38bdf8;
        max-height: 280px;
        overflow-y: auto;
        line-height: 1.5;
        box-shadow: inset 0 2px 6px rgba(0,0,0,0.5);
    }
    .agent-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .agent-card.running {
        border-color: #06b6d4;
        box-shadow: 0 0 12px rgba(6, 182, 212, 0.3);
        background: rgba(6, 182, 212, 0.08);
    }
    .agent-card.done {
        border-color: #10b981;
        background: rgba(16, 185, 129, 0.08);
    }
    .arch-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 16px;
    }
    .arch-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #818cf8;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

settings = get_settings()

# Sidebar
st.sidebar.markdown("### 🎓 Thông Tin Học Viên")
st.sidebar.success(
    "**Họ và tên:** Nguyễn Đăng Nam\n\n"
    "**Mã học viên:** `2A202601307`\n\n"
    "**Khóa:** VinUni AI20k - K3 Track 3"
)
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Cấu Hình Tìm Kiếm & LLM")

search_mode = st.sidebar.radio(
    "Nguồn Tìm Kiếm (Search Provider):",
    ["📚 Offline 30-Topics Corpus", "🌐 Tavily Live Web Search"],
    index=0,
)

if search_mode == "🌐 Tavily Live Web Search":
    current_key = os.environ.get("TAVILY_API_KEY", "") or (settings.tavily_api_key or "")
    tavily_input = st.sidebar.text_input(
        "Nhập Tavily API Key:",
        value=current_key,
        type="password",
        help="Đăng ký miễn phí tại https://tavily.com để lấy API key",
    )
    if tavily_input:
        os.environ["TAVILY_API_KEY"] = tavily_input
        settings.tavily_api_key = tavily_input
        st.sidebar.success("✅ Đã kết nối Tavily Live Web Search!")
    else:
        st.sidebar.warning("⚠️ Chưa có Tavily API Key (lấy miễn phí tại tavily.com).")
else:
    # Nếu chọn offline, tạm thời bỏ key để ưu tiên offline corpus
    settings.tavily_api_key = None

st.sidebar.info(
    f"**LLM Model:** `{settings.openai_model}`\n\n"
    f"**Chế độ tìm kiếm:** `{search_mode}`\n\n"
    f"**Max Iterations:** `{settings.max_iterations}`\n\n"
    f"**Timeout:** `{settings.timeout_seconds}s`"
)

PRESET_TOPICS = [
    "So sánh single-agent vs multi-agent trong bài toán nghiên cứu sâu",
    "So sánh ưu nhược điểm của GraphRAG và Vector RAG truyền thống",
    "Đánh giá phương pháp Tool Calling so với ReAct Pattern trong Agentic AI",
    "Phân tích giới hạn của Long-Context LLMs so với Agentic RAG",
    "So sánh framework AutoGen, CrewAI và LangGraph trong lập trình Multi-Agent",
]

st.sidebar.markdown("### 📚 Chủ Đề Mẫu (Offline Corpus)")
selected_preset = st.sidebar.selectbox("Chọn câu hỏi nghiên cứu:", PRESET_TOPICS)

# Header
st.markdown(
    '<div class="main-title">🤖 Multi-Agent Deep Research System</div>',
    unsafe_allow_html=True,
)
st.caption("Lab 20 • VinUni AI20k Track 3 • Trực Quan Hóa Quá Trình Thực Thi & Benchmark")


def format_log(msg: str, level: str = "INFO") -> str:
    now = datetime.now().strftime("%H:%M:%S")
    return f"[{now}] [{level}] {msg}"


def execute_multi_agent_live(
    query: str,
    log_fn: Callable[[str], None],
    step_fn: Callable[[str, str, dict[str, Any]], None],
) -> tuple[ResearchState, float]:
    """Execute Multi-Agent Workflow step-by-step while streaming logs and UI updates."""
    state = ResearchState(request=ResearchQuery(query=query))
    supervisor = SupervisorAgent()
    researcher = ResearcherAgent()
    analyst = AnalystAgent()
    writer = WriterAgent()

    started = perf_counter()
    log_fn(format_log(f"Khởi động Multi-Agent Workflow cho query: '{query}'", "INIT"))

    # Step 1: Supervisor evaluate initial state
    step_fn("supervisor", "running", {"detail": "Đang kiểm tra State ban đầu..."})
    log_fn(format_log("Supervisor: Chưa có sources -> Route: 'researcher'", "ROUTER"))
    state = supervisor.run(state)
    step_fn("supervisor", "done", {"detail": "Đã điều phối ➔ Researcher"})

    # Step 2: Researcher execute
    step_fn("researcher", "running", {"detail": "Đang tìm kiếm & thu thập tài liệu..."})
    log_fn(format_log("ResearcherAgent bắt đầu tìm kiếm qua SearchClient...", "AGENT"))
    r_start = perf_counter()
    state = researcher.run(state)
    r_dur = perf_counter() - r_start
    log_fn(format_log(f"Researcher ({r_dur:.2f}s): Tìm thấy {len(state.sources)} nguồn", "SUCCESS"))
    step_detail = f"Đã thu thập {len(state.sources)} nguồn ({r_dur:.2f}s)"
    step_fn("researcher", "done", {"detail": step_detail})

    # Step 3: Supervisor route to Analyst
    step_fn("supervisor", "running", {"detail": "Đang kiểm tra State sau Researcher..."})
    log_fn(format_log("Supervisor: Đã có sources -> Route: 'analyst'", "ROUTER"))
    state = supervisor.run(state)
    step_fn("supervisor", "done", {"detail": "Đã điều phối ➔ Analyst"})

    # Step 4: Analyst execute
    step_fn("analyst", "running", {"detail": "Đang phân tích, phản biện & trade-offs..."})
    log_fn(format_log("AnalystAgent bắt đầu phân tích đối chiếu mâu thuẫn...", "AGENT"))
    a_start = perf_counter()
    state = analyst.run(state)
    a_dur = perf_counter() - a_start
    log_fn(format_log(f"Analyst ({a_dur:.2f}s): Đã trích xuất Key Insights", "SUCCESS"))
    step_fn("analyst", "done", {"detail": f"Đã trích xuất phản biện ({a_dur:.2f}s)"})

    # Step 5: Supervisor route to Writer
    step_fn("supervisor", "running", {"detail": "Đang kiểm tra State sau Analyst..."})
    log_fn(format_log("Supervisor: Đã có analysis -> Route: 'writer'", "ROUTER"))
    state = supervisor.run(state)
    step_fn("supervisor", "done", {"detail": "Đã điều phối ➔ Writer"})

    # Step 6: Writer execute
    step_fn("writer", "running", {"detail": "Đang soạn thảo & gắn citations [1], [2]..."})
    log_fn(format_log("WriterAgent bắt đầu tổng hợp báo cáo Markdown...", "AGENT"))
    w_start = perf_counter()
    state = writer.run(state)
    w_dur = perf_counter() - w_start
    log_fn(format_log(f"Writer ({w_dur:.2f}s): Báo cáo hoàn thiện 100% trích dẫn", "SUCCESS"))
    step_fn("writer", "done", {"detail": f"Đã hoàn thành bài báo cáo ({w_dur:.2f}s)"})

    # Step 7: Supervisor finalize (Done)
    state = supervisor.run(state)
    total_latency = perf_counter() - started
    log_fn(
        format_log(
            f"Hoàn tất trong {total_latency:.2f}s (Route: {' ➔ '.join(state.route_history)})",
            "COMPLETE",
        )
    )

    return state, total_latency


def execute_baseline_live(
    query: str,
    log_fn: Callable[[str], None],
) -> tuple[ResearchState, float]:
    """Execute Single-Agent Baseline with live log feedback."""
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    search_client = SearchClient()
    llm_client = LLMClient()

    started = perf_counter()
    log_fn(format_log(f"Khởi động Single-Agent Baseline cho query: '{query}'", "INIT"))

    log_fn(format_log("Đang tìm kiếm tài liệu từ SearchClient...", "SEARCH"))
    sources = search_client.search(request.query, max_results=request.max_sources)
    state.sources = sources
    log_fn(format_log(f"Đã lấy {len(sources)} tài liệu nguồn", "SUCCESS"))

    log_fn(format_log("Xây dựng Monolithic Prompt và gửi 1-shot completion...", "LLM"))
    context_lines = [
        f"[{i}] Tiêu đề: {src.title}\n    URL: {src.url or 'N/A'}\n    Nội dung: {src.snippet}"
        for i, src in enumerate(sources, 1)
    ]
    context_str = "\n\n".join(context_lines)
    system_prompt = (
        "Bạn là trợ lý nghiên cứu độc lập (Single-Agent Baseline).\n"
        "Đọc tài liệu và viết báo cáo nghiên cứu 300-500 từ có trích dẫn [1], [2]."
    )
    user_prompt = f"Câu hỏi: {request.query}\n\nDanh sách tài liệu:\n{context_str}\n\nViết báo cáo:"

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
        },
    )
    log_fn(format_log(f"Single-Agent xong ({latency:.2f}s, ${resp.cost_usd:.6f})", "COMPLETE"))
    return state, latency


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚀 1. Multi-Agent Deep Research (Live)",
    "⚡ 2. Single-Agent Baseline (Live)",
    "📊 3. Head-to-Head Benchmark Arena",
    "🏛️ 4. Báo Cáo & Exit Ticket",
    "🧩 5. Kiến Trúc Hệ Thống & Luồng Hoạt Động",
])

# -------------------------------------------------------------------------------------------------
# TAB 1: Multi-Agent Deep Research
# -------------------------------------------------------------------------------------------------
with tab1:
    st.subheader("🚀 Trực Quan Hóa Tiến Trình Multi-Agent Workflow")
    st.write(
        "Theo dõi trực tiếp quá trình điều phối giữa **Supervisor** và từng **Worker Agent**."
    )

    col_q1, col_b1 = st.columns([4, 1])
    with col_q1:
        query_ma = st.text_input("Câu hỏi nghiên cứu:", value=selected_preset, key="query_ma")
    with col_b1:
        st.write("")
        st.write("")
        btn_run_ma = st.button("▶ Chạy Multi-Agent", type="primary", use_container_width=True)

    if btn_run_ma and query_ma:
        st.markdown("### 🔄 Tiến Trình Luân Chuyển Tác Vụ (Live Pipeline)")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            p_sup = st.empty()
            p_sup.markdown(
                '<div class="agent-card">👑 <strong>Supervisor</strong><br>'
                '<small style="color:#94a3b8;">⏸️ Chờ điều phối</small></div>',
                unsafe_allow_html=True,
            )
        with c2:
            p_res = st.empty()
            p_res.markdown(
                '<div class="agent-card">🔎 <strong>Researcher</strong><br>'
                '<small style="color:#94a3b8;">⏸️ Chờ tìm kiếm</small></div>',
                unsafe_allow_html=True,
            )
        with c3:
            p_ana = st.empty()
            p_ana.markdown(
                '<div class="agent-card">⚖️ <strong>Analyst</strong><br>'
                '<small style="color:#94a3b8;">⏸️ Chờ phân tích</small></div>',
                unsafe_allow_html=True,
            )
        with c4:
            p_wri = st.empty()
            p_wri.markdown(
                '<div class="agent-card">✍️ <strong>Writer</strong><br>'
                '<small style="color:#94a3b8;">⏸️ Chờ viết bài</small></div>',
                unsafe_allow_html=True,
            )

        st.markdown("### 📟 Nhật Ký Hoạt Động Thời Gian Thực (Live Terminal Logs)")
        log_placeholder = st.empty()
        logs_list: list[str] = []

        def append_log(log_line: str) -> None:
            logs_list.append(log_line)
            log_placeholder.markdown(
                f'<div class="terminal-box">{"<br>".join(logs_list)}</div>',
                unsafe_allow_html=True,
            )

        def update_step_ui(agent: str, status: str, payload: dict[str, Any]) -> None:
            status_text = "⏳ Đang thực thi..." if status == "running" else "✅ Hoàn tất"
            color_class = "running" if status == "running" else "done"
            detail = payload.get("detail", "")

            if agent == "supervisor":
                p_sup.markdown(
                    f'<div class="agent-card {color_class}">👑 <strong>Supervisor</strong><br>'
                    f'<small>{status_text}<br>{detail}</small></div>',
                    unsafe_allow_html=True,
                )
            elif agent == "researcher":
                p_res.markdown(
                    f'<div class="agent-card {color_class}">🔎 <strong>Researcher</strong><br>'
                    f'<small>{status_text}<br>{detail}</small></div>',
                    unsafe_allow_html=True,
                )
            elif agent == "analyst":
                p_ana.markdown(
                    f'<div class="agent-card {color_class}">⚖️ <strong>Analyst</strong><br>'
                    f'<small>{status_text}<br>{detail}</small></div>',
                    unsafe_allow_html=True,
                )
            elif agent == "writer":
                p_wri.markdown(
                    f'<div class="agent-card {color_class}">✍️ <strong>Writer</strong><br>'
                    f'<small>{status_text}<br>{detail}</small></div>',
                    unsafe_allow_html=True,
                )

        result_state, latency_val = execute_multi_agent_live(query_ma, append_log, update_step_ui)

        cov_val = compute_citation_coverage(result_state.final_answer, result_state.sources)
        cost_val = compute_estimated_cost(result_state)
        qual_val = compute_quality_score(result_state, cov_val)

        st.markdown("---")
        st.markdown("### 📊 Chỉ Số Hiệu Năng Thực Tế")
        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        mcol1.metric("⏱️ Tổng Thời Gian", f"{latency_val:.2f} s")
        mcol2.metric("💰 Chi Phí Token", f"${cost_val:.6f}")
        mcol3.metric("🎯 Điểm Chất Lượng", f"{qual_val:.1f} / 10")
        mcol4.metric("📚 Độ Phủ Trích Dẫn", f"{cov_val:.0%}")
        mcol5.metric("🔄 Lộ Trình", f"{len(result_state.route_history)} bước")

        st.markdown("### 📄 Bài Báo Cáo Nghiên Cứu Hoàn Chỉnh (Writer Agent)")
        st.markdown(result_state.final_answer or "Không có kết quả.")

        with st.expander("🔎 Chi tiết tài liệu nguồn & Research Notes từ Researcher"):
            st.markdown(f"**Số lượng tài liệu:** {len(result_state.sources)}")
            for i, s in enumerate(result_state.sources, 1):
                st.markdown(f"- **[{i}] {s.title}** ({s.url or 'N/A'})\n  _{s.snippet}_")
            st.markdown("---")
            st.markdown(f"**Research Notes thô:**\n\n{result_state.research_notes}")

        with st.expander("⚖️ Chi tiết phân tích & Trade-offs từ Analyst"):
            st.markdown(result_state.analysis_notes or "Không có dữ liệu.")


# -------------------------------------------------------------------------------------------------
# TAB 2: Single-Agent Baseline
# -------------------------------------------------------------------------------------------------
with tab2:
    st.subheader("⚡ Trực Quan Hóa Single-Agent Baseline")
    st.write("Một prompt duy nhất gửi tới LLM cùng toàn bộ ngữ cảnh tài liệu.")

    col_q2, col_b2 = st.columns([4, 1])
    with col_q2:
        query_base = st.text_input("Câu hỏi nghiên cứu:", value=selected_preset, key="query_base")
    with col_b2:
        st.write("")
        st.write("")
        btn_run_base = st.button("▶ Chạy Baseline", type="primary", use_container_width=True)

    if btn_run_base and query_base:
        st.markdown("### 📟 Nhật Ký Hoạt Động (Live Terminal Logs)")
        base_log_placeholder = st.empty()
        base_logs_list: list[str] = []

        def append_base_log(line: str) -> None:
            base_logs_list.append(line)
            base_log_placeholder.markdown(
                f'<div class="terminal-box">{"<br>".join(base_logs_list)}</div>',
                unsafe_allow_html=True,
            )

        b_state, b_lat = execute_baseline_live(query_base, append_base_log)
        b_cov = compute_citation_coverage(b_state.final_answer, b_state.sources)
        b_cost = compute_estimated_cost(b_state)
        b_qual = compute_quality_score(b_state, b_cov)

        st.markdown("---")
        bmcol1, bmcol2, bmcol3, bmcol4 = st.columns(4)
        bmcol1.metric("⏱️ Thời Gian Chạy", f"{b_lat:.2f} s")
        bmcol2.metric("💰 Chi Phí Token", f"${b_cost:.6f}")
        bmcol3.metric("🎯 Điểm Chất Lượng", f"{b_qual:.1f} / 10")
        bmcol4.metric("📚 Độ Phủ Trích Dẫn", f"{b_cov:.0%}")

        st.markdown("### 📄 Bài Báo Cáo Từ Single-Agent Baseline")
        st.markdown(b_state.final_answer or "Không có kết quả.")


# -------------------------------------------------------------------------------------------------
# TAB 3: Head-to-Head Benchmark Arena
# -------------------------------------------------------------------------------------------------
with tab3:
    st.subheader("📊 Head-to-Head Benchmark Arena (Chạy Đối Chứng Trực Quan)")
    st.write("Quan sát trực tiếp tiến trình chạy song song và nhật ký log của cả hai kiến trúc.")

    query_bench = st.text_input(
        "Câu hỏi đo lường Benchmark:",
        value=selected_preset,
        key="query_bench",
    )
    if st.button("🏁 Bắt Đầu Chạy Đối Chứng Head-to-Head", type="primary"):
        bench_progress = st.progress(0, text="Khởi động Benchmark Suite...")

        st.markdown("### 📟 Nhật Ký Thực Thi Trực Tiếp (Live Execution Stream)")
        arena_log_placeholder = st.empty()
        arena_logs: list[str] = []

        def append_arena_log(line: str) -> None:
            arena_logs.append(line)
            arena_log_placeholder.markdown(
                f'<div class="terminal-box">{"<br>".join(arena_logs)}</div>',
                unsafe_allow_html=True,
            )

        # 1. Run Baseline
        bench_progress.progress(15, text="[1/2] Đang thực thi Single-Agent Baseline...")
        append_arena_log(format_log("=== BẮT ĐẦU PHA 1: SINGLE-AGENT BASELINE ===", "BENCHMARK"))
        b_state, b_lat = execute_baseline_live(query_bench, append_arena_log)
        b_cov = compute_citation_coverage(b_state.final_answer, b_state.sources)
        b_cost = compute_estimated_cost(b_state)
        b_qual = compute_quality_score(b_state, b_cov)

        # 2. Run Multi-Agent
        bench_progress.progress(50, text="[2/2] Đang thực thi LangGraph Multi-Agent Workflow...")
        append_arena_log(format_log("=== BẮT ĐẦU PHA 2: MULTI-AGENT WORKFLOW ===", "BENCHMARK"))

        def dummy_step(a: str, s: str, p: dict[str, Any]) -> None:
            pass

        m_state, m_lat = execute_multi_agent_live(query_bench, append_arena_log, dummy_step)
        m_cov = compute_citation_coverage(m_state.final_answer, m_state.sources)
        m_cost = compute_estimated_cost(m_state)
        m_qual = compute_quality_score(m_state, m_cov)

        bench_progress.progress(100, text="✅ Benchmark Hoàn Tất Cả 2 Mô Hình!")

        st.markdown("---")
        st.markdown("### 🏆 Bảng So Sánh Hiệu Năng Thực Tế")
        data = {
            "Mô Hình": ["Single-Agent Baseline", "Multi-Agent System"],
            "Thời Gian (s)": [f"{b_lat:.2f}s", f"{m_lat:.2f}s"],
            "Chi Phí ($ USD)": [f"${b_cost:.6f}", f"${m_cost:.6f}"],
            "Chất Lượng (0-10)": [f"{b_qual:.1f} / 10", f"{m_qual:.1f} / 10"],
            "Độ Phủ Nguồn": [f"{b_cov:.0%}", f"{m_cov:.0%}"],
            "Số Lượng Nguồn": [len(b_state.sources), len(m_state.sources)],
        }
        st.table(data)

        # Comparative Charts
        st.markdown("### 📈 Biểu Đồ Trực Quan")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("**Thời Gian Chạy (Latency - Giây)**")
            st.bar_chart({"Single-Agent": b_lat, "Multi-Agent": m_lat})
        with chart_col2:
            st.markdown("**Chi Phí Token (Cost - USD)**")
            st.bar_chart({"Single-Agent": b_cost, "Multi-Agent": m_cost})

        # Side by Side Reports
        st.markdown("### 📑 So Sánh Bài Báo Cáo Đầu Ra")
        rep_col1, rep_col2 = st.columns(2)
        with rep_col1:
            st.markdown("#### ⚡ Single-Agent Baseline Output")
            st.markdown(b_state.final_answer or "N/A")
        with rep_col2:
            st.markdown("#### 🚀 Multi-Agent System Output")
            st.markdown(m_state.final_answer or "N/A")


# -------------------------------------------------------------------------------------------------
# TAB 4: Báo Cáo & Exit Ticket
# -------------------------------------------------------------------------------------------------
with tab4:
    st.subheader("🏛️ Ma Trận Quyết Định Kiến Trúc & Exit Ticket")

    st.markdown(
        """
        | Tiêu chí | Single-Agent | Multi-Agent | Khuyến nghị |
        |---|---|---|---|
        | **Tính chất** | Tra cứu nhanh, 1 nguồn | Nghiên cứu sâu, đa nguồn | **Multi-Agent** |
        | **Yêu cầu SLA** | Real-time (< 5s) | Background job (20-45s) | **Single-Agent** |
        | **Ngân sách** | Tối ưu chi phí tối đa | Chấp nhận chi phí cao hơn | **Single-Agent** |
        | **Ảo giác** | Dễ bỏ sót mâu thuẫn | Thấp nhờ kiểm chứng chéo | **Multi-Agent** |
        """
    )

    st.markdown("---")
    st.markdown("### ❓ Câu Trả Lời Exit Ticket")

    st.markdown(
        """
        > **1. Trường hợp nào BẮT BUỘC NÊN dùng Multi-Agent?**  
        > Khi bài toán đòi hỏi **nghiên cứu sâu (Deep Research)** với nhiều nguồn thông tin
        > đối nghịch nhau, cần phân tách rõ ràng giữa việc thu thập dữ liệu (Researcher), phản biện
        > kiểm chứng (Analyst), và soạn thảo báo cáo chuẩn học thuật (Writer). Multi-Agent giúp
        > cô lập lỗi, kiểm chứng chéo và duy trì kỷ luật trích dẫn nguồn mà Single-Agent
        > không thể đảm bảo một cách nhất quán.

        > **2. Trường hợp nào KHÔNG NÊN dùng Multi-Agent?**  
        > Với các tác vụ **đơn bước (Single-step lookup)**, hỏi đáp tài liệu ngắn, chatbot
        > hội thoại thời gian thực, hoặc các hệ thống có ngân sách token hạn hẹp và yêu cầu
        > độ trễ cực thấp (< 3 giây). Khi đó, chi phí điều phối (Orchestration Overhead)
        > và độ trễ của Multi-Agent là sự lãng phí không cần thiết.
        """
    )


# -------------------------------------------------------------------------------------------------
# TAB 5: Kiến Trúc Hệ Thống & Luồng Hoạt Động (Architecture & Flow)
# -------------------------------------------------------------------------------------------------
with tab5:
    st.subheader("🧩 Kiến Trúc Multi-Agent & Chu Trình Điều Phối Dữ Liệu")
    st.write(
        "Giải thích chi tiết mô hình **Supervisor Centralized Orchestration** "
        "và cấu trúc **Shared State Machine** trong LangGraph."
    )

    st.markdown("### 🗺️ 1. Sơ Đồ Khối Luồng Điều Phối (Data Flow & Handoff Loop)")
    st.code(
        """
                   ┌──────────────────────────────────────────────┐
                   │            👤 USER / CLIENT QUERY            │
                   │   "So sánh GraphRAG vs Vector RAG..."        │
                   └──────────────────────┬───────────────────────┘
                                          │
                                          ▼
                ┌────────────────────────────────────────────────────┐
                │              🧭 SUPERVISOR / ROUTER                │
                │             (agents/supervisor.py)                 │
                │  1. Đọc State: kiểm tra trường nào còn thiếu?      │
                │  2. Điều hướng: chọn Worker phù hợp                │
                │  3. Guardrail: chặn lặp khi iteration >= max_limit │
                └───────┬──────────────────┬──────────────────┬──────┘
                        │ (1)              │ (2)              │ (3)
        ┌───────────────┘                  │                  └───────────────┐
        ▼                                  ▼                                  ▼
┌───────────────┐                  ┌───────────────┐                  ┌───────────────┐
│  RESEARCHER   │                  │    ANALYST    │                  │    WRITER     │
│ • Tạo search  │                  │ • Đọc notes   │                  │ • Soạn báo cáo│
│ • Lấy sources │                  │ • Lọc mâu     │                  │ • Gắn citation│
│ ➔ Ghi: sources│                  │ ➔ Ghi: analysis                  │ ➔ Ghi: answer │
└───────┬───────┘                  └───────┬───────┘                  └───────┬───────┘
        │                                  │                                  │
        └──────────────────────────────────┼──────────────────────────────────┘
                                           │
                                           ▼
                             [ Trả quyền lại cho SUPERVISOR ]
        """,
        language="text",
    )

    st.markdown("### 👥 2. Phân Tách Trách Nhiệm Từng Tác Nhân (Agents Breakdown)")
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown(
            '<div class="arch-card">'
            '<div class="arch-title">🧭 1. Supervisor / Router (agents/supervisor.py)</div>'
            '<p><strong>Vai trò:</strong> Bộ não trung tâm điều phối toàn bộ chu trình.</p>'
            '<ul>'
            '<li><strong>Routing Policy:</strong>'
            '<ul>'
            '<li>Chưa có sources ➔ Gọi <strong>Researcher</strong>.</li>'
            '<li>Đã có sources, chưa có analysis ➔ Gọi <strong>Analyst</strong>.</li>'
            '<li>Đã có analysis, chưa có final_answer ➔ Gọi <strong>Writer</strong>.</li>'
            '<li>Đã có final_answer ➔ Kết thúc (<strong>Done</strong>).</li>'
            '</ul></li>'
            '<li><strong>Safety Guardrail:</strong> Ngắt ngay khi iteration >= max_iterations.</li>'
            '</ul></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="arch-card">'
            '<div class="arch-title">🔎 2. Researcher Agent (agents/researcher.py)</div>'
            '<p><strong>Vai trò:</strong> Chuyên gia thu thập và chuẩn hóa dữ liệu ngoài.</p>'
            '<ul>'
            '<li>Tối ưu hóa câu truy vấn từ yêu cầu nghiên cứu gốc.</li>'
            '<li>Giao tiếp với SearchClient (Tavily + 30 Topics Offline Corpus).</li>'
            '<li>Chuẩn hóa dữ liệu vào <code>sources: list[SourceDocument]</code>.</li>'
            '<li>Tóm tắt phát hiện thô vào <code>research_notes</code> (chống State Bloat).</li>'
            '</ul></div>',
            unsafe_allow_html=True,
        )

    with col_a2:
        st.markdown(
            '<div class="arch-card">'
            '<div class="arch-title">⚖️ 3. Analyst Agent (agents/analyst.py)</div>'
            '<p><strong>Vai trò:</strong> Chuyên gia phản biện, kiểm chứng và rút Insights.</p>'
            '<ul>'
            '<li>Đọc toàn bộ phát hiện thô trong <code>research_notes</code>.</li>'
            '<li><strong>Fact-Checking:</strong> Kiểm chứng chéo chống ảo giác thông tin.</li>'
            '<li>Phát hiện mâu thuẫn giữa các tài liệu và tổng hợp đánh đổi (trade-offs).</li>'
            '<li>Ghi nhận luận điểm vào <code>analysis_notes</code>.</li>'
            '</ul></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="arch-card">'
            '<div class="arch-title">✍️ 4. Writer Agent (agents/writer.py)</div>'
            '<p><strong>Vai trò:</strong> Chuyên gia tổng hợp báo cáo học thuật & trích dẫn.</p>'
            '<ul>'
            '<li>Đọc <code>analysis_notes</code> và danh sách URL gốc.</li>'
            '<li>Biên tập báo cáo Markdown theo cấu trúc mạch lạc.</li>'
            '<li><strong>Citation Discipline:</strong> Bắt buộc gắn trích dẫn [1], [2].</li>'
            '<li>Ghi bài viết hoàn chỉnh vào <code>final_answer</code>.</li>'
            '</ul></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### 💾 3. Cấu Trúc Trạng Thái Chia Sẻ (Shared State - ResearchState)")
    st.markdown(
        r"""
        | Tên Trường | Kiểu Dữ Liệu | Mục Đích Sử Dụng |
        |---|---|---|
        | `request` | `ResearchQuery` | Lưu câu hỏi nghiên cứu, max_sources và audience. |
        | `sources` | `list[SourceDocument]` | Danh sách tài liệu, snippet, URL nguồn. |
        | `research_notes` | `str \| None` | Tóm tắt phát hiện thô của **Researcher**. |
        | `analysis_notes` | `str \| None` | Luận điểm phản biện & trade-offs của **Analyst**. |
        | `final_answer` | `str \| None` | Báo cáo Markdown của **Writer** có trích dẫn. |
        | `route_history` | `list[str]` | Nhật ký điều phối: `['res', 'ana', 'wri']`. |
        | `agent_results` | `list[AgentResult]` | Chi tiết kết quả, tokens và cost của từng agent. |
        | `iteration` | `int` | Biến đếm vòng lặp phục vụ guardrail `max_iterations`. |
        """
    )

    st.markdown("### 🛡️ 4. Cơ Chế Điều Phối Handoff & Rào Chắn Vận Hành (Guardrails)")
    st.info(
        "💡 **Tại sao hệ thống không bao giờ bị kẹt vòng lặp vô hạn?**\n\n"
        "1. **LangGraph StateGraph & Conditional Edges:** Các cạnh có điều kiện chỉ chuyển tiếp "
        "khi thỏa mãn điều kiện logic rõ ràng.\n\n"
        "2. **Guardrail `max_iterations`:** Biến đếm `iteration` tăng sau mỗi lượt Supervisor. "
        "Nếu chạm ngưỡng tối đa (mặc định là 6), Supervisor sẽ dừng chu trình an toàn.\n\n"
        "3. **Tách Biệt Context (Chống State Bloat):** Researcher và Analyst cô đọng thông tin thô "
        "thành notes có cấu trúc, giúp Context Window của LLM luôn ngắn gọn và chính xác."
    )
