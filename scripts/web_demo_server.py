"""Lightweight Live Web Demo Server for Multi-Agent Research System.

Connects the HTML dashboard directly to real Python Single-Agent Baseline & LangGraph Multi-Agent Workflow.
Zero external dependencies required (uses built-in http.server with Threading).
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import logging
from pathlib import Path
from socketserver import ThreadingMixIn
import sys
from time import perf_counter
from typing import Any
import urllib.parse

from multi_agent_research_lab.cli import run_single_agent_baseline
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    compute_citation_coverage,
    compute_estimated_cost,
    compute_quality_score,
)
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WebDemoServer")

ROOT_DIR = Path(__file__).resolve().parent.parent
HTML_FILE = ROOT_DIR / "docs-for-dev" / "benchmark_dashboard.html"


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in separate threads for non-blocking concurrent execution."""

    daemon_threads = True


class LiveDemoHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler serving the live web UI and live Agent APIs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(ROOT_DIR), **kwargs)

    def _send_json_response(self, data: dict[str, Any], status: int = 200) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed_path = urllib.parse.urlparse(self.path).path
        if parsed_path in ["/", "/index.html", "/demo", "/dashboard"]:
            if HTML_FILE.exists():
                content = HTML_FILE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            self.send_error(404, "HTML file not found")
            return

        if parsed_path == "/api/status":
            self._send_json_response({"status": "ready", "model": get_settings().openai_model})
            return

        super().do_GET()

    def do_POST(self) -> None:
        parsed_path = urllib.parse.urlparse(self.path).path
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"

        try:
            req_data = json.loads(body)
        except json.JSONDecodeError:
            req_data = {}

        query = req_data.get("query", "So sánh single-agent vs multi-agent trong bài toán nghiên cứu")

        # 1. API: Chạy Single-Agent Baseline THẬT
        if parsed_path == "/api/run-baseline":
            logger.info("⚡ [API] Đang thực thi Single-Agent Baseline THẬT cho query: '%s'", query)
            try:
                started = perf_counter()
                state = run_single_agent_baseline(query)
                latency = perf_counter() - started

                cov = compute_citation_coverage(state.final_answer, state.sources)
                cost = compute_estimated_cost(state)
                quality = compute_quality_score(state, cov)

                resp = {
                    "success": True,
                    "model_type": "Single-Agent Baseline",
                    "latency_seconds": round(latency, 2),
                    "cost_usd": cost,
                    "quality_score": quality,
                    "citation_coverage": cov,
                    "sources": [s.model_dump() for s in state.sources],
                    "final_answer": state.final_answer,
                }
                self._send_json_response(resp)
            except Exception as exc:
                logger.exception("Lỗi thực thi Baseline: %s", exc)
                self._send_json_response({"success": False, "error": str(exc)}, status=500)
            return

        # 2. API: Chạy LangGraph Multi-Agent Workflow THẬT
        if parsed_path == "/api/run-multiagent":
            logger.info("🚀 [API] Đang thực thi Multi-Agent Workflow THẬT cho query: '%s'", query)
            try:
                started = perf_counter()
                state = ResearchState(request=ResearchQuery(query=query))
                workflow = MultiAgentWorkflow()
                result = workflow.run(state)
                latency = perf_counter() - started

                cov = compute_citation_coverage(result.final_answer, result.sources)
                cost = compute_estimated_cost(result)
                quality = compute_quality_score(result, cov)

                resp = {
                    "success": True,
                    "model_type": "Multi-Agent System",
                    "latency_seconds": round(latency, 2),
                    "cost_usd": cost,
                    "quality_score": quality,
                    "citation_coverage": cov,
                    "iterations": result.iteration,
                    "route_history": result.route_history,
                    "research_notes": result.research_notes,
                    "analysis_notes": result.analysis_notes,
                    "sources": [s.model_dump() for s in result.sources],
                    "agent_results": [r.model_dump() for r in result.agent_results],
                    "final_answer": result.final_answer,
                }
                self._send_json_response(resp)
            except Exception as exc:
                logger.exception("Lỗi thực thi Multi-Agent: %s", exc)
                self._send_json_response({"success": False, "error": str(exc)}, status=500)
            return

        # 3. API: Chạy Head-to-Head Benchmark Đối Chứng Cả Hai
        if parsed_path == "/api/run-benchmark":
            logger.info("📊 [API] Đang chạy Benchmark Head-to-Head THẬT cho query: '%s'", query)
            try:
                # Run Baseline
                b_start = perf_counter()
                b_state = run_single_agent_baseline(query)
                b_lat = perf_counter() - b_start
                b_cov = compute_citation_coverage(b_state.final_answer, b_state.sources)
                b_cost = compute_estimated_cost(b_state)
                b_qual = compute_quality_score(b_state, b_cov)

                # Run Multi-Agent
                m_start = perf_counter()
                m_state = ResearchState(request=ResearchQuery(query=query))
                m_result = MultiAgentWorkflow().run(m_state)
                m_lat = perf_counter() - m_start
                m_cov = compute_citation_coverage(m_result.final_answer, m_result.sources)
                m_cost = compute_estimated_cost(m_result)
                m_qual = compute_quality_score(m_result, m_cov)

                resp = {
                    "success": True,
                    "baseline": {
                        "latency": round(b_lat, 2),
                        "cost": b_cost,
                        "quality": b_qual,
                        "citation_cov": b_cov,
                        "final_answer": b_state.final_answer,
                        "sources_count": len(b_state.sources),
                    },
                    "multi_agent": {
                        "latency": round(m_lat, 2),
                        "cost": m_cost,
                        "quality": m_qual,
                        "citation_cov": m_cov,
                        "iterations": m_result.iteration,
                        "route_history": m_result.route_history,
                        "research_notes": m_result.research_notes,
                        "analysis_notes": m_result.analysis_notes,
                        "final_answer": m_result.final_answer,
                        "sources_count": len(m_result.sources),
                    },
                }
                self._send_json_response(resp)
            except Exception as exc:
                logger.exception("Lỗi thực thi Benchmark: %s", exc)
                self._send_json_response({"success": False, "error": str(exc)}, status=500)
            return

        self.send_error(404, "Endpoint not found")


def start_server(port: int = 8000) -> None:
    """Start the live demo web server."""
    settings = get_settings()
    configure_logging(settings.log_level)

    server_address = ("0.0.0.0", port)
    httpd = ThreadedHTTPServer(server_address, LiveDemoHandler)
    logger.info("==================================================================")
    logger.info("🚀 Web Demo Server ĐANG CHẠY THẬT TẠI: http://localhost:%d", port)
    logger.info("👉 Mở trình duyệt và truy cập: http://localhost:%d", port)
    logger.info("==================================================================")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nĐang dừng Web Demo Server...")
        httpd.shutdown()


if __name__ == "__main__":
    port_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    start_server(port_arg)
