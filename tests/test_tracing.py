"""Unit tests for trace span nesting and local trace export."""

import json

from multi_agent_research_lab.observability import tracing


def test_trace_span_records_duration_and_nesting(tmp_path, monkeypatch) -> None:
    """Span con phải ghi nhận đúng span cha để trace hiện thành một cây."""
    trace_dir = tmp_path / "traces"
    monkeypatch.setattr(tracing, "TRACE_DIR", trace_dir)
    monkeypatch.setattr(tracing, "LOCAL_TRACE_FILE", trace_dir / "local_trace.jsonl")

    # Cố ý lồng `with`: đây chính là hành vi đang được kiểm tra, không gộp lại được.
    with tracing.trace_span("multi_agent_run", {"query": "q"}) as outer:  # noqa: SIM117
        with tracing.trace_span("researcher") as inner:
            pass

    assert outer["duration_seconds"] is not None
    assert inner["duration_seconds"] is not None

    rows = [json.loads(line) for line in (trace_dir / "local_trace.jsonl").read_text().splitlines()]
    by_name = {r["name"]: r for r in rows}
    assert by_name["researcher"]["parent_run_id"] == by_name["multi_agent_run"]["run_id"]
    assert by_name["multi_agent_run"]["parent_run_id"] is None


def test_trace_span_never_breaks_on_exception(tmp_path, monkeypatch) -> None:
    """Lỗi bên trong span vẫn phải được ghi trace rồi ném tiếp, không nuốt lỗi."""
    trace_dir = tmp_path / "traces"
    monkeypatch.setattr(tracing, "TRACE_DIR", trace_dir)
    monkeypatch.setattr(tracing, "LOCAL_TRACE_FILE", trace_dir / "local_trace.jsonl")

    try:
        with tracing.trace_span("boom"):
            raise ValueError("agent failed")
    except ValueError:
        pass

    rows = [json.loads(line) for line in (trace_dir / "local_trace.jsonl").read_text().splitlines()]
    assert rows[0]["name"] == "boom"
    assert rows[0]["duration_seconds"] is not None


def test_export_state_trace(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tracing, "TRACE_DIR", tmp_path / "traces")
    path = tracing.export_state_trace("run1", [{"name": "supervisor_route", "payload": {}}])
    assert path.exists()
    assert json.loads(path.read_text())[0]["name"] == "supervisor_route"
