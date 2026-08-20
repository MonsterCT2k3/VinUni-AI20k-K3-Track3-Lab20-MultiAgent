"""Tracing hooks and span lifecycle management.

Mỗi span luôn được ghi vào file JSONL cục bộ dưới `reports/traces/`, nên một lần chạy luôn
có bằng chứng kiểm tra được kể cả khi hoàn toàn offline. Khi `LANGSMITH_API_KEY` được cấu
hình, span còn được đẩy lên LangSmith để xem trên UI.

**Lồng span (nesting):** một `trace_span` gọi bên trong `trace_span` khác sẽ trở thành *con*
của span ngoài. Hãy bọc trọn một lần chạy workflow trong một span cha
`trace_span("multi_agent_run", ...)` để toàn bộ chuỗi Supervisor → Researcher → Analyst →
Writer hiện lên thành **một cây duy nhất** thay vì các span rời rạc.

Tracing không bao giờ được phép làm hỏng workflow: mọi lỗi của provider đều bị nuốt và chỉ
ghi log cảnh báo.
"""

import json
import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

TRACE_DIR = Path("reports") / "traces"
LOCAL_TRACE_FILE = TRACE_DIR / "local_trace.jsonl"

# Theo dõi span đang hoạt động để span con gắn đúng vào span cha.
_current_run_id: ContextVar[str | None] = ContextVar("current_run_id", default=None)


def _langsmith_enabled() -> bool:
    return bool(get_settings().langsmith_api_key)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Đo thời gian một bước chạy, ghi trace cục bộ và (tuỳ chọn) đẩy lên LangSmith."""
    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}

    # Sinh id vô điều kiện để quan hệ cha-con hiện đúng cả trong trace cục bộ (offline),
    # không chỉ khi LangSmith được bật.
    parent_run_id = _current_run_id.get()
    run_id = str(uuid.uuid4())
    langsmith_run_id = (
        _start_langsmith_run(name, span, parent_run_id, run_id) if _langsmith_enabled() else None
    )
    token = _current_run_id.set(run_id)

    try:
        yield span
    finally:
        span["duration_seconds"] = round(perf_counter() - started, 4)
        logger.debug("Trace span '%s' hoàn thành trong %.4fs", name, span["duration_seconds"])
        _record_local(span, run_id, parent_run_id)
        if langsmith_run_id is not None:
            _end_langsmith_run(langsmith_run_id, span)
        _current_run_id.reset(token)


def _record_local(span: dict[str, Any], run_id: str | None, parent_run_id: str | None) -> None:
    """Ghi span ra file JSONL cục bộ (best-effort, không bao giờ làm hỏng workflow)."""
    try:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            **span,
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        with LOCAL_TRACE_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:  # pragma: no cover - tracing là best-effort
        logger.warning("Không ghi được trace cục bộ: %s", exc)


def _start_langsmith_run(
    name: str, span: dict[str, Any], parent_run_id: str | None, run_id: str
) -> str | None:
    try:
        from langsmith import Client

        settings = get_settings()
        Client(api_key=settings.langsmith_api_key).create_run(
            id=run_id,
            name=name,
            run_type="chain",
            inputs=span.get("attributes", {}),
            parent_run_id=parent_run_id,
            project_name=settings.langsmith_project,
        )
        return run_id
    except Exception as exc:  # pragma: no cover - tracing không được làm hỏng workflow
        logger.warning("Không khởi tạo được LangSmith run: %s", exc)
        return None


def _end_langsmith_run(run_id: str, span: dict[str, Any]) -> None:
    try:
        from langsmith import Client

        Client(api_key=get_settings().langsmith_api_key).update_run(
            run_id, outputs={"duration_seconds": span.get("duration_seconds")}
        )
    except Exception as exc:  # pragma: no cover - tracing không được làm hỏng workflow
        logger.warning("Không đóng được LangSmith run: %s", exc)


def export_state_trace(run_name: str, trace: list[dict[str, Any]]) -> Path:
    """Ghi toàn bộ `ResearchState.trace` của một lần chạy ra một file JSON."""
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    path = TRACE_DIR / f"{run_name}.json"
    path.write_text(json.dumps(trace, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
