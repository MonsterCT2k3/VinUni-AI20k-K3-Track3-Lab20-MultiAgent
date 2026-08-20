"""Tracing hooks and span lifecycle management."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Context manager to measure and log execution spans."""
    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    try:
        yield span
    finally:
        duration = perf_counter() - started
        span["duration_seconds"] = round(duration, 4)
        logger.debug("Trace span '%s' hoàn thành trong %.4fs", name, duration)
