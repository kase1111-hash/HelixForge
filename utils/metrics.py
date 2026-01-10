"""Metrics and telemetry for HelixForge.

Provides Prometheus metrics for monitoring performance,
tracking operations, and alerting on issues.
"""

import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Optional

# Try to import prometheus_client, gracefully handle if not available
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        multiprocess,
        REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# Default registry
_registry = REGISTRY if PROMETHEUS_AVAILABLE else None


# Application info
if PROMETHEUS_AVAILABLE:
    APP_INFO = Info(
        "helixforge",
        "HelixForge application information",
        registry=_registry
    )
    APP_INFO.info({
        "version": "1.0.0",
        "name": "HelixForge",
        "description": "Cross-Dataset Insight Synthesizer"
    })


# Request metrics
if PROMETHEUS_AVAILABLE:
    REQUEST_COUNT = Counter(
        "helixforge_requests_total",
        "Total number of requests",
        ["method", "endpoint", "status"],
        registry=_registry
    )

    REQUEST_LATENCY = Histogram(
        "helixforge_request_latency_seconds",
        "Request latency in seconds",
        ["method", "endpoint"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        registry=_registry
    )


# Agent metrics
if PROMETHEUS_AVAILABLE:
    AGENT_OPERATIONS = Counter(
        "helixforge_agent_operations_total",
        "Total agent operations",
        ["agent", "operation", "status"],
        registry=_registry
    )

    AGENT_LATENCY = Histogram(
        "helixforge_agent_latency_seconds",
        "Agent operation latency in seconds",
        ["agent", "operation"],
        buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
        registry=_registry
    )

    AGENT_ACTIVE = Gauge(
        "helixforge_agent_active",
        "Number of active agent operations",
        ["agent"],
        registry=_registry
    )


# Data metrics
if PROMETHEUS_AVAILABLE:
    DATASETS_INGESTED = Counter(
        "helixforge_datasets_ingested_total",
        "Total datasets ingested",
        ["source_type"],
        registry=_registry
    )

    ROWS_PROCESSED = Counter(
        "helixforge_rows_processed_total",
        "Total rows processed",
        ["operation"],
        registry=_registry
    )

    DATASET_SIZE = Histogram(
        "helixforge_dataset_size_rows",
        "Dataset size in rows",
        ["source_type"],
        buckets=[100, 1000, 10000, 100000, 1000000, 10000000],
        registry=_registry
    )


# LLM metrics
if PROMETHEUS_AVAILABLE:
    LLM_REQUESTS = Counter(
        "helixforge_llm_requests_total",
        "Total LLM API requests",
        ["provider", "model", "status"],
        registry=_registry
    )

    LLM_TOKENS = Counter(
        "helixforge_llm_tokens_total",
        "Total LLM tokens used",
        ["provider", "model", "type"],
        registry=_registry
    )

    LLM_LATENCY = Histogram(
        "helixforge_llm_latency_seconds",
        "LLM request latency in seconds",
        ["provider", "model"],
        buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
        registry=_registry
    )


# Fusion metrics
if PROMETHEUS_AVAILABLE:
    FUSIONS_EXECUTED = Counter(
        "helixforge_fusions_total",
        "Total dataset fusions executed",
        ["strategy", "status"],
        registry=_registry
    )

    ALIGNMENT_SCORE = Histogram(
        "helixforge_alignment_score",
        "Schema alignment scores",
        ["alignment_type"],
        buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
        registry=_registry
    )


# Error metrics
if PROMETHEUS_AVAILABLE:
    ERRORS = Counter(
        "helixforge_errors_total",
        "Total errors",
        ["type", "component"],
        registry=_registry
    )


def get_metrics() -> bytes:
    """Get Prometheus metrics output.

    Returns:
        Metrics in Prometheus format.
    """
    if not PROMETHEUS_AVAILABLE:
        return b"# Prometheus client not installed\n"
    return generate_latest(_registry)


def get_metrics_content_type() -> str:
    """Get content type for metrics endpoint.

    Returns:
        Content type string.
    """
    if not PROMETHEUS_AVAILABLE:
        return "text/plain"
    return CONTENT_TYPE_LATEST


@contextmanager
def track_time(histogram, labels: dict):
    """Context manager to track operation time.

    Args:
        histogram: Prometheus histogram.
        labels: Label values for the metric.

    Yields:
        None
    """
    if not PROMETHEUS_AVAILABLE:
        yield
        return

    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        histogram.labels(**labels).observe(duration)


def track_request(method: str, endpoint: str, status: int, duration: float) -> None:
    """Track an API request.

    Args:
        method: HTTP method.
        endpoint: API endpoint.
        status: HTTP status code.
        duration: Request duration in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return

    REQUEST_COUNT.labels(
        method=method,
        endpoint=endpoint,
        status=str(status)
    ).inc()

    REQUEST_LATENCY.labels(
        method=method,
        endpoint=endpoint
    ).observe(duration)


def track_agent_operation(
    agent: str,
    operation: str,
    status: str = "success",
    duration: Optional[float] = None
) -> None:
    """Track an agent operation.

    Args:
        agent: Agent name.
        operation: Operation name.
        status: Operation status (success/error).
        duration: Operation duration in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return

    AGENT_OPERATIONS.labels(
        agent=agent,
        operation=operation,
        status=status
    ).inc()

    if duration is not None:
        AGENT_LATENCY.labels(
            agent=agent,
            operation=operation
        ).observe(duration)


def track_dataset_ingestion(source_type: str, row_count: int) -> None:
    """Track dataset ingestion.

    Args:
        source_type: Type of data source.
        row_count: Number of rows ingested.
    """
    if not PROMETHEUS_AVAILABLE:
        return

    DATASETS_INGESTED.labels(source_type=source_type).inc()
    ROWS_PROCESSED.labels(operation="ingest").inc(row_count)
    DATASET_SIZE.labels(source_type=source_type).observe(row_count)


def track_llm_request(
    provider: str,
    model: str,
    status: str,
    duration: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0
) -> None:
    """Track an LLM API request.

    Args:
        provider: LLM provider (openai, anthropic, etc.).
        model: Model name.
        status: Request status (success/error).
        duration: Request duration in seconds.
        prompt_tokens: Number of prompt tokens.
        completion_tokens: Number of completion tokens.
    """
    if not PROMETHEUS_AVAILABLE:
        return

    LLM_REQUESTS.labels(
        provider=provider,
        model=model,
        status=status
    ).inc()

    LLM_LATENCY.labels(
        provider=provider,
        model=model
    ).observe(duration)

    if prompt_tokens > 0:
        LLM_TOKENS.labels(
            provider=provider,
            model=model,
            type="prompt"
        ).inc(prompt_tokens)

    if completion_tokens > 0:
        LLM_TOKENS.labels(
            provider=provider,
            model=model,
            type="completion"
        ).inc(completion_tokens)


def track_fusion(strategy: str, status: str = "success") -> None:
    """Track a fusion operation.

    Args:
        strategy: Fusion strategy used.
        status: Operation status.
    """
    if not PROMETHEUS_AVAILABLE:
        return

    FUSIONS_EXECUTED.labels(strategy=strategy, status=status).inc()


def track_alignment_score(alignment_type: str, score: float) -> None:
    """Track an alignment score.

    Args:
        alignment_type: Type of alignment.
        score: Alignment score (0-1).
    """
    if not PROMETHEUS_AVAILABLE:
        return

    ALIGNMENT_SCORE.labels(alignment_type=alignment_type).observe(score)


def track_error(error_type: str, component: str) -> None:
    """Track an error.

    Args:
        error_type: Type of error.
        component: Component where error occurred.
    """
    if not PROMETHEUS_AVAILABLE:
        return

    ERRORS.labels(type=error_type, component=component).inc()


def metrics_middleware(func: Callable) -> Callable:
    """Decorator to add metrics tracking to a function.

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function with metrics tracking.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        if not PROMETHEUS_AVAILABLE:
            return await func(*args, **kwargs)

        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            status = "success"
            return result
        except Exception as e:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            track_agent_operation(
                agent=func.__module__,
                operation=func.__name__,
                status=status,
                duration=duration
            )

    return wrapper
