"""Logging utility for HelixForge.

Provides structured logging with correlation IDs for tracing
operations across the agent pipeline.
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional


# Configure root logger
_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_JSON_FORMAT = True


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "correlation_id"):
            log_data["correlation_id"] = record.correlation_id
        if hasattr(record, "agent"):
            log_data["agent"] = record.agent
        if hasattr(record, "event_type"):
            log_data["event_type"] = record.event_type

        # Add any other extra attributes
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "correlation_id", "agent", "event_type", "message"
            ):
                if not key.startswith("_"):
                    log_data[key] = value

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class StandardFormatter(logging.Formatter):
    """Standard text formatter with correlation ID support."""

    def format(self, record: logging.LogRecord) -> str:
        # Add correlation_id to message if present
        extra = ""
        if hasattr(record, "correlation_id"):
            extra += f" [corr_id={record.correlation_id}]"
        if hasattr(record, "agent"):
            extra += f" [agent={record.agent}]"

        base_format = f"%(asctime)s | %(levelname)-8s | %(name)s{extra} | %(message)s"
        formatter = logging.Formatter(base_format)
        return formatter.format(record)


def get_logger(
    name: str,
    level: str = "INFO",
    json_format: bool = True
) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name (typically __name__).
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: If True, use JSON formatting.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    if json_format:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(StandardFormatter())

    logger.addHandler(handler)
    logger.propagate = False

    return logger


def log_event(
    logger: logging.Logger,
    event_type: str,
    payload: Dict[str, Any],
    correlation_id: Optional[str] = None,
    level: str = "INFO"
) -> None:
    """Log a structured event.

    Args:
        logger: Logger instance.
        event_type: Type of event (e.g., 'data.ingested').
        payload: Event data.
        correlation_id: Optional correlation ID for tracing.
        level: Log level.
    """
    extra = {"event_type": event_type}
    if correlation_id:
        extra["correlation_id"] = correlation_id

    # Flatten payload into extra for structured logging
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            extra[key] = str(value) if value is not None else ""
        else:
            extra[key] = str(value)

    log_method = getattr(logger, level.lower())
    log_method(f"Event: {event_type}", extra=extra)


def log_metric(
    logger: logging.Logger,
    metric_name: str,
    value: float,
    tags: Optional[Dict[str, str]] = None,
    correlation_id: Optional[str] = None
) -> None:
    """Log a metric value.

    Args:
        logger: Logger instance.
        metric_name: Name of the metric.
        value: Metric value.
        tags: Optional tags for the metric.
        correlation_id: Optional correlation ID.
    """
    extra = {
        "metric_name": metric_name,
        "metric_value": value,
    }
    if tags:
        extra["tags"] = tags
    if correlation_id:
        extra["correlation_id"] = correlation_id

    logger.info(f"Metric: {metric_name}={value}", extra=extra)


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds correlation ID to all messages."""

    def __init__(
        self,
        logger: logging.Logger,
        correlation_id: str,
        agent: Optional[str] = None
    ):
        extra = {"correlation_id": correlation_id}
        if agent:
            extra["agent"] = agent
        super().__init__(logger, extra)

    def process(self, msg: str, kwargs):  # type: ignore[override]
        # Merge extra into kwargs
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


def get_correlated_logger(
    name: str,
    correlation_id: str,
    agent: Optional[str] = None
) -> LoggerAdapter:
    """Get a logger with correlation ID baked in.

    Args:
        name: Logger name.
        correlation_id: Correlation ID for all log messages.
        agent: Optional agent name.

    Returns:
        LoggerAdapter with correlation ID.
    """
    base_logger = get_logger(name)
    return LoggerAdapter(base_logger, correlation_id, agent)
