"""Structured JSON logging configuration."""
import json
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from apps.runtime.signal import Signal

from loguru import logger


class JSONFormatter:
    """JSON formatter for structured logging."""

    def __call__(self, record: dict[str, Any]) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record["level"].name,
            "message": record["message"],
            "module": record.get("module", ""),
            "function": record.get("function", ""),
            "line": record.get("line", 0),
        }

        # Add extra fields if present
        if "extra" in record:
            log_data.update(record["extra"])

        return json.dumps(log_data)


def setup_structured_logging(log_format: str = "json", log_level: str = "INFO") -> None:
    """
    Setup structured logging.

    Args:
        log_format: Log format ('json' or 'text')
        log_level: Log level (DEBUG, INFO, WARNING, ERROR)
    """
    # Remove default handler
    logger.remove()

    if log_format == "json":
        # Add JSON formatter
        logger.add(sys.stdout, format=JSONFormatter(), level=log_level, serialize=True)
    else:
        # Add text formatter
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=log_level,
        )

    # Also log to file
    logger.add(
        "logs/runtime_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level=log_level,
        format=JSONFormatter() if log_format == "json" else None,
        serialize=(log_format == "json"),
    )

    logger.info(f"Structured logging configured: format={log_format}, level={log_level}")


def log_signal(signal: "Signal") -> None:  # noqa: F821
    """
    Log a signal with structured fields.

    Args:
        signal: Signal object
    """
    logger.info(
        "Signal generated",
        extra={
            "event": "signal",
            "mode": signal.mode,
            "pair": signal.pair,
            "direction": signal.direction,
            "tier": signal.tier,
            "confidence": signal.confidence,
            "entry": signal.entry_price,
            "tp": signal.tp_price,
            "sl": signal.sl_price,
            "rr": signal.rr_ratio,
            "lat_ms": signal.latency_ms,
            "spread_bps": signal.spread_bps,
            "slip_bps": signal.slippage_bps,
        },
    )

