#!/usr/bin/env python3
"""Export runtime observation metrics to JSON."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func
logger = logging.getLogger("export_metrics")


from libs.config.config import Settings
from libs.db.models import Signal, get_session


def _serialize_datetime(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def export_metrics(window_hours: int, output_path: Path) -> dict[str, Any]:
    """Collect signal and risk metrics for the observation window."""
    settings = Settings()
    settings.validate()

    session = get_session(settings.db_url)

    window_start = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    # Signal timestamps are stored as naive UTC; normalize comparison
    cutoff = window_start.replace(tzinfo=None)

    signals = (
        session.query(Signal)
        .filter(Signal.timestamp >= cutoff)
        .order_by(Signal.timestamp.asc())
        .all()
    )

    total_signals = len(signals)
    by_tier: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    by_symbol: dict[str, int] = {}
    avg_confidence = 0.0

    for signal in signals:
        tier = signal.tier or "unknown"
        by_tier[tier] = by_tier.get(tier, 0) + 1
        by_symbol[signal.symbol] = by_symbol.get(signal.symbol, 0) + 1
        avg_confidence += signal.confidence or 0.0

    avg_confidence = avg_confidence / total_signals if total_signals else 0.0

    latency_stats = (
        session.query(
            func.avg(Signal.latency_ms),
            func.max(Signal.latency_ms),
        )
        .filter(Signal.timestamp >= cutoff)
        .first()
    )
    avg_latency_ms = float(latency_stats[0]) if latency_stats and latency_stats[0] else 0.0
    max_latency_ms = float(latency_stats[1]) if latency_stats and latency_stats[1] else 0.0

    metrics: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_hours": window_hours,
        "signal_sample_start": _serialize_datetime(signals[0].timestamp) if signals else None,
        "signal_sample_end": _serialize_datetime(signals[-1].timestamp) if signals else None,
        "totals": {
            "signals": total_signals,
            "avg_confidence": avg_confidence,
        },
        "by_tier": by_tier,
        "by_symbol": by_symbol,
        "latency_ms": {
            "avg": avg_latency_ms,
            "max": max_latency_ms,
        },
        "risk": {
            "snapshots_collected": 0,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    logger.info("Exported observation metrics to %s (signals=%s)", output_path, total_signals)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Export observation metrics to JSON")
    parser.add_argument(
        "--window",
        type=int,
        default=24,
        help="Number of hours to include in the sample window",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/phase6_5/metrics_latest.json"),
        help="Output JSON file",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    export_metrics(args.window, args.out)


if __name__ == "__main__":
    main()

