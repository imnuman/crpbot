#!/usr/bin/env python3
"""Automate daily Phase 6.5 observation logging."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import boto3
except ImportError:  # pragma: no cover - boto3 should be available via project deps
    boto3 = None  # type: ignore[assignment]

# Ensure project root on path so we can import local modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_metrics import export_metrics  # type: ignore[import]

logger = logging.getLogger("phase6_5_daily")

DAY_FILES = {
    "day0": "day0.md",
    "day1": "day1.md",
    "day2": "day2.md",
    "day3": "day3.md",
    "day4": "day4.md",
    "day5": "day4.md",  # optional extension reuses day4 template
}


def _load_alarm_summary(region: str | None) -> dict[str, Any]:
    """Fetch CloudWatch alarm state summary."""
    if boto3 is None:
        logger.warning("boto3 unavailable; skipping alarm summary")
        return {"available": False, "note": "boto3 not installed in environment"}

    try:
        session = boto3.session.Session(region_name=region) if region else boto3.session.Session()
        client = session.client("cloudwatch")
        alarms = client.describe_alarms(StateValue="ALARM")
        names = [alarm["AlarmName"] for alarm in alarms.get("MetricAlarms", [])]
        return {"available": True, "count": len(names), "alarms": names}
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning("Unable to fetch alarm summary: %s", exc)
        return {"available": False, "error": str(exc)}


def _append_snapshot(day_file: Path, metrics: dict[str, Any], metrics_path: Path, alarm_summary: dict[str, Any]) -> None:
    """Append a formatted snapshot to the daily markdown file."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "",
        f"## Automated Snapshot ({now})",
        f"- Metrics window: {metrics['window_hours']}h → `{metrics_path.relative_to(PROJECT_ROOT)}`",
        f"- Signals produced: {metrics['totals']['signals']}",
        f"- Avg confidence: {metrics['totals']['avg_confidence']:.3f}",
        "- Tier distribution:",
    ]

    for tier, count in metrics["by_tier"].items():
        lines.append(f"  - {tier.title()}: {count}")

    lines.extend(
        [
            "- Symbols observed: " + (", ".join(sorted(metrics["by_symbol"])) if metrics["by_symbol"] else "None"),
            "- Latency (ms): avg "
            f"{metrics['latency_ms']['avg']:.2f}, max {metrics['latency_ms']['max']:.2f}",
        ]
    )

    if alarm_summary.get("available"):
        count = alarm_summary.get("count", 0)
        if count:
            alarms = ", ".join(alarm_summary.get("alarms", []))
            lines.append(f"- CloudWatch alarms in ALARM state: {count} → {alarms}")
        else:
            lines.append("- CloudWatch alarms in ALARM state: 0 ✅")
    else:
        note = alarm_summary.get("note") or alarm_summary.get("error", "Unavailable")
        lines.append(f"- CloudWatch alarm summary: unavailable ({note})")

    day_file.write_text(day_file.read_text() + "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6.5 daily automation helper")
    parser.add_argument("--day", required=True, help="Day identifier (day0-day4)")
    parser.add_argument("--window", type=int, default=24, help="Metrics window in hours (default 24)")
    parser.add_argument(
        "--region",
        help="AWS region override for CloudWatch queries (defaults to environment configuration)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    day_key = args.day.lower()
    if day_key not in DAY_FILES:
        raise ValueError(f"Unsupported day '{args.day}'. Expected one of: {', '.join(DAY_FILES)}")

    reports_dir = PROJECT_ROOT / "reports" / "phase6_5"
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = reports_dir / f"{day_key}_metrics.json"
    metrics = export_metrics(args.window, metrics_path)

    day_file = reports_dir / DAY_FILES[day_key]
    if not day_file.exists():
        raise FileNotFoundError(f"Daily log template missing: {day_file}")

    alarm_summary = _load_alarm_summary(args.region)

    _append_snapshot(day_file, metrics, metrics_path, alarm_summary)
    logger.info("Daily snapshot appended to %s", day_file.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()

