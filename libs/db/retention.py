"""Database retention policy and archival utilities."""
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

from libs.db.database import get_database
from libs.db.models import RiskBookSnapshot


def archive_dryrun_trades(archive_after_days: int = 90) -> int:
    """
    Archive dry-run trades older than specified days.

    Args:
        archive_after_days: Days to keep dry-run trades (default: 90)

    Returns:
        Number of trades archived
    """
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=archive_after_days)

    db = get_database()
    with db.get_session() as session:
        # Count trades to archive
        count = (
            session.query(RiskBookSnapshot)
            .filter(RiskBookSnapshot.mode == "dryrun")
            .filter(RiskBookSnapshot.entry_time < cutoff_date)
            .count()
        )

        if count == 0:
            logger.info(f"No dry-run trades to archive (cutoff: {cutoff_date})")
            return 0

        # Delete old dry-run trades
        deleted = (
            session.query(RiskBookSnapshot)
            .filter(RiskBookSnapshot.mode == "dryrun")
            .filter(RiskBookSnapshot.entry_time < cutoff_date)
            .delete()
        )

        session.commit()
        logger.info(f"Archived {deleted} dry-run trades older than {archive_after_days} days")

        return deleted


def export_to_parquet(
    output_path: str,
    days: int | None = None,
    mode: str | None = None,
    pair: str | None = None,
) -> None:
    """
    Export trades to Parquet for long-term analysis.

    Args:
        output_path: Output file path
        days: Number of days to export (None = all)
        mode: Filter by mode ('dryrun' or 'live')
        pair: Filter by pair
    """
    import pandas as pd

    db = get_database()

    cutoff_date = None
    if days:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

    with db.get_session() as session:
        query = session.query(RiskBookSnapshot)

        if cutoff_date:
            query = query.filter(RiskBookSnapshot.entry_time >= cutoff_date)
        if mode:
            query = query.filter(RiskBookSnapshot.mode == mode)
        if pair:
            query = query.filter(RiskBookSnapshot.pair == pair)

        trades = query.all()

        if not trades:
            logger.warning("No trades found to export")
            return

        # Convert to DataFrame
        data = []
        for trade in trades:
            data.append(
                {
                    "signal_id": trade.signal_id,
                    "pair": trade.pair,
                    "tier": trade.tier,
                    "entry_time": trade.entry_time,
                    "entry_price": trade.entry_price,
                    "tp_price": trade.tp_price,
                    "sl_price": trade.sl_price,
                    "rr_expected": trade.rr_expected,
                    "result": trade.result,
                    "exit_time": trade.exit_time,
                    "exit_price": trade.exit_price,
                    "r_realized": trade.r_realized,
                    "time_to_tp_sl_seconds": trade.time_to_tp_sl_seconds,
                    "slippage_bps": trade.slippage_bps,
                    "slippage_expected_bps": trade.slippage_expected_bps,
                    "spread_bps": trade.spread_bps,
                    "latency_ms": trade.latency_ms,
                    "mode": trade.mode,
                    "created_at": trade.created_at,
                }
            )

        df = pd.DataFrame(data)
        df.to_parquet(output_path, index=False)
        logger.info(f"Exported {len(df)} trades to {output_path}")

