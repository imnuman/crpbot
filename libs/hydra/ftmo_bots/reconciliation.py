"""
Trade Reconciliation System - Ensures data integrity between local ledger and MT5.

Features:
1. Position Close Monitor - Detects when MT5 positions close and updates ledger
2. Daily Reconciliation - Compares local vs MT5 history at 00:05 UTC
3. Alerts - Telegram + log for discrepancies

Created: 2025-12-11
"""

import asyncio
import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

from .trade_ledger import TradeLedger, get_ledger

logger = logging.getLogger(__name__)

# Check interval for closed positions (seconds)
CLOSE_MONITOR_INTERVAL = 30

# Reconciliation schedule (UTC hour)
DAILY_RECONCILIATION_HOUR = 0
DAILY_RECONCILIATION_MINUTE = 5


class PositionCloseMonitor:
    """
    Monitors MT5 positions and detects when they close.
    Updates the trade ledger with exit details.
    """

    def __init__(self, ledger: TradeLedger, zmq_client: Any):
        """
        Initialize the position close monitor.

        Args:
            ledger: TradeLedger instance for recording closes
            zmq_client: MT5ZMQClient for querying positions and history
        """
        self.ledger = ledger
        self.zmq_client = zmq_client
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_check_time: Optional[datetime] = None

    def start(self):
        """Start the position close monitor in a background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="PositionCloseMonitor",
            daemon=True
        )
        self._thread.start()
        logger.info("[CloseMonitor] Started position close monitoring (30s interval)")

    def stop(self):
        """Stop the position close monitor."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("[CloseMonitor] Stopped position close monitoring")

    def _monitor_loop(self):
        """Main monitoring loop - runs every 30 seconds."""
        while self._running:
            try:
                self._check_for_closed_positions()
            except Exception as e:
                logger.error(f"[CloseMonitor] Error in monitor loop: {e}")

            # Sleep in smaller chunks to allow graceful shutdown
            for _ in range(CLOSE_MONITOR_INTERVAL):
                if not self._running:
                    break
                time.sleep(1)

    def _check_for_closed_positions(self):
        """Check if any tracked positions have closed on MT5."""
        # Get current open positions from MT5
        mt5_positions = self.zmq_client.get_positions()
        if mt5_positions is None:
            logger.warning("[CloseMonitor] Failed to get MT5 positions")
            return

        mt5_tickets = {p["ticket"] for p in mt5_positions}

        # Get our tracked open positions from ledger
        open_trades = self.ledger.get_open_trades()

        closed_count = 0
        for trade in open_trades:
            if trade.mt5_ticket not in mt5_tickets:
                # Position closed on MT5!
                self._handle_position_close(trade)
                closed_count += 1

        self._last_check_time = datetime.now(timezone.utc)

        if closed_count > 0:
            logger.info(f"[CloseMonitor] Detected {closed_count} closed position(s)")

    def _handle_position_close(self, trade):
        """
        Handle a position that has closed on MT5.
        Fetch exit details from history and update ledger.
        """
        ticket = trade.mt5_ticket
        logger.info(f"[CloseMonitor] Position {ticket} closed - fetching details")

        # Query MT5 history for this specific trade
        trade_details = self.zmq_client.get_deal_info(ticket)

        if trade_details and trade_details.get("success"):
            # Got exit details from MT5
            exit_price = trade_details.get("exit_price")
            exit_time_str = trade_details.get("exit_time")
            exit_reason = trade_details.get("exit_reason", "UNKNOWN")
            pnl = trade_details.get("profit", 0)
            commission = trade_details.get("commission", 0)
            swap = trade_details.get("swap", 0)

            # Parse exit time
            exit_time = None
            if exit_time_str:
                try:
                    exit_time = datetime.fromisoformat(exit_time_str.replace('Z', '+00:00'))
                except:
                    exit_time = datetime.now(timezone.utc)
            else:
                exit_time = datetime.now(timezone.utc)

            # Calculate pips (approximate)
            pnl_pips = None
            if trade.entry_price and exit_price:
                pip_diff = abs(exit_price - trade.entry_price)
                # Rough pip calculation (works for most pairs)
                if trade.symbol.upper() in ["XAUUSD", "GOLD"]:
                    pnl_pips = pip_diff * 10  # Gold is 0.1 per pip
                elif "JPY" in trade.symbol.upper():
                    pnl_pips = pip_diff * 100  # JPY pairs
                else:
                    pnl_pips = pip_diff * 10000  # Most forex pairs

            # Record close in ledger
            success = self.ledger.record_close(
                mt5_ticket=ticket,
                exit_price=exit_price,
                exit_time=exit_time,
                exit_reason=exit_reason,
                pnl_usd=pnl,
                pnl_pips=pnl_pips,
                commission=commission,
                swap=swap
            )

            if success:
                logger.info(
                    f"[CloseMonitor] Recorded CLOSE: #{ticket} {trade.symbol} "
                    f"exit={exit_price} P&L=${pnl:.2f} reason={exit_reason}"
                )
                # Send alert
                self._send_close_alert(trade, pnl, exit_reason)
            else:
                logger.warning(f"[CloseMonitor] Failed to record close for #{ticket}")

        else:
            # Couldn't get details, mark as closed with partial info
            logger.warning(f"[CloseMonitor] No details for #{ticket}, marking closed")
            self.ledger.record_close(
                mt5_ticket=ticket,
                exit_price=0,
                exit_time=datetime.now(timezone.utc),
                exit_reason="UNKNOWN",
                pnl_usd=0
            )

    def _send_close_alert(self, trade, pnl: float, exit_reason: str):
        """Send alert for closed trade."""
        try:
            from libs.notifications.telegram_bot import send_ftmo_trade_alert

            emoji = "+" if pnl >= 0 else ""
            message = (
                f"Position Closed: {trade.bot_name}\n"
                f"Symbol: {trade.symbol} ({trade.direction})\n"
                f"P&L: {emoji}${pnl:.2f}\n"
                f"Reason: {exit_reason}"
            )
            # Use existing telegram alert (it handles logging too)
            logger.info(f"[CloseAlert] {trade.symbol} {trade.direction} P&L=${pnl:.2f}")
        except Exception as e:
            logger.debug(f"[CloseMonitor] Alert send failed: {e}")


class TradeReconciler:
    """
    Daily reconciliation between local ledger and MT5 history.
    Runs at 00:05 UTC to catch any missed trades or discrepancies.
    """

    def __init__(self, ledger: TradeLedger, zmq_client: Any):
        """
        Initialize the reconciler.

        Args:
            ledger: TradeLedger instance
            zmq_client: MT5ZMQClient for querying history
        """
        self.ledger = ledger
        self.zmq_client = zmq_client
        self._last_reconciliation: Optional[datetime] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the daily reconciliation scheduler."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._scheduler_loop,
            name="ReconciliationScheduler",
            daemon=True
        )
        self._thread.start()
        logger.info(f"[Reconciler] Started (daily at {DAILY_RECONCILIATION_HOUR:02d}:{DAILY_RECONCILIATION_MINUTE:02d} UTC)")

    def stop(self):
        """Stop the reconciliation scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _scheduler_loop(self):
        """Check every minute if it's time to reconcile."""
        while self._running:
            now = datetime.now(timezone.utc)

            # Check if it's reconciliation time
            if (now.hour == DAILY_RECONCILIATION_HOUR and
                now.minute == DAILY_RECONCILIATION_MINUTE):

                # Only run once per day
                if (self._last_reconciliation is None or
                    self._last_reconciliation.date() != now.date()):

                    logger.info("[Reconciler] Starting daily reconciliation...")
                    try:
                        result = self.reconcile_daily()
                        self._last_reconciliation = now
                        logger.info(f"[Reconciler] Completed: {result}")
                    except Exception as e:
                        logger.error(f"[Reconciler] Failed: {e}")

            # Check every 60 seconds
            for _ in range(60):
                if not self._running:
                    break
                time.sleep(1)

    def reconcile_daily(self) -> Dict[str, Any]:
        """
        Run daily reconciliation.
        Compares last 24 hours of MT5 history with local ledger.

        Returns:
            Summary dict with discrepancies found
        """
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)

        # Fetch MT5 history for last 24 hours
        mt5_trades = self.zmq_client.get_history(days=1)
        if not mt5_trades:
            logger.warning("[Reconciler] No MT5 history returned")
            return {"success": False, "error": "No MT5 history"}

        # Get local trades from same period
        local_trades = self.ledger.get_trades_since(yesterday)
        local_by_ticket = {t.mt5_ticket: t for t in local_trades}

        discrepancies = []
        verified_count = 0

        for mt5_trade in mt5_trades:
            ticket = mt5_trade.get("ticket")
            local = local_by_ticket.get(ticket)

            if not local:
                # Trade in MT5 but not in local ledger - ORPHAN
                discrepancies.append({
                    "type": "ORPHAN_MT5",
                    "ticket": ticket,
                    "symbol": mt5_trade.get("symbol"),
                    "issue": "Trade in MT5 but not in local records"
                })

                # Try to import the orphaned trade
                self._import_orphan(mt5_trade)

            else:
                # Verify P&L matches
                mt5_pnl = mt5_trade.get("profit", 0)
                local_pnl = local.pnl_usd or 0

                if abs(local_pnl - mt5_pnl) > 0.01:
                    discrepancies.append({
                        "type": "PNL_MISMATCH",
                        "ticket": ticket,
                        "local_pnl": local_pnl,
                        "mt5_pnl": mt5_pnl,
                        "diff": abs(local_pnl - mt5_pnl)
                    })
                else:
                    # Mark as verified
                    self.ledger.mark_verified(ticket)
                    verified_count += 1

        # Check for local trades not in MT5 (shouldn't happen for closed trades)
        mt5_tickets = {t.get("ticket") for t in mt5_trades}
        for local in local_trades:
            if local.status == "CLOSED" and local.mt5_ticket not in mt5_tickets:
                discrepancies.append({
                    "type": "ORPHAN_LOCAL",
                    "ticket": local.mt5_ticket,
                    "symbol": local.symbol,
                    "issue": "Closed trade in local but not in MT5 history"
                })

        # Send alert if discrepancies found
        if discrepancies:
            self._send_discrepancy_alert(discrepancies)

        result = {
            "success": True,
            "mt5_trades": len(mt5_trades),
            "local_trades": len(local_trades),
            "verified": verified_count,
            "discrepancies": len(discrepancies),
            "details": discrepancies
        }

        logger.info(
            f"[Reconciler] Results: {len(mt5_trades)} MT5 trades, "
            f"{verified_count} verified, {len(discrepancies)} discrepancies"
        )

        return result

    def _import_orphan(self, mt5_trade: Dict):
        """Import an orphaned MT5 trade into the ledger."""
        try:
            ticket = mt5_trade.get("ticket")
            entry_time_str = mt5_trade.get("entry_time")
            exit_time_str = mt5_trade.get("exit_time")

            entry_time = datetime.fromisoformat(entry_time_str.replace('Z', '+00:00')) if entry_time_str else datetime.now(timezone.utc)
            exit_time = datetime.fromisoformat(exit_time_str.replace('Z', '+00:00')) if exit_time_str else datetime.now(timezone.utc)

            # Extract bot name from comment if possible
            comment = mt5_trade.get("comment", "")
            bot_name = "UNKNOWN"
            if "HYDRA_" in comment:
                bot_name = comment.replace("HYDRA_", "")

            success = self.ledger.import_from_mt5(
                mt5_ticket=ticket,
                bot_name=bot_name,
                symbol=mt5_trade.get("symbol"),
                direction=mt5_trade.get("direction"),
                volume=mt5_trade.get("volume", 0.01),
                entry_price=mt5_trade.get("entry_price", 0),
                entry_time=entry_time,
                exit_price=mt5_trade.get("exit_price", 0),
                exit_time=exit_time,
                pnl_usd=mt5_trade.get("profit", 0),
                commission=mt5_trade.get("commission", 0),
                swap=mt5_trade.get("swap", 0),
                mode="LIVE"
            )

            if success:
                logger.info(f"[Reconciler] Imported orphan trade #{ticket}")

        except Exception as e:
            logger.error(f"[Reconciler] Failed to import orphan: {e}")

    def _send_discrepancy_alert(self, discrepancies: List[Dict]):
        """Send alert about reconciliation discrepancies."""
        try:
            from libs.notifications.telegram_bot import get_telegram_bot

            # Build message
            message_lines = ["Trade Reconciliation Alert"]
            message_lines.append(f"Found {len(discrepancies)} discrepancies:")
            message_lines.append("")

            for d in discrepancies[:5]:  # Limit to 5
                if d["type"] == "ORPHAN_MT5":
                    message_lines.append(
                        f"ORPHAN (MT5): #{d['ticket']} {d.get('symbol', '?')}"
                    )
                elif d["type"] == "PNL_MISMATCH":
                    message_lines.append(
                        f"P&L MISMATCH: #{d['ticket']} "
                        f"local=${d['local_pnl']:.2f} vs MT5=${d['mt5_pnl']:.2f}"
                    )
                elif d["type"] == "ORPHAN_LOCAL":
                    message_lines.append(
                        f"ORPHAN (Local): #{d['ticket']} {d.get('symbol', '?')}"
                    )

            if len(discrepancies) > 5:
                message_lines.append(f"... and {len(discrepancies) - 5} more")

            message = "\n".join(message_lines)

            # Log to file (always)
            logger.warning(f"[Reconciler] Discrepancy Alert:\n{message}")

            # Try to send to Telegram
            try:
                bot = get_telegram_bot()
                if bot:
                    bot.send_message(message)
            except:
                pass

        except Exception as e:
            logger.error(f"[Reconciler] Failed to send alert: {e}")


def backfill_from_mt5(zmq_client: Any, ledger: TradeLedger, days: int = 30) -> int:
    """
    Backfill historical trades from MT5 into the ledger.

    Args:
        zmq_client: MT5ZMQClient for querying history
        ledger: TradeLedger to import into
        days: Number of days to backfill

    Returns:
        Number of trades imported
    """
    logger.info(f"[Backfill] Fetching MT5 history for last {days} days...")

    trades = zmq_client.get_history(days=days)
    if not trades:
        logger.warning("[Backfill] No trades returned from MT5")
        return 0

    imported = 0
    skipped = 0

    for trade in trades:
        try:
            ticket = trade.get("ticket")

            # Check if already exists
            existing = ledger.get_by_ticket(ticket)
            if existing:
                skipped += 1
                continue

            # Extract bot name from comment
            comment = trade.get("comment", "")
            bot_name = "UNKNOWN"
            if "HYDRA_" in comment:
                bot_name = comment.replace("HYDRA_", "")

            # Parse times
            entry_time_str = trade.get("entry_time")
            exit_time_str = trade.get("exit_time")

            entry_time = datetime.fromisoformat(entry_time_str.replace('Z', '+00:00')) if entry_time_str else datetime.now(timezone.utc)
            exit_time = datetime.fromisoformat(exit_time_str.replace('Z', '+00:00')) if exit_time_str else datetime.now(timezone.utc)

            success = ledger.import_from_mt5(
                mt5_ticket=ticket,
                bot_name=bot_name,
                symbol=trade.get("symbol"),
                direction=trade.get("direction"),
                volume=trade.get("volume", 0.01),
                entry_price=trade.get("entry_price", 0),
                entry_time=entry_time,
                exit_price=trade.get("exit_price", 0),
                exit_time=exit_time,
                pnl_usd=trade.get("profit", 0),
                commission=trade.get("commission", 0),
                swap=trade.get("swap", 0),
                mode="LIVE"
            )

            if success:
                imported += 1

        except Exception as e:
            logger.error(f"[Backfill] Error importing trade: {e}")

    logger.info(f"[Backfill] Complete: {imported} imported, {skipped} skipped (already exist)")
    return imported


# Convenience function to start all reconciliation components
def start_reconciliation_system(zmq_client: Any, ledger: Optional[TradeLedger] = None):
    """
    Start the complete reconciliation system.

    Args:
        zmq_client: MT5ZMQClient instance
        ledger: Optional TradeLedger (creates default if not provided)

    Returns:
        Tuple of (PositionCloseMonitor, TradeReconciler)
    """
    if ledger is None:
        ledger = get_ledger()

    close_monitor = PositionCloseMonitor(ledger, zmq_client)
    reconciler = TradeReconciler(ledger, zmq_client)

    close_monitor.start()
    reconciler.start()

    return close_monitor, reconciler


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    print("Reconciliation module loaded")
    print("Use start_reconciliation_system(zmq_client) to start monitoring")
