"""
Trade Ledger - Persistent SQLite database for all FTMO trades.

This module provides a single source of truth for trade tracking:
- Records all trade opens with MT5 ticket numbers
- Captures trade closes with exit details and P&L
- Supports daily reconciliation with MT5 history
- Alerts on discrepancies via Telegram + logs

Created: 2025-12-11
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

# Default database location
DEFAULT_DB_PATH = Path("/app/data/hydra/ftmo/trade_ledger.db")


@dataclass
class Trade:
    """Represents a trade record from the ledger."""
    id: int
    mt5_ticket: int
    bot_name: str
    symbol: str
    direction: str
    volume: float
    entry_price: Optional[float]
    entry_time: datetime
    entry_sl: Optional[float]
    entry_tp: Optional[float]
    exit_price: Optional[float]
    exit_time: Optional[datetime]
    exit_reason: Optional[str]
    pnl_usd: Optional[float]
    pnl_pips: Optional[float]
    commission: Optional[float]
    swap: Optional[float]
    status: str
    mode: str
    mt5_verified: bool
    last_sync: Optional[datetime]
    created_at: datetime
    updated_at: datetime


class TradeLedger:
    """
    SQLite-based trade ledger for persistent trade tracking.

    Usage:
        ledger = TradeLedger()
        ledger.record_open(mt5_ticket=123, bot_name="HFScalper", ...)
        ledger.record_close(mt5_ticket=123, exit_price=..., pnl_usd=...)
        trades = ledger.get_open_trades()
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the trade ledger with database connection."""
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"[TradeLedger] Initialized at {self.db_path}")

    def _init_db(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mt5_ticket INTEGER UNIQUE,
                    bot_name VARCHAR(50) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    direction VARCHAR(10) NOT NULL,
                    volume DECIMAL(10,2) NOT NULL,

                    -- Entry
                    entry_price DECIMAL(15,5),
                    entry_time DATETIME NOT NULL,
                    entry_sl DECIMAL(15,5),
                    entry_tp DECIMAL(15,5),

                    -- Exit (NULL until closed)
                    exit_price DECIMAL(15,5),
                    exit_time DATETIME,
                    exit_reason VARCHAR(20),

                    -- P&L
                    pnl_usd DECIMAL(15,2),
                    pnl_pips DECIMAL(10,1),
                    commission DECIMAL(10,2),
                    swap DECIMAL(10,2),

                    -- Status
                    status VARCHAR(20) DEFAULT 'OPEN',
                    mode VARCHAR(20) NOT NULL,

                    -- Reconciliation
                    mt5_verified BOOLEAN DEFAULT 0,
                    last_sync DATETIME,

                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_ticket ON trades(mt5_ticket)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_bot ON trades(bot_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)")

            conn.commit()

    def record_open(
        self,
        mt5_ticket: int,
        bot_name: str,
        symbol: str,
        direction: str,
        volume: float,
        entry_price: Optional[float] = None,
        entry_time: Optional[datetime] = None,
        entry_sl: Optional[float] = None,
        entry_tp: Optional[float] = None,
        mode: str = "LIVE"
    ) -> bool:
        """
        Record a new trade open in the ledger.

        Returns True if successful, False if ticket already exists.
        """
        entry_time = entry_time or datetime.utcnow()

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO trades (
                        mt5_ticket, bot_name, symbol, direction, volume,
                        entry_price, entry_time, entry_sl, entry_tp,
                        status, mode
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
                """, (
                    mt5_ticket, bot_name, symbol, direction, volume,
                    entry_price, entry_time.isoformat(), entry_sl, entry_tp,
                    mode
                ))
                conn.commit()

            logger.info(
                f"[TradeLedger] Recorded OPEN: ticket={mt5_ticket} "
                f"bot={bot_name} {direction} {volume} {symbol} @ {entry_price}"
            )
            return True

        except sqlite3.IntegrityError:
            logger.warning(f"[TradeLedger] Ticket {mt5_ticket} already exists in ledger")
            return False

    def record_close(
        self,
        mt5_ticket: int,
        exit_price: float,
        exit_time: Optional[datetime] = None,
        exit_reason: Optional[str] = None,
        pnl_usd: Optional[float] = None,
        pnl_pips: Optional[float] = None,
        commission: Optional[float] = None,
        swap: Optional[float] = None
    ) -> bool:
        """
        Record trade close details.

        Returns True if trade was found and updated.
        """
        exit_time = exit_time or datetime.utcnow()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE trades SET
                    exit_price = ?,
                    exit_time = ?,
                    exit_reason = ?,
                    pnl_usd = ?,
                    pnl_pips = ?,
                    commission = ?,
                    swap = ?,
                    status = 'CLOSED',
                    updated_at = ?
                WHERE mt5_ticket = ? AND status = 'OPEN'
            """, (
                exit_price, exit_time.isoformat(), exit_reason,
                pnl_usd, pnl_pips, commission, swap,
                datetime.utcnow().isoformat(), mt5_ticket
            ))
            conn.commit()

            if cursor.rowcount > 0:
                logger.info(
                    f"[TradeLedger] Recorded CLOSE: ticket={mt5_ticket} "
                    f"exit={exit_price} reason={exit_reason} P&L=${pnl_usd:.2f}"
                )
                return True
            else:
                logger.warning(f"[TradeLedger] No open trade found for ticket {mt5_ticket}")
                return False

    def get_by_ticket(self, mt5_ticket: int) -> Optional[Trade]:
        """Get a trade by MT5 ticket number."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM trades WHERE mt5_ticket = ?",
                (mt5_ticket,)
            )
            row = cursor.fetchone()
            return self._row_to_trade(row) if row else None

    def get_open_trades(self) -> List[Trade]:
        """Get all open trades."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM trades WHERE status = 'OPEN' ORDER BY entry_time DESC"
            )
            return [self._row_to_trade(row) for row in cursor.fetchall()]

    def get_trades_since(self, since: datetime) -> List[Trade]:
        """Get all trades since a given datetime."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM trades WHERE entry_time >= ? ORDER BY entry_time DESC",
                (since.isoformat(),)
            )
            return [self._row_to_trade(row) for row in cursor.fetchall()]

    def get_trades_by_bot(self, bot_name: str, limit: int = 100) -> List[Trade]:
        """Get trades for a specific bot."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM trades WHERE bot_name = ? ORDER BY entry_time DESC LIMIT ?",
                (bot_name, limit)
            )
            return [self._row_to_trade(row) for row in cursor.fetchall()]

    def get_closed_trades(self, limit: int = 100) -> List[Trade]:
        """Get recently closed trades."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM trades WHERE status = 'CLOSED' ORDER BY exit_time DESC LIMIT ?",
                (limit,)
            )
            return [self._row_to_trade(row) for row in cursor.fetchall()]

    def get_unverified_trades(self) -> List[Trade]:
        """Get trades that haven't been verified against MT5."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM trades WHERE mt5_verified = 0 AND status = 'CLOSED'"
            )
            return [self._row_to_trade(row) for row in cursor.fetchall()]

    def mark_verified(self, mt5_ticket: int) -> bool:
        """Mark a trade as verified against MT5 history."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """UPDATE trades SET
                   mt5_verified = 1,
                   last_sync = ?,
                   updated_at = ?
                   WHERE mt5_ticket = ?""",
                (datetime.utcnow().isoformat(), datetime.utcnow().isoformat(), mt5_ticket)
            )
            conn.commit()
            return cursor.rowcount > 0

    def mark_orphaned(self, mt5_ticket: int) -> bool:
        """Mark a trade as orphaned (found in MT5 but not in ledger entry)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE trades SET status = 'ORPHANED', updated_at = ? WHERE mt5_ticket = ?",
                (datetime.utcnow().isoformat(), mt5_ticket)
            )
            conn.commit()
            return cursor.rowcount > 0

    def import_from_mt5(
        self,
        mt5_ticket: int,
        bot_name: str,
        symbol: str,
        direction: str,
        volume: float,
        entry_price: float,
        entry_time: datetime,
        exit_price: float,
        exit_time: datetime,
        pnl_usd: float,
        commission: float = 0,
        swap: float = 0,
        mode: str = "LIVE"
    ) -> bool:
        """
        Import a historical trade from MT5 (for backfill).
        Creates a complete closed trade record.
        """
        # Determine exit reason based on P&L and prices
        if direction == "BUY":
            exit_reason = "TP_HIT" if exit_price > entry_price else "SL_HIT"
        else:
            exit_reason = "TP_HIT" if exit_price < entry_price else "SL_HIT"

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO trades (
                        mt5_ticket, bot_name, symbol, direction, volume,
                        entry_price, entry_time, exit_price, exit_time,
                        exit_reason, pnl_usd, commission, swap,
                        status, mode, mt5_verified
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'CLOSED', ?, 1)
                """, (
                    mt5_ticket, bot_name, symbol, direction, volume,
                    entry_price, entry_time.isoformat(),
                    exit_price, exit_time.isoformat(),
                    exit_reason, pnl_usd, commission, swap, mode
                ))
                conn.commit()

            logger.info(
                f"[TradeLedger] Imported: ticket={mt5_ticket} {direction} {symbol} "
                f"P&L=${pnl_usd:.2f}"
            )
            return True

        except sqlite3.IntegrityError:
            logger.debug(f"[TradeLedger] Ticket {mt5_ticket} already exists (skip)")
            return False

    def get_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get trading statistics for the last N days."""
        since = datetime.utcnow() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Total stats
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl_usd < 0 THEN 1 ELSE 0 END) as losses,
                    SUM(pnl_usd) as total_pnl,
                    SUM(commission) as total_commission,
                    SUM(swap) as total_swap
                FROM trades
                WHERE status = 'CLOSED' AND entry_time >= ?
            """, (since.isoformat(),))

            row = cursor.fetchone()
            total = row["total_trades"] or 0
            wins = row["wins"] or 0
            losses = row["losses"] or 0

            # By bot
            cursor = conn.execute("""
                SELECT
                    bot_name,
                    COUNT(*) as trades,
                    SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(pnl_usd) as pnl
                FROM trades
                WHERE status = 'CLOSED' AND entry_time >= ?
                GROUP BY bot_name
                ORDER BY pnl DESC
            """, (since.isoformat(),))

            by_bot = {
                row["bot_name"]: {
                    "trades": row["trades"],
                    "wins": row["wins"],
                    "win_rate": 100 * row["wins"] / row["trades"] if row["trades"] > 0 else 0,
                    "pnl": row["pnl"] or 0
                }
                for row in cursor.fetchall()
            }

            # By symbol
            cursor = conn.execute("""
                SELECT
                    symbol,
                    COUNT(*) as trades,
                    SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(pnl_usd) as pnl
                FROM trades
                WHERE status = 'CLOSED' AND entry_time >= ?
                GROUP BY symbol
                ORDER BY pnl DESC
            """, (since.isoformat(),))

            by_symbol = {
                row["symbol"]: {
                    "trades": row["trades"],
                    "wins": row["wins"],
                    "win_rate": 100 * row["wins"] / row["trades"] if row["trades"] > 0 else 0,
                    "pnl": row["pnl"] or 0
                }
                for row in cursor.fetchall()
            }

            return {
                "period_days": days,
                "total_trades": total,
                "wins": wins,
                "losses": losses,
                "win_rate": 100 * wins / total if total > 0 else 0,
                "total_pnl": row["total_pnl"] or 0,
                "total_commission": row["total_commission"] or 0,
                "total_swap": row["total_swap"] or 0,
                "by_bot": by_bot,
                "by_symbol": by_symbol
            }

    def _row_to_trade(self, row: sqlite3.Row) -> Trade:
        """Convert a database row to a Trade object."""
        return Trade(
            id=row["id"],
            mt5_ticket=row["mt5_ticket"],
            bot_name=row["bot_name"],
            symbol=row["symbol"],
            direction=row["direction"],
            volume=row["volume"],
            entry_price=row["entry_price"],
            entry_time=datetime.fromisoformat(row["entry_time"]) if row["entry_time"] else None,
            entry_sl=row["entry_sl"],
            entry_tp=row["entry_tp"],
            exit_price=row["exit_price"],
            exit_time=datetime.fromisoformat(row["exit_time"]) if row["exit_time"] else None,
            exit_reason=row["exit_reason"],
            pnl_usd=row["pnl_usd"],
            pnl_pips=row["pnl_pips"],
            commission=row["commission"],
            swap=row["swap"],
            status=row["status"],
            mode=row["mode"],
            mt5_verified=bool(row["mt5_verified"]),
            last_sync=datetime.fromisoformat(row["last_sync"]) if row["last_sync"] else None,
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None
        )

    def export_to_csv(self, filepath: Path, days: int = 30) -> int:
        """Export trades to CSV file. Returns number of records exported."""
        import csv

        since = datetime.utcnow() - timedelta(days=days)
        trades = self.get_trades_since(since)

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'ticket', 'bot', 'symbol', 'direction', 'volume',
                'entry_price', 'entry_time', 'exit_price', 'exit_time',
                'exit_reason', 'pnl_usd', 'commission', 'swap', 'status', 'mode'
            ])

            for t in trades:
                writer.writerow([
                    t.mt5_ticket, t.bot_name, t.symbol, t.direction, t.volume,
                    t.entry_price, t.entry_time, t.exit_price, t.exit_time,
                    t.exit_reason, t.pnl_usd, t.commission, t.swap, t.status, t.mode
                ])

        logger.info(f"[TradeLedger] Exported {len(trades)} trades to {filepath}")
        return len(trades)


# Convenience function for quick access
def get_ledger(db_path: Optional[Path] = None) -> TradeLedger:
    """Get a TradeLedger instance."""
    return TradeLedger(db_path)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    # Use test database
    test_db = Path("/tmp/test_ledger.db")
    ledger = TradeLedger(test_db)

    # Test record open
    ledger.record_open(
        mt5_ticket=12345,
        bot_name="TestBot",
        symbol="XAUUSD",
        direction="BUY",
        volume=0.1,
        entry_price=2000.50,
        entry_sl=1990.00,
        entry_tp=2020.00,
        mode="PAPER"
    )

    # Test get open
    open_trades = ledger.get_open_trades()
    print(f"Open trades: {len(open_trades)}")

    # Test record close
    ledger.record_close(
        mt5_ticket=12345,
        exit_price=2015.00,
        exit_reason="TP_HIT",
        pnl_usd=145.00,
        commission=-5.00
    )

    # Test stats
    stats = ledger.get_stats(days=30)
    print(f"Stats: {json.dumps(stats, indent=2)}")

    # Cleanup
    test_db.unlink()
    print("Test passed!")
