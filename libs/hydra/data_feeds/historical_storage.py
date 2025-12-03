"""
HYDRA 3.0 - 72-Hour Historical Storage

Stores rolling 72-hour history of all data feeds:
- Order book snapshots
- Funding rates
- Liquidation data
- Market metrics

Uses SQLite for efficient storage and querying.
Automatic cleanup of data older than 72 hours.
"""

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Any
import threading

logger = logging.getLogger(__name__)


# Configuration
RETENTION_HOURS = 72
CLEANUP_INTERVAL_MINUTES = 30
MAX_RECORDS_PER_TYPE = 10000  # Safety limit


class DataType(Enum):
    """Types of data stored."""
    ORDER_BOOK = "order_book"
    FUNDING_RATE = "funding_rate"
    LIQUIDATION = "liquidation"
    MARKET_METRICS = "market_metrics"
    SEARCH_RESULT = "search_result"
    ENGINE_SIGNAL = "engine_signal"
    PRICE = "price"


@dataclass
class HistoricalRecord:
    """A single historical record."""
    id: int
    data_type: DataType
    symbol: str
    timestamp: datetime
    data: dict

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "data_type": self.data_type.value,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }


class HistoricalStorage:
    """
    Rolling 72-hour historical data storage.

    Features:
    - SQLite-based for efficiency
    - Automatic cleanup of old data
    - Query by type, symbol, and time range
    - Thread-safe operations
    """

    _instance: Optional["HistoricalStorage"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, data_dir: Optional[Path] = None):
        if self._initialized:
            return

        # Auto-detect data directory
        if data_dir is None:
            if os.path.exists("/root/crpbot"):
                data_dir = Path("/root/crpbot/data/hydra")
            else:
                data_dir = Path.home() / "crpbot" / "data" / "hydra"

        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.data_dir / "historical_72h.db"
        self._conn_local = threading.local()

        # Initialize database
        self._init_db()

        # Track last cleanup
        self._last_cleanup = datetime.now()

        self._initialized = True
        logger.info(f"HistoricalStorage initialized at {self.db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._conn_local, 'conn') or self._conn_local.conn is None:
            self._conn_local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._conn_local.conn.row_factory = sqlite3.Row
        return self._conn_local.conn

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for efficient querying
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_type_symbol_time
            ON historical_data (data_type, symbol, timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON historical_data (timestamp)
        """)

        conn.commit()
        logger.info("Historical database initialized")

    def store(
        self,
        data_type: DataType,
        symbol: str,
        data: dict,
        timestamp: Optional[datetime] = None
    ) -> int:
        """
        Store a data record.

        Args:
            data_type: Type of data
            symbol: Symbol/identifier
            data: Data payload (dict)
            timestamp: Record timestamp (default: now)

        Returns:
            Record ID
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Periodic cleanup check
        self._maybe_cleanup()

        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO historical_data (data_type, symbol, timestamp, data)
            VALUES (?, ?, ?, ?)
        """, (
            data_type.value,
            symbol,
            timestamp.isoformat(),
            json.dumps(data)
        ))

        conn.commit()
        return cursor.lastrowid

    def store_batch(
        self,
        records: list[tuple[DataType, str, dict, Optional[datetime]]]
    ) -> int:
        """
        Store multiple records efficiently.

        Args:
            records: List of (data_type, symbol, data, timestamp) tuples

        Returns:
            Number of records stored
        """
        self._maybe_cleanup()

        conn = self._get_conn()
        cursor = conn.cursor()

        rows = []
        for data_type, symbol, data, timestamp in records:
            ts = timestamp or datetime.now()
            rows.append((
                data_type.value,
                symbol,
                ts.isoformat(),
                json.dumps(data)
            ))

        cursor.executemany("""
            INSERT INTO historical_data (data_type, symbol, timestamp, data)
            VALUES (?, ?, ?, ?)
        """, rows)

        conn.commit()
        return len(rows)

    def query(
        self,
        data_type: Optional[DataType] = None,
        symbol: Optional[str] = None,
        hours_back: int = 24,
        limit: int = 1000
    ) -> list[HistoricalRecord]:
        """
        Query historical records.

        Args:
            data_type: Filter by type (optional)
            symbol: Filter by symbol (optional)
            hours_back: How many hours back to look
            limit: Max records to return

        Returns:
            List of HistoricalRecord
        """
        cutoff = datetime.now() - timedelta(hours=hours_back)

        conn = self._get_conn()
        cursor = conn.cursor()

        query = "SELECT id, data_type, symbol, timestamp, data FROM historical_data WHERE timestamp > ?"
        params = [cutoff.isoformat()]

        if data_type:
            query += " AND data_type = ?"
            params.append(data_type.value)

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)

        records = []
        for row in cursor.fetchall():
            records.append(HistoricalRecord(
                id=row["id"],
                data_type=DataType(row["data_type"]),
                symbol=row["symbol"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                data=json.loads(row["data"]),
            ))

        return records

    def get_latest(
        self,
        data_type: DataType,
        symbol: str
    ) -> Optional[HistoricalRecord]:
        """Get the most recent record of a type/symbol."""
        records = self.query(data_type=data_type, symbol=symbol, hours_back=72, limit=1)
        return records[0] if records else None

    def get_time_series(
        self,
        data_type: DataType,
        symbol: str,
        field: str,
        hours_back: int = 24
    ) -> list[tuple[datetime, Any]]:
        """
        Get time series of a specific field.

        Args:
            data_type: Data type to query
            symbol: Symbol to query
            field: Field to extract from data
            hours_back: Hours to look back

        Returns:
            List of (timestamp, value) tuples
        """
        records = self.query(
            data_type=data_type,
            symbol=symbol,
            hours_back=hours_back,
            limit=5000
        )

        series = []
        for record in reversed(records):  # Chronological order
            value = record.data.get(field)
            if value is not None:
                series.append((record.timestamp, value))

        return series

    def get_aggregates(
        self,
        data_type: DataType,
        symbol: str,
        field: str,
        hours_back: int = 24
    ) -> dict:
        """
        Get aggregated statistics for a field.

        Args:
            data_type: Data type to query
            symbol: Symbol to query
            field: Field to aggregate
            hours_back: Hours to look back

        Returns:
            Dict with min, max, avg, count, first, last
        """
        series = self.get_time_series(data_type, symbol, field, hours_back)

        if not series:
            return {
                "min": None,
                "max": None,
                "avg": None,
                "count": 0,
                "first": None,
                "last": None,
                "change": None,
                "change_pct": None,
            }

        values = [v for _, v in series if isinstance(v, (int, float))]

        if not values:
            return {
                "min": None,
                "max": None,
                "avg": None,
                "count": len(series),
                "first": series[0][1] if series else None,
                "last": series[-1][1] if series else None,
                "change": None,
                "change_pct": None,
            }

        first_val = values[0]
        last_val = values[-1]
        change = last_val - first_val
        change_pct = (change / first_val * 100) if first_val != 0 else 0

        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "count": len(values),
            "first": first_val,
            "last": last_val,
            "change": change,
            "change_pct": change_pct,
        }

    def _maybe_cleanup(self):
        """Run cleanup if enough time has passed."""
        now = datetime.now()
        if (now - self._last_cleanup).total_seconds() > CLEANUP_INTERVAL_MINUTES * 60:
            self.cleanup()
            self._last_cleanup = now

    def cleanup(self):
        """Remove data older than retention period."""
        cutoff = datetime.now() - timedelta(hours=RETENTION_HOURS)

        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM historical_data WHERE timestamp < ?
        """, (cutoff.isoformat(),))

        deleted = cursor.rowcount
        conn.commit()

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old historical records")

        # Also vacuum periodically to reclaim space
        cursor.execute("VACUUM")

        return deleted

    def get_stats(self) -> dict:
        """Get storage statistics."""
        conn = self._get_conn()
        cursor = conn.cursor()

        # Total records
        cursor.execute("SELECT COUNT(*) FROM historical_data")
        total = cursor.fetchone()[0]

        # Records by type
        cursor.execute("""
            SELECT data_type, COUNT(*) as count
            FROM historical_data
            GROUP BY data_type
        """)
        by_type = {row["data_type"]: row["count"] for row in cursor.fetchall()}

        # Records by symbol
        cursor.execute("""
            SELECT symbol, COUNT(*) as count
            FROM historical_data
            GROUP BY symbol
            ORDER BY count DESC
            LIMIT 10
        """)
        by_symbol = {row["symbol"]: row["count"] for row in cursor.fetchall()}

        # Time range
        cursor.execute("""
            SELECT MIN(timestamp), MAX(timestamp) FROM historical_data
        """)
        row = cursor.fetchone()
        oldest = row[0]
        newest = row[1]

        # Database size
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        return {
            "total_records": total,
            "by_type": by_type,
            "by_symbol": by_symbol,
            "oldest_record": oldest,
            "newest_record": newest,
            "db_size_mb": round(db_size / 1024 / 1024, 2),
            "retention_hours": RETENTION_HOURS,
        }

    def format_summary(self) -> str:
        """Get human-readable summary."""
        stats = self.get_stats()

        lines = [
            f"=== Historical Storage (72h) ===",
            f"Total Records: {stats['total_records']:,}",
            f"Database Size: {stats['db_size_mb']:.2f} MB",
            f"",
            f"By Type:",
        ]

        for dtype, count in stats["by_type"].items():
            lines.append(f"  {dtype}: {count:,}")

        lines.append(f"")
        lines.append(f"Top Symbols:")
        for symbol, count in list(stats["by_symbol"].items())[:5]:
            lines.append(f"  {symbol}: {count:,}")

        if stats["oldest_record"]:
            lines.append(f"")
            lines.append(f"Range: {stats['oldest_record'][:19]} to {stats['newest_record'][:19]}")

        return "\n".join(lines)


# Singleton accessor
_storage_instance: Optional[HistoricalStorage] = None


def get_historical_storage() -> HistoricalStorage:
    """Get or create the historical storage singleton."""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = HistoricalStorage()
    return _storage_instance


# Convenience functions for common operations
def store_order_book(symbol: str, metrics: dict) -> int:
    """Store order book snapshot."""
    storage = get_historical_storage()
    return storage.store(DataType.ORDER_BOOK, symbol, metrics)


def store_funding_rate(symbol: str, data: dict) -> int:
    """Store funding rate data."""
    storage = get_historical_storage()
    return storage.store(DataType.FUNDING_RATE, symbol, data)


def store_liquidation(symbol: str, data: dict) -> int:
    """Store liquidation data."""
    storage = get_historical_storage()
    return storage.store(DataType.LIQUIDATION, symbol, data)


def store_price(symbol: str, price: float, volume: float = 0) -> int:
    """Store price data point."""
    storage = get_historical_storage()
    return storage.store(DataType.PRICE, symbol, {"price": price, "volume": volume})


def store_engine_signal(engine_id: str, signal: dict) -> int:
    """Store engine signal."""
    storage = get_historical_storage()
    return storage.store(DataType.ENGINE_SIGNAL, engine_id, signal)


def get_price_history(symbol: str, hours: int = 24) -> list[tuple[datetime, float]]:
    """Get price history for a symbol."""
    storage = get_historical_storage()
    return storage.get_time_series(DataType.PRICE, symbol, "price", hours)


def get_funding_history(symbol: str, hours: int = 24) -> list[tuple[datetime, float]]:
    """Get funding rate history for a symbol."""
    storage = get_historical_storage()
    return storage.get_time_series(DataType.FUNDING_RATE, symbol, "funding_rate_annual", hours)
