"""
Counter-Trade Intelligence Module

The "Fade the AI" System - Detects when the AI is CONSISTENTLY WRONG
and automatically inverts signals to turn losses into wins.

Core Insight: If a condition historically has <40% win rate,
taking the OPPOSITE trade has >60% win rate.

Example:
    - SELL trades had 27% WR → If inverted to BUY: 73% WR
    - The AI doesn't need to be RIGHT, just CONSISTENTLY wrong!

Usage:
    from libs.hydra.counter_trade_intelligence import get_counter_trade_intelligence

    cti = get_counter_trade_intelligence()
    final_direction, meta = cti.get_smart_direction(
        original_direction="SELL",
        engine="A",
        symbol="BTC-USD",
        regime="TRENDING_UP",
        rsi=55.0,
        session="london",
    )

    if meta["inverted"]:
        print(f"Inverted {meta['original_direction']} → {final_direction}")
"""

import json
import os
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from loguru import logger


# ==================== DATA CLASSES ====================

@dataclass
class ConditionStats:
    """Statistics for a specific condition fingerprint."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl_percent: float = 0.0
    last_updated: Optional[str] = None

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades

    @property
    def avg_pnl(self) -> float:
        """Calculate average P&L per trade."""
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl_percent / self.total_trades

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "ConditionStats":
        return cls(
            total_trades=data.get("total_trades", 0),
            wins=data.get("wins", 0),
            losses=data.get("losses", 0),
            total_pnl_percent=data.get("total_pnl_percent", 0.0),
            last_updated=data.get("last_updated"),
        )


@dataclass
class InversionResult:
    """Result of an inversion decision."""
    should_invert: bool
    original_direction: str
    final_direction: str
    reason: str
    fingerprint: str
    historical_wr: Optional[float] = None
    inverted_wr: Optional[float] = None
    sample_size: int = 0


# ==================== MAIN CLASS ====================

class CounterTradeIntelligence:
    """
    Tracks when the AI is WRONG and inverts signals automatically.

    Key concept: If a condition historically loses >60% of time,
    taking the OPPOSITE trade wins >60% of time.

    Fingerprint Components:
    - engine: Which AI engine (A, B, C, D)
    - direction: Original signal direction (BUY, SELL)
    - symbol: Trading pair (BTC-USD, ETH-USD, XAUUSD, etc.)
    - regime: Market regime (TRENDING_UP, TRENDING_DOWN, RANGING, etc.)
    - rsi_zone: RSI classification (oversold, neutral, overbought)
    - session: Trading session (asia, london, new_york, overlap)
    """

    # Configuration
    INVERT_THRESHOLD = 0.40  # If WR < 40%, INVERT the signal
    MIN_TRADES_FOR_INVERSION = 10  # Need enough data before inverting
    PERSISTENCE_FILE = "data/hydra/counter_trade_stats.json"
    BACKUP_FILE = "data/hydra/counter_trade_stats.backup.json"

    def __init__(self, data_dir: str = "data/hydra"):
        """Initialize Counter-Trade Intelligence."""
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._condition_stats: Dict[str, ConditionStats] = {}
        self._lock = threading.Lock()
        self._dirty = False  # Track if we need to save

        # Load existing data
        self._load_history()

        logger.info(
            f"[CounterTrade] Initialized with {len(self._condition_stats)} conditions tracked"
        )

    # ==================== PUBLIC METHODS ====================

    def should_invert(
        self,
        engine: str,
        direction: str,
        symbol: str,
        regime: str,
        rsi: float,
        session: str,
    ) -> Tuple[bool, str]:
        """
        Check if this signal should be INVERTED based on historical performance.

        Args:
            engine: AI engine identifier (A, B, C, D)
            direction: Original signal direction (BUY, SELL)
            symbol: Trading symbol (BTC-USD, XAUUSD, etc.)
            regime: Market regime classification
            rsi: Current RSI value (0-100)
            session: Trading session (asia, london, new_york, overlap)

        Returns:
            Tuple of (should_invert, reason)
        """
        fingerprint = self._create_fingerprint(engine, direction, symbol, regime, rsi, session)

        with self._lock:
            stats = self._condition_stats.get(fingerprint)

            if not stats or stats.total_trades < self.MIN_TRADES_FOR_INVERSION:
                return False, f"Not enough data ({stats.total_trades if stats else 0}/{self.MIN_TRADES_FOR_INVERSION} trades)"

            win_rate = stats.win_rate

            if win_rate < self.INVERT_THRESHOLD:
                inverted_wr = 1.0 - win_rate
                return True, f"Historical WR={win_rate:.0%} < {self.INVERT_THRESHOLD:.0%}, Inverted WR={inverted_wr:.0%} ({stats.total_trades} trades)"

            return False, f"Historical WR={win_rate:.0%} >= {self.INVERT_THRESHOLD:.0%}, No inversion ({stats.total_trades} trades)"

    def get_smart_direction(
        self,
        original_direction: str,
        engine: str,
        symbol: str,
        regime: str,
        rsi: float,
        session: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get the SMART direction (original or inverted).

        This is the main entry point for the Counter-Trade system.

        Args:
            original_direction: The AI's original signal (BUY, SELL)
            engine: AI engine identifier
            symbol: Trading symbol
            regime: Market regime
            rsi: Current RSI value
            session: Trading session

        Returns:
            Tuple of (final_direction, metadata_dict)
        """
        should_invert, reason = self.should_invert(
            engine, original_direction, symbol, regime, rsi, session
        )

        fingerprint = self._create_fingerprint(engine, original_direction, symbol, regime, rsi, session)

        with self._lock:
            stats = self._condition_stats.get(fingerprint)
            historical_wr = stats.win_rate if stats else None
            sample_size = stats.total_trades if stats else 0

        if should_invert:
            # INVERT: BUY → SELL, SELL → BUY
            final_direction = "SELL" if original_direction.upper() == "BUY" else "BUY"
            inverted_wr = 1.0 - historical_wr if historical_wr else None

            logger.info(
                f"[CounterTrade] INVERTING {original_direction} → {final_direction} "
                f"({reason})"
            )

            return final_direction, {
                "inverted": True,
                "original_direction": original_direction,
                "final_direction": final_direction,
                "reason": reason,
                "fingerprint": fingerprint,
                "historical_wr": historical_wr,
                "inverted_wr": inverted_wr,
                "sample_size": sample_size,
            }

        return original_direction, {
            "inverted": False,
            "original_direction": original_direction,
            "final_direction": original_direction,
            "reason": reason,
            "fingerprint": fingerprint,
            "historical_wr": historical_wr,
            "inverted_wr": None,
            "sample_size": sample_size,
        }

    def record_outcome(
        self,
        engine: str,
        direction: str,
        symbol: str,
        regime: str,
        rsi: float,
        session: str,
        won: bool,
        pnl_percent: float = 0.0,
    ) -> None:
        """
        Record trade outcome to improve future inversion decisions.

        IMPORTANT: Record the ORIGINAL direction, not the inverted one.
        This lets us track how the AI performs, not how we performed after inversion.

        Args:
            engine: AI engine identifier
            direction: ORIGINAL signal direction (before any inversion)
            symbol: Trading symbol
            regime: Market regime
            rsi: RSI at time of signal
            session: Trading session
            won: Whether the trade won
            pnl_percent: P&L percentage of the trade
        """
        fingerprint = self._create_fingerprint(engine, direction, symbol, regime, rsi, session)

        with self._lock:
            if fingerprint not in self._condition_stats:
                self._condition_stats[fingerprint] = ConditionStats()

            stats = self._condition_stats[fingerprint]
            stats.total_trades += 1
            stats.total_pnl_percent += pnl_percent

            if won:
                stats.wins += 1
            else:
                stats.losses += 1

            stats.last_updated = datetime.now(timezone.utc).isoformat()
            self._dirty = True

        # Log the update
        logger.debug(
            f"[CounterTrade] Recorded: {fingerprint} "
            f"→ WR={stats.win_rate:.0%} ({stats.total_trades} trades)"
        )

        # Auto-save after recording
        self._save_history()

    def get_inversion_candidates(self) -> List[Dict[str, Any]]:
        """
        Get all conditions that should be inverted (for dashboard/analysis).

        Returns:
            List of dicts with fingerprint, win_rate, inverted_wr, trades
        """
        candidates = []

        with self._lock:
            for fingerprint, stats in self._condition_stats.items():
                if stats.total_trades >= self.MIN_TRADES_FOR_INVERSION:
                    win_rate = stats.win_rate
                    if win_rate < self.INVERT_THRESHOLD:
                        candidates.append({
                            "fingerprint": fingerprint,
                            "win_rate": win_rate,
                            "inverted_wr": 1.0 - win_rate,
                            "trades": stats.total_trades,
                            "wins": stats.wins,
                            "losses": stats.losses,
                            "avg_pnl": stats.avg_pnl,
                            "last_updated": stats.last_updated,
                        })

        # Sort by win rate (lowest first = best inversion candidates)
        return sorted(candidates, key=lambda x: x["win_rate"])

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get all tracked condition stats."""
        with self._lock:
            return {
                fp: {
                    "win_rate": stats.win_rate,
                    "trades": stats.total_trades,
                    "wins": stats.wins,
                    "losses": stats.losses,
                    "avg_pnl": stats.avg_pnl,
                    "should_invert": stats.win_rate < self.INVERT_THRESHOLD and stats.total_trades >= self.MIN_TRADES_FOR_INVERSION,
                }
                for fp, stats in self._condition_stats.items()
            }

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the Counter-Trade Intelligence state."""
        with self._lock:
            total_conditions = len(self._condition_stats)
            inversion_candidates = sum(
                1 for stats in self._condition_stats.values()
                if stats.win_rate < self.INVERT_THRESHOLD
                and stats.total_trades >= self.MIN_TRADES_FOR_INVERSION
            )
            total_trades_tracked = sum(
                stats.total_trades for stats in self._condition_stats.values()
            )

            return {
                "total_conditions_tracked": total_conditions,
                "inversion_candidates": inversion_candidates,
                "total_trades_tracked": total_trades_tracked,
                "invert_threshold": self.INVERT_THRESHOLD,
                "min_trades_required": self.MIN_TRADES_FOR_INVERSION,
            }

    # ==================== FINGERPRINT CREATION ====================

    def _create_fingerprint(
        self,
        engine: str,
        direction: str,
        symbol: str,
        regime: str,
        rsi: float,
        session: str,
    ) -> str:
        """
        Create a condition fingerprint for tracking.

        Fingerprint format: {engine}_{direction}_{symbol}_{regime}_{rsi_zone}_{session}

        Example: "A_SELL_BTC-USD_TRENDING_UP_neutral_london"
        """
        # Normalize inputs
        engine = str(engine).upper()
        direction = str(direction).upper()
        symbol = str(symbol).upper().replace("/", "-")
        regime = str(regime).upper().replace(" ", "_")
        session = str(session).lower()

        # RSI zones: oversold (<=30), neutral (30-70), overbought (>=70)
        if rsi <= 30:
            rsi_zone = "oversold"
        elif rsi >= 70:
            rsi_zone = "overbought"
        else:
            rsi_zone = "neutral"

        return f"{engine}_{direction}_{symbol}_{regime}_{rsi_zone}_{session}"

    @staticmethod
    def get_current_session() -> str:
        """
        Determine the current trading session based on UTC time.

        Sessions:
        - asia: 00:00 - 08:00 UTC
        - london: 08:00 - 13:00 UTC
        - overlap: 13:00 - 17:00 UTC (London + NY)
        - new_york: 17:00 - 22:00 UTC
        - off_hours: 22:00 - 00:00 UTC
        """
        hour = datetime.now(timezone.utc).hour

        if 0 <= hour < 8:
            return "asia"
        elif 8 <= hour < 13:
            return "london"
        elif 13 <= hour < 17:
            return "overlap"
        elif 17 <= hour < 22:
            return "new_york"
        else:
            return "off_hours"

    # ==================== PERSISTENCE ====================

    def _load_history(self) -> None:
        """Load condition stats from disk."""
        persistence_path = self._data_dir / "counter_trade_stats.json"
        backup_path = self._data_dir / "counter_trade_stats.backup.json"

        # Try main file first
        if persistence_path.exists():
            try:
                with open(persistence_path, "r") as f:
                    data = json.load(f)
                    self._condition_stats = {
                        fp: ConditionStats.from_dict(stats_dict)
                        for fp, stats_dict in data.items()
                    }
                logger.info(
                    f"[CounterTrade] Loaded {len(self._condition_stats)} conditions from {persistence_path}"
                )
                return
            except Exception as e:
                logger.warning(f"[CounterTrade] Failed to load main file: {e}")

        # Try backup file
        if backup_path.exists():
            try:
                with open(backup_path, "r") as f:
                    data = json.load(f)
                    self._condition_stats = {
                        fp: ConditionStats.from_dict(stats_dict)
                        for fp, stats_dict in data.items()
                    }
                logger.info(
                    f"[CounterTrade] Loaded {len(self._condition_stats)} conditions from backup"
                )
                return
            except Exception as e:
                logger.warning(f"[CounterTrade] Failed to load backup: {e}")

        # Start fresh
        logger.info("[CounterTrade] Starting with empty condition stats")
        self._condition_stats = {}

    def _save_history(self) -> None:
        """Save condition stats to disk (atomic write)."""
        if not self._dirty:
            return

        persistence_path = self._data_dir / "counter_trade_stats.json"
        backup_path = self._data_dir / "counter_trade_stats.backup.json"
        temp_path = self._data_dir / "counter_trade_stats.tmp"

        try:
            with self._lock:
                data = {
                    fp: stats.to_dict()
                    for fp, stats in self._condition_stats.items()
                }

            # Write to temp file first
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            # Create backup of existing file
            if persistence_path.exists():
                import shutil
                shutil.copy2(persistence_path, backup_path)

            # Atomic rename
            temp_path.rename(persistence_path)
            self._dirty = False

            logger.debug(
                f"[CounterTrade] Saved {len(self._condition_stats)} conditions to {persistence_path}"
            )

        except Exception as e:
            logger.error(f"[CounterTrade] Failed to save history: {e}")

    # ==================== BOOTSTRAP FROM TRADE HISTORY ====================

    def bootstrap_from_trade_history(self, trades: List[Dict[str, Any]]) -> int:
        """
        Bootstrap condition stats from existing trade history.

        Args:
            trades: List of trade dicts with keys:
                - engine/gladiator: AI engine
                - direction/action: Trade direction
                - asset/symbol: Trading symbol
                - regime: Market regime
                - rsi: RSI value (optional, default 50)
                - session: Trading session (optional, will be inferred from timestamp)
                - outcome: "win" or "loss"
                - pnl_percent: P&L percentage

        Returns:
            Number of trades processed
        """
        processed = 0

        for trade in trades:
            try:
                # Extract fields with fallbacks
                engine = trade.get("engine") or trade.get("gladiator", "unknown")
                direction = trade.get("direction") or trade.get("action", "unknown")
                symbol = trade.get("asset") or trade.get("symbol", "unknown")
                regime = trade.get("regime", "unknown")
                rsi = trade.get("rsi", 50.0)  # Default to neutral if not available

                # Infer session from timestamp if not provided
                session = trade.get("session")
                if not session and "entry_timestamp" in trade:
                    try:
                        ts = trade["entry_timestamp"]
                        if isinstance(ts, str):
                            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        hour = ts.hour
                        if 0 <= hour < 8:
                            session = "asia"
                        elif 8 <= hour < 13:
                            session = "london"
                        elif 13 <= hour < 17:
                            session = "overlap"
                        elif 17 <= hour < 22:
                            session = "new_york"
                        else:
                            session = "off_hours"
                    except Exception:
                        session = "unknown"
                session = session or "unknown"

                # Determine outcome
                outcome = trade.get("outcome", "").lower()
                won = outcome == "win"
                pnl_percent = trade.get("pnl_percent", 0.0)

                # Record the outcome
                self.record_outcome(
                    engine=engine,
                    direction=direction,
                    symbol=symbol,
                    regime=regime,
                    rsi=rsi,
                    session=session,
                    won=won,
                    pnl_percent=pnl_percent,
                )
                processed += 1

            except Exception as e:
                logger.warning(f"[CounterTrade] Failed to process trade: {e}")
                continue

        # Force save after bootstrap
        self._dirty = True
        self._save_history()

        logger.info(f"[CounterTrade] Bootstrapped from {processed} historical trades")
        return processed


# ==================== SINGLETON ACCESSOR ====================

_instance: Optional[CounterTradeIntelligence] = None
_instance_lock = threading.Lock()


def get_counter_trade_intelligence(data_dir: str = "data/hydra") -> CounterTradeIntelligence:
    """
    Get the singleton instance of CounterTradeIntelligence.

    Args:
        data_dir: Directory for persistence files

    Returns:
        CounterTradeIntelligence instance
    """
    global _instance

    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = CounterTradeIntelligence(data_dir=data_dir)

    return _instance


def reset_counter_trade_intelligence() -> None:
    """Reset the singleton instance (for testing)."""
    global _instance
    with _instance_lock:
        _instance = None


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Example usage
    cti = get_counter_trade_intelligence()

    # Simulate some historical trades (Engine A SELL trades losing)
    for i in range(12):
        # Engine A SELL in TRENDING_UP loses 80% of time
        cti.record_outcome(
            engine="A",
            direction="SELL",
            symbol="BTC-USD",
            regime="TRENDING_UP",
            rsi=55.0,
            session="london",
            won=(i < 2),  # Only 2 wins out of 12
            pnl_percent=-1.5 if i >= 2 else 2.5,
        )

    # Now check if we should invert
    final_dir, meta = cti.get_smart_direction(
        original_direction="SELL",
        engine="A",
        symbol="BTC-USD",
        regime="TRENDING_UP",
        rsi=55.0,
        session="london",
    )

    print(f"Original: SELL → Final: {final_dir}")
    print(f"Inverted: {meta['inverted']}")
    print(f"Reason: {meta['reason']}")

    # Show inversion candidates
    print("\nInversion Candidates:")
    for candidate in cti.get_inversion_candidates():
        print(f"  {candidate['fingerprint']}: WR={candidate['win_rate']:.0%} → Inverted={candidate['inverted_wr']:.0%}")
