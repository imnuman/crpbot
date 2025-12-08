"""
HYDRA 3.0 - Paper Trading System

Simulates trades with $0 risk to validate strategy performance.

Paper Trade Lifecycle:
1. Signal generated → Create paper trade entry
2. Monitor price → Check for SL/TP hit
3. Exit detected → Close trade, record result
4. Update tournament → Feed performance back
5. Learn from losses → Extract lessons

This runs continuously in the background, monitoring all open paper trades.
"""

from typing import Dict, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field
from loguru import logger
from pathlib import Path
import json
import os
import tempfile
import shutil
import threading


@dataclass
class PaperTrade:
    """
    A single paper trade (simulated).

    Tracks entry, SL, TP, and monitors for exit.
    """
    trade_id: str
    asset: str
    regime: str
    strategy_id: str
    gladiator: str

    # Entry details
    direction: str  # "BUY" | "SELL"
    entry_price: float
    entry_timestamp: datetime
    position_size_usd: float

    # Exit targets
    stop_loss: float
    take_profit: float

    # Exit details (filled when closed)
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    exit_reason: Optional[str] = None  # "stop_loss" | "take_profit" | "manual"

    # Performance
    pnl_usd: float = 0.0
    pnl_percent: float = 0.0
    outcome: Optional[str] = None  # "win" | "loss"
    rr_actual: float = 0.0

    # Metadata
    status: str = "OPEN"  # "OPEN" | "CLOSED"
    slippage_est: float = 0.0005  # 0.05% estimated slippage

    def to_dict(self) -> Dict:
        """Convert to dict for storage."""
        return {
            "trade_id": self.trade_id,
            "asset": self.asset,
            "regime": self.regime,
            "strategy_id": self.strategy_id,
            "gladiator": self.gladiator,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "entry_timestamp": self.entry_timestamp.isoformat(),
            "position_size_usd": self.position_size_usd,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "exit_price": self.exit_price,
            "exit_timestamp": self.exit_timestamp.isoformat() if self.exit_timestamp else None,
            "exit_reason": self.exit_reason,
            "pnl_usd": self.pnl_usd,
            "pnl_percent": self.pnl_percent,
            "outcome": self.outcome,
            "rr_actual": self.rr_actual,
            "status": self.status,
            "slippage_est": self.slippage_est
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PaperTrade':
        """Load from dict."""
        trade = cls(
            trade_id=data["trade_id"],
            asset=data["asset"],
            regime=data["regime"],
            strategy_id=data["strategy_id"],
            gladiator=data["gladiator"],
            direction=data["direction"],
            entry_price=data["entry_price"],
            entry_timestamp=datetime.fromisoformat(data["entry_timestamp"]),
            position_size_usd=data["position_size_usd"],
            stop_loss=data["stop_loss"],
            take_profit=data["take_profit"]
        )

        trade.exit_price = data.get("exit_price")
        trade.exit_timestamp = datetime.fromisoformat(data["exit_timestamp"]) if data.get("exit_timestamp") else None
        trade.exit_reason = data.get("exit_reason")
        trade.pnl_usd = data.get("pnl_usd", 0.0)
        trade.pnl_percent = data.get("pnl_percent", 0.0)
        trade.outcome = data.get("outcome")
        trade.rr_actual = data.get("rr_actual", 0.0)
        trade.status = data.get("status", "OPEN")
        trade.slippage_est = data.get("slippage_est", 0.0005)

        return trade


class PaperTradingSystem:
    """
    Manages all paper trades for HYDRA.

    Responsibilities:
    - Create paper trades from signals
    - Monitor open trades for SL/TP hits
    - Close trades and calculate P&L
    - Feed results back to tournament
    - Extract lessons from losses
    """

    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            from .config import PAPER_TRADES_FILE
            storage_path = PAPER_TRADES_FILE

        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Backup path for recovery
        self.backup_path = Path(str(storage_path) + ".backup")

        # In-memory tracking
        self.open_trades: Dict[str, PaperTrade] = {}
        self.closed_trades: List[PaperTrade] = []

        # Statistics
        self.total_trades = 0
        self.wins = 0
        self.losses = 0

        # Thread safety for file operations
        self._file_lock = threading.Lock()

        # Load existing trades
        self._load_trades()

        # Sync duplicate guard with open positions
        try:
            from .duplicate_order_guard import sync_guard_with_paper_trades
            sync_guard_with_paper_trades(str(self.storage_path))
        except Exception as e:
            logger.warning(f"Failed to sync duplicate guard: {e}")

        logger.info(
            f"Paper Trading System initialized "
            f"({len(self.open_trades)} open, {len(self.closed_trades)} closed)"
        )

    # ==================== TRADE CREATION ====================

    def create_paper_trade(
        self,
        asset: str,
        regime: str,
        strategy_id: str,
        gladiator: str,
        signal: Dict,
        market_data: Optional[Dict] = None
    ) -> Optional[PaperTrade]:
        """
        Create new paper trade from signal.

        Args:
            signal: {
                "action": "BUY" | "SELL",
                "entry_price": 97500,
                "stop_loss_pct": 0.015,
                "take_profit_pct": 0.025,
                "position_size_usd": 100,
                "consensus_level": "STRONG"
            }
            market_data: Optional dict with price data for trend filtering

        Returns:
            PaperTrade if created, None if blocked by filters
        """
        direction = signal["action"]
        entry_price = signal["entry_price"]

        # Check duplicate guard (prevents conflicting positions)
        try:
            from .duplicate_order_guard import get_duplicate_guard
            guard = get_duplicate_guard()
            can_place, reason = guard.can_place_order(asset, direction, gladiator, entry_price)
            if not can_place:
                logger.warning(f"[DuplicateGuard] Trade blocked: {reason}")
                return None
        except Exception as e:
            logger.warning(f"Duplicate guard check failed: {e}")

        # Check trend filter (prevents counter-trend trades)
        if market_data:
            try:
                from .trend_filter import get_trend_filter
                trend_filter = get_trend_filter()
                can_trade, reason = trend_filter.check_trend_alignment(asset, direction, market_data)
                if not can_trade:
                    logger.warning(f"[TrendFilter] Trade blocked: {reason}")
                    return None
            except Exception as e:
                logger.warning(f"Trend filter check failed: {e}")

        trade_id = f"{asset}_{int(datetime.now(timezone.utc).timestamp())}"
        sl_pct = signal.get("stop_loss_pct", 0.015)
        tp_pct = signal.get("take_profit_pct", 0.025)
        position_size = signal.get("position_size_usd", 100)

        # Calculate SL/TP levels
        if direction == "BUY":
            stop_loss = entry_price * (1 - sl_pct)
            take_profit = entry_price * (1 + tp_pct)
        else:  # SELL
            stop_loss = entry_price * (1 + sl_pct)
            take_profit = entry_price * (1 - tp_pct)

        trade = PaperTrade(
            trade_id=trade_id,
            asset=asset,
            regime=regime,
            strategy_id=strategy_id,
            gladiator=gladiator,
            direction=direction,
            entry_price=entry_price,
            entry_timestamp=datetime.now(timezone.utc),
            position_size_usd=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        self.open_trades[trade_id] = trade
        self.total_trades += 1

        # Record in duplicate guard
        try:
            from .duplicate_order_guard import get_duplicate_guard
            guard = get_duplicate_guard()
            guard.record_order(asset, direction, gladiator, trade_id, entry_price)
        except Exception as e:
            logger.warning(f"Failed to record order in guard: {e}")

        logger.success(
            f"Paper trade created: {direction} {asset} @ {entry_price:.2f} "
            f"(SL: {stop_loss:.2f}, TP: {take_profit:.2f}, size: ${position_size:.0f})"
        )

        self._save_trades()

        return trade

    # ==================== TRADE MONITORING ====================

    def check_open_trades(self, market_data: Dict[str, List[Dict]]):
        """
        Check all open trades for SL/TP hits.

        Args:
            market_data: {
                "BTC-USD": [{"high": 98000, "low": 97000, "close": 97500, ...}, ...],
                "ETH-USD": [...],
                ...
            }
        """
        if not self.open_trades:
            return

        trades_to_close = []

        for trade_id, trade in self.open_trades.items():
            asset_data = market_data.get(trade.asset)

            if not asset_data or len(asset_data) == 0:
                continue

            # Get latest candle
            latest = asset_data[-1]
            high = latest["high"]
            low = latest["low"]
            close = latest["close"]

            # Check for SL/TP hit
            exit_price = None
            exit_reason = None

            if trade.direction == "BUY":
                # Check stop loss (price went below SL)
                if low <= trade.stop_loss:
                    exit_price = trade.stop_loss
                    exit_reason = "stop_loss"
                # Check take profit (price went above TP)
                elif high >= trade.take_profit:
                    exit_price = trade.take_profit
                    exit_reason = "take_profit"

            else:  # SELL
                # Check stop loss (price went above SL)
                if high >= trade.stop_loss:
                    exit_price = trade.stop_loss
                    exit_reason = "stop_loss"
                # Check take profit (price went below TP)
                elif low <= trade.take_profit:
                    exit_price = trade.take_profit
                    exit_reason = "take_profit"

            if exit_price and exit_reason:
                trades_to_close.append((trade_id, exit_price, exit_reason))

        # Close trades
        for trade_id, exit_price, exit_reason in trades_to_close:
            self.close_paper_trade(trade_id, exit_price, exit_reason)

    def close_paper_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str
    ) -> Optional[PaperTrade]:
        """
        Close paper trade and calculate P&L.

        Returns:
            Closed trade with results
        """
        if trade_id not in self.open_trades:
            logger.warning(f"Trade {trade_id} not found in open trades")
            return None

        trade = self.open_trades[trade_id]

        # Validate entry price to prevent division by zero
        if trade.entry_price <= 0:
            logger.error(f"Invalid entry price {trade.entry_price} for trade {trade_id}")
            del self.open_trades[trade_id]
            return None

        # Apply slippage ONLY to SL (market order)
        # TP orders are limit orders and execute at exact price
        if exit_reason == "stop_loss":
            # Slippage works against you on SL (market order)
            if trade.direction == "BUY":
                exit_price = exit_price * (1 - trade.slippage_est)
            else:  # SELL
                exit_price = exit_price * (1 + trade.slippage_est)
        # TP: No slippage - limit orders fill at exact price

        # Calculate P&L (entry_price validated above)
        if trade.direction == "BUY":
            pnl_percent = (exit_price - trade.entry_price) / trade.entry_price
        else:  # SELL
            pnl_percent = (trade.entry_price - exit_price) / trade.entry_price

        pnl_usd = trade.position_size_usd * pnl_percent

        # Determine outcome
        outcome = "win" if pnl_percent > 0 else "loss"

        # Calculate actual R:R with safe division
        if trade.direction == "BUY":
            risk = trade.entry_price - trade.stop_loss
            reward = exit_price - trade.entry_price
        else:  # SELL
            risk = trade.stop_loss - trade.entry_price
            reward = trade.entry_price - exit_price

        # Safe R:R calculation - avoid division by zero or tiny numbers
        rr_actual = abs(reward / risk) if abs(risk) > 0.0001 else 0.0

        # Update trade
        trade.exit_price = exit_price
        trade.exit_timestamp = datetime.now(timezone.utc)
        trade.exit_reason = exit_reason
        trade.pnl_usd = pnl_usd
        trade.pnl_percent = pnl_percent
        trade.outcome = outcome
        trade.rr_actual = rr_actual
        trade.status = "CLOSED"

        # Move to closed trades
        del self.open_trades[trade_id]
        self.closed_trades.append(trade)

        # Notify duplicate guard that position is closed
        try:
            from .duplicate_order_guard import get_duplicate_guard
            guard = get_duplicate_guard()
            guard.record_position_closed(trade.asset, trade.gladiator)
        except Exception as e:
            logger.warning(f"Failed to notify guard of position close: {e}")

        # Update statistics
        if outcome == "win":
            self.wins += 1
        else:
            self.losses += 1

        logger.info(
            f"Paper trade closed: {trade.asset} {trade.direction} "
            f"({outcome.upper()}, {exit_reason}) - "
            f"P&L: {pnl_percent:+.2%} (${pnl_usd:+.2f}), R:R: {rr_actual:.2f}"
        )

        self._save_trades()

        return trade

    # ==================== STATISTICS & REPORTING ====================

    def get_closed_trades(self) -> List[Dict]:
        """
        Get all closed trades as list of dicts.

        Returns:
            List of closed trade dictionaries
        """
        return [
            {
                "trade_id": t.trade_id,
                "asset": t.asset,
                "regime": t.regime,
                "strategy_id": t.strategy_id,
                "gladiator": t.gladiator,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "position_size_usd": t.position_size_usd,
                "entry_timestamp": t.entry_timestamp.isoformat() if t.entry_timestamp else None,
                "exit_timestamp": t.exit_timestamp.isoformat() if t.exit_timestamp else None,
                "exit_reason": t.exit_reason,
                "pnl_usd": t.pnl_usd,
                "pnl_percent": t.pnl_percent,
                "outcome": t.outcome,
                "rr_actual": t.rr_actual,
                "status": t.status
            }
            for t in self.closed_trades
        ]

    def get_overall_stats(self) -> Dict:
        """Get overall performance statistics."""
        if self.total_trades == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl_usd": 0.0,
                "total_pnl_percent": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "avg_rr": 0.0,
                "sharpe_ratio": 0.0,
                "open_trades": 0
            }

        win_rate = self.wins / self.total_trades

        # Calculate average win/loss
        wins = [t.pnl_percent for t in self.closed_trades if t.outcome == "win"]
        losses = [t.pnl_percent for t in self.closed_trades if t.outcome == "loss"]

        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0

        # Total P&L
        total_pnl_usd = sum(t.pnl_usd for t in self.closed_trades)
        total_pnl_percent = sum(t.pnl_percent for t in self.closed_trades)

        # Average R:R
        rrs = [t.rr_actual for t in self.closed_trades if t.rr_actual > 0]
        avg_rr = sum(rrs) / len(rrs) if rrs else 0.0

        # Sharpe ratio (simplified)
        if len(self.closed_trades) > 1:
            returns = [t.pnl_percent for t in self.closed_trades]
            avg_return = sum(returns) / len(returns)
            variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
            std_dev = variance ** 0.5
            sharpe = avg_return / std_dev if std_dev > 0 else 0.0
        else:
            sharpe = 0.0

        return {
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": win_rate,
            "total_pnl_usd": total_pnl_usd,
            "total_pnl_percent": total_pnl_percent,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_rr": avg_rr,
            "sharpe_ratio": sharpe,
            "open_trades": len(self.open_trades)
        }

    def get_stats_by_asset(self, asset: str) -> Dict:
        """Get statistics for specific asset."""
        asset_trades = [t for t in self.closed_trades if t.asset == asset]

        if not asset_trades:
            return {
                "asset": asset,
                "total_trades": 0,
                "win_rate": 0.0
            }

        wins = sum(1 for t in asset_trades if t.outcome == "win")

        return {
            "asset": asset,
            "total_trades": len(asset_trades),
            "wins": wins,
            "losses": len(asset_trades) - wins,
            "win_rate": wins / len(asset_trades),
            "total_pnl_percent": sum(t.pnl_percent for t in asset_trades),
            "avg_pnl_percent": sum(t.pnl_percent for t in asset_trades) / len(asset_trades)
        }

    def get_stats_by_regime(self, regime: str) -> Dict:
        """Get statistics for specific regime."""
        regime_trades = [t for t in self.closed_trades if t.regime == regime]

        if not regime_trades:
            return {
                "regime": regime,
                "total_trades": 0,
                "win_rate": 0.0
            }

        wins = sum(1 for t in regime_trades if t.outcome == "win")

        return {
            "regime": regime,
            "total_trades": len(regime_trades),
            "wins": wins,
            "losses": len(regime_trades) - wins,
            "win_rate": wins / len(regime_trades),
            "total_pnl_percent": sum(t.pnl_percent for t in regime_trades),
            "avg_pnl_percent": sum(t.pnl_percent for t in regime_trades) / len(regime_trades)
        }

    def get_stats_by_strategy(self, strategy_id: str) -> Dict:
        """Get statistics for specific strategy."""
        strategy_trades = [t for t in self.closed_trades if t.strategy_id == strategy_id]

        if not strategy_trades:
            return {
                "strategy_id": strategy_id,
                "total_trades": 0,
                "win_rate": 0.0
            }

        wins = sum(1 for t in strategy_trades if t.outcome == "win")

        return {
            "strategy_id": strategy_id,
            "total_trades": len(strategy_trades),
            "wins": wins,
            "losses": len(strategy_trades) - wins,
            "win_rate": wins / len(strategy_trades),
            "total_pnl_percent": sum(t.pnl_percent for t in strategy_trades),
            "avg_pnl_percent": sum(t.pnl_percent for t in strategy_trades) / len(strategy_trades),
            "avg_rr": sum(t.rr_actual for t in strategy_trades) / len(strategy_trades)
        }

    # ==================== PERSISTENCE ====================

    def _save_trades(self):
        """
        Save trades to JSONL file using atomic write pattern.

        This ensures data integrity even if process crashes mid-write:
        1. Write to temporary file
        2. Sync to disk (fsync)
        3. Atomically rename temp -> target
        4. Create backup copy for recovery
        """
        with self._file_lock:
            try:
                # Get directory of storage path
                storage_dir = self.storage_path.parent

                # Create temp file in same directory (required for atomic rename)
                fd, temp_path = tempfile.mkstemp(
                    dir=storage_dir,
                    prefix=".paper_trades_",
                    suffix=".tmp"
                )

                try:
                    # Write all trades to temp file
                    with os.fdopen(fd, 'w') as f:
                        # Save closed trades
                        for trade in self.closed_trades:
                            f.write(json.dumps(trade.to_dict()) + "\n")

                        # Save open trades
                        for trade in self.open_trades.values():
                            f.write(json.dumps(trade.to_dict()) + "\n")

                        # Flush Python buffers
                        f.flush()
                        # Sync to disk (ensure data hits storage)
                        os.fsync(f.fileno())

                    # Create backup of current file before replacing
                    if self.storage_path.exists():
                        shutil.copy2(self.storage_path, self.backup_path)

                    # Atomic rename (POSIX guarantees atomicity)
                    shutil.move(temp_path, self.storage_path)

                    logger.debug(
                        f"Saved {len(self.closed_trades)} closed + "
                        f"{len(self.open_trades)} open trades (atomic)"
                    )

                except Exception as e:
                    # Clean up temp file on error
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    raise

            except Exception as e:
                logger.error(f"Failed to save trades atomically: {e}")
                # Fall back to non-atomic write as last resort
                self._save_trades_fallback()

    def _save_trades_fallback(self):
        """
        Fallback non-atomic save for when atomic save fails.

        Only used as last resort - better to lose in-progress data
        than crash entirely.
        """
        try:
            with open(self.storage_path, "w") as f:
                for trade in self.closed_trades:
                    f.write(json.dumps(trade.to_dict()) + "\n")
                for trade in self.open_trades.values():
                    f.write(json.dumps(trade.to_dict()) + "\n")
            logger.warning("Used fallback (non-atomic) save")
        except Exception as e:
            logger.error(f"Fallback save also failed: {e}")

    def _load_trades(self):
        """
        Load trades from JSONL file with backup recovery.

        If main file is corrupted, attempts to recover from backup.
        """
        loaded_from_backup = False

        # Try main file first
        if self.storage_path.exists():
            try:
                self._load_trades_from_file(self.storage_path)
                return
            except Exception as e:
                logger.error(f"Failed to load from main file: {e}")

                # Try backup if main file failed
                if self.backup_path.exists():
                    logger.warning("Attempting recovery from backup...")
                    try:
                        self._load_trades_from_file(self.backup_path)
                        loaded_from_backup = True
                        logger.success("Recovered trades from backup!")
                    except Exception as backup_e:
                        logger.error(f"Backup recovery also failed: {backup_e}")
        elif self.backup_path.exists():
            # Main file doesn't exist but backup does
            logger.warning("Main file missing, loading from backup...")
            try:
                self._load_trades_from_file(self.backup_path)
                loaded_from_backup = True
            except Exception as e:
                logger.error(f"Failed to load backup: {e}")

        if loaded_from_backup:
            # Restore main file from backup
            try:
                shutil.copy2(self.backup_path, self.storage_path)
                logger.info("Restored main file from backup")
            except Exception as e:
                logger.error(f"Failed to restore main file: {e}")

        logger.info(
            f"Loaded {len(self.closed_trades)} closed trades, "
            f"{len(self.open_trades)} open trades"
        )

    def _load_trades_from_file(self, file_path: Path):
        """
        Load trades from specific file path.

        Args:
            file_path: Path to JSONL file
        """
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        trade = PaperTrade.from_dict(data)

                        if trade.status == "OPEN":
                            self.open_trades[trade.trade_id] = trade
                        else:
                            self.closed_trades.append(trade)

                            # Update statistics
                            self.total_trades += 1
                            if trade.outcome == "win":
                                self.wins += 1
                            else:
                                self.losses += 1

                    except Exception as e:
                        logger.error(f"Failed to load trade: {e}")


# Global singleton instance
_paper_trader = None

def get_paper_trader() -> PaperTradingSystem:
    """Get global PaperTradingSystem singleton."""
    global _paper_trader
    if _paper_trader is None:
        _paper_trader = PaperTradingSystem()
    return _paper_trader
