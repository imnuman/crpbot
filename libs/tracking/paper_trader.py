"""
Paper Trading System for V7 Ultimate

Automatically executes paper trades based on V7 signals and tracks results.
This allows DeepSeek to trade aggressively to measure performance without real money.

Features:
- Automatic entry on signal generation
- Smart exit strategies (take profit, stop loss, time-based)
- Real-time P&L tracking
- Full trade history with timestamps
- Integration with PerformanceTracker
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import json

from loguru import logger
from sqlalchemy import text

from libs.tracking.performance_tracker import PerformanceTracker
from libs.db.models import Signal, get_session
from libs.config.config import Settings


class ExitReason(Enum):
    """Reasons for exiting a paper trade"""
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TIME_LIMIT = "time_limit"
    REVERSE_SIGNAL = "reverse_signal"
    MANUAL = "manual"


@dataclass
class PaperTradeConfig:
    """Configuration for paper trading"""
    enabled: bool = True
    aggressive_mode: bool = True  # Trade all signals regardless of confidence
    min_confidence: float = 0.0  # Minimum confidence to trade (0.0 = all signals)
    max_hold_minutes: int = 240  # Max hold time (4 hours)
    use_signal_targets: bool = True  # Use TP/SL from signal
    default_take_profit_pct: float = 2.0  # Default TP if signal has none
    default_stop_loss_pct: float = 1.5  # Default SL if signal has none
    position_size_usd: float = 1000.0  # Virtual position size


class PaperTrader:
    """
    Automated paper trading system
    
    Executes virtual trades based on V7 signals and tracks results.
    """
    
    def __init__(self, config: Optional[PaperTradeConfig] = None, settings: Optional[Settings] = None):
        """
        Initialize paper trader
        
        Args:
            config: Paper trading configuration
            settings: Database settings
        """
        self.config = config or PaperTradeConfig()
        self.settings = settings or Settings()
        self.tracker = PerformanceTracker()
        
        logger.info(f"📊 Paper Trader initialized | Aggressive: {self.config.aggressive_mode} | Min Confidence: {self.config.min_confidence:.0%}")
    
    def should_trade_signal(self, signal: Signal) -> bool:
        """
        Check if we should paper trade this signal

        Args:
            signal: Signal object from database

        Returns:
            True if we should trade this signal
        """
        if not self.config.enabled:
            return False

        # CRITICAL: Never trade HOLD signals - they mean "do nothing"
        if signal.direction and signal.direction.lower() == 'hold':
            logger.debug(f"Signal {signal.id} is HOLD - skipping paper trade")
            return False

        # Check if already have an open position for this signal
        open_positions = self.tracker.get_open_positions()
        for pos in open_positions:
            if pos['signal_id'] == signal.id:
                logger.debug(f"Signal {signal.id} already has open position")
                return False

        # In aggressive mode, trade everything (except HOLD)
        if self.config.aggressive_mode:
            return True

        # Otherwise check confidence threshold
        if signal.confidence and signal.confidence >= self.config.min_confidence:
            return True

        logger.debug(f"Signal {signal.id} confidence {signal.confidence:.1%} below threshold {self.config.min_confidence:.0%}")
        return False
    
    def enter_paper_trade(self, signal_id: int) -> bool:
        """
        Enter a paper trade based on signal
        
        Args:
            signal_id: ID of signal to trade
            
        Returns:
            True if entry was recorded
        """
        try:
            # Get signal from database
            session = get_session(self.settings.db_url)
            signal = session.query(Signal).filter(Signal.id == signal_id).first()
            
            if not signal:
                logger.error(f"Signal {signal_id} not found")
                session.close()
                return False
            
            # Check if we should trade this signal
            if not self.should_trade_signal(signal):
                session.close()
                return False
            
            # Get entry price (use signal's entry price or current price)
            entry_price = signal.entry_price
            if not entry_price:
                logger.warning(f"Signal {signal_id} has no entry price, skipping")
                session.close()
                return False
            
            # Record entry
            entry_time = datetime.now(timezone.utc)
            success = self.tracker.record_entry(
                signal_id=signal_id,
                entry_price=entry_price,
                entry_timestamp=entry_time
            )

            if success:
                # Format timestamp in EST for logging
                from libs.utils.timezone import format_timestamp_est
                entry_est = format_timestamp_est(entry_time)
                logger.info(f"📈 PAPER ENTRY: {signal.symbol} {signal.direction.upper()} @ ${entry_price:,.2f} at {entry_est} EST (Signal #{signal_id}, Confidence: {signal.confidence:.1%})")
            
            session.close()
            return success
            
        except Exception as e:
            logger.error(f"Failed to enter paper trade for signal {signal_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def check_and_exit_trades(self, current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Check all open positions and exit if conditions met
        
        Args:
            current_prices: Dict of symbol -> current price
            
        Returns:
            List of trades that were exited
        """
        exited_trades = []
        
        try:
            open_positions = self.tracker.get_open_positions()
            
            for position in open_positions:
                signal_id = position['signal_id']
                symbol = position['symbol']
                direction = position['direction']
                entry_price = position['entry_price']
                entry_timestamp = position['entry_timestamp']
                
                # Get current price for this symbol
                if symbol not in current_prices:
                    logger.warning(f"No current price for {symbol}, skipping exit check")
                    continue
                
                current_price = current_prices[symbol]
                
                # Parse entry timestamp if it's a string
                if isinstance(entry_timestamp, str):
                    from dateutil import parser
                    entry_timestamp = parser.parse(entry_timestamp)
                
                # Check exit conditions
                exit_reason, should_exit = self._check_exit_conditions(
                    signal_id=signal_id,
                    direction=direction,
                    entry_price=entry_price,
                    current_price=current_price,
                    entry_timestamp=entry_timestamp
                )
                
                if should_exit:
                    # Exit the trade
                    exit_time = datetime.now(timezone.utc)
                    success = self.tracker.record_exit(
                        signal_id=signal_id,
                        exit_price=current_price,
                        exit_timestamp=exit_time,
                        exit_reason=exit_reason.value
                    )
                    
                    if success:
                        # Calculate P&L
                        if direction == 'long':
                            pnl_pct = ((current_price - entry_price) / entry_price) * 100
                        else:  # short
                            pnl_pct = ((entry_price - current_price) / entry_price) * 100

                        # Format timestamps in EST
                        from libs.utils.timezone import format_timestamp_est
                        exit_est = format_timestamp_est(exit_time)

                        outcome_emoji = '✅' if pnl_pct > 0 else '❌'
                        logger.info(f"{outcome_emoji} PAPER EXIT: {symbol} {direction.upper()} @ ${current_price:,.2f} at {exit_est} EST | P&L: {pnl_pct:+.2f}% | Reason: {exit_reason.value}")
                        
                        exited_trades.append({
                            'signal_id': signal_id,
                            'symbol': symbol,
                            'direction': direction,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'exit_time': exit_time,  # Add missing exit_time
                            'pnl_percent': pnl_pct,
                            'exit_reason': exit_reason.value
                        })
            
        except Exception as e:
            logger.error(f"Error checking and exiting trades: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return exited_trades
    
    def _check_exit_conditions(
        self,
        signal_id: int,
        direction: str,
        entry_price: float,
        current_price: float,
        entry_timestamp: datetime
    ) -> tuple[ExitReason, bool]:
        """
        Check if trade should be exited
        
        Returns:
            Tuple of (exit_reason, should_exit)
        """
        try:
            # Get signal to check TP/SL targets
            session = get_session(self.settings.db_url)
            signal = session.query(Signal).filter(Signal.id == signal_id).first()
            session.close()
            
            if not signal:
                return ExitReason.MANUAL, False
            
            # Calculate current P&L percentage
            if direction == 'long':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:  # short
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            # Check time limit
            now = datetime.now(timezone.utc)
            if entry_timestamp.tzinfo is None:
                entry_timestamp = entry_timestamp.replace(tzinfo=timezone.utc)
            
            hold_minutes = (now - entry_timestamp).total_seconds() / 60
            if hold_minutes >= self.config.max_hold_minutes:
                return ExitReason.TIME_LIMIT, True
            
            # Check take profit
            if self.config.use_signal_targets and signal.tp_price:
                if direction == 'long' and current_price >= signal.tp_price:
                    return ExitReason.TAKE_PROFIT, True
                elif direction == 'short' and current_price <= signal.tp_price:
                    return ExitReason.TAKE_PROFIT, True
            else:
                # Use default TP percentage
                if pnl_pct >= self.config.default_take_profit_pct:
                    return ExitReason.TAKE_PROFIT, True
            
            # Check stop loss
            if self.config.use_signal_targets and signal.sl_price:
                if direction == 'long' and current_price <= signal.sl_price:
                    return ExitReason.STOP_LOSS, True
                elif direction == 'short' and current_price >= signal.sl_price:
                    return ExitReason.STOP_LOSS, True
            else:
                # Use default SL percentage
                if pnl_pct <= -self.config.default_stop_loss_pct:
                    return ExitReason.STOP_LOSS, True
            
            return ExitReason.MANUAL, False
            
        except Exception as e:
            logger.error(f"Error checking exit conditions for signal {signal_id}: {e}")
            return ExitReason.MANUAL, False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary
        
        Returns:
            Dict with performance metrics
        """
        stats = self.tracker.get_win_rate(days=30)
        open_positions = self.tracker.get_open_positions()
        recent_trades = self.tracker.get_recent_trades(limit=20)
        
        return {
            'stats': stats,
            'open_positions': len(open_positions),
            'open_positions_list': open_positions,
            'recent_trades': recent_trades,
            'config': {
                'aggressive_mode': self.config.aggressive_mode,
                'min_confidence': self.config.min_confidence,
                'max_hold_minutes': self.config.max_hold_minutes,
                'position_size_usd': self.config.position_size_usd
            }
        }
    
    def force_exit_all(self, current_prices: Dict[str, float], reason: str = "manual") -> int:
        """
        Force exit all open positions
        
        Args:
            current_prices: Dict of symbol -> current price
            reason: Reason for force exit
            
        Returns:
            Number of positions closed
        """
        closed_count = 0
        open_positions = self.tracker.get_open_positions()
        
        for position in open_positions:
            signal_id = position['signal_id']
            symbol = position['symbol']
            
            if symbol not in current_prices:
                logger.warning(f"Cannot force exit {symbol}: no current price")
                continue
            
            success = self.tracker.record_exit(
                signal_id=signal_id,
                exit_price=current_prices[symbol],
                exit_timestamp=datetime.now(timezone.utc),
                exit_reason=reason
            )
            
            if success:
                closed_count += 1
        
        logger.info(f"Force closed {closed_count} positions")
        return closed_count


# Convenience function for use in V7 runtime
def auto_trade_signal(signal_id: int, aggressive: bool = True) -> bool:
    """
    Convenience function to automatically paper trade a signal
    
    Args:
        signal_id: Signal ID to trade
        aggressive: If True, trade regardless of confidence
        
    Returns:
        True if trade was entered
    """
    config = PaperTradeConfig(aggressive_mode=aggressive)
    trader = PaperTrader(config=config)
    return trader.enter_paper_trade(signal_id)
