"""
V7 Ultimate - Trade Performance Tracker

Tracks trade outcomes, calculates win rates, PnL, and provides
performance statistics for V7 signals.

This module is designed for manual trading where users manually
execute V7 signals and then record the outcomes.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass

from loguru import logger
from sqlalchemy import func, and_

from libs.db.models import Signal, get_session


@dataclass
class PerformanceStats:
    """Performance statistics for V7 trading"""
    # Win rate
    total_trades: int
    wins: int
    losses: int
    pending: int
    win_rate: float  # Percentage

    # PnL
    total_pnl: float
    avg_win: float
    avg_loss: float
    max_win: float
    max_loss: float
    profit_factor: float  # Gross profit / Gross loss

    # Risk metrics
    sharpe_ratio: Optional[float]
    max_drawdown: float
    max_drawdown_pct: float

    # By signal type
    buy_win_rate: float
    sell_win_rate: float
    hold_skipped: int

    # Time period
    period_days: int
    first_trade: Optional[datetime]
    last_trade: Optional[datetime]


class TradeTracker:
    """
    Trade performance tracker for V7 Ultimate

    Tracks manually executed trades, calculates performance metrics,
    and provides statistics for continuous improvement.
    """

    def __init__(self, db_url: str):
        """
        Initialize trade tracker

        Args:
            db_url: Database connection string
        """
        self.db_url = db_url

    def record_trade_execution(
        self,
        signal_id: int,
        execution_time: datetime,
        execution_price: float
    ) -> bool:
        """
        Record that a signal was manually executed

        Args:
            signal_id: ID of the signal that was executed
            execution_time: When the trade was executed
            execution_price: Price at execution

        Returns:
            True if recorded successfully
        """
        try:
            session = get_session(self.db_url)
            try:
                signal = session.query(Signal).filter(Signal.id == signal_id).first()

                if not signal:
                    logger.error(f"Signal {signal_id} not found")
                    return False

                signal.executed = 1
                signal.execution_time = execution_time
                signal.execution_price = execution_price
                signal.result = 'pending'  # Will be updated when trade closes

                session.commit()
                logger.info(f"Trade execution recorded: Signal {signal_id} @ {execution_price}")
                return True

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error recording trade execution: {e}")
            return False

    def record_trade_outcome(
        self,
        signal_id: int,
        exit_time: datetime,
        exit_price: float,
        result: str  # 'win' or 'loss'
    ) -> bool:
        """
        Record the outcome of a closed trade

        Args:
            signal_id: ID of the signal
            exit_time: When the trade was closed
            exit_price: Price at exit
            result: 'win' or 'loss'

        Returns:
            True if recorded successfully
        """
        try:
            session = get_session(self.db_url)
            try:
                signal = session.query(Signal).filter(Signal.id == signal_id).first()

                if not signal:
                    logger.error(f"Signal {signal_id} not found")
                    return False

                if not signal.executed:
                    logger.warning(f"Signal {signal_id} was not executed, cannot record outcome")
                    return False

                # Calculate PnL
                if signal.direction == 'long':
                    pnl_pct = ((exit_price - signal.execution_price) / signal.execution_price) * 100
                elif signal.direction == 'short':
                    pnl_pct = ((signal.execution_price - exit_price) / signal.execution_price) * 100
                else:  # hold
                    pnl_pct = 0.0

                signal.exit_time = exit_time
                signal.exit_price = exit_price
                signal.result = result
                signal.pnl = pnl_pct

                session.commit()
                logger.info(f"Trade outcome recorded: Signal {signal_id} | Result: {result} | PnL: {pnl_pct:.2f}%")
                return True

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error recording trade outcome: {e}")
            return False

    def get_performance_stats(self, days: int = 30) -> PerformanceStats:
        """
        Get performance statistics for the last N days

        Args:
            days: Number of days to analyze

        Returns:
            PerformanceStats object with all metrics
        """
        try:
            session = get_session(self.db_url)
            try:
                # Get all executed trades from last N days
                since = datetime.utcnow() - timedelta(days=days)

                trades = session.query(Signal).filter(
                    and_(
                        Signal.model_version == 'v7_ultimate',
                        Signal.executed == 1,
                        Signal.execution_time >= since,
                        Signal.result.in_(['win', 'loss'])  # Only closed trades
                    )
                ).all()

                if not trades:
                    # Return empty stats
                    return PerformanceStats(
                        total_trades=0, wins=0, losses=0, pending=0, win_rate=0.0,
                        total_pnl=0.0, avg_win=0.0, avg_loss=0.0,
                        max_win=0.0, max_loss=0.0, profit_factor=0.0,
                        sharpe_ratio=None, max_drawdown=0.0, max_drawdown_pct=0.0,
                        buy_win_rate=0.0, sell_win_rate=0.0, hold_skipped=0,
                        period_days=days, first_trade=None, last_trade=None
                    )

                # Basic counts
                total_trades = len(trades)
                wins = sum(1 for t in trades if t.result == 'win')
                losses = sum(1 for t in trades if t.result == 'loss')

                # Get pending trades count
                pending = session.query(Signal).filter(
                    and_(
                        Signal.model_version == 'v7_ultimate',
                        Signal.executed == 1,
                        Signal.result == 'pending'
                    )
                ).count()

                # PnL calculations
                win_trades = [t for t in trades if t.result == 'win']
                loss_trades = [t for t in trades if t.result == 'loss']

                total_pnl = sum(t.pnl for t in trades)
                avg_win = sum(t.pnl for t in win_trades) / len(win_trades) if win_trades else 0.0
                avg_loss = sum(t.pnl for t in loss_trades) / len(loss_trades) if loss_trades else 0.0
                max_win = max((t.pnl for t in win_trades), default=0.0)
                max_loss = min((t.pnl for t in loss_trades), default=0.0)

                # Profit factor
                gross_profit = sum(t.pnl for t in win_trades)
                gross_loss = abs(sum(t.pnl for t in loss_trades))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

                # Win rate
                win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

                # By signal type
                buy_trades = [t for t in trades if t.direction == 'long']
                sell_trades = [t for t in trades if t.direction == 'short']

                buy_wins = sum(1 for t in buy_trades if t.result == 'win')
                sell_wins = sum(1 for t in sell_trades if t.result == 'win')

                buy_win_rate = (buy_wins / len(buy_trades) * 100) if buy_trades else 0.0
                sell_win_rate = (sell_wins / len(sell_trades) * 100) if sell_trades else 0.0

                # Hold signals that were skipped
                hold_skipped = session.query(Signal).filter(
                    and_(
                        Signal.model_version == 'v7_ultimate',
                        Signal.direction == 'hold',
                        Signal.execution_time >= since
                    )
                ).count()

                # Sharpe ratio (simplified - using PnL variance)
                if len(trades) > 1:
                    pnl_list = [t.pnl for t in trades]
                    mean_pnl = sum(pnl_list) / len(pnl_list)
                    variance = sum((p - mean_pnl) ** 2 for p in pnl_list) / len(pnl_list)
                    std_dev = variance ** 0.5
                    sharpe_ratio = (mean_pnl / std_dev) if std_dev > 0 else None
                else:
                    sharpe_ratio = None

                # Max drawdown
                cumulative_pnl = []
                running_total = 0.0
                for trade in sorted(trades, key=lambda x: x.exit_time):
                    running_total += trade.pnl
                    cumulative_pnl.append(running_total)

                if cumulative_pnl:
                    peak = cumulative_pnl[0]
                    max_drawdown = 0.0
                    for pnl in cumulative_pnl:
                        if pnl > peak:
                            peak = pnl
                        drawdown = peak - pnl
                        if drawdown > max_drawdown:
                            max_drawdown = drawdown

                    max_drawdown_pct = (max_drawdown / abs(peak) * 100) if peak != 0 else 0.0
                else:
                    max_drawdown = 0.0
                    max_drawdown_pct = 0.0

                # Time period
                first_trade = min(t.execution_time for t in trades) if trades else None
                last_trade = max(t.exit_time for t in trades) if trades else None

                return PerformanceStats(
                    total_trades=total_trades,
                    wins=wins,
                    losses=losses,
                    pending=pending,
                    win_rate=win_rate,
                    total_pnl=total_pnl,
                    avg_win=avg_win,
                    avg_loss=avg_loss,
                    max_win=max_win,
                    max_loss=max_loss,
                    profit_factor=profit_factor,
                    sharpe_ratio=sharpe_ratio,
                    max_drawdown=max_drawdown,
                    max_drawdown_pct=max_drawdown_pct,
                    buy_win_rate=buy_win_rate,
                    sell_win_rate=sell_win_rate,
                    hold_skipped=hold_skipped,
                    period_days=days,
                    first_trade=first_trade,
                    last_trade=last_trade
                )

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            # Return empty stats on error
            return PerformanceStats(
                total_trades=0, wins=0, losses=0, pending=0, win_rate=0.0,
                total_pnl=0.0, avg_win=0.0, avg_loss=0.0,
                max_win=0.0, max_loss=0.0, profit_factor=0.0,
                sharpe_ratio=None, max_drawdown=0.0, max_drawdown_pct=0.0,
                buy_win_rate=0.0, sell_win_rate=0.0, hold_skipped=0,
                period_days=days, first_trade=None, last_trade=None
            )

    def get_recent_trades(self, limit: int = 10) -> List[Signal]:
        """
        Get recent executed trades

        Args:
            limit: Number of trades to return

        Returns:
            List of Signal objects
        """
        try:
            session = get_session(self.db_url)
            try:
                trades = session.query(Signal).filter(
                    and_(
                        Signal.model_version == 'v7_ultimate',
                        Signal.executed == 1
                    )
                ).order_by(Signal.execution_time.desc()).limit(limit).all()

                return trades

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            return []
