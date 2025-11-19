"""
V7 Ultimate - Paper Trading Module

Simplified paper trading system for manually tracking V7 signal execution.
Integrates with Telegram for easy trade recording and performance tracking.

Usage:
    When you receive a V7 signal via Telegram and decide to execute:
    1. /execute <signal_id> - Mark signal as executed at current price
    2. /close <signal_id> win|loss - Record outcome when trade closes
    3. /trades - View open trades
    4. /performance - View performance stats

The Bayesian learner automatically updates from recorded outcomes.
"""

from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass

from loguru import logger
from sqlalchemy import and_

from libs.db.models import Signal, get_session
from libs.performance.trade_tracker import TradeTracker, PerformanceStats
from libs.bayesian.bayesian_learner import BayesianLearner


@dataclass
class OpenTrade:
    """Represents an open paper trade"""
    signal_id: int
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    execution_time: datetime
    confidence: float
    tier: str


class PaperTrader:
    """
    Paper trading tracker for V7 Ultimate signals

    Workflow:
    1. User receives V7 signal via Telegram
    2. User manually executes trade on exchange
    3. User records execution: /execute <signal_id>
    4. System marks signal as executed, tracks entry
    5. When trade closes, user records outcome: /close <signal_id> win|loss
    6. System calculates PnL, feeds to Bayesian learner
    7. Bayesian learner adapts confidence calibration
    """

    def __init__(self, db_url: str = "sqlite:///tradingai.db"):
        """
        Initialize paper trader

        Args:
            db_url: Database connection string
        """
        self.db_url = db_url
        self.trade_tracker = TradeTracker(db_url)
        self.bayesian_learner = BayesianLearner(db_url)

        logger.info("Paper trader initialized")

    def execute_signal(
        self,
        signal_id: int,
        execution_price: Optional[float] = None
    ) -> Dict:
        """
        Mark a signal as executed (paper trade entry)

        Args:
            signal_id: ID of the signal to execute
            execution_price: Price at execution (if None, uses signal's entry_price)

        Returns:
            Dict with execution details
        """
        try:
            session = get_session(self.db_url)
            try:
                # Get signal
                signal = session.query(Signal).filter(Signal.id == signal_id).first()

                if not signal:
                    return {
                        "success": False,
                        "error": f"Signal {signal_id} not found"
                    }

                if signal.executed:
                    return {
                        "success": False,
                        "error": f"Signal {signal_id} already executed"
                    }

                # Use current price or signal's entry price
                exec_price = execution_price or signal.entry_price
                exec_time = datetime.utcnow()

                # Record execution
                success = self.trade_tracker.record_trade_execution(
                    signal_id=signal_id,
                    execution_time=exec_time,
                    execution_price=exec_price
                )

                if not success:
                    return {
                        "success": False,
                        "error": "Failed to record execution"
                    }

                logger.info(
                    f"Paper trade executed: {signal.symbol} {signal.direction.upper()} "
                    f"@ ${exec_price:.2f} (Signal #{signal_id})"
                )

                return {
                    "success": True,
                    "signal_id": signal_id,
                    "symbol": signal.symbol,
                    "direction": signal.direction,
                    "entry_price": exec_price,
                    "execution_time": exec_time.isoformat(),
                    "confidence": signal.confidence,
                    "tier": signal.tier
                }

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def close_trade(
        self,
        signal_id: int,
        result: str,  # 'win' or 'loss'
        exit_price: Optional[float] = None,
        pnl_pct: Optional[float] = None
    ) -> Dict:
        """
        Close a paper trade and record outcome

        Args:
            signal_id: ID of the signal
            result: 'win' or 'loss'
            exit_price: Price at exit (required if pnl_pct not provided)
            pnl_pct: Manual PnL percentage (optional)

        Returns:
            Dict with trade outcome details
        """
        try:
            session = get_session(self.db_url)
            try:
                # Get signal
                signal = session.query(Signal).filter(Signal.id == signal_id).first()

                if not signal:
                    return {
                        "success": False,
                        "error": f"Signal {signal_id} not found"
                    }

                if not signal.executed:
                    return {
                        "success": False,
                        "error": f"Signal {signal_id} was not executed"
                    }

                if signal.result in ['win', 'loss']:
                    return {
                        "success": False,
                        "error": f"Signal {signal_id} already closed ({signal.result})"
                    }

                # Validate result
                if result not in ['win', 'loss']:
                    return {
                        "success": False,
                        "error": f"Invalid result: {result} (must be 'win' or 'loss')"
                    }

                # Determine exit price
                if exit_price is None and pnl_pct is None:
                    return {
                        "success": False,
                        "error": "Must provide either exit_price or pnl_pct"
                    }

                # If pnl_pct provided, calculate exit price
                if pnl_pct is not None:
                    if signal.direction == 'long':
                        exit_price = signal.execution_price * (1 + pnl_pct / 100)
                    else:  # short
                        exit_price = signal.execution_price * (1 - pnl_pct / 100)

                # Record outcome
                exit_time = datetime.utcnow()
                success = self.trade_tracker.record_trade_outcome(
                    signal_id=signal_id,
                    exit_time=exit_time,
                    exit_price=exit_price,
                    result=result
                )

                if not success:
                    return {
                        "success": False,
                        "error": "Failed to record outcome"
                    }

                # Calculate PnL
                if signal.direction == 'long':
                    pnl = ((exit_price - signal.execution_price) / signal.execution_price) * 100
                elif signal.direction == 'short':
                    pnl = ((signal.execution_price - exit_price) / signal.execution_price) * 100
                else:
                    pnl = 0.0

                logger.info(
                    f"Paper trade closed: {signal.symbol} {signal.direction.upper()} "
                    f"Result: {result.upper()} | PnL: {pnl:+.2f}% (Signal #{signal_id})"
                )

                return {
                    "success": True,
                    "signal_id": signal_id,
                    "symbol": signal.symbol,
                    "direction": signal.direction,
                    "result": result,
                    "entry_price": signal.execution_price,
                    "exit_price": exit_price,
                    "pnl_pct": pnl,
                    "execution_time": signal.execution_time.isoformat() if signal.execution_time else None,
                    "exit_time": exit_time.isoformat()
                }

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error closing trade: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_open_trades(self) -> List[OpenTrade]:
        """
        Get all open paper trades

        Returns:
            List of OpenTrade objects
        """
        try:
            session = get_session(self.db_url)
            try:
                # Query open trades (executed but not closed)
                signals = session.query(Signal).filter(
                    and_(
                        Signal.model_version == 'v7_ultimate',
                        Signal.executed == 1,
                        Signal.result == 'pending'
                    )
                ).order_by(Signal.execution_time.desc()).all()

                open_trades = []
                for signal in signals:
                    open_trades.append(OpenTrade(
                        signal_id=signal.id,
                        symbol=signal.symbol,
                        direction=signal.direction,
                        entry_price=signal.execution_price,
                        execution_time=signal.execution_time,
                        confidence=signal.confidence,
                        tier=signal.tier
                    ))

                return open_trades

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error getting open trades: {e}")
            return []

    def get_performance_stats(self, days: int = 30) -> PerformanceStats:
        """
        Get performance statistics

        Args:
            days: Number of days to analyze

        Returns:
            PerformanceStats object
        """
        return self.trade_tracker.get_performance_stats(days=days)

    def get_bayesian_summary(self, days: int = 30) -> Dict:
        """
        Get Bayesian learning summary

        Args:
            days: Number of days to analyze

        Returns:
            Dict with Bayesian learning insights
        """
        return self.bayesian_learner.get_learning_summary(days=days)

    def get_recent_trades(self, limit: int = 10) -> List[Signal]:
        """
        Get recent closed trades

        Args:
            limit: Number of trades to return

        Returns:
            List of Signal objects
        """
        return self.trade_tracker.get_recent_trades(limit=limit)


# Convenience functions for Telegram bot integration
def format_open_trades_message(trades: List[OpenTrade]) -> str:
    """Format open trades for Telegram message"""
    if not trades:
        return "No open paper trades."

    msg = f"*Open Paper Trades ({len(trades)}):*\n\n"

    for trade in trades:
        duration = datetime.utcnow() - trade.execution_time
        hours = int(duration.total_seconds() / 3600)
        minutes = int((duration.total_seconds() % 3600) / 60)

        msg += (
            f"*#{trade.signal_id}* - {trade.symbol} {trade.direction.upper()}\n"
            f"  Entry: ${trade.execution_price:.2f}\n"
            f"  Confidence: {trade.confidence*100:.1f}% ({trade.tier})\n"
            f"  Duration: {hours}h {minutes}m\n"
            f"  _Close with:_ /close {trade.signal_id} win|loss\n\n"
        )

    return msg


def format_performance_message(stats: PerformanceStats) -> str:
    """Format performance stats for Telegram message"""
    if stats.total_trades == 0:
        return "No closed paper trades yet. Execute signals with /execute <signal_id>"

    msg = f"*Paper Trading Performance ({stats.period_days} days):*\n\n"

    msg += f"*Overall:*\n"
    msg += f"  Trades: {stats.total_trades} ({stats.wins}W / {stats.losses}L / {stats.pending} open)\n"
    msg += f"  Win Rate: {stats.win_rate:.1f}%\n"
    msg += f"  Total PnL: {stats.total_pnl:+.2f}%\n"
    msg += f"  Avg Win: +{stats.avg_win:.2f}%\n"
    msg += f"  Avg Loss: {stats.avg_loss:.2f}%\n"
    msg += f"  Profit Factor: {stats.profit_factor:.2f}\n\n"

    if stats.buy_win_rate > 0 or stats.sell_win_rate > 0:
        msg += f"*By Direction:*\n"
        if stats.buy_win_rate > 0:
            msg += f"  Long: {stats.buy_win_rate:.1f}% win rate\n"
        if stats.sell_win_rate > 0:
            msg += f"  Short: {stats.sell_win_rate:.1f}% win rate\n"
        msg += "\n"

    if stats.sharpe_ratio:
        msg += f"*Risk Metrics:*\n"
        msg += f"  Sharpe Ratio: {stats.sharpe_ratio:.2f}\n"
        msg += f"  Max Drawdown: {stats.max_drawdown:.2f}% ({stats.max_drawdown_pct:.1f}%)\n"

    return msg


if __name__ == "__main__":
    # Test paper trader
    import sys

    print("=" * 80)
    print("V7 Ultimate - Paper Trading Module Test")
    print("=" * 80)

    trader = PaperTrader()

    # Test: Get open trades
    print("\n1. Testing get_open_trades()...")
    open_trades = trader.get_open_trades()
    print(f"   Found {len(open_trades)} open trades")
    if open_trades:
        print(format_open_trades_message(open_trades))

    # Test: Get performance stats
    print("\n2. Testing get_performance_stats()...")
    stats = trader.get_performance_stats(days=30)
    print(format_performance_message(stats))

    # Test: Get Bayesian summary
    print("\n3. Testing get_bayesian_summary()...")
    bayesian = trader.get_bayesian_summary(days=30)
    print(f"   Overall Win Rate: {bayesian.get('overall', {}).get('win_rate', 0)*100:.1f}%")
    print(f"   Sample Size: {bayesian.get('overall', {}).get('sample_size', 0)} trades")

    print("\n" + "=" * 80)
    print("Paper Trading Module Test Complete!")
    print("=" * 80)
