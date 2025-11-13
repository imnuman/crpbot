"""Backtest engine with empirical FTMO execution model."""
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger

from apps.trainer.features import get_trading_session
from libs.constants import (
    BASIS_POINTS_CONVERSION,
    EPSILON_DIV_BY_ZERO,
    EXPECTED_WIN_RATE_DEFAULT,
    EXPECTED_WIN_RATE_HIGH,
    EXPECTED_WIN_RATE_LOW,
    EXPECTED_WIN_RATE_MEDIUM,
    INITIAL_BALANCE,
    LATENCY_BUDGET_MS,
    LATENCY_PENALTY_MULTIPLIER,
    P90_PERCENTILE,
    RISK_PER_TRADE,
    RISK_REWARD_RATIO,
    TRADING_DAYS_PER_YEAR,
)
from libs.rl_env.execution_model import ExecutionModel


@dataclass
class Trade:
    """Represents a single trade."""

    entry_time: datetime
    entry_price: float
    exit_time: datetime | None
    exit_price: float | None
    direction: str  # 'long' or 'short'
    signal_confidence: float
    tier: str  # 'high', 'medium', 'low'
    symbol: str
    session: str
    tp_price: float | None = None
    sl_price: float | None = None
    rr_expected: float | None = None
    latency_ms: float = 0.0
    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    result: str | None = None  # 'win', 'loss', None (open)
    pnl: float = 0.0
    r_realized: float = 0.0


@dataclass
class BacktestMetrics:
    """Comprehensive backtest metrics."""

    # Basic metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl_per_trade: float

    # Risk metrics
    max_drawdown: float
    avg_drawdown: float
    sharpe_ratio: float

    # Per-tier metrics
    tier_metrics: dict[str, dict[str, float]]

    # Session metrics
    session_metrics: dict[str, dict[str, float]]

    # Calibration metrics
    brier_score: float
    calibration_error: float

    # Latency metrics
    avg_latency_ms: float
    p90_latency_ms: float
    latency_penalized_pnl: float

    # Hit rate by session
    hit_rate_by_session: dict[str, float]

    def __str__(self) -> str:
        """String representation."""
        lines = [
            "=" * 60,
            "Backtest Metrics",
            "=" * 60,
            f"Total Trades: {self.total_trades}",
            f"Win Rate: {self.win_rate:.2%}",
            f"Winning Trades: {self.winning_trades}",
            f"Losing Trades: {self.losing_trades}",
            f"Total PnL: ${self.total_pnl:.2f}",
            f"Avg PnL per Trade: ${self.avg_pnl_per_trade:.2f}",
            f"Max Drawdown: {self.max_drawdown:.2%}",
            f"Avg Drawdown: {self.avg_drawdown:.2%}",
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}",
            f"Brier Score: {self.brier_score:.4f}",
            f"Calibration Error: {self.calibration_error:.2%}",
            f"Avg Latency: {self.avg_latency_ms:.2f}ms",
            f"P90 Latency: {self.p90_latency_ms:.2f}ms",
        ]
        return "\n".join(lines)


class BacktestEngine:
    """
    Backtest engine with empirical FTMO execution model.

    Features:
    - Uses execution model for realistic costs
    - Measures latency and applies penalties
    - Tracks trades with full metrics
    - Calculates enhanced metrics (precision/recall, Brier score, drawdown, etc.)
    """

    def __init__(
        self,
        execution_model: ExecutionModel | None = None,
        latency_budget_ms: float = LATENCY_BUDGET_MS,
        initial_balance: float = INITIAL_BALANCE,
        risk_per_trade: float = RISK_PER_TRADE,
        rr_ratio: float = RISK_REWARD_RATIO,
    ):
        """
        Initialize backtest engine.

        Args:
            execution_model: Execution model for realistic costs
            latency_budget_ms: Latency budget in milliseconds
            initial_balance: Initial account balance
            risk_per_trade: Risk per trade as fraction of balance
            rr_ratio: Risk:Reward ratio (default: 2.0)
        """
        self.execution_model = execution_model or ExecutionModel()
        self.latency_budget_ms = latency_budget_ms
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.rr_ratio = rr_ratio

        self.trades: list[Trade] = []
        self.balance_history: list[tuple[datetime, float]] = [(datetime.now(timezone.utc), initial_balance)]
        self.current_balance = initial_balance

    def execute_trade(
        self,
        entry_time: datetime,
        entry_price: float,
        direction: str,
        signal_confidence: float,
        tier: str,
        symbol: str,
        latency_ms: float = 0.0,
    ) -> Trade:
        """
        Execute a trade with execution costs.

        Args:
            entry_time: Entry timestamp
            entry_price: Theoretical entry price
            direction: 'long' or 'short'
            signal_confidence: Signal confidence (0-1)
            tier: Confidence tier ('high', 'medium', 'low')
            symbol: Trading pair
            latency_ms: Decision latency in milliseconds

        Returns:
            Trade object with execution details
        """
        session = get_trading_session(entry_time)

        # Sample execution costs
        spread_bps = self.execution_model.sample_spread(
            symbol=symbol, session=session, timestamp=entry_time, use_p90=False
        )
        slippage_bps = self.execution_model.sample_slippage(
            symbol=symbol,
            session=session,
            timestamp=entry_time,
            latency_ms=latency_ms,
            latency_budget_ms=self.latency_budget_ms,
            use_p90=False,
        )
        total_cost_bps = spread_bps + slippage_bps

        # Calculate actual entry price with execution costs
        price_impact = entry_price * (total_cost_bps / BASIS_POINTS_CONVERSION)
        if direction == "long":
            actual_entry = entry_price + price_impact
        else:  # short
            actual_entry = entry_price - price_impact

        # Calculate position size based on risk
        risk_amount = self.current_balance * self.risk_per_trade
        if direction == "long":
            sl_price = actual_entry * (1 - risk_amount / (actual_entry * 100))
            tp_price = actual_entry + (actual_entry - sl_price) * self.rr_ratio
        else:
            sl_price = actual_entry * (1 + risk_amount / (actual_entry * 100))
            tp_price = actual_entry - (sl_price - actual_entry) * self.rr_ratio

        trade = Trade(
            entry_time=entry_time,
            entry_price=actual_entry,
            exit_time=None,
            exit_price=None,
            direction=direction,
            signal_confidence=signal_confidence,
            tier=tier,
            symbol=symbol,
            session=session,
            tp_price=tp_price,
            sl_price=sl_price,
            rr_expected=self.rr_ratio,
            latency_ms=latency_ms,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
        )

        self.trades.append(trade)
        return trade

    def close_trade(self, trade: Trade, exit_time: datetime, exit_price: float, reason: str = "manual") -> None:
        """
        Close a trade and calculate PnL.

        Args:
            trade: Trade to close
            exit_time: Exit timestamp
            exit_price: Exit price
            reason: Exit reason ('tp', 'sl', 'manual')
        """
        trade.exit_time = exit_time
        trade.exit_price = exit_price

        # Calculate PnL
        if trade.direction == "long":
            pnl = (exit_price - trade.entry_price) / trade.entry_price
        else:
            pnl = (trade.entry_price - exit_price) / trade.entry_price

        trade.pnl = pnl * self.current_balance * self.risk_per_trade
        trade.result = "win" if pnl > 0 else "loss"

        # Calculate R realized
        if trade.sl_price and trade.direction == "long":
            risk = trade.entry_price - trade.sl_price
            if risk > 0:
                trade.r_realized = (exit_price - trade.entry_price) / risk
        elif trade.sl_price and trade.direction == "short":
            risk = trade.sl_price - trade.entry_price
            if risk > 0:
                trade.r_realized = (trade.entry_price - exit_price) / risk

        # Update balance
        self.current_balance += trade.pnl
        self.balance_history.append((exit_time, self.current_balance))

    def calculate_metrics(self) -> BacktestMetrics:
        """
        Calculate comprehensive backtest metrics.

        Returns:
            BacktestMetrics object
        """
        if not self.trades:
            logger.warning("No trades to calculate metrics")
            return BacktestMetrics(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                avg_pnl_per_trade=0.0,
                max_drawdown=0.0,
                avg_drawdown=0.0,
                sharpe_ratio=0.0,
                tier_metrics={},
                session_metrics={},
                brier_score=0.0,
                calibration_error=0.0,
                avg_latency_ms=0.0,
                p90_latency_ms=0.0,
                latency_penalized_pnl=0.0,
                hit_rate_by_session={},
            )

        # Basic metrics
        closed_trades = [t for t in self.trades if t.result is not None]
        winning_trades = [t for t in closed_trades if t.result == "win"]
        losing_trades = [t for t in closed_trades if t.result == "loss"]

        total_trades = len(closed_trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0

        total_pnl = sum(t.pnl for t in closed_trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0.0

        # Drawdown calculation (balance_history is already chronological, no need to sort)
        balances = [b for _, b in self.balance_history]
        if len(balances) > 1:
            peak = balances[0]
            drawdowns = []
            for balance in balances[1:]:
                if balance > peak:
                    peak = balance
                drawdown = (peak - balance) / peak if peak > 0 else 0.0
                drawdowns.append(drawdown)
            max_drawdown = max(drawdowns) if drawdowns else 0.0
            avg_drawdown = np.mean(drawdowns) if drawdowns else 0.0
        else:
            max_drawdown = 0.0
            avg_drawdown = 0.0

        # Sharpe ratio (simplified)
        returns = [t.pnl / self.initial_balance for t in closed_trades]
        sharpe_ratio = (
            np.mean(returns) / (np.std(returns) + EPSILON_DIV_BY_ZERO) * np.sqrt(TRADING_DAYS_PER_YEAR)
            if len(returns) > 1
            else 0.0
        )

        # Per-tier metrics
        tier_metrics = {}
        for tier in ["high", "medium", "low"]:
            tier_trades = [t for t in closed_trades if t.tier == tier]
            if tier_trades:
                tier_win_rate = sum(1 for t in tier_trades if t.result == "win") / len(tier_trades)
                tier_pnl = sum(t.pnl for t in tier_trades)
                tier_metrics[tier] = {
                    "trades": len(tier_trades),
                    "win_rate": tier_win_rate,
                    "total_pnl": tier_pnl,
                    "avg_pnl": tier_pnl / len(tier_trades) if tier_trades else 0.0,
                }

        # Session metrics
        session_metrics = {}
        for session in ["tokyo", "london", "new_york"]:
            session_trades = [t for t in closed_trades if t.session == session]
            if session_trades:
                session_win_rate = sum(1 for t in session_trades if t.result == "win") / len(session_trades)
                session_pnl = sum(t.pnl for t in session_trades)
                session_metrics[session] = {
                    "trades": len(session_trades),
                    "win_rate": session_win_rate,
                    "total_pnl": session_pnl,
                    "avg_pnl": session_pnl / len(session_trades) if session_trades else 0.0,
                }

        # Brier score (calibration metric)
        confidences = [t.signal_confidence for t in closed_trades]
        outcomes = [1.0 if t.result == "win" else 0.0 for t in closed_trades]
        if confidences and outcomes:
            brier_score = np.mean([(c - o) ** 2 for c, o in zip(confidences, outcomes)])
        else:
            brier_score = 0.0

        # Calibration error (tier MAE)
        tier_calibration_errors = []
        expected_win_rates = {
            "high": EXPECTED_WIN_RATE_HIGH,
            "medium": EXPECTED_WIN_RATE_MEDIUM,
            "low": EXPECTED_WIN_RATE_LOW,
        }
        for tier, metrics in tier_metrics.items():
            expected_win_rate = expected_win_rates.get(tier, EXPECTED_WIN_RATE_DEFAULT)
            actual_win_rate = metrics["win_rate"]
            tier_calibration_errors.append(abs(expected_win_rate - actual_win_rate))
        calibration_error = np.mean(tier_calibration_errors) if tier_calibration_errors else 0.0

        # Latency metrics
        latencies = [t.latency_ms for t in closed_trades]
        avg_latency = np.mean(latencies) if latencies else 0.0
        p90_latency = np.percentile(latencies, P90_PERCENTILE) if latencies else 0.0

        # Latency-penalized PnL
        latency_penalized_pnl = sum(
            t.pnl * (1.0 if t.latency_ms <= self.latency_budget_ms else LATENCY_PENALTY_MULTIPLIER)
            for t in closed_trades
        )

        # Hit rate by session
        hit_rate_by_session = {}
        for session, metrics in session_metrics.items():
            hit_rate_by_session[session] = metrics["win_rate"]

        return BacktestMetrics(
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_pnl_per_trade=avg_pnl,
            max_drawdown=max_drawdown,
            avg_drawdown=avg_drawdown,
            sharpe_ratio=sharpe_ratio,
            tier_metrics=tier_metrics,
            session_metrics=session_metrics,
            brier_score=brier_score,
            calibration_error=calibration_error,
            avg_latency_ms=avg_latency,
            p90_latency_ms=p90_latency,
            latency_penalized_pnl=latency_penalized_pnl,
            hit_rate_by_session=hit_rate_by_session,
        )

