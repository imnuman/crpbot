"""
HYDRA 4.0 - Turbo Tournament (Quick Historical Ranker)

Rapidly backtests and ranks strategies on 30 days of historical data.
Target: <2 seconds per strategy.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import threading

from libs.hydra.turbo_generator import GeneratedStrategy, StrategyType


@dataclass
class BacktestResult:
    """Results from quick backtest."""
    strategy_id: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl_percent: float
    avg_win_percent: float
    avg_loss_percent: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_holding_hours: float
    rank_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "total_pnl_percent": self.total_pnl_percent,
            "avg_win_percent": self.avg_win_percent,
            "avg_loss_percent": self.avg_loss_percent,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "avg_holding_hours": self.avg_holding_hours,
            "rank_score": self.rank_score,
        }


class TurboTournament:
    """
    Quick backtester and ranker for generated strategies.

    Features:
    - Fast backtest on 30 days historical data
    - Calculates: WR, Sharpe, Max DD, R:R
    - Multi-threaded for parallel processing
    - Target: <2 sec per strategy
    """

    # Ranking weights
    RANK_WEIGHTS = {
        "win_rate": 0.25,
        "sharpe_ratio": 0.30,
        "profit_factor": 0.20,
        "max_drawdown": 0.15,  # Penalize high DD
        "trade_count": 0.10,   # Reward more trades (statistical significance)
    }

    # Minimum thresholds for ranking
    MIN_TRADES = 5
    MIN_WIN_RATE = 0.40
    MAX_DRAWDOWN_LIMIT = 0.15  # 15% max DD

    def __init__(self, data_provider=None):
        """Initialize the turbo tournament."""
        self.data_provider = data_provider
        self._historical_cache: Dict[str, pd.DataFrame] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        logger.info("[TurboTournament] Initialized")

    def quick_rank_strategy(
        self,
        strategy: GeneratedStrategy,
        historical_data: Optional[pd.DataFrame] = None
    ) -> BacktestResult:
        """
        Quick backtest and rank a single strategy.

        Args:
            strategy: The strategy to test
            historical_data: Optional pre-loaded historical data

        Returns:
            BacktestResult with performance metrics
        """
        # Get historical data if not provided
        if historical_data is None:
            historical_data = self._get_historical_data(strategy.asset_class)

        if historical_data is None or len(historical_data) < 100:
            logger.warning(f"[TurboTournament] Insufficient data for {strategy.strategy_id}")
            return self._empty_result(strategy.strategy_id)

        # Run quick backtest
        trades = self._simulate_strategy(strategy, historical_data)

        if len(trades) < self.MIN_TRADES:
            return self._empty_result(strategy.strategy_id)

        # Calculate metrics
        result = self._calculate_metrics(strategy.strategy_id, trades)

        # Calculate rank score
        result.rank_score = self._calculate_rank(result)

        return result

    def rank_batch(
        self,
        strategies: List[GeneratedStrategy],
        max_workers: int = 4
    ) -> List[Tuple[GeneratedStrategy, BacktestResult]]:
        """
        Rank a batch of strategies in parallel.

        Args:
            strategies: List of strategies to rank
            max_workers: Number of parallel workers

        Returns:
            List of (strategy, result) tuples, sorted by rank_score descending
        """
        results = []

        # Pre-load historical data for each asset class
        asset_classes = set(s.asset_class for s in strategies)
        historical_data = {}
        for asset_class in asset_classes:
            historical_data[asset_class] = self._get_historical_data(asset_class)

        logger.info(f"[TurboTournament] Ranking {len(strategies)} strategies with {max_workers} workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_strategy = {
                executor.submit(
                    self.quick_rank_strategy,
                    strategy,
                    historical_data.get(strategy.asset_class)
                ): strategy
                for strategy in strategies
            }

            for future in as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    result = future.result()
                    # Update strategy with backtest results
                    strategy.backtest_wr = result.win_rate
                    strategy.backtest_sharpe = result.sharpe_ratio
                    strategy.backtest_max_dd = result.max_drawdown
                    strategy.backtest_trades = result.total_trades
                    strategy.rank_score = result.rank_score
                    results.append((strategy, result))
                except Exception as e:
                    logger.error(f"[TurboTournament] Error ranking {strategy.strategy_id}: {e}")
                    results.append((strategy, self._empty_result(strategy.strategy_id)))

        # Sort by rank score descending
        results.sort(key=lambda x: x[1].rank_score, reverse=True)

        logger.info(f"[TurboTournament] Ranked {len(results)} strategies")
        if results:
            top = results[0]
            logger.info(f"[TurboTournament] Top strategy: {top[0].strategy_id} (score: {top[1].rank_score:.3f})")

        return results

    def _get_historical_data(self, asset_class: str) -> Optional[pd.DataFrame]:
        """Get 30 days of historical data for asset class."""
        # Check cache
        cache_key = asset_class
        if cache_key in self._historical_cache:
            expiry = self._cache_expiry.get(cache_key)
            if expiry and datetime.now(timezone.utc) < expiry:
                return self._historical_cache[cache_key]

        # Map asset class to symbol
        symbol_map = {
            "BTC": "BTC-USD",
            "ETH": "ETH-USD",
            "ALTCOIN": "SOL-USD",  # Use SOL as proxy for altcoins
            "ALL": "BTC-USD",  # Default to BTC
        }
        symbol = symbol_map.get(asset_class, "BTC-USD")

        try:
            if self.data_provider:
                # Use real data provider
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(days=30)
                df = self.data_provider.fetch_klines(
                    symbol=symbol,
                    interval="1h",
                    start_time=start_time,
                    end_time=end_time
                )
            else:
                # Generate mock data for testing
                df = self._generate_mock_data(symbol, days=30)

            if df is not None and len(df) > 0:
                # Add technical indicators
                df = self._add_indicators(df)
                self._historical_cache[cache_key] = df
                self._cache_expiry[cache_key] = datetime.now(timezone.utc) + timedelta(hours=1)

            return df

        except Exception as e:
            logger.error(f"[TurboTournament] Error fetching data for {asset_class}: {e}")
            return self._generate_mock_data(symbol, days=30)

    def _generate_mock_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Generate mock OHLCV data for testing."""
        hours = days * 24
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=hours, freq='h')

        # Base price based on symbol
        base_prices = {"BTC-USD": 95000, "ETH-USD": 3500, "SOL-USD": 230}
        base = base_prices.get(symbol, 100)

        # Generate random walk
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(0, 0.01, hours)  # 1% hourly volatility
        prices = base * np.cumprod(1 + returns)

        # Generate OHLCV
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.uniform(-0.005, 0.005, hours)),
            'high': prices * (1 + np.random.uniform(0, 0.01, hours)),
            'low': prices * (1 - np.random.uniform(0, 0.01, hours)),
            'close': prices,
            'volume': np.random.uniform(1000, 10000, hours)
        })

        return df

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators for strategy simulation."""
        df = df.copy()

        # ATR (14-period)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()

        # Simple momentum
        df['momentum'] = df['close'].pct_change(periods=10)

        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(20).std()

        # Regime proxy (based on ADX-like calculation)
        df['trend_strength'] = np.abs(df['momentum']) / (df['volatility'] + 0.0001)

        return df.dropna()

    def _simulate_strategy(
        self,
        strategy: GeneratedStrategy,
        data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Simulate strategy on historical data.

        Returns list of simulated trades.
        """
        trades = []
        position = None
        atr_col = 'atr' if 'atr' in data.columns else None

        for i in range(20, len(data)):  # Start after warmup period
            row = data.iloc[i]
            prev_row = data.iloc[i-1]

            # Check for exit if in position
            if position:
                exit_price = None
                exit_reason = None

                # Check stop loss
                if position['direction'] == 'BUY':
                    if row['low'] <= position['stop_loss']:
                        exit_price = position['stop_loss']
                        exit_reason = 'stop_loss'
                    elif row['high'] >= position['take_profit']:
                        exit_price = position['take_profit']
                        exit_reason = 'take_profit'
                else:  # SELL
                    if row['high'] >= position['stop_loss']:
                        exit_price = position['stop_loss']
                        exit_reason = 'stop_loss'
                    elif row['low'] <= position['take_profit']:
                        exit_price = position['take_profit']
                        exit_reason = 'take_profit'

                # Check time exit
                if not exit_price:
                    hours_held = i - position['entry_idx']
                    if hours_held >= strategy.max_holding_hours:
                        exit_price = row['close']
                        exit_reason = 'time_exit'

                if exit_price:
                    # Calculate P&L
                    if position['direction'] == 'BUY':
                        pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
                    else:
                        pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']

                    trades.append({
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'direction': position['direction'],
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason,
                        'holding_hours': i - position['entry_idx'],
                        'outcome': 'win' if pnl_pct > 0 else 'loss'
                    })
                    position = None
                    continue

            # Check for entry if not in position
            if position is None and self._check_entry_signal(strategy, data, i):
                entry_price = row['close']
                atr = row[atr_col] if atr_col else entry_price * 0.01

                # Determine direction based on momentum
                direction = 'BUY' if row.get('momentum', 0) > 0 else 'SELL'

                # Calculate SL/TP
                if direction == 'BUY':
                    stop_loss = entry_price - (atr * strategy.stop_loss_atr_mult)
                    take_profit = entry_price + (atr * strategy.take_profit_atr_mult)
                else:
                    stop_loss = entry_price + (atr * strategy.stop_loss_atr_mult)
                    take_profit = entry_price - (atr * strategy.take_profit_atr_mult)

                position = {
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'direction': direction,
                    'entry_idx': i
                }

        return trades

    def _check_entry_signal(
        self,
        strategy: GeneratedStrategy,
        data: pd.DataFrame,
        idx: int
    ) -> bool:
        """
        Check if entry signal is triggered.

        Simplified signal based on strategy parameters.
        """
        row = data.iloc[idx]

        # Random entry with probability based on min_confidence
        # (In production, this would check actual specialty triggers)
        import random
        entry_probability = 0.02  # ~2% of bars trigger entry

        # Adjust based on regime match
        if 'trend_strength' in row:
            if strategy.regime == 'TRENDING_UP' and row.get('momentum', 0) > 0:
                entry_probability *= 1.5
            elif strategy.regime == 'TRENDING_DOWN' and row.get('momentum', 0) < 0:
                entry_probability *= 1.5
            elif strategy.regime == 'RANGING' and abs(row.get('momentum', 0)) < 0.01:
                entry_probability *= 1.5

        return random.random() < entry_probability

    def _calculate_metrics(self, strategy_id: str, trades: List[Dict]) -> BacktestResult:
        """Calculate performance metrics from trades."""
        if not trades:
            return self._empty_result(strategy_id)

        wins = [t for t in trades if t['outcome'] == 'win']
        losses = [t for t in trades if t['outcome'] == 'loss']

        total_trades = len(trades)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total_trades if total_trades > 0 else 0

        # P&L calculations
        pnls = [t['pnl_pct'] for t in trades]
        total_pnl = sum(pnls)

        avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0

        # Profit factor
        gross_profit = sum(t['pnl_pct'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['pnl_pct'] for t in losses)) if losses else 0.0001
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Sharpe ratio (simplified)
        if len(pnls) > 1:
            returns_std = np.std(pnls)
            sharpe = (np.mean(pnls) / returns_std * np.sqrt(252)) if returns_std > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

        # Avg holding hours
        avg_holding = np.mean([t['holding_hours'] for t in trades])

        return BacktestResult(
            strategy_id=strategy_id,
            total_trades=total_trades,
            wins=win_count,
            losses=loss_count,
            win_rate=win_rate,
            total_pnl_percent=total_pnl,
            avg_win_percent=avg_win,
            avg_loss_percent=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            avg_holding_hours=avg_holding,
            rank_score=0.0
        )

    def _calculate_rank(self, result: BacktestResult) -> float:
        """
        Calculate composite rank score.

        Higher is better. Range: 0-100
        """
        # Fail fast checks
        if result.total_trades < self.MIN_TRADES:
            return 0.0
        if result.win_rate < self.MIN_WIN_RATE:
            return 0.0
        if result.max_drawdown > self.MAX_DRAWDOWN_LIMIT:
            return 0.0

        # Normalize components to 0-1 range
        wr_score = min(result.win_rate, 0.70) / 0.70  # Cap at 70%
        sharpe_score = min(max(result.sharpe_ratio, 0), 3) / 3  # Cap at 3
        pf_score = min(max(result.profit_factor, 0), 3) / 3  # Cap at 3
        dd_score = 1 - (result.max_drawdown / self.MAX_DRAWDOWN_LIMIT)  # Lower is better
        trade_score = min(result.total_trades, 50) / 50  # Cap at 50 trades

        # Weighted sum
        rank = (
            self.RANK_WEIGHTS["win_rate"] * wr_score +
            self.RANK_WEIGHTS["sharpe_ratio"] * sharpe_score +
            self.RANK_WEIGHTS["profit_factor"] * pf_score +
            self.RANK_WEIGHTS["max_drawdown"] * dd_score +
            self.RANK_WEIGHTS["trade_count"] * trade_score
        ) * 100

        return round(rank, 2)

    def _empty_result(self, strategy_id: str) -> BacktestResult:
        """Return empty result for failed backtests."""
        return BacktestResult(
            strategy_id=strategy_id,
            total_trades=0,
            wins=0,
            losses=0,
            win_rate=0.0,
            total_pnl_percent=0.0,
            avg_win_percent=0.0,
            avg_loss_percent=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            avg_holding_hours=0.0,
            rank_score=0.0
        )

    def get_top_strategies(
        self,
        ranked_results: List[Tuple[GeneratedStrategy, BacktestResult]],
        top_n: int = 100
    ) -> List[GeneratedStrategy]:
        """Get top N strategies by rank score."""
        return [strategy for strategy, _ in ranked_results[:top_n]]


# Singleton instance
_tournament_instance: Optional[TurboTournament] = None
_tournament_lock = threading.Lock()


def get_turbo_tournament() -> TurboTournament:
    """Get or create the turbo tournament singleton (thread-safe)."""
    global _tournament_instance
    if _tournament_instance is None:
        with _tournament_lock:
            if _tournament_instance is None:
                _tournament_instance = TurboTournament()
    return _tournament_instance
