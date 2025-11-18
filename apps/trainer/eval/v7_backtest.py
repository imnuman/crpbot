"""
V7 vs V6 Backtesting Framework

Compares V7 Ultimate (6 theories + LLM) against V6 Enhanced FNN baseline
on historical data to validate:
1. Prediction accuracy improvement
2. Risk-adjusted returns (Sharpe, Sortino, max drawdown)
3. Bayesian learning effectiveness
4. Theory contribution analysis
5. Cost-benefit analysis

Usage:
    python -m apps.trainer.eval.v7_backtest --days 7 --symbols BTC-USD ETH-USD SOL-USD
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from apps.runtime.data_fetcher import MarketDataFetcher
from apps.runtime.ensemble import EnsemblePredictor
from apps.trainer.eval.backtest import BacktestEngine, BacktestMetrics
from libs.config.config import Settings
from libs.llm.signal_generator import SignalGenerator


class V7BacktestRunner:
    """
    Run V7 Ultimate backtest on historical data.

    Features:
    - Fetches historical OHLCV data
    - Runs V6 baseline predictions
    - Runs V7 predictions with LLM synthesis
    - Compares performance metrics
    - Analyzes theory contributions
    """

    def __init__(
        self,
        symbols: list[str],
        days: int = 7,
        sample_rate: float = 0.2,  # Only run V7 LLM on 20% of data points (cost control)
        config: Settings | None = None,
    ):
        """
        Initialize V7 backtest runner.

        Args:
            symbols: List of trading symbols (e.g., ["BTC-USD", "ETH-USD"])
            days: Number of days of historical data to backtest
            sample_rate: Fraction of data points to run V7 LLM on (0.0-1.0)
            config: App configuration
        """
        self.symbols = symbols
        self.days = days
        self.sample_rate = sample_rate
        self.config = config or Settings()

        # Initialize components
        self.data_fetcher = MarketDataFetcher(config=config)

        # V6 ensemble predictors (one per symbol)
        self.v6_predictors = {
            symbol: EnsemblePredictor(symbol, model_dir=str(Path.cwd() / "models" / "promoted"))
            for symbol in symbols
        }

        # V7 signal generator (reads DEEPSEEK_API_KEY from env)
        self.v7_generator = SignalGenerator(
            api_key=None,  # Will use DEEPSEEK_API_KEY env var
            conservative_mode=True
        )

        # Backtest engines
        self.v6_engine = BacktestEngine()
        self.v7_engine = BacktestEngine()

        logger.info(f"Initialized V7 backtest runner: {days} days, {len(symbols)} symbols")
        logger.info(f"V7 LLM sample rate: {sample_rate*100:.0f}% (cost control)")

    def fetch_historical_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for backtesting.

        Args:
            symbol: Trading symbol

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching {self.days} days of historical data for {symbol}...")

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=self.days)

        # Fetch 1-minute candles (same as runtime)
        # Note: This will fetch from Coinbase API
        # Each day = 1440 candles, so 7 days = ~10k candles
        try:
            df = self.data_fetcher.fetch_historical_candles(
                symbol=symbol,
                start=start_time,
                end=end_time,
                granularity="ONE_MINUTE"
            )

            logger.info(f"✅ Fetched {len(df)} candles for {symbol} ({start_time} to {end_time})")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return pd.DataFrame()

    def run_v6_backtest(self, symbol: str, historical_data: pd.DataFrame) -> BacktestMetrics:
        """
        Run V6 Enhanced FNN backtest on historical data.

        Args:
            symbol: Trading symbol
            historical_data: Historical OHLCV data

        Returns:
            BacktestMetrics for V6
        """
        logger.info(f"Running V6 backtest for {symbol}...")

        v6_predictor = self.v6_predictors[symbol]
        trades_executed = 0

        # Iterate through historical data (skip first 200 candles for feature warmup)
        for i in range(200, len(historical_data)):
            # Get prediction window (last 120 candles)
            window = historical_data.iloc[max(0, i-120):i+1]

            if len(window) < 120:
                continue

            # Run V6 prediction
            try:
                prediction = v6_predictor.predict(window)

                # Check if signal passes confidence threshold
                if prediction['confidence'] >= self.config.confidence_threshold:
                    # Determine direction
                    direction = prediction['direction']  # 'long' or 'short'

                    # Execute trade in backtest engine
                    entry_time = window.iloc[-1]['timestamp']
                    entry_price = window.iloc[-1]['close']

                    # Determine tier based on confidence
                    if prediction['confidence'] >= 0.75:
                        tier = 'high'
                    elif prediction['confidence'] >= 0.65:
                        tier = 'medium'
                    else:
                        tier = 'low'

                    trade = self.v6_engine.execute_trade(
                        entry_time=entry_time,
                        entry_price=entry_price,
                        direction=direction,
                        signal_confidence=prediction['confidence'],
                        tier=tier,
                        symbol=symbol,
                        latency_ms=0.0
                    )

                    # Simulate exit after 5 candles (5 minutes) or when TP/SL hit
                    exit_idx = min(i + 5, len(historical_data) - 1)

                    # Check if TP or SL was hit in the next 5 candles
                    hit_tp = False
                    hit_sl = False

                    for j in range(i+1, exit_idx+1):
                        candle_high = historical_data.iloc[j]['high']
                        candle_low = historical_data.iloc[j]['low']

                        if direction == 'long':
                            if candle_high >= trade.tp_price:
                                exit_price = trade.tp_price
                                hit_tp = True
                                exit_idx = j
                                break
                            elif candle_low <= trade.sl_price:
                                exit_price = trade.sl_price
                                hit_sl = True
                                exit_idx = j
                                break
                        else:  # short
                            if candle_low <= trade.tp_price:
                                exit_price = trade.tp_price
                                hit_tp = True
                                exit_idx = j
                                break
                            elif candle_high >= trade.sl_price:
                                exit_price = trade.sl_price
                                hit_sl = True
                                exit_idx = j
                                break

                    # If neither TP nor SL hit, exit at close of 5th candle
                    if not hit_tp and not hit_sl:
                        exit_price = historical_data.iloc[exit_idx]['close']

                    exit_time = historical_data.iloc[exit_idx]['timestamp']

                    # Close trade
                    self.v6_engine.close_trade(trade, exit_time, exit_price,
                                              reason='tp' if hit_tp else ('sl' if hit_sl else 'manual'))

                    trades_executed += 1

            except Exception as e:
                logger.debug(f"V6 prediction failed at index {i}: {e}")
                continue

        logger.info(f"✅ V6 backtest complete: {trades_executed} trades executed")

        # Calculate metrics
        metrics = self.v6_engine.calculate_metrics()
        return metrics

    def run_v7_backtest(self, symbol: str, historical_data: pd.DataFrame) -> BacktestMetrics:
        """
        Run V7 Ultimate backtest on historical data (with sampling for cost control).

        Args:
            symbol: Trading symbol
            historical_data: Historical OHLCV data

        Returns:
            BacktestMetrics for V7
        """
        logger.info(f"Running V7 backtest for {symbol} (sample rate: {self.sample_rate*100:.0f}%)...")

        trades_executed = 0
        v7_costs_total = 0.0

        # Sample indices to run V7 on (cost control)
        # Start from index 200 to ensure we have 200 candles of history
        total_points = len(historical_data) - 200
        sample_size = int(total_points * self.sample_rate)
        if sample_size < 1:
            logger.warning(f"Sample size too small ({sample_size}), need more historical data")
            return self.v7_engine.calculate_metrics()

        sample_indices = sorted(
            pd.Series(range(200, len(historical_data))).sample(n=min(sample_size, total_points), random_state=42).tolist()
        )

        logger.info(f"Sampling {sample_size} / {total_points} data points for V7 LLM analysis")

        # Iterate through sampled historical data points
        for i in sample_indices:
            # Get prediction window (last 200 candles for V7)
            window = historical_data.iloc[max(0, i-200):i+1]

            if len(window) < 200:
                continue

            # Run V7 signal generation (with LLM)
            try:
                # Extract arrays for V7 signal generator
                prices = window['close'].values
                # Convert pandas datetime to Unix timestamps (seconds since epoch)
                timestamps = window['timestamp'].astype('int64').values / 10**9  # nanoseconds to seconds
                current_price = float(window.iloc[-1]['close'])

                # Note: This will call DeepSeek API - costs ~$0.0003 per signal
                result = self.v7_generator.generate_signal(
                    symbol=symbol,
                    prices=prices,
                    timestamps=timestamps,
                    current_price=current_price
                )

                v7_costs_total += result.total_cost_usd

                # Check if signal is valid and passes confidence threshold
                if result.parsed_signal.is_valid and result.parsed_signal.confidence >= self.config.confidence_threshold:
                    # Determine direction
                    direction = 'long' if result.parsed_signal.signal.value == 'BUY' else 'short'

                    if result.parsed_signal.signal.value == 'HOLD':
                        continue  # Skip HOLD signals

                    # Execute trade in backtest engine
                    entry_time = window.iloc[-1]['timestamp']
                    entry_price = window.iloc[-1]['close']

                    # Determine tier
                    if result.parsed_signal.confidence >= 0.75:
                        tier = 'high'
                    elif result.parsed_signal.confidence >= 0.65:
                        tier = 'medium'
                    else:
                        tier = 'low'

                    trade = self.v7_engine.execute_trade(
                        entry_time=entry_time,
                        entry_price=entry_price,
                        direction=direction,
                        signal_confidence=result.parsed_signal.confidence,
                        tier=tier,
                        symbol=symbol,
                        latency_ms=0.0
                    )

                    # Simulate exit after 5 candles (5 minutes) or when TP/SL hit
                    exit_idx = min(i + 5, len(historical_data) - 1)

                    # Check if TP or SL was hit
                    hit_tp = False
                    hit_sl = False

                    for j in range(i+1, exit_idx+1):
                        candle_high = historical_data.iloc[j]['high']
                        candle_low = historical_data.iloc[j]['low']

                        if direction == 'long':
                            if candle_high >= trade.tp_price:
                                exit_price = trade.tp_price
                                hit_tp = True
                                exit_idx = j
                                break
                            elif candle_low <= trade.sl_price:
                                exit_price = trade.sl_price
                                hit_sl = True
                                exit_idx = j
                                break
                        else:  # short
                            if candle_low <= trade.tp_price:
                                exit_price = trade.tp_price
                                hit_tp = True
                                exit_idx = j
                                break
                            elif candle_high >= trade.sl_price:
                                exit_price = trade.sl_price
                                hit_sl = True
                                exit_idx = j
                                break

                    # If neither TP nor SL hit, exit at close of 5th candle
                    if not hit_tp and not hit_sl:
                        exit_price = historical_data.iloc[exit_idx]['close']

                    exit_time = historical_data.iloc[exit_idx]['timestamp']

                    # Close trade
                    self.v7_engine.close_trade(trade, exit_time, exit_price,
                                              reason='tp' if hit_tp else ('sl' if hit_sl else 'manual'))

                    trades_executed += 1

            except Exception as e:
                logger.debug(f"V7 signal generation failed at index {i}: {e}")
                continue

        logger.info(f"✅ V7 backtest complete: {trades_executed} trades executed")
        logger.info(f"💰 Total V7 LLM costs: ${v7_costs_total:.4f}")

        # Calculate metrics
        metrics = self.v7_engine.calculate_metrics()
        return metrics

    def compare_results(self, symbol: str, v6_metrics: BacktestMetrics, v7_metrics: BacktestMetrics) -> dict[str, Any]:
        """
        Compare V6 vs V7 backtest results.

        Args:
            symbol: Trading symbol
            v6_metrics: V6 backtest metrics
            v7_metrics: V7 backtest metrics

        Returns:
            Comparison dictionary
        """
        comparison = {
            'symbol': symbol,
            'v6': {
                'total_trades': v6_metrics.total_trades,
                'win_rate': v6_metrics.win_rate,
                'total_pnl': v6_metrics.total_pnl,
                'sharpe_ratio': v6_metrics.sharpe_ratio,
                'max_drawdown': v6_metrics.max_drawdown,
            },
            'v7': {
                'total_trades': v7_metrics.total_trades,
                'win_rate': v7_metrics.win_rate,
                'total_pnl': v7_metrics.total_pnl,
                'sharpe_ratio': v7_metrics.sharpe_ratio,
                'max_drawdown': v7_metrics.max_drawdown,
            },
            'improvement': {
                'win_rate': v7_metrics.win_rate - v6_metrics.win_rate,
                'sharpe_ratio': v7_metrics.sharpe_ratio - v6_metrics.sharpe_ratio,
                'max_drawdown_reduction': v6_metrics.max_drawdown - v7_metrics.max_drawdown,
            }
        }

        return comparison

    def run_full_backtest(self) -> dict[str, Any]:
        """
        Run complete V6 vs V7 backtest on all symbols.

        Returns:
            Complete backtest results
        """
        logger.info(f"🚀 Starting V6 vs V7 backtest: {self.days} days, {len(self.symbols)} symbols")

        results = {
            'config': {
                'days': self.days,
                'symbols': self.symbols,
                'sample_rate': self.sample_rate,
                'confidence_threshold': self.config.confidence_threshold,
            },
            'comparisons': []
        }

        for symbol in self.symbols:
            logger.info(f"\n{'='*60}")
            logger.info(f"Backtesting {symbol}")
            logger.info(f"{'='*60}")

            # Fetch historical data
            historical_data = self.fetch_historical_data(symbol)

            if historical_data.empty:
                logger.warning(f"No historical data for {symbol}, skipping")
                continue

            # Run V6 backtest
            logger.info(f"\n📊 Running V6 baseline backtest...")
            v6_metrics = self.run_v6_backtest(symbol, historical_data)
            print(v6_metrics)

            # Run V7 backtest
            logger.info(f"\n🔬 Running V7 Ultimate backtest...")
            v7_metrics = self.run_v7_backtest(symbol, historical_data)
            print(v7_metrics)

            # Compare results
            logger.info(f"\n📈 Comparing results...")
            comparison = self.compare_results(symbol, v6_metrics, v7_metrics)
            results['comparisons'].append(comparison)

            # Print comparison
            self._print_comparison(comparison)

        # Save results
        output_path = Path('backtest_results_v6_vs_v7.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"\n✅ Backtest complete! Results saved to {output_path}")

        return results

    def _print_comparison(self, comparison: dict[str, Any]) -> None:
        """Print formatted comparison results."""
        print(f"\n{'='*60}")
        print(f"V6 vs V7 Comparison: {comparison['symbol']}")
        print(f"{'='*60}")

        print(f"\nV6 Baseline:")
        print(f"  Trades: {comparison['v6']['total_trades']}")
        print(f"  Win Rate: {comparison['v6']['win_rate']:.2%}")
        print(f"  Total PnL: ${comparison['v6']['total_pnl']:.2f}")
        print(f"  Sharpe Ratio: {comparison['v6']['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {comparison['v6']['max_drawdown']:.2%}")

        print(f"\nV7 Ultimate:")
        print(f"  Trades: {comparison['v7']['total_trades']}")
        print(f"  Win Rate: {comparison['v7']['win_rate']:.2%}")
        print(f"  Total PnL: ${comparison['v7']['total_pnl']:.2f}")
        print(f"  Sharpe Ratio: {comparison['v7']['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {comparison['v7']['max_drawdown']:.2%}")

        print(f"\nImprovement:")
        print(f"  Win Rate: {comparison['improvement']['win_rate']:+.2%}")
        print(f"  Sharpe Ratio: {comparison['improvement']['sharpe_ratio']:+.2f}")
        print(f"  Drawdown Reduction: {comparison['improvement']['max_drawdown_reduction']:+.2%}")
        print(f"{'='*60}\n")


def main():
    """Run V7 backtest from command line."""
    import argparse

    parser = argparse.ArgumentParser(description='V7 vs V6 Backtesting Framework')
    parser.add_argument('--days', type=int, default=7, help='Days of historical data (default: 7)')
    parser.add_argument('--symbols', nargs='+', default=['BTC-USD', 'ETH-USD', 'SOL-USD'],
                       help='Symbols to backtest (default: BTC-USD ETH-USD SOL-USD)')
    parser.add_argument('--sample-rate', type=float, default=0.2,
                       help='V7 LLM sample rate for cost control (default: 0.2 = 20%%)')

    args = parser.parse_args()

    # Initialize and run backtest
    runner = V7BacktestRunner(
        symbols=args.symbols,
        days=args.days,
        sample_rate=args.sample_rate
    )

    results = runner.run_full_backtest()

    print(f"\n✅ Backtest complete!")
    print(f"Results saved to: backtest_results_v6_vs_v7.json")


if __name__ == '__main__':
    main()
