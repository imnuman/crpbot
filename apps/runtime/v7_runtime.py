"""
V7 Ultimate Trading Runtime
Integrates 6 mathematical theories + DeepSeek LLM for signal generation.

Architecture:
1. Fetch live market data (100+ minutes of 1m candles)
2. Run mathematical analysis (Shannon, Hurst, Markov, Kalman, Bayesian, Monte Carlo)
3. Send analysis to DeepSeek LLM for signal synthesis
4. Parse LLM response into structured signal
5. Apply FTMO rules and rate limiting
6. Output signal (console, database, Telegram)

Cost Controls:
- Track cumulative API costs
- Enforce daily/monthly budget limits
- Rate limit signal generation (max signals per hour)

Usage:
    runtime = V7TradingRuntime()
    runtime.run(iterations=-1, sleep_seconds=120)  # Run continuously
"""
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass

import pandas as pd
import numpy as np
from loguru import logger

from libs.llm import SignalGenerator, SignalType, SignalGenerationResult
from apps.runtime.data_fetcher import get_data_fetcher, MarketDataFetcher
from apps.runtime.ftmo_rules import (
    check_daily_loss_limit,
    check_position_size,
    check_total_loss_limit,
)
from libs.config.config import Settings
from libs.db.models import Signal, create_tables, get_session
from libs.constants import INITIAL_BALANCE


@dataclass
class V7RuntimeConfig:
    """Configuration for V7 runtime"""
    symbols: list[str]
    min_data_points: int = 200  # Minimum candles needed for analysis (signal generator requires 200)
    max_signals_per_hour: int = 6  # Max signals per hour (10 signals/day budget)
    max_cost_per_day: float = 3.00  # Max $3/day ($90/month)
    max_cost_per_month: float = 100.00  # Hard monthly limit
    signal_interval_seconds: int = 120  # 2 minutes between scans
    conservative_mode: bool = True  # Conservative LLM prompting


class V7TradingRuntime:
    """
    V7 Ultimate Trading Runtime

    Combines 6 mathematical theories with DeepSeek LLM for signal generation.

    Workflow:
    1. Fetch live market data
    2. Run mathematical analysis (6 theories)
    3. Generate LLM signal
    4. Apply FTMO rules
    5. Rate limiting and cost controls
    6. Output signal
    """

    def __init__(self, config: Optional[Settings] = None, runtime_config: Optional[V7RuntimeConfig] = None):
        """
        Initialize V7 Trading Runtime

        Args:
            config: Settings object (loads from .env if None)
            runtime_config: V7-specific runtime configuration
        """
        self.config = config or Settings()
        self.runtime_config = runtime_config or V7RuntimeConfig(
            symbols=["BTC-USD", "ETH-USD", "SOL-USD"]
        )

        # Initialize database
        create_tables(self.config.db_url)
        logger.info(f"✅ Database initialized: {self.config.db_url}")

        # Initialize market data fetcher
        self.data_fetcher: MarketDataFetcher = get_data_fetcher(self.config)
        logger.info("✅ Market data fetcher initialized")

        # Initialize V7 signal generator
        self.signal_generator = SignalGenerator(
            api_key=self.config.deepseek_api_key,
            conservative_mode=self.runtime_config.conservative_mode
        )
        logger.info("✅ V7 SignalGenerator initialized (6 theories + DeepSeek LLM)")

        # Account state (for FTMO rules)
        self.initial_balance = INITIAL_BALANCE
        self.current_balance = INITIAL_BALANCE
        self.daily_pnl = 0.0

        # Rate limiting state
        self.signal_history: list[Dict[str, Any]] = []  # Last hour of signals
        self.last_signal_time: Optional[datetime] = None

        # Cost tracking
        self.daily_cost = 0.0
        self.monthly_cost = 0.0
        self.cost_reset_day = datetime.now().day
        self.cost_reset_month = datetime.now().month

        logger.info(f"V7 Runtime initialized | Symbols: {len(self.runtime_config.symbols)} | Conservative: {self.runtime_config.conservative_mode}")

    def _check_rate_limits(self) -> tuple[bool, str]:
        """
        Check if we can generate a signal (rate limits)

        Returns:
            Tuple of (allowed, reason)
        """
        now = datetime.now()

        # Remove signals older than 1 hour
        one_hour_ago = now - timedelta(hours=1)
        self.signal_history = [
            s for s in self.signal_history
            if s['timestamp'] > one_hour_ago
        ]

        # Check hourly signal limit
        signals_last_hour = len(self.signal_history)
        if signals_last_hour >= self.runtime_config.max_signals_per_hour:
            return False, f"Rate limit: {signals_last_hour}/{self.runtime_config.max_signals_per_hour} signals in last hour"

        # Check minimum interval since last signal (prevent rapid-fire)
        if self.last_signal_time:
            time_since_last = (now - self.last_signal_time).total_seconds()
            min_interval = 60  # 1 minute minimum between signals
            if time_since_last < min_interval:
                return False, f"Too soon: {time_since_last:.0f}s since last signal (min {min_interval}s)"

        return True, "OK"

    def _check_cost_limits(self, estimated_cost: float = 0.0003) -> tuple[bool, str]:
        """
        Check if we have budget remaining

        Args:
            estimated_cost: Estimated cost of next signal (default $0.0003)

        Returns:
            Tuple of (allowed, reason)
        """
        # Reset daily cost if new day
        now = datetime.now()
        if now.day != self.cost_reset_day:
            logger.info(f"Daily cost reset: ${self.daily_cost:.4f} spent yesterday")
            self.daily_cost = 0.0
            self.cost_reset_day = now.day

        # Reset monthly cost if new month
        if now.month != self.cost_reset_month:
            logger.info(f"Monthly cost reset: ${self.monthly_cost:.2f} spent last month")
            self.monthly_cost = 0.0
            self.cost_reset_month = now.month

        # Check daily limit
        if self.daily_cost + estimated_cost > self.runtime_config.max_cost_per_day:
            return False, f"Daily budget exceeded: ${self.daily_cost:.4f} / ${self.runtime_config.max_cost_per_day:.2f}"

        # Check monthly limit
        if self.monthly_cost + estimated_cost > self.runtime_config.max_cost_per_month:
            return False, f"Monthly budget exceeded: ${self.monthly_cost:.2f} / ${self.runtime_config.max_cost_per_month:.2f}"

        return True, "OK"

    def _update_costs(self, actual_cost: float):
        """Update cost tracking after signal generation"""
        self.daily_cost += actual_cost
        self.monthly_cost += actual_cost
        logger.debug(f"Cost updated: +${actual_cost:.6f} | Daily: ${self.daily_cost:.4f} | Monthly: ${self.monthly_cost:.2f}")

    def _check_ftmo_rules(self, signal_type: SignalType, current_price: float) -> tuple[bool, str]:
        """
        Check FTMO compliance rules

        Args:
            signal_type: BUY/SELL/HOLD
            current_price: Current market price

        Returns:
            Tuple of (allowed, reason)
        """
        # HOLD signals always allowed
        if signal_type == SignalType.HOLD:
            return True, "OK"

        # Check daily loss limit
        if not check_daily_loss_limit(
            current_balance=self.current_balance,
            initial_balance=self.initial_balance,
            daily_pnl=self.daily_pnl
        ):
            return False, "Daily loss limit exceeded (5%)"

        # Check total loss limit
        if not check_total_loss_limit(
            current_balance=self.current_balance,
            initial_balance=self.initial_balance
        ):
            return False, "Total loss limit exceeded (10%)"

        # Check position size
        position_size = self.initial_balance * 0.01  # 1% risk per trade
        if not check_position_size(position_size, current_price):
            return False, f"Position size too large: ${position_size:.2f}"

        return True, "OK"

    def _format_signal_output(
        self,
        symbol: str,
        result: SignalGenerationResult,
        current_price: float
    ) -> str:
        """
        Format signal for console output

        Args:
            symbol: Trading symbol
            result: Signal generation result
            current_price: Current market price

        Returns:
            Formatted string
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"V7 ULTIMATE SIGNAL | {symbol}")
        lines.append("=" * 80)
        lines.append(f"Timestamp:    {result.parsed_signal.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"Current Price: ${current_price:,.2f}")
        lines.append("")
        lines.append(f"SIGNAL:       {result.parsed_signal.signal.value}")
        lines.append(f"CONFIDENCE:   {result.parsed_signal.confidence:.1%}")
        lines.append(f"REASONING:    {result.parsed_signal.reasoning}")
        lines.append("")
        lines.append("MATHEMATICAL ANALYSIS:")
        lines.append(f"  Shannon Entropy:     {result.theory_analysis.entropy:.3f} ({result.theory_analysis.entropy_interpretation.get('predictability', 'N/A')})")
        lines.append(f"  Hurst Exponent:      {result.theory_analysis.hurst:.3f} ({result.theory_analysis.hurst_interpretation})")
        lines.append(f"  Markov Regime:       {result.theory_analysis.current_regime}")
        lines.append(f"  Kalman Price:        ${result.theory_analysis.denoised_price:,.2f}")
        lines.append(f"  Kalman Momentum:     {result.theory_analysis.price_momentum:+.6f}")
        lines.append(f"  Bayesian Win Rate:   {result.theory_analysis.win_rate_estimate:.1%}")

        if 'sharpe_ratio' in result.theory_analysis.risk_metrics:
            lines.append(f"  Sharpe Ratio:        {result.theory_analysis.risk_metrics['sharpe_ratio']:.2f}")
        if 'var_95' in result.theory_analysis.risk_metrics:
            lines.append(f"  VaR (95%):           {result.theory_analysis.risk_metrics['var_95']:.1%}")

        lines.append("")
        lines.append(f"DeepSeek Cost:  ${result.total_cost_usd:.6f}")
        lines.append(f"Valid Signal:   {'✅ Yes' if result.parsed_signal.is_valid else '❌ No'}")
        lines.append("=" * 80)

        return "\n".join(lines)

    def _save_signal_to_db(
        self,
        symbol: str,
        result: SignalGenerationResult,
        current_price: float
    ):
        """Save signal to database"""
        try:
            session = get_session(self.config.db_url)

            # Map V7 signal type to direction
            signal_value = result.parsed_signal.signal.value
            if signal_value == "BUY":
                direction = "long"
            elif signal_value == "SELL":
                direction = "short"
            else:  # HOLD
                direction = "hold"

            # Determine tier based on confidence
            if result.parsed_signal.confidence >= 0.70:
                tier = "high"
            elif result.parsed_signal.confidence >= 0.55:
                tier = "medium"
            else:
                tier = "low"

            signal = Signal(
                timestamp=result.parsed_signal.timestamp,
                symbol=symbol,
                direction=direction,
                confidence=result.parsed_signal.confidence,
                tier=tier,
                ensemble_prediction=result.parsed_signal.confidence,  # V7 uses single confidence value
                entry_price=current_price,
                model_version="v7_ultimate",
                notes=result.parsed_signal.reasoning
            )

            session.add(signal)
            session.commit()
            session.close()

            logger.debug(f"Signal saved to database: {symbol} {direction} @ {result.parsed_signal.confidence:.1%}")

        except Exception as e:
            logger.error(f"Failed to save signal to database: {e}")

    def generate_signal_for_symbol(self, symbol: str) -> Optional[SignalGenerationResult]:
        """
        Generate V7 signal for a specific symbol

        Args:
            symbol: Trading symbol (e.g., "BTC-USD")

        Returns:
            SignalGenerationResult or None if failed
        """
        try:
            # Fetch live market data (fetch extra buffer for signal generator)
            logger.info(f"Fetching live data for {symbol}...")
            df = self.data_fetcher.fetch_latest_candles(
                symbol=symbol,
                num_candles=max(250, self.runtime_config.min_data_points + 50)  # Buffer for lookback window
            )

            if df.empty:
                logger.warning(f"No data available for {symbol}")
                return None

            if len(df) < self.runtime_config.min_data_points:
                logger.warning(
                    f"Insufficient data for {symbol}: {len(df)}/{self.runtime_config.min_data_points} candles"
                )
                return None

            # Extract price series and timestamps
            prices = df['close'].values
            timestamps = (df['timestamp'].astype('int64') // 10**9).values  # Convert to Unix timestamps as numpy array
            current_price = float(prices[-1])

            logger.info(f"Generating V7 signal for {symbol} (price: ${current_price:,.2f}, {len(prices)} candles)")

            # Generate signal using V7 system
            result = self.signal_generator.generate_signal(
                symbol=symbol,
                prices=prices,
                timestamps=timestamps,
                current_price=current_price,
                timeframe="1m"
            )

            # Update costs
            self._update_costs(result.total_cost_usd)

            # Check FTMO rules
            ftmo_ok, ftmo_reason = self._check_ftmo_rules(result.parsed_signal.signal, current_price)
            if not ftmo_ok:
                logger.warning(f"FTMO rules block signal for {symbol}: {ftmo_reason}")
                # Still return result but mark as invalid
                result.parsed_signal.is_valid = False
                result.parsed_signal.reasoning += f" [BLOCKED: {ftmo_reason}]"

            return result

        except Exception as e:
            logger.error(f"Failed to generate signal for {symbol}: {e}")
            return None

    def run_single_scan(self) -> int:
        """
        Run a single scan across all symbols

        Returns:
            Number of valid signals generated
        """
        logger.info(f"Starting V7 scan across {len(self.runtime_config.symbols)} symbols...")

        valid_signals = 0

        for symbol in self.runtime_config.symbols:
            try:
                # Check rate limits
                rate_ok, rate_reason = self._check_rate_limits()
                if not rate_ok:
                    logger.info(f"Rate limit reached: {rate_reason}")
                    break

                # Check cost limits
                cost_ok, cost_reason = self._check_cost_limits()
                if not cost_ok:
                    logger.warning(f"Cost limit reached: {cost_reason}")
                    break

                # Generate signal
                result = self.generate_signal_for_symbol(symbol)

                if result is None:
                    continue

                # Get current price for formatting
                df = self.data_fetcher.fetch_latest_candles(symbol=symbol, num_candles=1)
                current_price = float(df['close'].iloc[-1]) if not df.empty else 0.0

                # Format and output signal
                output = self._format_signal_output(symbol, result, current_price)
                print(output)

                # Save to database
                self._save_signal_to_db(symbol, result, current_price)

                # Update signal history (for rate limiting)
                self.signal_history.append({
                    'timestamp': result.parsed_signal.timestamp,
                    'symbol': symbol,
                    'signal': result.parsed_signal.signal.value,
                    'confidence': result.parsed_signal.confidence
                })
                self.last_signal_time = result.parsed_signal.timestamp

                # Count valid signals
                if result.parsed_signal.is_valid and result.parsed_signal.signal != SignalType.HOLD:
                    valid_signals += 1

                # Small delay between symbols
                time.sleep(2)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue

        logger.info(f"Scan complete | Valid signals: {valid_signals}/{len(self.runtime_config.symbols)}")
        return valid_signals

    def run(self, iterations: int = -1, sleep_seconds: int = 120):
        """
        Run V7 trading runtime continuously

        Args:
            iterations: Number of scans to run (-1 = infinite)
            sleep_seconds: Seconds to sleep between scans
        """
        logger.info("=" * 80)
        logger.info("V7 ULTIMATE TRADING RUNTIME - STARTED")
        logger.info("=" * 80)
        logger.info(f"Symbols: {', '.join(self.runtime_config.symbols)}")
        logger.info(f"Scan Interval: {sleep_seconds}s")
        logger.info(f"Rate Limit: {self.runtime_config.max_signals_per_hour} signals/hour")
        logger.info(f"Daily Budget: ${self.runtime_config.max_cost_per_day:.2f}")
        logger.info(f"Monthly Budget: ${self.runtime_config.max_cost_per_month:.2f}")
        logger.info(f"Conservative Mode: {self.runtime_config.conservative_mode}")
        logger.info("=" * 80)

        iteration = 0

        try:
            while iterations == -1 or iteration < iterations:
                iteration += 1

                logger.info(f"\n{'='*80}")
                logger.info(f"ITERATION {iteration} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'='*80}")

                # Run single scan
                valid_signals = self.run_single_scan()

                # Print statistics
                stats = self.signal_generator.get_statistics()
                logger.info(f"\nV7 Statistics:")
                logger.info(f"  DeepSeek API Calls:    {stats['deepseek_api']['total_requests']}")
                logger.info(f"  Total API Cost:        ${stats['deepseek_api']['total_cost_usd']:.6f}")
                logger.info(f"  Bayesian Win Rate:     {stats['bayesian_win_rate']:.1%}")
                logger.info(f"  Bayesian Total Trades: {stats['bayesian_total_trades']}")
                logger.info(f"  Daily Cost:            ${self.daily_cost:.4f} / ${self.runtime_config.max_cost_per_day:.2f}")
                logger.info(f"  Monthly Cost:          ${self.monthly_cost:.2f} / ${self.runtime_config.max_cost_per_month:.2f}")

                # Sleep until next iteration
                if iterations == -1 or iteration < iterations:
                    logger.info(f"\nSleeping {sleep_seconds}s until next scan...")
                    time.sleep(sleep_seconds)

        except KeyboardInterrupt:
            logger.info("\n\nV7 Runtime stopped by user (Ctrl+C)")

        except Exception as e:
            logger.error(f"\n\nV7 Runtime crashed: {e}")
            raise

        finally:
            logger.info("=" * 80)
            logger.info("V7 ULTIMATE TRADING RUNTIME - STOPPED")
            logger.info("=" * 80)

            # Final statistics
            stats = self.signal_generator.get_statistics()
            logger.info(f"Final Statistics:")
            logger.info(f"  Total Iterations:      {iteration}")
            logger.info(f"  DeepSeek API Calls:    {stats['deepseek_api']['total_requests']}")
            logger.info(f"  Total API Cost:        ${stats['deepseek_api']['total_cost_usd']:.6f}")
            logger.info(f"  Daily Cost:            ${self.daily_cost:.4f}")
            logger.info(f"  Monthly Cost:          ${self.monthly_cost:.2f}")
            logger.info("=" * 80)


def main():
    """CLI entry point for V7 runtime"""
    import argparse

    parser = argparse.ArgumentParser(description="V7 Ultimate Trading Runtime")
    parser.add_argument(
        "--iterations",
        type=int,
        default=-1,
        help="Number of scans to run (-1 = infinite, default: -1)"
    )
    parser.add_argument(
        "--sleep-seconds",
        type=int,
        default=120,
        help="Seconds between scans (default: 120)"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC-USD", "ETH-USD", "SOL-USD"],
        help="Symbols to scan (default: BTC-USD ETH-USD SOL-USD)"
    )
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help="Use aggressive mode (default: conservative)"
    )
    parser.add_argument(
        "--max-signals-per-hour",
        type=int,
        default=6,
        help="Max signals per hour (default: 6)"
    )
    parser.add_argument(
        "--daily-budget",
        type=float,
        default=3.0,
        help="Max daily cost in USD (default: 3.0)"
    )

    args = parser.parse_args()

    # Create runtime config
    runtime_config = V7RuntimeConfig(
        symbols=args.symbols,
        max_signals_per_hour=args.max_signals_per_hour,
        max_cost_per_day=args.daily_budget,
        signal_interval_seconds=args.sleep_seconds,
        conservative_mode=not args.aggressive
    )

    # Initialize and run
    runtime = V7TradingRuntime(runtime_config=runtime_config)
    runtime.run(iterations=args.iterations, sleep_seconds=args.sleep_seconds)


if __name__ == "__main__":
    main()
