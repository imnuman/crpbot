"""
V7 Ultimate Trading Runtime - Phase 1 Enhanced
Integrates Phase 1 risk management enhancements with V7 Ultimate.

Phase 1 Enhancements:
1. Kelly Criterion Position Sizing - Optimal position sizes based on historical performance
2. Exit Strategy Enhancement - Trailing stops, break-even stops, time-based exits
3. Correlation Analysis - Prevent highly correlated positions
4. Market Regime Strategy - Filter signals based on market regime

Architecture:
1. Fetch live market data
2. Run mathematical analysis (6-11 theories)
3. Apply regime-based signal filtering (NEW)
4. Check correlation with open positions (NEW)
5. Send analysis to DeepSeek LLM
6. Parse LLM response
7. Apply FTMO rules and rate limiting
8. Calculate Kelly position size (NEW)
9. Setup dynamic exit strategy (NEW)
10. Output signal

Usage:
    runtime = V7Phase1Runtime()
    runtime.run(iterations=-1, sleep_seconds=120)  # Run continuously
"""
from datetime import datetime
from typing import Optional, Dict, Any

from loguru import logger

# Import base V7 runtime
from apps.runtime.v7_runtime import V7TradingRuntime, V7RuntimeConfig
from libs.llm import SignalType, SignalGenerationResult

# Import Phase 1 components
from libs.risk.kelly_criterion import KellyCriterion
from libs.risk.exit_strategy import ExitStrategy
from libs.risk.correlation_analyzer import CorrelationAnalyzer
from libs.risk.regime_strategy import RegimeStrategyManager

# For database queries
import pandas as pd
from libs.db.models import get_session, Signal, SignalResult


class V7Phase1Config(V7RuntimeConfig):
    """Extended configuration for Phase 1 runtime"""
    # Kelly Criterion
    fractional_kelly: float = 0.5  # Use 50% Kelly for safety
    kelly_lookback_trades: int = 50  # Recent trades for Kelly calculation

    # Exit Strategy
    trailing_stop_activation: float = 0.005  # 0.5% profit to activate trailing stop
    trailing_stop_distance: float = 0.002    # 0.2% distance from peak
    max_hold_hours: int = 24                 # 24-hour max hold time
    breakeven_profit_threshold: float = 0.0025  # 0.25% profit to move SL to breakeven

    # Correlation Analysis
    correlation_threshold: float = 0.7  # Block if correlation > 0.7
    correlation_lookback_hours: int = 168  # 7 days for correlation calculation

    # Regime Strategy
    enable_regime_filtering: bool = True  # Enable regime-based signal filtering


class V7Phase1Runtime(V7TradingRuntime):
    """
    V7 Phase 1 Enhanced Runtime

    Extends V7TradingRuntime with Phase 1 risk management enhancements.
    """

    def __init__(self, config=None, runtime_config: Optional[V7Phase1Config] = None):
        """
        Initialize V7 Phase 1 Enhanced Runtime

        Args:
            config: Settings object
            runtime_config: Phase 1 extended configuration
        """
        # Initialize base V7 runtime
        phase1_config = runtime_config or V7Phase1Config(
            symbols=[
                "BTC-USD", "ETH-USD", "SOL-USD",
                "XRP-USD", "DOGE-USD", "ADA-USD", "AVAX-USD",
                "LINK-USD", "POL-USD", "LTC-USD"
            ]
        )
        super().__init__(config=config, runtime_config=phase1_config)

        self.phase1_config: V7Phase1Config = phase1_config  # Type hint for IDE

        # Initialize Phase 1 components
        logger.info("🚀 Initializing Phase 1 enhancements...")

        # 1. Kelly Criterion for position sizing
        self.kelly_calculator = KellyCriterion(
            fractional_kelly=self.phase1_config.fractional_kelly
        )
        logger.info(f"✅ Kelly Criterion initialized (fractional={self.phase1_config.fractional_kelly*100:.0f}%)")

        # 2. Exit Strategy for profit protection
        self.exit_strategy = ExitStrategy(
            trailing_stop_activation=self.phase1_config.trailing_stop_activation,
            trailing_stop_distance=self.phase1_config.trailing_stop_distance,
            max_hold_hours=self.phase1_config.max_hold_hours,
            breakeven_profit_threshold=self.phase1_config.breakeven_profit_threshold
        )
        logger.info(
            f"✅ Exit Strategy initialized "
            f"(trailing={self.phase1_config.trailing_stop_activation*100:.1f}%, "
            f"max_hold={self.phase1_config.max_hold_hours}h)"
        )

        # 3. Correlation Analyzer for diversification
        self.correlation_analyzer = CorrelationAnalyzer(
            correlation_threshold=self.phase1_config.correlation_threshold,
            lookback_hours=self.phase1_config.correlation_lookback_hours
        )
        logger.info(
            f"✅ Correlation Analyzer initialized "
            f"(threshold={self.phase1_config.correlation_threshold*100:.0f}%)"
        )

        # 4. Regime Strategy Manager for market-aware trading
        self.regime_strategy = RegimeStrategyManager()
        logger.info("✅ Regime Strategy Manager initialized")

        # Cache for Kelly fraction (updated periodically)
        self.current_kelly_fraction = 0.10  # Default 10% if no data
        self.last_kelly_update = datetime.now()

        logger.info("🎉 V7 Phase 1 Enhanced Runtime initialized successfully!")

    def _update_kelly_fraction(self) -> float:
        """
        Update Kelly fraction from recent trades

        Returns:
            Current Kelly fraction
        """
        try:
            # Load recent trades from database
            session = get_session(self.config.db_url)
            try:
                query = session.query(SignalResult).filter(
                    SignalResult.outcome.in_(['win', 'loss'])
                ).order_by(SignalResult.created_at.desc()).limit(
                    self.phase1_config.kelly_lookback_trades
                )

                trades = query.all()

                if len(trades) < 10:
                    logger.debug(
                        f"Insufficient trades for Kelly ({len(trades)}/10 minimum). "
                        f"Using default: {self.current_kelly_fraction*100:.1f}%"
                    )
                    return self.current_kelly_fraction

                # Convert to DataFrame
                trades_df = pd.DataFrame([
                    {'pnl_percent': t.pnl_percent, 'outcome': t.outcome}
                    for t in trades
                ])

                # Calculate Kelly
                analysis = self.kelly_calculator.analyze_historical_trades(trades_df)
                kelly_fraction = analysis['kelly_fraction']

                # Log update
                logger.info(
                    f"📊 Kelly Updated: {self.current_kelly_fraction*100:.1f}% → {kelly_fraction*100:.1f}% "
                    f"(WR: {analysis['win_rate']*100:.1f}%, EV: {analysis['expected_value']:.2f}%)"
                )

                self.current_kelly_fraction = kelly_fraction
                self.last_kelly_update = datetime.now()

                return kelly_fraction

            finally:
                session.close()

        except Exception as e:
            logger.warning(f"Failed to update Kelly fraction: {e}")
            return self.current_kelly_fraction

    def _get_open_positions(self) -> list[Dict[str, Any]]:
        """
        Get currently open paper trading positions

        Returns:
            List of open positions with symbol, direction, entry_price
        """
        try:
            session = get_session(self.config.db_url)
            try:
                # Query open positions (paper trades with outcome='open')
                query = session.query(SignalResult, Signal).join(
                    Signal, SignalResult.signal_id == Signal.id
                ).filter(
                    SignalResult.outcome == 'open'
                )

                open_positions = []
                for result, signal in query.all():
                    open_positions.append({
                        'symbol': signal.symbol,
                        'direction': signal.direction,
                        'entry_price': result.entry_price,
                        'confidence': signal.confidence,
                        'timestamp': result.entry_timestamp
                    })

                return open_positions

            finally:
                session.close()

        except Exception as e:
            logger.warning(f"Failed to get open positions: {e}")
            return []

    def _update_correlation_matrix(self):
        """Update correlation matrix with recent price data"""
        try:
            from datetime import timedelta

            # Fetch recent price data for all symbols
            price_data = {}
            lookback_hours = self.phase1_config.correlation_lookback_hours

            for symbol in self.runtime_config.symbols[:5]:  # Limit to 5 for performance
                try:
                    df = self.data_fetcher.fetch_latest_candles(
                        symbol=symbol,
                        num_candles=lookback_hours  # Hourly data
                    )
                    if not df.empty:
                        price_series = pd.Series(
                            df['close'].values,
                            index=pd.to_datetime(df['timestamp'])
                        )
                        price_data[symbol] = price_series
                except Exception as e:
                    logger.debug(f"Skipped {symbol} for correlation: {e}")

            if len(price_data) >= 2:
                self.correlation_analyzer.calculate_correlation_matrix(price_data)
                logger.debug(f"Correlation matrix updated ({len(price_data)} symbols)")

        except Exception as e:
            logger.warning(f"Failed to update correlation matrix: {e}")

    def generate_signal_for_symbol(self, symbol: str, strategy: str = "v7_phase1") -> Optional[SignalGenerationResult]:
        """
        Generate Phase 1 enhanced signal for a symbol

        Phase 1 Enhancements:
        1. Apply regime-based signal filtering
        2. Check correlation with open positions
        3. Calculate Kelly position size
        4. Setup dynamic exit strategy

        Args:
            symbol: Trading symbol
            strategy: Strategy identifier (defaults to "v7_phase1" for tracking)

        Returns:
            Enhanced SignalGenerationResult or None
        """
        try:
            # PHASE 1 ENHANCEMENT #1: Update Kelly fraction periodically (every hour)
            time_since_kelly_update = (datetime.now() - self.last_kelly_update).total_seconds()
            if time_since_kelly_update > 3600:  # Update hourly
                self._update_kelly_fraction()

            # Generate base V7 signal (calls parent method)
            logger.info(f"🔬 Generating V7 signal for {symbol}...")
            result = super().generate_signal_for_symbol(symbol, strategy="v7_full_math")

            if result is None or result.parsed_signal.signal == SignalType.HOLD:
                return result

            # Extract signal details
            signal_direction = 'long' if result.parsed_signal.signal == SignalType.BUY else 'short'
            signal_confidence = result.parsed_signal.confidence

            # PHASE 1 ENHANCEMENT #2: Regime-based signal filtering
            if self.phase1_config.enable_regime_filtering:
                # Get current regime from theory analysis
                regime_name = result.theory_analysis.regime_name if hasattr(result.theory_analysis, 'regime_name') else "High Volatility Range"
                regime_confidence = result.theory_analysis.regime_confidence if hasattr(result.theory_analysis, 'regime_confidence') else 0.7

                # Filter signal through regime strategy
                is_allowed, filter_reason = self.regime_strategy.filter_signal(
                    signal_direction=signal_direction,
                    signal_confidence=signal_confidence,
                    regime_name=regime_name
                )

                if not is_allowed:
                    logger.warning(f"❌ Signal filtered by regime: {filter_reason}")
                    # Mark as HOLD
                    result.parsed_signal.signal = SignalType.HOLD
                    result.parsed_signal.reasoning += f" [FILTERED: {filter_reason}]"
                    return result

                logger.info(f"✅ Regime filter passed: {regime_name} allows {signal_direction.upper()}")

                # Adjust position size based on regime
                base_position_size = self.current_kelly_fraction
                adjusted_position_size = self.regime_strategy.adjust_position_size(
                    base_position_size,
                    regime_name
                )

                # Adjust stop-loss based on regime volatility
                base_sl_distance = abs(result.parsed_signal.entry_price - result.parsed_signal.stop_loss) / result.parsed_signal.entry_price
                adjusted_sl_distance = self.regime_strategy.adjust_stop_loss(
                    base_sl_distance,
                    regime_name
                )

                # Update stop-loss with regime adjustment
                if signal_direction == 'long':
                    result.parsed_signal.stop_loss = result.parsed_signal.entry_price * (1 - adjusted_sl_distance)
                else:
                    result.parsed_signal.stop_loss = result.parsed_signal.entry_price * (1 + adjusted_sl_distance)

            else:
                adjusted_position_size = self.current_kelly_fraction

            # PHASE 1 ENHANCEMENT #3: Correlation check with open positions
            open_positions = self._get_open_positions()
            if open_positions:
                # Update correlation matrix if needed
                self._update_correlation_matrix()

                # Check correlation
                open_symbols = [pos['symbol'] for pos in open_positions]
                is_diversified, conflicts = self.correlation_analyzer.check_position_correlation(
                    symbol,
                    open_symbols
                )

                if not is_diversified:
                    conflict_str = ', '.join([f"{sym} ({corr:+.2f})" for sym, corr in conflicts])
                    logger.warning(
                        f"❌ Signal blocked by correlation: {symbol} highly correlated with {conflict_str}"
                    )
                    # Mark as HOLD
                    result.parsed_signal.signal = SignalType.HOLD
                    result.parsed_signal.reasoning += f" [BLOCKED: High correlation with {conflict_str}]"
                    return result

                logger.info(f"✅ Correlation check passed: {symbol} diversifies portfolio")

            # PHASE 1 ENHANCEMENT #4: Calculate Kelly position size
            position_size_pct = adjusted_position_size * 100  # Convert to percentage

            logger.info(
                f"📊 Position Size: {position_size_pct:.1f}% "
                f"(Kelly: {self.current_kelly_fraction*100:.1f}% × Regime adjustment)"
            )

            # PHASE 1 ENHANCEMENT #5: Setup dynamic exit strategy
            exit_params = self.exit_strategy.calculate_exit_levels(
                entry_price=result.parsed_signal.entry_price,
                direction=signal_direction,
                initial_stop_loss=result.parsed_signal.stop_loss,
                initial_take_profit=result.parsed_signal.take_profit,
                entry_timestamp=datetime.now()
            )

            logger.info(f"🎯 Exit Strategy: {self.exit_strategy.get_exit_summary(exit_params)}")

            # Store Phase 1 metadata in result (for tracking)
            if not hasattr(result, 'phase1_metadata'):
                result.phase1_metadata = {}

            result.phase1_metadata.update({
                'kelly_fraction': self.current_kelly_fraction,
                'position_size_pct': position_size_pct,
                'regime_name': regime_name if self.phase1_config.enable_regime_filtering else 'N/A',
                'regime_adjusted_size': adjusted_position_size,
                'exit_params': exit_params,
                'correlation_checked': len(open_positions) > 0,
                'diversification_score': self.correlation_analyzer.get_diversification_score(
                    [pos['symbol'] for pos in open_positions] + [symbol]
                ) if open_positions else 1.0
            })

            # Update strategy tag for A/B testing
            result.strategy = strategy

            logger.info(f"🎉 Phase 1 enhanced signal generated for {symbol}")

            return result

        except Exception as e:
            logger.error(f"Error in Phase 1 signal generation for {symbol}: {e}", exc_info=True)
            return None


# CLI usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='V7 Phase 1 Enhanced Runtime')
    parser.add_argument('--iterations', type=int, default=-1, help='Number of iterations (-1 = infinite)')
    parser.add_argument('--sleep-seconds', type=int, default=300, help='Sleep between iterations (default: 5 min)')
    parser.add_argument('--max-signals-per-hour', type=int, default=3, help='Max signals per hour')
    parser.add_argument('--variant', type=str, default='v7_phase1', help='Strategy variant for A/B testing')

    args = parser.parse_args()

    # Create Phase 1 config
    phase1_config = V7Phase1Config(
        symbols=["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD"],
        max_signals_per_hour=args.max_signals_per_hour,
        enable_paper_trading=True,
        paper_trading_aggressive=True
    )

    # Initialize Phase 1 runtime
    runtime = V7Phase1Runtime(runtime_config=phase1_config)

    logger.info("="*70)
    logger.info("V7 PHASE 1 ENHANCED RUNTIME")
    logger.info("="*70)
    logger.info(f"Variant: {args.variant}")
    logger.info(f"Symbols: {len(phase1_config.symbols)}")
    logger.info(f"Max signals/hour: {phase1_config.max_signals_per_hour}")
    logger.info(f"Kelly Fraction: {phase1_config.fractional_kelly*100:.0f}%")
    logger.info(f"Regime Filtering: {'ENABLED' if phase1_config.enable_regime_filtering else 'DISABLED'}")
    logger.info("="*70)

    # Run runtime
    try:
        runtime.run(
            iterations=args.iterations,
            sleep_seconds=args.sleep_seconds
        )
    except KeyboardInterrupt:
        logger.info("Runtime stopped by user")
    except Exception as e:
        logger.error(f"Runtime error: {e}", exc_info=True)
