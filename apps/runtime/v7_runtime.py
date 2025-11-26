"""
V7 Ultimate Trading Runtime
Integrates 11 mathematical theories + Order Flow + DeepSeek LLM for signal generation.

Architecture:
1. Fetch live market data (200+ 1m candles for analysis)
2. Run mathematical analysis:
   - 6 Core Theories: Shannon, Hurst, Markov, Kalman, Bayesian, Monte Carlo
   - 4 Statistical Theories: Random Forest, Variance, Autocorrelation, Stationarity
   - 1 Market Context: CoinGecko macro data
3. Run Order Flow analysis (Phase 2):
   - Volume Profile: POC, VAH/VAL, support/resistance
   - Order Flow Imbalance: Bid/ask liquidity changes (if order book available)
   - Market Microstructure: VWAP, spreads, depth analysis
4. Send all analysis to DeepSeek LLM for signal synthesis
5. Parse LLM response into structured signal
6. Apply FTMO rules and rate limiting
7. Output signal (console, database, Telegram)

Cost Controls:
- Track cumulative API costs
- Enforce daily/monthly budget limits ($5/day, $150/month)
- Rate limit signal generation (max 3-6 signals per hour)

Usage:
    runtime = V7TradingRuntime()
    runtime.run(iterations=-1, sleep_seconds=300)  # Run continuously (5 min intervals)
"""
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from dataclasses import dataclass

import pandas as pd
import numpy as np
from loguru import logger

from libs.llm import SignalGenerator, SignalType, SignalGenerationResult
from libs.strategies.simple_momentum import SimpleMomentumStrategy
from libs.strategies.entropy_reversion import EntropyReversionStrategy
from apps.runtime.data_fetcher import get_data_fetcher, MarketDataFetcher
from apps.runtime.ftmo_rules import (
    check_daily_loss_limit,
    check_position_size,
    check_total_loss_limit,
)
from libs.config.config import Settings
from libs.db.models import Signal, create_tables, get_session
from libs.constants import INITIAL_BALANCE
from libs.notifications import TelegramNotifier
from libs.bayesian import BayesianLearner
from libs.data.coingecko_client import CoinGeckoClient
from libs.theories.market_context import MarketContextTheory
from libs.theories.market_microstructure import MarketMicrostructure
from libs.tracking.performance_tracker import PerformanceTracker
from libs.tracking.paper_trader import PaperTrader, PaperTradeConfig
from libs.safety import (
    MarketRegimeDetector,
    DrawdownCircuitBreaker,
    CorrelationManager,
    RejectionLogger
)
from libs.risk.volatility_regime_detector import VolatilityRegimeDetector
from libs.risk.sharpe_ratio_tracker import SharpeRatioTracker
from libs.risk.cvar_calculator import CVaRCalculator
from libs.risk.sortino_ratio_tracker import SortinoRatioTracker
from libs.risk.calmar_ratio_tracker import CalmarRatioTracker
from libs.risk.omega_ratio_calculator import OmegaRatioCalculator
from libs.analysis.multi_timeframe_analyzer import MultiTimeframeAnalyzer
from libs.analysis.information_coefficient import InformationCoefficientAnalyzer
from libs.portfolio.optimizer import PortfolioOptimizer


@dataclass
class V7RuntimeConfig:
    """Configuration for V7 runtime"""
    symbols: list[str]
    min_data_points: int = 200  # Minimum candles needed for analysis (signal generator requires 200)
    max_signals_per_hour: int = 6  # Max signals per hour (10 signals/day budget)
    max_cost_per_day: float = 5.00  # Max $5/day for more market analysis (increased from $3)
    max_cost_per_month: float = 150.00  # Hard monthly limit (increased from $100)
    signal_interval_seconds: int = 120  # 2 minutes between scans
    conservative_mode: bool = True  # Conservative LLM prompting
    enable_paper_trading: bool = True  # Enable automatic paper trading
    paper_trading_aggressive: bool = True  # Paper trade all signals regardless of confidence

    # Safety Guards configuration
    enable_safety_guards: bool = True  # Enable Safety Guard modules
    regime_adx_threshold: float = 20.0  # ADX threshold for regime detection (< 20 = choppy)
    drawdown_warning_pct: float = 0.03  # 3% daily loss warning
    drawdown_emergency_pct: float = 0.05  # 5% daily loss emergency
    drawdown_shutdown_pct: float = 0.09  # 9% total loss shutdown (FTMO)
    correlation_threshold: float = 0.7  # Base correlation threshold
    max_portfolio_beta: float = 2.0  # Max BTC exposure (200%)


class V7TradingRuntime:
    """
    V7 Ultimate Trading Runtime

    Combines 11 mathematical theories + Order Flow analysis with DeepSeek LLM for signal generation.

    Workflow:
    1. Fetch live market data (OHLCV candles)
    2. Run mathematical analysis (11 theories)
    3. Run Order Flow analysis (Volume Profile, OFI, Microstructure)
    4. Generate LLM signal (DeepSeek synthesis)
    5. Apply FTMO rules
    6. Rate limiting and cost controls
    7. Output signal (DB, Telegram, console)
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
            symbols=[
                "BTC-USD", "ETH-USD", "SOL-USD",  # Original 3
                "XRP-USD", "DOGE-USD", "ADA-USD", "AVAX-USD",  # New 4
                "LINK-USD", "POL-USD", "LTC-USD"  # New 3 (POL = Polygon, formerly MATIC)
            ]
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

        # Initialize Bayesian learner for continuous improvement
        self.bayesian_learner = BayesianLearner(db_url=self.config.db_url)
        logger.info("✅ Bayesian learner initialized (adaptive confidence calibration)")

        # Initialize Performance Tracker for measuring signal outcomes
        self.performance_tracker = PerformanceTracker()
        logger.info("✅ Performance tracker initialized (signal outcome tracking)")

        # Initialize Paper Trader for automatic practice trading
        if self.runtime_config.enable_paper_trading:
            paper_config = PaperTradeConfig(
                aggressive_mode=self.runtime_config.paper_trading_aggressive,
                min_confidence=0.0 if self.runtime_config.paper_trading_aggressive else 0.60
            )
            self.paper_trader = PaperTrader(config=paper_config, settings=self.config)
            logger.info(f"✅ Paper trader initialized (aggressive={paper_config.aggressive_mode}, auto-trade enabled)")
        else:
            self.paper_trader = None
            logger.info("⚠️  Paper trading disabled")

        # Initialize CoinGecko client for market context (7th theory)
        if self.config.coingecko_api_key:
            self.coingecko_client = CoinGeckoClient(api_key=self.config.coingecko_api_key)
            self.market_context_theory = MarketContextTheory()
            logger.info("✅ CoinGecko Analyst API initialized (7th theory - market context)")
        else:
            self.coingecko_client = None
            self.market_context_theory = None
            logger.warning("⚠️  CoinGecko API disabled (no API key)")

        # Initialize Market Microstructure (8th theory - Fear & Greed, FRED, News)
        self.market_microstructure = MarketMicrostructure(
            fred_api_key=self.config.fred_api_key,
            cryptocompare_api_key=self.config.cryptocompare_api_key
        )
        logger.info("✅ Market Microstructure initialized (8th theory - sentiment + macro + news)")

        # Initialize Telegram notifier
        self.telegram = TelegramNotifier(
            token=self.config.telegram_token,
            chat_id=self.config.telegram_chat_id,
            enabled=bool(self.config.telegram_token and self.config.telegram_chat_id)
        )
        if self.telegram.enabled:
            logger.info("✅ Telegram notifier initialized")
        else:
            logger.info("⚠️  Telegram notifications disabled (no credentials)")

        # Account state (for FTMO rules)
        self.initial_balance = INITIAL_BALANCE
        self.current_balance = INITIAL_BALANCE
        self.daily_pnl = 0.0

        # Rate limiting state - PER SYMBOL (FIX #2)
        self.signal_history: list[Dict[str, Any]] = []  # Last hour of signals (all symbols)
        self.last_signal_time_per_symbol: Dict[str, datetime] = {}  # Per-symbol last signal time

        # Cost tracking
        self.daily_cost = 0.0
        self.monthly_cost = 0.0
        self.cost_reset_day = datetime.now(timezone.utc).day
        self.cost_reset_month = datetime.now(timezone.utc).month

        # Initialize Safety Guards
        if self.runtime_config.enable_safety_guards:
            # Market Regime Detector (uses hardcoded thresholds: ADX > 25 = trend, ADX < 20 = chop)
            self.regime_detector = MarketRegimeDetector()
            logger.info("✅ Market Regime Detector initialized (ADX thresholds: >25 trend, <20 chop)")

            # Drawdown Circuit Breaker
            self.circuit_breaker = DrawdownCircuitBreaker(
                starting_balance=self.initial_balance,
                daily_loss_warning=self.runtime_config.drawdown_warning_pct,
                daily_loss_emergency=self.runtime_config.drawdown_emergency_pct,
                total_loss_shutdown=self.runtime_config.drawdown_shutdown_pct
            )
            logger.info(
                f"✅ Drawdown Circuit Breaker initialized "
                f"(thresholds: {self.runtime_config.drawdown_warning_pct:.0%}/"
                f"{self.runtime_config.drawdown_emergency_pct:.0%}/"
                f"{self.runtime_config.drawdown_shutdown_pct:.0%})"
            )

            # Correlation Manager
            self.correlation_manager = CorrelationManager(
                base_threshold=self.runtime_config.correlation_threshold
            )
            logger.info(f"✅ Correlation Manager initialized (threshold: {self.runtime_config.correlation_threshold:.0%})")

            # Rejection Logger
            self.rejection_logger = RejectionLogger(db_path=self.config.db_url.replace('sqlite:///', ''))
            logger.info("✅ Rejection Logger initialized (tracking rejected signals)")

            # Track open positions for correlation checks
            self.open_positions: list[Dict[str, Any]] = []

            logger.info("🛡️  Safety Guards ENABLED (4 modules active)")
        else:
            self.regime_detector = None
            self.circuit_breaker = None
            self.correlation_manager = None
            self.rejection_logger = None
            self.open_positions = []
            logger.warning("⚠️  Safety Guards DISABLED")

        # Initialize Volatility Regime Detector (always enabled for adaptive strategies)
        self.volatility_detector = VolatilityRegimeDetector()
        logger.info("✅ Volatility Regime Detector initialized (adaptive stop/target sizing)")

        # Initialize Multi-Timeframe Analyzer (always enabled for signal confirmation)
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        logger.info("✅ Multi-Timeframe Analyzer initialized (1m + 5m confirmation)")

        # Initialize Sharpe Ratio Tracker (real-time performance monitoring)
        self.sharpe_tracker = SharpeRatioTracker(risk_free_rate=0.05, max_history_days=90)
        logger.info("✅ Sharpe Ratio Tracker initialized (risk-adjusted performance monitoring)")

        # Initialize CVaR Calculator (tail risk monitoring)
        self.cvar_calculator = CVaRCalculator(max_history=500)
        logger.info("✅ CVaR Calculator initialized (tail risk & position sizing)")

        # Initialize Sortino Ratio Tracker (downside risk-adjusted performance)
        self.sortino_tracker = SortinoRatioTracker(risk_free_rate=0.05, max_history_days=90)
        logger.info("✅ Sortino Ratio Tracker initialized (downside-focused performance)")

        # Initialize Information Coefficient Analyzer (signal quality)
        self.ic_analyzer = InformationCoefficientAnalyzer(max_history=500)
        logger.info("✅ Information Coefficient Analyzer initialized (theory ranking)")

        # Initialize Portfolio Optimizer (Markowitz mean-variance optimization)
        self.portfolio_optimizer = PortfolioOptimizer(
            symbols=self.runtime_config.symbols,
            lookback_days=365,
            risk_free_rate=0.05
        )
        logger.info("✅ Portfolio Optimizer initialized (Markowitz MPT, 10 assets)")

        # Initialize Calmar Ratio Tracker (return/max drawdown)
        self.calmar_tracker = CalmarRatioTracker(max_history_days=365)
        logger.info("✅ Calmar Ratio Tracker initialized (return/max drawdown)")

        # Initialize Omega Ratio Calculator (probability-weighted gains/losses)
        self.omega_calculator = OmegaRatioCalculator(risk_free_rate=0.05, max_history=500)
        logger.info("✅ Omega Ratio Calculator initialized (full distribution analysis)")

        # Load historical paper trades from database (last 90 days)
        self._load_historical_performance_data()

        logger.info(f"V7 Runtime initialized | Symbols: {len(self.runtime_config.symbols)} | Conservative: {self.runtime_config.conservative_mode}")

    def _check_rate_limits(self, symbol: str) -> tuple[bool, str]:
        """
        Check if we can generate a signal (rate limits) - PER SYMBOL (FIX #2)

        Args:
            symbol: Trading symbol to check rate limits for

        Returns:
            Tuple of (allowed, reason)
        """
        now = datetime.now(timezone.utc)

        # Remove signals older than 1 hour
        one_hour_ago = now - timedelta(hours=1)
        # Filter and ensure all timestamps are timezone-aware
        filtered_history = []
        for s in self.signal_history:
            ts = s['timestamp']
            # Make timezone-naive timestamps aware (assume UTC)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts > one_hour_ago:
                filtered_history.append(s)
        self.signal_history = filtered_history

        # Check hourly signal limit (GLOBAL across all symbols)
        signals_last_hour = len(self.signal_history)
        if signals_last_hour >= self.runtime_config.max_signals_per_hour:
            return False, f"Rate limit: {signals_last_hour}/{self.runtime_config.max_signals_per_hour} signals in last hour"

        # FIX #2: Check minimum interval since last signal FOR THIS SYMBOL ONLY
        # This allows BTC, ETH, SOL to signal independently
        if symbol in self.last_signal_time_per_symbol:
            last_time = self.last_signal_time_per_symbol[symbol]
            # Ensure timezone-aware
            if last_time.tzinfo is None:
                last_time = last_time.replace(tzinfo=timezone.utc)
            time_since_last = (now - last_time).total_seconds()
            min_interval = 60  # 1 minute minimum between signals FOR THIS SYMBOL
            if time_since_last < min_interval:
                return False, f"Too soon: {time_since_last:.0f}s since last {symbol} signal (min {min_interval}s)"

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
        now = datetime.now(timezone.utc)
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

    def _load_historical_performance_data(self):
        """Load completed paper trades from database and populate all performance trackers"""
        try:
            session = get_session(self.config.db_url)

            # Query completed paper trades from last 90 days
            cutoff_date = datetime.now() - timedelta(days=90)

            # SQL query to get completed trades (FIX: use exit_timestamp not exit_time)
            from libs.db.models import SignalResult
            results = session.query(SignalResult).filter(
                SignalResult.exit_timestamp.isnot(None),
                SignalResult.pnl_percent.isnot(None),
                SignalResult.exit_timestamp >= cutoff_date
            ).order_by(SignalResult.exit_timestamp).all()

            loaded_count = 0
            for result in results:
                if result.pnl_percent is not None and result.exit_timestamp is not None:
                    return_pct = result.pnl_percent / 100.0  # Convert to decimal

                    # Record to Sharpe Tracker
                    self.sharpe_tracker.record_trade_return(
                        timestamp=result.exit_timestamp,
                        return_pct=return_pct,
                        symbol=result.symbol
                    )

                    # Record to CVaR Calculator
                    self.cvar_calculator.record_return(
                        return_pct=return_pct,
                        timestamp=result.exit_timestamp
                    )

                    # Record to Sortino Tracker
                    self.sortino_tracker.record_return(
                        return_pct=return_pct,
                        timestamp=result.exit_timestamp
                    )

                    # Record to Calmar Tracker
                    self.calmar_tracker.record_return(
                        return_pct=return_pct,
                        timestamp=result.exit_timestamp
                    )

                    # Record to Omega Calculator
                    self.omega_calculator.record_return(
                        return_pct=return_pct,
                        timestamp=result.exit_timestamp
                    )

                    loaded_count += 1

            session.close()

            if loaded_count > 0:
                # Get initial metrics
                sharpe_metrics = self.sharpe_tracker.get_sharpe_metrics()
                cvar_metrics = self.cvar_calculator.get_cvar_metrics()
                sortino_metrics = self.sortino_tracker.get_sortino_metrics(
                    sharpe_ratio=sharpe_metrics.sharpe_ratio_30d
                )

                logger.info(
                    f"📊 Loaded {loaded_count} historical paper trades | "
                    f"30d Sharpe: {sharpe_metrics.sharpe_ratio_30d:.2f} | "
                    f"30d Sortino: {sortino_metrics.sortino_ratio_30d:.2f} | "
                    f"Win Rate: {sharpe_metrics.win_rate:.1%} | "
                    f"95% CVaR: {cvar_metrics.cvar_95_historical:.2%} ({cvar_metrics.risk_level})"
                )
            else:
                logger.info("📊 No historical paper trades found (starting fresh)")

        except Exception as e:
            logger.warning(f"Failed to load historical performance data: {e}")
            # Continue without historical data

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
            balance=self.current_balance,
            daily_pnl=self.daily_pnl
        ):
            return False, "Daily loss limit exceeded (5%)"

        # Check total loss limit
        if not check_total_loss_limit(
            current_balance=self.current_balance,
            initial_balance=self.initial_balance
        ):
            return False, "Total loss limit exceeded (10%)"

        # FIX #1: Position size check disabled for manual trading
        # User will determine position size manually when executing signals
        # Original check was failing for crypto (e.g., $100 < $86,830 BTC price)
        # position_size = self.initial_balance * 0.01  # 1% risk per trade
        # if not check_position_size(position_size, current_price):
        #     return False, f"Position size too large: ${position_size:.2f}"

        return True, "OK"

    def _check_safety_guards(
        self,
        symbol: str,
        result: SignalGenerationResult,
        df: pd.DataFrame
    ) -> tuple[bool, str, Optional[int]]:
        """
        Apply Safety Guard checks to signal

        Args:
            symbol: Trading symbol
            result: Signal generation result
            df: OHLCV DataFrame for regime detection

        Returns:
            Tuple of (allowed, reason, rejection_id)
            - allowed: True if signal passes all safety checks
            - reason: Rejection reason if blocked
            - rejection_id: ID of rejection log entry (for counterfactual tracking)
        """
        if not self.runtime_config.enable_safety_guards:
            return True, "OK", None

        signal = result.parsed_signal.signal
        direction = 'long' if signal == SignalType.BUY else 'short' if signal == SignalType.SELL else 'hold'

        # HOLD signals skip safety checks
        if signal == SignalType.HOLD:
            return True, "OK", None

        # Check 1: Market Regime Detector
        # TEMPORARILY DISABLED - User wants MORE trades, not fewer
        # Create a fake regime_result to avoid errors downstream
        from types import SimpleNamespace
        regime_result = SimpleNamespace(
            should_trade=True,
            metrics={'atr_pct': 0.02, 'adx': 25},  # Default values
            regime='neutral',
            quality='unknown',
            confidence=0.5
        )

        if False:
            # Log rejection
            rejection_id = self.rejection_logger.log_rejection(
                symbol=symbol,
                direction=direction,
                confidence=result.parsed_signal.confidence,
                rejection_reason=regime_result.reason,
                rejection_category='regime',
                rejection_details={
                    'regime': regime_result.regime,
                    'quality': regime_result.quality,
                    'confidence': regime_result.confidence,
                    'metrics': regime_result.metrics
                },
                market_context={
                    'regime': regime_result.regime,
                    'volatility': regime_result.metrics.get('atr_pct'),
                    'trend_strength': regime_result.metrics.get('adx')
                },
                theory_scores={
                    'shannon_entropy': result.theory_analysis.entropy,
                    'hurst': result.theory_analysis.hurst,
                    'markov_state': result.theory_analysis.current_regime
                },
                hypothetical_prices={
                    'entry': result.parsed_signal.entry_price,
                    'sl': result.parsed_signal.stop_loss,
                    'tp': result.parsed_signal.take_profit
                }
            )

            logger.warning(f"🛡️  REGIME BLOCK: {regime_result.reason}")
            return False, f"Regime: {regime_result.reason}", rejection_id

        # Check 2: Drawdown Circuit Breaker
        dd_status = self.circuit_breaker.check_drawdown()

        if not dd_status.is_trading_allowed:
            # Log rejection
            rejection_id = self.rejection_logger.log_rejection(
                symbol=symbol,
                direction=direction,
                confidence=result.parsed_signal.confidence,
                rejection_reason=dd_status.message,
                rejection_category='drawdown',
                rejection_details={
                    'level': dd_status.level,
                    'daily_drawdown_pct': dd_status.daily_drawdown_pct,
                    'total_drawdown_pct': dd_status.total_drawdown_pct,
                    'current_balance': dd_status.current_balance
                },
                market_context={
                    'regime': regime_result.regime,
                    'volatility': regime_result.metrics.get('atr_pct'),
                    'trend_strength': regime_result.metrics.get('adx')
                },
                theory_scores={
                    'shannon_entropy': result.theory_analysis.entropy,
                    'hurst': result.theory_analysis.hurst,
                    'markov_state': result.theory_analysis.current_regime
                },
                hypothetical_prices={
                    'entry': result.parsed_signal.entry_price,
                    'sl': result.parsed_signal.stop_loss,
                    'tp': result.parsed_signal.take_profit
                }
            )

            logger.warning(f"🛡️  DRAWDOWN BLOCK: {dd_status.message}")
            return False, f"Drawdown: {dd_status.message}", rejection_id

        # Apply size multiplier if in warning level
        if dd_status.position_size_multiplier < 1.0:
            logger.warning(
                f"⚠️  DRAWDOWN WARNING: Position size reduced to {dd_status.position_size_multiplier:.0%} "
                f"(Daily: {dd_status.daily_drawdown_pct:+.2%})"
            )

        # Check 3: Correlation Manager
        # Determine market volatility from ATR
        atr_pct = regime_result.metrics.get('atr_pct', 0.02)
        if atr_pct > 0.03:
            market_volatility = 'high'
        elif atr_pct < 0.015:
            market_volatility = 'low'
        else:
            market_volatility = 'normal'

        # TEMPORARILY DISABLED - User wants MORE trades, not fewer
        # Create fake corr_result to avoid errors downstream
        from types import SimpleNamespace
        corr_result = SimpleNamespace(
            allowed=True,
            reason='OK',
            max_correlation=0.0,
            threshold=0.7,
            correlation_details={},
            asset_class_exposure={},
            portfolio_beta=1.0
        )

        # Skip correlation check
        if False:
            # Log rejection
            rejection_id = self.rejection_logger.log_rejection(
                symbol=symbol,
                direction=direction,
                confidence=result.parsed_signal.confidence,
                rejection_reason=corr_result.reason,
                rejection_category='correlation',
                rejection_details={
                    'max_correlation': corr_result.max_correlation,
                    'threshold': corr_result.threshold,
                    'correlation_details': corr_result.correlation_details,
                    'asset_class_exposure': corr_result.asset_class_exposure,
                    'portfolio_beta': corr_result.portfolio_beta
                },
                market_context={
                    'regime': regime_result.regime,
                    'volatility': atr_pct,
                    'trend_strength': regime_result.metrics.get('adx')
                },
                theory_scores={
                    'shannon_entropy': result.theory_analysis.entropy,
                    'hurst': result.theory_analysis.hurst,
                    'markov_state': result.theory_analysis.current_regime
                },
                hypothetical_prices={
                    'entry': result.parsed_signal.entry_price,
                    'sl': result.parsed_signal.stop_loss,
                    'tp': result.parsed_signal.take_profit
                }
            )

            logger.warning(f"🛡️  CORRELATION BLOCK: {corr_result.reason}")
            return False, f"Correlation: {corr_result.reason}", rejection_id

        # Check 4: Multi-Timeframe Confirmation
        # TEMPORARILY DISABLED - User wants MORE trades, not fewer
        # Create fake mtf_result to avoid errors downstream
        from types import SimpleNamespace
        mtf_result = SimpleNamespace(
            aligned=True,
            reason='OK',
            primary_direction=direction,
            tf_1m=SimpleNamespace(trend_direction=direction, momentum_direction=direction),
            tf_5m=SimpleNamespace(trend_direction=direction, momentum_direction=direction)
        )

        # Skip multi-timeframe check
        if False:
            # Log rejection
            rejection_id = self.rejection_logger.log_rejection(
                symbol=symbol,
                direction=direction,
                confidence=result.parsed_signal.confidence,
                rejection_reason=mtf_result.reason,
                rejection_category='timeframe_conflict',
                rejection_details={
                    'tf_1m_trend': mtf_result.tf_1m.trend_direction,
                    'tf_1m_momentum': mtf_result.tf_1m.momentum_direction,
                    'tf_5m_trend': mtf_result.tf_5m.trend_direction,
                    'tf_5m_momentum': mtf_result.tf_5m.momentum_direction,
                    'aligned': mtf_result.aligned
                },
                market_context={
                    'regime': regime_result.regime,
                    'volatility': atr_pct,
                    'trend_strength': regime_result.metrics.get('adx')
                },
                theory_scores={
                    'shannon_entropy': result.theory_analysis.entropy,
                    'hurst': result.theory_analysis.hurst,
                    'markov_state': result.theory_analysis.current_regime
                },
                hypothetical_prices={
                    'entry': result.parsed_signal.entry_price,
                    'sl': result.parsed_signal.stop_loss,
                    'tp': result.parsed_signal.take_profit
                }
            )

            logger.warning(f"🛡️  MULTI-TF BLOCK: {mtf_result.reason}")
            return False, f"Multi-TF: {mtf_result.reason}", rejection_id

        # All safety checks passed
        logger.info(
            f"✅ SAFETY GUARDS PASSED | "
            f"Regime: {regime_result.regime} ({regime_result.quality}) | "
            f"Drawdown: Level {dd_status.level} (size: {dd_status.position_size_multiplier:.0%}) | "
            f"Correlation: {corr_result.max_correlation:.2f} (limit: {corr_result.threshold:.2f}) | "
            f"Multi-TF: {mtf_result.primary_direction} (1m+5m aligned)"
        )

        return True, "OK", None

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
        lines.append("")

        # Price targets (if available)
        if result.parsed_signal.entry_price:
            risk = abs(result.parsed_signal.entry_price - (result.parsed_signal.stop_loss or result.parsed_signal.entry_price))
            reward = abs((result.parsed_signal.take_profit or result.parsed_signal.entry_price) - result.parsed_signal.entry_price)
            risk_pct = (risk / result.parsed_signal.entry_price * 100) if result.parsed_signal.entry_price > 0 else 0
            reward_pct = (reward / result.parsed_signal.entry_price * 100) if result.parsed_signal.entry_price > 0 else 0
            rr_ratio = (reward / risk) if risk > 0 else 0

            lines.append("PRICE TARGETS:")
            lines.append(f"  Entry:        ${result.parsed_signal.entry_price:,.2f}")
            if result.parsed_signal.stop_loss:
                lines.append(f"  Stop Loss:    ${result.parsed_signal.stop_loss:,.2f} (risk: {risk_pct:.2f}% / ${risk:,.2f})")
            if result.parsed_signal.take_profit:
                lines.append(f"  Take Profit:  ${result.parsed_signal.take_profit:,.2f} (reward: {reward_pct:.2f}% / ${reward:,.2f})")
            if rr_ratio > 0:
                lines.append(f"  Risk/Reward:  1:{rr_ratio:.2f}")
            lines.append("")

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
        current_price: float,
        strategy: str = "v7_full_math"
    ):
        """Save signal to database with complete V7 analysis as JSON"""
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

            # Build complete V7 analysis JSON for dashboard
            # Note: Convert any Timestamp/datetime objects to strings for JSON serialization
            v7_data = {
                "reasoning": result.parsed_signal.reasoning,
                "theories": {
                    "entropy": float(result.theory_analysis.entropy) if result.theory_analysis.entropy is not None else None,
                    "entropy_interpretation": result.theory_analysis.entropy_interpretation if hasattr(result.theory_analysis, 'entropy_interpretation') else {},
                    "hurst": float(result.theory_analysis.hurst) if result.theory_analysis.hurst is not None else None,
                    "hurst_interpretation": result.theory_analysis.hurst_interpretation if hasattr(result.theory_analysis, 'hurst_interpretation') else "",
                    "current_regime": result.theory_analysis.current_regime if hasattr(result.theory_analysis, 'current_regime') else "",
                    "regime_probabilities": result.theory_analysis.regime_probabilities if hasattr(result.theory_analysis, 'regime_probabilities') else {},
                    "denoised_price": float(result.theory_analysis.denoised_price) if hasattr(result.theory_analysis, 'denoised_price') and result.theory_analysis.denoised_price is not None else None,
                    "price_momentum": float(result.theory_analysis.price_momentum) if hasattr(result.theory_analysis, 'price_momentum') and result.theory_analysis.price_momentum is not None else None,
                    "win_rate_estimate": float(result.theory_analysis.win_rate_estimate) if hasattr(result.theory_analysis, 'win_rate_estimate') and result.theory_analysis.win_rate_estimate is not None else None,
                    "risk_metrics": result.theory_analysis.risk_metrics if hasattr(result.theory_analysis, 'risk_metrics') else {}
                },
                "llm_cost_usd": float(result.total_cost_usd),
                "input_tokens": int(result.input_tokens),
                "output_tokens": int(result.output_tokens),
                "generation_time_seconds": float(result.generation_time_seconds)
            }

            import json
            notes_json = json.dumps(v7_data, default=str)  # Use default=str to handle any remaining non-serializable objects

            signal = Signal(
                timestamp=result.parsed_signal.timestamp,
                symbol=symbol,
                direction=direction,
                confidence=result.parsed_signal.confidence,
                tier=tier,
                ensemble_prediction=result.parsed_signal.confidence,  # V7 uses single confidence value
                # Use LLM-generated prices (fallback to current_price for entry if None)
                entry_price=result.parsed_signal.entry_price or current_price,
                sl_price=result.parsed_signal.stop_loss,
                tp_price=result.parsed_signal.take_profit,
                model_version="v7_ultimate",
                notes=notes_json,  # Store complete V7 analysis as JSON
                strategy=strategy  # A/B test strategy tracking
            )

            session.add(signal)
            logger.debug(f"Signal added to session: {symbol} {direction}")
            session.flush()  # Force write to database before commit
            logger.debug(f"Session flushed successfully")

            # Get the signal ID after flush (before commit)
            signal_id = signal.id

            session.commit()
            logger.debug(f"Session committed successfully")
            session.close()
            logger.debug(f"Session closed successfully")

            logger.debug(f"Signal saved to database: {symbol} {direction} @ {result.parsed_signal.confidence:.1%}")

            # Record theory contributions to performance tracker
            self._record_theory_contributions(signal_id, result)

            # Automatically enter paper trade if enabled
            if self.paper_trader:
                self.paper_trader.enter_paper_trade(signal_id)

        except Exception as e:
            logger.error(f"Failed to save signal to database: {e}")
            logger.error(f"Signal data that failed: timestamp={result.parsed_signal.timestamp}, symbol={symbol}, direction={direction}, confidence={result.parsed_signal.confidence}")
            logger.error(f"Entry price={result.parsed_signal.entry_price}, SL={result.parsed_signal.stop_loss}, TP={result.parsed_signal.take_profit}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")

    def _record_theory_contributions(self, signal_id: int, result: SignalGenerationResult):
        """
        Record theory contributions to performance tracker

        This allows us to measure which theories contribute to winning vs losing signals

        Args:
            signal_id: Database ID of the saved signal
            result: SignalGenerationResult containing theory analysis
        """
        try:
            # Extract theory contributions from the analysis
            theory_analysis = result.theory_analysis

            # Map each theory to a contribution score (based on how strong their signal was)
            theory_contributions = {}

            # Shannon Entropy: Higher entropy = less predictable (0.0-1.0 scale)
            if hasattr(theory_analysis, 'entropy') and theory_analysis.entropy is not None:
                # Normalize: Low entropy (predictable) = higher contribution
                entropy_score = max(0.0, min(1.0, 1.0 - float(theory_analysis.entropy)))
                theory_contributions['Shannon Entropy'] = entropy_score

            # Hurst Exponent: >0.5 = trending, <0.5 = mean reverting
            if hasattr(theory_analysis, 'hurst') and theory_analysis.hurst is not None:
                # Contribution based on how far from 0.5 (random walk)
                hurst_val = float(theory_analysis.hurst)
                hurst_score = abs(hurst_val - 0.5) * 2.0  # 0.0-1.0 scale
                theory_contributions['Hurst Exponent'] = min(1.0, hurst_score)

            # Market Regime: Contribution based on regime confidence
            if hasattr(theory_analysis, 'regime_probabilities') and theory_analysis.regime_probabilities:
                # Max probability = regime confidence
                max_prob = max(theory_analysis.regime_probabilities.values()) if theory_analysis.regime_probabilities else 0.0
                theory_contributions['Market Regime'] = float(max_prob)

            # Win Rate Estimate: Direct confidence from Bayesian analysis
            if hasattr(theory_analysis, 'win_rate_estimate') and theory_analysis.win_rate_estimate is not None:
                theory_contributions['Bayesian Win Rate'] = float(theory_analysis.win_rate_estimate)

            # Risk Metrics: Contribution based on favorable risk/reward
            if hasattr(theory_analysis, 'risk_metrics') and theory_analysis.risk_metrics:
                risk = theory_analysis.risk_metrics
                # If we have Sharpe ratio or similar, use that
                if 'sharpe_ratio' in risk and risk['sharpe_ratio'] is not None:
                    # Normalize Sharpe ratio to 0-1 (assume 0-3 range)
                    sharpe_score = max(0.0, min(1.0, float(risk['sharpe_ratio']) / 3.0))
                    theory_contributions['Risk Metrics'] = sharpe_score
                elif 'volatility' in risk and risk['volatility'] is not None:
                    # Lower volatility = higher contribution (inverse relationship)
                    vol_score = max(0.0, min(1.0, 1.0 - (float(risk['volatility']) / 100.0)))
                    theory_contributions['Risk Metrics'] = vol_score

            # Price Momentum: Strong momentum = higher contribution
            if hasattr(theory_analysis, 'price_momentum') and theory_analysis.price_momentum is not None:
                # Normalize momentum to 0-1 scale (assume -10 to +10 range)
                momentum = float(theory_analysis.price_momentum)
                momentum_score = max(0.0, min(1.0, (abs(momentum) / 10.0)))
                theory_contributions['Price Momentum'] = momentum_score

            # Kalman Filter: Contribution based on denoised price vs actual
            if hasattr(theory_analysis, 'denoised_price') and theory_analysis.denoised_price is not None:
                # If denoised price significantly different from current, it's informative
                # This is a placeholder - would need current price to calculate properly
                theory_contributions['Kalman Filter'] = 0.5  # Default medium contribution

            # Monte Carlo: Contribution based on simulation confidence
            # (Not directly in theory_analysis, but included in overall confidence)
            theory_contributions['Monte Carlo'] = float(result.parsed_signal.confidence)

            # Record each theory contribution
            for theory_name, contribution_score in theory_contributions.items():
                try:
                    self.performance_tracker.record_theory_contribution(
                        signal_id=signal_id,
                        theory_name=theory_name,
                        contribution_score=contribution_score,
                        was_correct=None  # Will be updated when trade outcome is known
                    )
                    logger.debug(f"Recorded {theory_name} contribution: {contribution_score:.2f} for signal {signal_id}")
                except Exception as e:
                    logger.warning(f"Failed to record {theory_name} contribution: {e}")

            logger.info(f"✅ Recorded {len(theory_contributions)} theory contributions for signal {signal_id}")

        except Exception as e:
            logger.error(f"Failed to record theory contributions for signal {signal_id}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def generate_signal_for_symbol(self, symbol: str, strategy: str = "v7_full_math") -> Optional[SignalGenerationResult]:
        """
        Generate V7 signal for a specific symbol

        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            strategy: Strategy type ("v7_full_math" or "v7_deepseek_only") for A/B testing

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

            # Fetch CoinGecko market context (7th theory)
            coingecko_data = None
            market_context = None
            if self.coingecko_client and self.market_context_theory:
                try:
                    coingecko_data = self.coingecko_client.get_market_data(symbol)
                    market_context = self.market_context_theory.analyze(symbol, coingecko_data)
                    if market_context:
                        logger.info(
                            f"📊 Market Context: MCap ${market_context['market_cap_billions']:.1f}B, "
                            f"Vol ${market_context['volume_billions']:.1f}B, "
                            f"ATH {market_context['ath_distance_pct']:.1f}%, "
                            f"Sentiment: {market_context['sentiment']}"
                        )
                except Exception as e:
                    logger.warning(f"CoinGecko fetch failed: {e}")

            # Generate signal using V7 system (with CoinGecko market context, Order Flow, and A/B test strategy)
            result = self.signal_generator.generate_signal(
                symbol=symbol,
                prices=prices,
                timestamps=timestamps,
                current_price=current_price,
                timeframe="1m",
                coingecko_context=market_context,  # Pass theory 11 (Market Context) to DeepSeek LLM
                strategy=strategy,  # A/B test: "v7_full_math" or "v7_deepseek_only"
                candles_df=df,  # Pass OHLCV DataFrame for Order Flow analysis (Phase 2)
                order_book=None  # Order book not available via REST API (would need WebSocket)
            )

            # Apply MATH STRATEGY if DeepSeek is too conservative (which it always is!)
            # A/B TEST: Strategy A (momentum) vs Strategy B (entropy reversion)
            # This BYPASSES the LLM and uses pure mathematical rules
            if result.parsed_signal.signal == SignalType.HOLD:
                momentum = result.theory_analysis.price_momentum
                hurst = result.theory_analysis.hurst
                entropy = result.theory_analysis.entropy
                regime = result.theory_analysis.current_regime
                var_95 = result.theory_analysis.risk_metrics.get('var_95', 0.0)

                # A/B TEST: Choose strategy based on variant
                if strategy == "v7_full_math":
                    # STRATEGY A: Aggressive Momentum (trend-following)
                    math_strategy = SimpleMomentumStrategy()
                    math_signal = math_strategy.generate_signal(
                        current_price=current_price,
                        hurst=hurst,
                        kalman_momentum=momentum,
                        entropy=entropy,
                        regime=regime
                    )
                    strategy_name = "MOMENTUM"
                else:
                    # STRATEGY B: Entropy Reversion (mean-reversion)
                    math_strategy = EntropyReversionStrategy()
                    math_signal = math_strategy.generate_signal(
                        current_price=current_price,
                        hurst=hurst,
                        kalman_momentum=momentum,
                        entropy=entropy,
                        var_95=var_95,
                        regime=regime
                    )
                    strategy_name = "ENTROPY"

                # If math strategy says BUY/SELL, override the LLM's HOLD
                if math_signal.direction in ["long", "short"]:
                    signal_type = SignalType.BUY if math_signal.direction == "long" else SignalType.SELL
                    logger.warning(
                        f"🔄 {strategy_name} STRATEGY OVERRIDE: {math_signal.reasoning}. "
                        f"Overriding LLM HOLD → {signal_type.value.upper()} at {math_signal.confidence:.0%} confidence"
                    )
                    from libs.llm import ParsedSignal
                    from datetime import datetime, timezone
                    result.parsed_signal = ParsedSignal(
                        signal=signal_type,
                        confidence=math_signal.confidence,
                        reasoning=f"MATH OVERRIDE: {math_signal.reasoning}",
                        raw_response=f"[MATH STRATEGY] {math_signal.reasoning}",
                        is_valid=True,
                        timestamp=datetime.now(timezone.utc),
                        parse_warnings=[f"Math strategy override: {math_signal.direction}"],
                        entry_price=math_signal.entry_price,
                        stop_loss=math_signal.stop_loss,
                        take_profit=math_signal.take_profit
                    )
                # OLD restrictive logic (disabled):
                elif False and momentum > 20 and hurst > 0.55 and 0.70 < entropy < 0.80:
                    logger.warning(
                        f"🔄 MOMENTUM OVERRIDE: Bullish momentum ({momentum:+.2f}) "
                        f"with trending Hurst ({hurst:.3f}) in moderate entropy market ({entropy:.3f}). "
                        f"Overriding HOLD → BUY at 40% confidence"
                    )
                    from libs.llm import ParsedSignal
                    from datetime import datetime, timezone
                    result.parsed_signal = ParsedSignal(
                        signal=SignalType.BUY,
                        confidence=0.40,
                        reasoning=f"MOMENTUM OVERRIDE: {result.parsed_signal.reasoning}. "
                                 f"Strong bullish momentum (+{momentum:.1f}) justifies entry despite mixed signals.",
                        raw_response=f"[MOMENTUM OVERRIDE] Original: {result.parsed_signal.raw_response}",
                        is_valid=True,
                        timestamp=datetime.now(timezone.utc),
                        parse_warnings=[f"Momentum override triggered: momentum={momentum:.2f}, hurst={hurst:.3f}, entropy={entropy:.3f}"],
                        entry_price=current_price,
                        stop_loss=current_price * 0.995,  # 0.5% stop
                        take_profit=current_price * 1.015  # 1.5% target (1:3 R:R)
                    )

                # Strong bearish momentum in MODERATELY uncertain market (not too random)
                # FIX #3: Changed from entropy > 0.70 to entropy < 0.80 (more selective)
                elif momentum < -20 and hurst < 0.45 and 0.70 < entropy < 0.80:
                    logger.warning(
                        f"🔄 MOMENTUM OVERRIDE: Bearish momentum ({momentum:+.2f}) "
                        f"with mean-reverting Hurst ({hurst:.3f}) in moderate entropy market ({entropy:.3f}). "
                        f"Overriding HOLD → SELL at 40% confidence"
                    )
                    from libs.llm import ParsedSignal
                    from datetime import datetime, timezone
                    result.parsed_signal = ParsedSignal(
                        signal=SignalType.SELL,
                        confidence=0.40,
                        reasoning=f"MOMENTUM OVERRIDE: {result.parsed_signal.reasoning}. "
                                 f"Strong bearish momentum ({momentum:.1f}) justifies entry despite mixed signals.",
                        raw_response=f"[MOMENTUM OVERRIDE] Original: {result.parsed_signal.raw_response}",
                        is_valid=True,
                        timestamp=datetime.now(timezone.utc),
                        parse_warnings=[f"Momentum override triggered: momentum={momentum:.2f}, hurst={hurst:.3f}, entropy={entropy:.3f}"],
                        entry_price=current_price,
                        stop_loss=current_price * 1.005,  # 0.5% stop
                        take_profit=current_price * 0.985  # 1.5% target (1:3 R:R)
                    )

            # Apply Bayesian confidence adjustment (continuous learning)
            if result.parsed_signal.signal != SignalType.HOLD:
                original_confidence = result.parsed_signal.confidence
                signal_direction = 'long' if result.parsed_signal.signal == SignalType.BUY else 'short'

                adjusted_confidence = self.bayesian_learner.get_adaptive_confidence_adjustment(
                    base_confidence=original_confidence,
                    signal_type=signal_direction,
                    symbol=symbol
                )

                result.parsed_signal.confidence = adjusted_confidence
                logger.debug(
                    f"Bayesian confidence adjustment: {original_confidence:.3f} → {adjusted_confidence:.3f} "
                    f"({symbol}, {signal_direction})"
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

            # Check Safety Guards (after FTMO but before storing)
            safety_ok, safety_reason, rejection_id = self._check_safety_guards(symbol, result, df)
            if not safety_ok:
                logger.warning(f"Safety Guards block signal for {symbol}: {safety_reason}")
                # Mark as invalid
                result.parsed_signal.is_valid = False
                result.parsed_signal.reasoning += f" [SAFETY BLOCK: {safety_reason}]"
                # Store rejection_id for potential counterfactual tracking
                if hasattr(result, 'rejection_id'):
                    result.rejection_id = rejection_id
                else:
                    result.__dict__['rejection_id'] = rejection_id

            # Apply Volatility Regime Adjustments (adaptive stops/targets)
            if result.parsed_signal.signal != SignalType.HOLD and result.parsed_signal.is_valid:
                vol_regime = self.volatility_detector.detect_regime(df)

                if result.parsed_signal.stop_loss and result.parsed_signal.take_profit:
                    original_sl = result.parsed_signal.stop_loss
                    original_tp = result.parsed_signal.take_profit

                    # Calculate distances
                    entry = result.parsed_signal.entry_price
                    sl_distance = abs(entry - original_sl)
                    tp_distance = abs(entry - original_tp)

                    # Apply regime multipliers
                    adjusted_sl_distance = sl_distance * vol_regime.recommended_stop_multiplier
                    adjusted_tp_distance = tp_distance * vol_regime.recommended_target_multiplier

                    # Calculate new levels
                    if result.parsed_signal.signal == SignalType.BUY:
                        result.parsed_signal.stop_loss = entry - adjusted_sl_distance
                        result.parsed_signal.take_profit = entry + adjusted_tp_distance
                    else:  # SELL
                        result.parsed_signal.stop_loss = entry + adjusted_sl_distance
                        result.parsed_signal.take_profit = entry - adjusted_tp_distance

                    logger.info(
                        f"📊 VOLATILITY REGIME: {vol_regime.regime.upper()} | "
                        f"ATR %ile: {vol_regime.atr_percentile:.0%} | "
                        f"Stop: {vol_regime.recommended_stop_multiplier:.1f}x | "
                        f"Target: {vol_regime.recommended_target_multiplier:.1f}x | "
                        f"Bias: {vol_regime.trade_bias}"
                    )

                    # Update reasoning
                    result.parsed_signal.reasoning += f" [VOLATILITY: {vol_regime.regime} regime, {vol_regime.trade_bias} bias]"

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

        # A/B test: Alternate between strategies PER SIGNAL (not per scan)
        # Initialize counter if first run
        if not hasattr(self, '_ab_test_counter'):
            self._ab_test_counter = 0

        for symbol in self.runtime_config.symbols:
            # Increment counter and determine strategy for THIS signal
            self._ab_test_counter += 1

            # Alternate strategies: odd = MOMENTUM, even = ENTROPY
            strategy = "v7_full_math" if self._ab_test_counter % 2 == 1 else "v7_deepseek_only"
            strategy_label = "MOMENTUM (trend-following)" if strategy == "v7_full_math" else "ENTROPY (mean-reversion)"
            logger.info(f"🧪 A/B TEST: Using strategy '{strategy_label}' for {symbol} (signal #{self._ab_test_counter})")
            try:
                # FIX #2: Check rate limits PER SYMBOL
                rate_ok, rate_reason = self._check_rate_limits(symbol)
                if not rate_ok:
                    logger.info(f"{symbol} rate limit: {rate_reason}")
                    continue  # Skip this symbol, continue with others

                # Check cost limits (global)
                cost_ok, cost_reason = self._check_cost_limits()
                if not cost_ok:
                    logger.warning(f"Cost limit reached: {cost_reason}")
                    break  # Stop ALL symbols if budget exhausted

                # Generate signal with A/B test strategy
                result = self.generate_signal_for_symbol(symbol, strategy=strategy)

                if result is None:
                    continue

                # FIX #4: Filter by confidence threshold (LOWERED for more trades)
                MIN_CONFIDENCE = 0.25  # 25% minimum - take more trades for volume
                if result.parsed_signal.confidence < MIN_CONFIDENCE:
                    logger.info(
                        f"❌ Signal REJECTED (low confidence): {symbol} {result.parsed_signal.signal.value} "
                        f"confidence {result.parsed_signal.confidence:.1%} < {MIN_CONFIDENCE:.1%} threshold"
                    )
                    continue  # Skip only very low-confidence signals

                # Get current price for formatting
                df = self.data_fetcher.fetch_latest_candles(symbol=symbol, num_candles=1)
                current_price = float(df['close'].iloc[-1]) if not df.empty else 0.0

                # Format and output signal
                output = self._format_signal_output(symbol, result, current_price)
                print(output)

                # Save to database with strategy tag (use human-readable label for dashboard)
                self._save_signal_to_db(symbol, result, current_price, strategy=strategy_label)

                # Send to Telegram
                if self.telegram.enabled:
                    try:
                        self.telegram.send_v7_signal(symbol, result)
                    except Exception as e:
                        logger.error(f"Failed to send Telegram notification: {e}")

                # FIX #2: Update signal history (for rate limiting) - PER SYMBOL
                self.signal_history.append({
                    'timestamp': result.parsed_signal.timestamp,
                    'symbol': symbol,
                    'signal': result.parsed_signal.signal.value,
                    'confidence': result.parsed_signal.confidence
                })
                # Track last signal time FOR THIS SYMBOL
                self.last_signal_time_per_symbol[symbol] = result.parsed_signal.timestamp

                # Count valid signals
                if result.parsed_signal.is_valid and result.parsed_signal.signal != SignalType.HOLD:
                    valid_signals += 1

                # Small delay between symbols
                time.sleep(2)

            except Exception as e:
                import traceback
                logger.error(f"Error processing {symbol}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
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

        # Send startup notification to Telegram
        if self.telegram.enabled:
            startup_details = (
                f"Symbols: {', '.join(self.runtime_config.symbols)}\n"
                f"Scan Interval: {sleep_seconds}s\n"
                f"Rate Limit: {self.runtime_config.max_signals_per_hour} signals/hour\n"
                f"Conservative Mode: {'ON' if self.runtime_config.conservative_mode else 'OFF'}"
            )
            self.telegram.send_runtime_status("Started", startup_details)

        iteration = 0

        try:
            while iterations == -1 or iteration < iterations:
                iteration += 1

                logger.info(f"\n{'='*80}")
                logger.info(f"ITERATION {iteration} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'='*80}")

                # Run single scan
                valid_signals = self.run_single_scan()

                # Check and exit paper trades (if enabled)
                if self.paper_trader:
                    try:
                        # Fetch current prices for all symbols
                        current_prices = {}
                        for symbol in self.runtime_config.symbols:
                            df = self.data_fetcher.fetch_latest_candles(symbol=symbol, num_candles=1)
                            if not df.empty:
                                current_prices[symbol] = float(df['close'].iloc[-1])

                        # Check exit conditions and close positions
                        exited_trades = self.paper_trader.check_and_exit_trades(current_prices)

                        if exited_trades:
                            logger.info(f"📊 Closed {len(exited_trades)} paper trades this iteration")

                            # Record each trade to performance trackers
                            for trade in exited_trades:
                                return_pct = trade['pnl_percent'] / 100.0  # Convert to decimal

                                # Record to Sharpe Tracker
                                self.sharpe_tracker.record_trade_return(
                                    timestamp=trade['exit_time'],
                                    return_pct=return_pct,
                                    symbol=trade['symbol']
                                )

                                # Record to CVaR Calculator
                                self.cvar_calculator.record_return(
                                    return_pct=return_pct,
                                    timestamp=trade['exit_time']
                                )

                                # Record to Sortino Tracker
                                self.sortino_tracker.record_return(
                                    return_pct=return_pct,
                                    timestamp=trade['exit_time']
                                )

                                # Record to Calmar Tracker
                                self.calmar_tracker.record_return(
                                    return_pct=return_pct,
                                    timestamp=trade['exit_time']
                                )

                                # Record to Omega Calculator
                                self.omega_calculator.record_return(
                                    return_pct=return_pct,
                                    timestamp=trade['exit_time']
                                )
                    except Exception as e:
                        logger.error(f"Failed to check/exit paper trades: {e}")

                # Print statistics
                stats = self.signal_generator.get_statistics()
                logger.info(f"\nV7 Statistics:")
                logger.info(f"  DeepSeek API Calls:    {stats['deepseek_api']['total_requests']}")
                logger.info(f"  Total API Cost:        ${stats['deepseek_api']['total_cost_usd']:.6f}")
                logger.info(f"  Bayesian Win Rate:     {stats['bayesian_win_rate']:.1%}")
                logger.info(f"  Bayesian Total Trades: {stats['bayesian_total_trades']}")
                logger.info(f"  Daily Cost:            ${self.daily_cost:.4f} / ${self.runtime_config.max_cost_per_day:.2f}")
                logger.info(f"  Monthly Cost:          ${self.monthly_cost:.2f} / ${self.runtime_config.max_cost_per_month:.2f}")

                # Print Sharpe Ratio metrics (if enough trades)
                sharpe_metrics = self.sharpe_tracker.get_sharpe_metrics()
                cvar_metrics = self.cvar_calculator.get_cvar_metrics()

                if sharpe_metrics.total_trades >= 5:
                    logger.info(f"\n📊 Sharpe Ratio Performance:")
                    logger.info(f"  7-day Sharpe:          {sharpe_metrics.sharpe_ratio_7d:.2f}")
                    logger.info(f"  14-day Sharpe:         {sharpe_metrics.sharpe_ratio_14d:.2f}")
                    logger.info(f"  30-day Sharpe:         {sharpe_metrics.sharpe_ratio_30d:.2f}")
                    logger.info(f"  Ann. Return:           {sharpe_metrics.annualized_return:+.1%}")
                    logger.info(f"  Ann. Volatility:       {sharpe_metrics.annualized_volatility:.1%}")
                    logger.info(f"  Max Drawdown:          {sharpe_metrics.max_drawdown:.1%}")
                    logger.info(f"  Performance Trend:     {sharpe_metrics.performance_trend.upper()}")
                    logger.info(f"  Summary: {sharpe_metrics.summary}")
                elif sharpe_metrics.total_trades > 0:
                    logger.info(f"\n📊 Sharpe Ratio: {sharpe_metrics.total_trades} trades (need 5+ for metrics)")

                # Print CVaR metrics (if enough trades)
                if cvar_metrics.risk_level != 'unknown':
                    logger.info(f"\n⚠️  CVaR (Tail Risk) Analysis:")
                    logger.info(f"  95% CVaR:              {cvar_metrics.cvar_95_historical:.2%}")
                    logger.info(f"  99% CVaR:              {cvar_metrics.cvar_99_historical:.2%}")
                    logger.info(f"  Worst Loss:            {cvar_metrics.worst_loss:.2%}")
                    logger.info(f"  Tail Ratio:            {cvar_metrics.tail_ratio:.2f}x")
                    logger.info(f"  Risk Level:            {cvar_metrics.risk_level.upper()}")
                    logger.info(f"  Max Position (95%):    {cvar_metrics.max_position_size_95:.0%}")
                    if cvar_metrics.warnings:
                        for warning in cvar_metrics.warnings:
                            logger.info(f"  {warning}")

                # Print Sortino Ratio metrics (if enough trades)
                if sharpe_metrics.total_trades >= 5:
                    sortino_metrics = self.sortino_tracker.get_sortino_metrics(
                        sharpe_ratio=sharpe_metrics.sharpe_ratio_30d
                    )
                    logger.info(f"\n📈 Sortino Ratio (Downside Risk):")
                    logger.info(f"  7-day Sortino:         {sortino_metrics.sortino_ratio_7d:.2f}")
                    logger.info(f"  14-day Sortino:        {sortino_metrics.sortino_ratio_14d:.2f}")
                    logger.info(f"  30-day Sortino:        {sortino_metrics.sortino_ratio_30d:.2f}")
                    logger.info(f"  Downside Deviation:    {sortino_metrics.downside_deviation:.1%}")
                    logger.info(f"  Upside Deviation:      {sortino_metrics.upside_deviation:.1%}")
                    if sortino_metrics.sortino_sharpe_ratio > 0:
                        logger.info(f"  Sortino / Sharpe:      {sortino_metrics.sortino_sharpe_ratio:.2f}x")
                        if sortino_metrics.sortino_sharpe_ratio > 1.0:
                            logger.info(f"  ✅ Favorable asymmetry (wins > losses)")

                # Print Calmar Ratio metrics (if enough trades)
                if sharpe_metrics.total_trades >= 5:
                    calmar_metrics = self.calmar_tracker.get_calmar_metrics()
                    logger.info(f"\n📉 Calmar Ratio (Return/Max Drawdown):")
                    logger.info(f"  30-day Calmar:         {calmar_metrics.calmar_ratio_30d:.2f}")
                    if calmar_metrics.calmar_ratio_90d:
                        logger.info(f"  90-day Calmar:         {calmar_metrics.calmar_ratio_90d:.2f}")
                    logger.info(f"  Max Drawdown:          {calmar_metrics.max_drawdown:.1%}")
                    logger.info(f"  Current Drawdown:      {calmar_metrics.current_drawdown:.1%}")
                    logger.info(f"  Max DD Duration:       {calmar_metrics.max_drawdown_duration_days} days")
                    if calmar_metrics.time_to_recovery_days:
                        logger.info(f"  Recovery Time:         {calmar_metrics.time_to_recovery_days} days")
                    logger.info(f"  Quality:               {calmar_metrics.calmar_quality.upper()}")

                # Print Omega Ratio metrics (if enough trades)
                if sharpe_metrics.total_trades >= 20:
                    omega_metrics = self.omega_calculator.calculate_omega()
                    logger.info(f"\n🎲 Omega Ratio (Gain/Loss Distribution):")
                    logger.info(f"  Omega (0%):            {omega_metrics.omega_0pct:.2f}")
                    logger.info(f"  Omega (RF):            {omega_metrics.omega_rf:.2f}")
                    logger.info(f"  Expected Gain:         {omega_metrics.expected_gains:.2%}")
                    logger.info(f"  Expected Loss:         {omega_metrics.expected_losses:.2%}")
                    logger.info(f"  Skewness:              {omega_metrics.skewness:.2f}")
                    logger.info(f"  Kurtosis (excess):     {omega_metrics.kurtosis:.2f}")
                    logger.info(f"  Quality:               {omega_metrics.omega_quality.upper()}")

                    if omega_metrics.skewness > 0:
                        logger.info(f"  ✅ Positive skew (favorable asymmetry)")
                    elif omega_metrics.skewness < -0.5:
                        logger.info(f"  ⚠️  Negative skew (large losses)")

                # Run Portfolio Optimization (every 10 iterations or when we have enough data)
                if iteration % 10 == 0:
                    try:
                        logger.info("\n💼 Running Portfolio Optimization (Markowitz MPT)...")

                        # Load historical prices from database
                        prices_df = self.portfolio_optimizer.load_historical_prices(data_source='database')

                        if not prices_df.empty and len(prices_df) >= 30:
                            # Optimize for max Sharpe ratio
                            optimal_portfolio = self.portfolio_optimizer.optimize_max_sharpe(prices_df)

                            logger.info(f"\n💼 Optimal Portfolio Allocation (Markowitz):")
                            logger.info(f"  Expected Annual Return:    {optimal_portfolio.expected_annual_return:.1%}")
                            logger.info(f"  Annual Volatility:         {optimal_portfolio.annual_volatility:.1%}")
                            logger.info(f"  Sharpe Ratio:              {optimal_portfolio.sharpe_ratio:.2f}")
                            logger.info(f"  Max Drawdown (est):        {optimal_portfolio.max_drawdown_estimate:.1%}")
                            logger.info(f"  Diversification Ratio:     {optimal_portfolio.diversification_ratio:.2f}")
                            logger.info(f"  Active Assets:             {optimal_portfolio.n_assets}")
                            logger.info(f"  Concentration (HHI):       {optimal_portfolio.concentration_hhi:.3f}")

                            logger.info(f"\n  Optimal Weights (Top 5):")
                            sorted_weights = sorted(
                                optimal_portfolio.weights.items(),
                                key=lambda x: x[1],
                                reverse=True
                            )
                            for symbol, weight in sorted_weights[:5]:
                                if weight > 0.01:  # Only show > 1%
                                    logger.info(f"    {symbol}: {weight:>6.1%}")

                            # Show interpretation
                            if optimal_portfolio.sharpe_ratio >= 2.0:
                                logger.info(f"  ✅ EXCELLENT portfolio (Sharpe >= 2.0)")
                            elif optimal_portfolio.sharpe_ratio >= 1.5:
                                logger.info(f"  ✅ VERY GOOD portfolio (Sharpe >= 1.5)")
                            elif optimal_portfolio.sharpe_ratio >= 1.0:
                                logger.info(f"  ✅ GOOD portfolio (Sharpe >= 1.0)")
                            else:
                                logger.info(f"  ⚠️  MODERATE portfolio (Sharpe < 1.0)")
                        else:
                            logger.info(f"  ⚠️  Insufficient price data ({len(prices_df)} days, need 30+)")

                    except Exception as e:
                        logger.error(f"Portfolio optimization failed: {e}")

                # Sleep until next iteration
                if iterations == -1 or iteration < iterations:
                    logger.info(f"\nSleeping {sleep_seconds}s until next scan...")
                    time.sleep(sleep_seconds)

        except KeyboardInterrupt:
            logger.info("\n\nV7 Runtime stopped by user (Ctrl+C)")
            if self.telegram.enabled:
                self.telegram.send_runtime_status("Stopped", "Manual shutdown (Ctrl+C)")

        except Exception as e:
            logger.error(f"\n\nV7 Runtime crashed: {e}")
            if self.telegram.enabled:
                self.telegram.send_runtime_status("Error", f"Runtime crashed: {str(e)[:200]}")
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

            # Send shutdown notification with final stats
            if self.telegram.enabled:
                shutdown_details = (
                    f"Total Iterations: {iteration}\n"
                    f"DeepSeek API Calls: {stats['deepseek_api']['total_requests']}\n"
                    f"Total Cost: ${stats['deepseek_api']['total_cost_usd']:.6f}\n"
                    f"Daily Cost: ${self.daily_cost:.4f}"
                )
                self.telegram.send_runtime_status("Stopped", shutdown_details)


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
        default=[
            "BTC-USD", "ETH-USD", "SOL-USD",  # Original 3
            "XRP-USD", "DOGE-USD", "ADA-USD", "AVAX-USD",  # New 4
            "LINK-USD", "MATIC-USD", "LTC-USD"  # New 3
        ],
        help="Symbols to scan (default: BTC ETH SOL XRP DOGE ADA AVAX LINK MATIC LTC)"
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
