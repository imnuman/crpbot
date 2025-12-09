"""
HYDRA 4.0 Prometheus Metrics Definitions

Defines all metrics exposed by the HYDRA runtime to Prometheus.
Uses prometheus_client library for metric types.

Metrics Categories:
1. Trading Performance (P&L, win rate, trades)
2. Engine Tournament (rankings, weights, scores)
3. Risk & Safety (drawdown, kill switch, limits)
4. Market Data (prices, API latency)
5. System Health (uptime, errors)
"""

from prometheus_client import Gauge, Counter, Histogram, Info
from typing import Dict, Any
import time


class HydraMetrics:
    """
    Central class for all HYDRA Prometheus metrics.

    All metrics are class-level (singleton pattern) for easy access
    from anywhere in the codebase.
    """

    # ========== Trading Performance ==========

    # Total P&L percentage
    pnl_total = Gauge(
        'hydra_pnl_total_percent',
        'Total P&L percentage across all trades'
    )

    # Daily P&L percentage
    pnl_daily = Gauge(
        'hydra_pnl_daily_percent',
        'Daily P&L percentage'
    )

    # Win rate (24h rolling)
    win_rate_24h = Gauge(
        'hydra_win_rate_24h',
        '24-hour rolling win rate percentage'
    )

    # Win rate (all time)
    win_rate_total = Gauge(
        'hydra_win_rate_total',
        'All-time win rate percentage'
    )

    # Total trades counter
    trades_total = Counter(
        'hydra_trades_total',
        'Total number of trades',
        ['asset', 'direction', 'engine', 'outcome']
    )

    # Open positions gauge
    open_positions = Gauge(
        'hydra_open_positions',
        'Number of currently open positions',
        ['asset', 'engine']
    )

    # Consecutive wins/losses
    consecutive_wins = Gauge(
        'hydra_consecutive_wins',
        'Current consecutive winning trades streak'
    )

    consecutive_losses = Gauge(
        'hydra_consecutive_losses',
        'Current consecutive losing trades streak'
    )

    # ========== Engine Tournament ==========

    # Engine tournament rank
    engine_rank = Gauge(
        'hydra_engine_rank',
        'Current tournament ranking (1-4)',
        ['engine']
    )

    # Engine weight (influence on trades)
    engine_weight = Gauge(
        'hydra_engine_weight',
        'Current tournament weight percentage',
        ['engine']
    )

    # Engine total points
    engine_points = Gauge(
        'hydra_engine_points',
        'Total tournament points',
        ['engine']
    )

    # Engine win rate
    engine_win_rate = Gauge(
        'hydra_engine_win_rate',
        'Per-engine win rate percentage',
        ['engine']
    )

    # Engine active status
    engine_active = Gauge(
        'hydra_engine_active',
        'Whether engine is active (1) or offline (0)',
        ['engine']
    )

    # ========== Risk & Safety ==========

    # Daily drawdown
    daily_drawdown = Gauge(
        'hydra_daily_drawdown_percent',
        'Current daily drawdown percentage'
    )

    # Total drawdown
    total_drawdown = Gauge(
        'hydra_total_drawdown_percent',
        'Total drawdown from peak'
    )

    # Kill switch status
    kill_switch_active = Gauge(
        'hydra_kill_switch_active',
        'Whether kill switch is active (1) or inactive (0)'
    )

    # Guardian status
    guardian_status = Gauge(
        'hydra_guardian_status',
        'Guardian status: 0=inactive, 1=normal, 2=warning, 3=critical'
    )

    # Risk exposure
    risk_exposure = Gauge(
        'hydra_risk_exposure_percent',
        'Current risk exposure as percentage of capital'
    )

    # ========== Per-Asset Performance (Phase 1) ==========

    # Per-asset P&L
    asset_pnl = Gauge(
        'hydra_asset_pnl_percent',
        'Total P&L percentage by asset',
        ['asset']
    )

    # Per-asset win rate
    asset_win_rate = Gauge(
        'hydra_asset_win_rate',
        'Win rate by asset (0-1)',
        ['asset']
    )

    # Per-asset trade count
    asset_trade_count = Gauge(
        'hydra_asset_trade_count',
        'Total trades by asset',
        ['asset']
    )

    # ========== Technical Indicators (Phase 1) ==========

    # Current market regime
    regime_current = Gauge(
        'hydra_regime_current',
        'Current regime: 0=CHOPPY, 1=RANGING, 2=TRENDING_UP, 3=TRENDING_DOWN, 4=VOLATILE',
        ['asset']
    )

    # Regime confidence
    regime_confidence = Gauge(
        'hydra_regime_confidence',
        'Regime detection confidence (0-1)',
        ['asset']
    )

    # ADX indicator
    indicator_adx = Gauge(
        'hydra_indicator_adx',
        'Average Directional Index (trend strength)',
        ['asset']
    )

    # ATR indicator
    indicator_atr = Gauge(
        'hydra_indicator_atr',
        'Average True Range (volatility)',
        ['asset']
    )

    # Bollinger Band width
    indicator_bb_width = Gauge(
        'hydra_indicator_bb_width',
        'Bollinger Band width (squeeze indicator)',
        ['asset']
    )

    # ========== Guardian Full State (Phase 1) ==========

    # Account balance
    account_balance = Gauge(
        'hydra_account_balance_usd',
        'Current account balance in USD'
    )

    # Peak balance (for drawdown calculation)
    peak_balance = Gauge(
        'hydra_peak_balance_usd',
        'Peak account balance in USD'
    )

    # Daily P&L USD
    daily_pnl_usd = Gauge(
        'hydra_daily_pnl_usd',
        'Daily P&L in USD'
    )

    # Daily P&L percent
    daily_pnl_percent = Gauge(
        'hydra_daily_pnl_percent',
        'Daily P&L as percentage'
    )

    # Circuit breaker active
    circuit_breaker_active = Gauge(
        'hydra_circuit_breaker_active',
        'Circuit breaker status: 1=active (reduced size), 0=normal'
    )

    # Emergency shutdown
    emergency_shutdown = Gauge(
        'hydra_emergency_shutdown_active',
        'Emergency shutdown: 1=active, 0=normal'
    )

    # Trading allowed
    trading_allowed = Gauge(
        'hydra_trading_allowed',
        'Trading allowed: 1=yes, 0=no (shutdown/circuit breaker)'
    )

    # Position size multiplier (affected by circuit breaker)
    position_size_multiplier = Gauge(
        'hydra_position_size_multiplier',
        'Current position size multiplier (1.0=normal, 0.5=reduced)'
    )

    # ========== Per-Regime Performance (Phase 2) ==========

    # Per-regime P&L
    regime_pnl = Gauge(
        'hydra_regime_pnl_percent',
        'Total P&L percentage by regime',
        ['regime']
    )

    # Per-regime win rate
    regime_win_rate = Gauge(
        'hydra_regime_win_rate',
        'Win rate by regime (0-1)',
        ['regime']
    )

    # Per-regime trade count
    regime_trade_count = Gauge(
        'hydra_regime_trade_count',
        'Total trades by regime',
        ['regime']
    )

    # Per-regime average P&L per trade
    regime_avg_pnl = Gauge(
        'hydra_regime_avg_pnl_percent',
        'Average P&L per trade by regime',
        ['regime']
    )

    # ========== Engine Analytics (Phase 3) ==========

    # Engine per-asset P&L
    engine_asset_pnl = Gauge(
        'hydra_engine_asset_pnl_percent',
        'Engine P&L by asset',
        ['engine', 'asset']
    )

    # Engine per-asset accuracy
    engine_asset_accuracy = Gauge(
        'hydra_engine_asset_accuracy',
        'Engine prediction accuracy by asset (0-1)',
        ['engine', 'asset']
    )

    # Engine votes by direction
    engine_votes = Gauge(
        'hydra_engine_votes_total',
        'Total votes by engine and direction',
        ['engine', 'direction']
    )

    # Engine agreement rate
    engine_agreement = Gauge(
        'hydra_engine_agreement_rate',
        'Rate at which engines agree on direction (0-1)'
    )

    # Engine last vote
    engine_last_vote = Gauge(
        'hydra_engine_last_vote',
        'Last vote direction: 0=HOLD, 1=BUY, -1=SELL',
        ['engine']
    )

    # ========== Advanced Statistics (Phase 4) ==========

    # Sharpe ratio
    sharpe_ratio = Gauge(
        'hydra_sharpe_ratio',
        'Risk-adjusted return (Sharpe ratio)'
    )

    # Sortino ratio
    sortino_ratio = Gauge(
        'hydra_sortino_ratio',
        'Downside risk-adjusted return (Sortino ratio)'
    )

    # Max drawdown
    max_drawdown = Gauge(
        'hydra_max_drawdown_percent',
        'Maximum drawdown from peak'
    )

    # Calmar ratio
    calmar_ratio = Gauge(
        'hydra_calmar_ratio',
        'Return / Max Drawdown ratio'
    )

    # Profit factor
    profit_factor = Gauge(
        'hydra_profit_factor',
        'Gross profit / Gross loss ratio'
    )

    # Expectancy
    expectancy = Gauge(
        'hydra_expectancy',
        'Average expected profit per trade'
    )

    # Average R:R ratio
    avg_risk_reward = Gauge(
        'hydra_avg_risk_reward',
        'Average risk-to-reward ratio'
    )

    # Total trades
    total_trades = Gauge(
        'hydra_total_trades',
        'Total number of completed trades'
    )

    # ========== Market Data ==========

    # Asset prices
    asset_price = Gauge(
        'hydra_asset_price_usd',
        'Current asset price in USD',
        ['asset']
    )

    # Price change 24h
    price_change_24h = Gauge(
        'hydra_price_change_24h_percent',
        '24-hour price change percentage',
        ['asset']
    )

    # API latency
    api_latency = Histogram(
        'hydra_api_latency_seconds',
        'API request latency in seconds',
        ['api', 'endpoint'],
        buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )

    # ========== 4-Engine Strategy Generation (HYDRA 4.0) ==========

    # Total strategies generated in last cycle
    generation_strategies_total = Gauge(
        'hydra_generation_strategies_total',
        'Total strategies generated in last cycle'
    )

    # Strategies generated by engine
    generation_by_engine = Gauge(
        'hydra_generation_by_engine',
        'Strategies generated by engine in last cycle',
        ['engine']
    )

    # Winning engine in last generation cycle
    generation_winning_engine = Gauge(
        'hydra_generation_winning_engine',
        'Engine that won last generation cycle: 1=A, 2=B, 3=C, 4=D'
    )

    # Vote breakdown from top 100 strategies
    generation_vote_breakdown = Gauge(
        'hydra_generation_votes',
        'Vote count from top 100 strategies by engine',
        ['engine']
    )

    # Last generation cycle timestamp
    generation_last_run = Gauge(
        'hydra_generation_last_run_timestamp',
        'Unix timestamp of last generation cycle'
    )

    # Generation cycle duration
    generation_duration_ms = Gauge(
        'hydra_generation_duration_ms',
        'Duration of last generation cycle in milliseconds'
    )

    # Next scheduled generation
    generation_next_run = Gauge(
        'hydra_generation_next_run_timestamp',
        'Unix timestamp of next scheduled generation (midnight UTC)'
    )

    # Generation mode (mock vs real)
    generation_mode = Gauge(
        'hydra_generation_mode',
        'Generation mode: 0=real API, 1=mock'
    )

    # ========== Independent Trading Mode (HYDRA 4.0) ==========

    # Independent mode active flag
    independent_mode_active = Gauge(
        'hydra_independent_mode_active',
        'Whether independent trading mode is active (1) or disabled (0)'
    )

    # Specialty trigger status per engine
    specialty_trigger = Gauge(
        'hydra_specialty_trigger',
        'Whether engine specialty is triggered for current market: 1=triggered, 0=not triggered',
        ['engine']
    )

    # Engine portfolio balance (each starts at $25k)
    engine_portfolio_balance = Gauge(
        'hydra_engine_portfolio_balance_usd',
        'Current engine portfolio balance in USD',
        ['engine']
    )

    # Engine portfolio trades
    engine_portfolio_trades = Gauge(
        'hydra_engine_portfolio_trades',
        'Total trades made by engine portfolio',
        ['engine']
    )

    # Engine portfolio wins
    engine_portfolio_wins = Gauge(
        'hydra_engine_portfolio_wins',
        'Total wins by engine portfolio',
        ['engine']
    )

    # Engine portfolio P&L
    engine_portfolio_pnl = Gauge(
        'hydra_engine_portfolio_pnl_usd',
        'Engine portfolio P&L in USD',
        ['engine']
    )

    # Engine specialty type
    engine_specialty = Info(
        'hydra_engine_specialty',
        'Engine specialty type (liquidation_cascade, funding_extreme, etc)'
    )

    # ========== Turbo Batch Generation (HYDRA 4.0) ==========

    # Turbo batch mode active flag
    turbo_batch_active = Gauge(
        'hydra_turbo_batch_active',
        'Whether turbo batch generation mode is active (1) or disabled (0)'
    )

    # Strategies generated per specialty
    turbo_strategies_per_specialty = Gauge(
        'hydra_turbo_strategies_per_specialty',
        'Number of strategies generated per specialty (target: 250)',
        ['specialty']
    )

    # Total strategies backtested
    turbo_backtested_count = Gauge(
        'hydra_turbo_backtested_count',
        'Total strategies backtested in turbo tournament'
    )

    # Top strategy Sharpe ratio
    turbo_top_sharpe = Gauge(
        'hydra_turbo_top_sharpe',
        'Sharpe ratio of top-ranked strategy from turbo tournament'
    )

    # Top strategy return
    turbo_top_return = Gauge(
        'hydra_turbo_top_return_percent',
        'Return percentage of top-ranked strategy'
    )

    # Turbo generation time
    turbo_generation_time_ms = Gauge(
        'hydra_turbo_generation_time_ms',
        'Time taken for turbo batch generation in milliseconds'
    )

    # Turbo tournament time
    turbo_tournament_time_ms = Gauge(
        'hydra_turbo_tournament_time_ms',
        'Time taken for turbo tournament ranking in milliseconds'
    )

    # Best specialty (which specialty won)
    turbo_best_specialty = Gauge(
        'hydra_turbo_best_specialty',
        'Specialty that produced winning strategy: 1=liquidation, 2=funding, 3=orderbook, 4=regime'
    )

    # ========== FTMO Multi-Bot System ==========

    # FTMO Bot status (active/inactive)
    ftmo_bot_active = Gauge(
        'ftmo_bot_active',
        'Whether FTMO bot is active (1) or inactive (0)',
        ['bot']
    )

    # FTMO Bot trades today
    ftmo_bot_trades_today = Gauge(
        'ftmo_bot_trades_today',
        'Number of trades taken today by each FTMO bot',
        ['bot']
    )

    # FTMO Bot win rate (rolling 30 days)
    ftmo_bot_win_rate = Gauge(
        'ftmo_bot_win_rate',
        'Win rate percentage for each FTMO bot',
        ['bot']
    )

    # FTMO Bot P&L today
    ftmo_bot_pnl_today = Gauge(
        'ftmo_bot_pnl_today_usd',
        'Today P&L in USD for each FTMO bot',
        ['bot']
    )

    # FTMO Bot total P&L
    ftmo_bot_pnl_total = Gauge(
        'ftmo_bot_pnl_total_usd',
        'Total P&L in USD for each FTMO bot',
        ['bot']
    )

    # FTMO Orchestrator open positions
    ftmo_open_positions = Gauge(
        'ftmo_open_positions',
        'Current number of open FTMO positions'
    )

    # FTMO Orchestrator max positions
    ftmo_max_positions = Gauge(
        'ftmo_max_positions',
        'Maximum allowed concurrent positions'
    )

    # FTMO Daily drawdown percent
    ftmo_daily_drawdown = Gauge(
        'ftmo_daily_drawdown_percent',
        'Current daily drawdown percentage (FTMO limit: 5%)'
    )

    # FTMO Total drawdown percent
    ftmo_total_drawdown = Gauge(
        'ftmo_total_drawdown_percent',
        'Current total drawdown percentage (FTMO limit: 10%)'
    )

    # FTMO Kill switch active
    ftmo_kill_switch = Gauge(
        'ftmo_kill_switch_active',
        'Whether FTMO kill switch is active (1) or not (0)'
    )

    # FTMO Account balance
    ftmo_account_balance = Gauge(
        'ftmo_account_balance_usd',
        'Current FTMO account balance in USD'
    )

    # FTMO Paper mode
    ftmo_paper_mode = Gauge(
        'ftmo_paper_mode_active',
        'Whether FTMO is in paper mode (1) or live (0)'
    )

    # FTMO Signal generated counter
    ftmo_signals_total = Counter(
        'ftmo_signals_total',
        'Total signals generated by FTMO bots',
        ['bot', 'direction']
    )

    # FTMO Trade outcome counter
    ftmo_trade_outcomes = Counter(
        'ftmo_trade_outcomes_total',
        'Total FTMO trade outcomes',
        ['bot', 'outcome']
    )

    # FTMO Bot last trade timestamp
    ftmo_bot_last_trade = Gauge(
        'ftmo_bot_last_trade_timestamp',
        'Unix timestamp of last trade for each FTMO bot',
        ['bot']
    )

    # FTMO Combined daily target progress
    ftmo_daily_target_progress = Gauge(
        'ftmo_daily_target_progress_percent',
        'Progress toward daily target (0-100%, target ~$300/day)'
    )

    # ========== Bug Detection & Monitoring ==========

    # Engine errors counter (for bug detection)
    engine_errors_total = Counter(
        'hydra_engine_errors_total',
        'Total errors by engine',
        ['engine']
    )

    # Low confidence strategy counter
    engine_low_confidence_total = Counter(
        'hydra_engine_low_confidence_total',
        'Total low confidence (<55%) strategies by engine',
        ['engine']
    )

    # Data feed errors counter
    data_feed_errors_total = Counter(
        'hydra_data_feed_errors_total',
        'Total data feed errors by source',
        ['source']
    )

    # Duplicate orders blocked
    duplicate_orders_blocked_total = Counter(
        'hydra_duplicate_orders_blocked_total',
        'Total duplicate orders blocked by guard'
    )

    # Last checkpoint timestamp
    last_checkpoint_timestamp = Gauge(
        'hydra_last_checkpoint_timestamp',
        'Unix timestamp of last successful state checkpoint'
    )

    # Iterations counter
    iterations_total = Counter(
        'hydra_iterations_total',
        'Total trading iterations completed'
    )

    # Trades closed counter (with outcome)
    trades_closed_total = Counter(
        'hydra_trades_closed_total',
        'Total trades closed by outcome',
        ['outcome']
    )

    # ========== System Health ==========

    # Runtime uptime
    uptime_seconds = Gauge(
        'hydra_uptime_seconds',
        'HYDRA runtime uptime in seconds'
    )

    # Last cycle timestamp
    last_cycle_timestamp = Gauge(
        'hydra_last_cycle_timestamp',
        'Unix timestamp of last completed cycle'
    )

    # Cycle duration
    cycle_duration = Histogram(
        'hydra_cycle_duration_seconds',
        'Duration of trading cycle in seconds',
        buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
    )

    # Error counter
    errors_total = Counter(
        'hydra_errors_total',
        'Total number of errors',
        ['type', 'component']
    )

    # System info
    system_info = Info(
        'hydra_system',
        'HYDRA system information'
    )

    # ========== Class Variables ==========
    _start_time: float = 0.0

    # ========== Convenience Methods ==========

    @classmethod
    def initialize(cls):
        """Initialize metrics with default values and record start time."""
        cls._start_time = time.time()

        # Set system info
        cls.system_info.info({
            'version': '4.0',
            'environment': 'production',
            'mode': 'paper_trading'
        })

        # Initialize engine metrics
        for engine in ['A', 'B', 'C', 'D']:
            cls.engine_active.labels(engine=engine).set(1)
            cls.engine_rank.labels(engine=engine).set(0)
            cls.engine_weight.labels(engine=engine).set(25)  # Default 25% each
            cls.engine_points.labels(engine=engine).set(0)
            cls.engine_win_rate.labels(engine=engine).set(0)

        # Initialize risk metrics
        cls.kill_switch_active.set(0)
        cls.guardian_status.set(1)  # Normal
        cls.daily_drawdown.set(0)
        cls.total_drawdown.set(0)
        cls.risk_exposure.set(0)

        # Initialize performance metrics
        cls.pnl_total.set(0)
        cls.pnl_daily.set(0)
        cls.win_rate_24h.set(0)
        cls.win_rate_total.set(0)
        cls.consecutive_wins.set(0)
        cls.consecutive_losses.set(0)

        # Initialize HYDRA 4.0 metrics
        cls.initialize_hydra_40()

    @classmethod
    def update_uptime(cls):
        """Update the uptime metric."""
        if cls._start_time > 0:
            cls.uptime_seconds.set(time.time() - cls._start_time)

    @classmethod
    def set_pnl(cls, total: float, daily: float = None):
        """Set P&L metrics."""
        cls.pnl_total.set(total)
        if daily is not None:
            cls.pnl_daily.set(daily)

    @classmethod
    def set_win_rate(cls, rate_24h: float, rate_total: float = None):
        """Set win rate metrics."""
        cls.win_rate_24h.set(rate_24h)
        if rate_total is not None:
            cls.win_rate_total.set(rate_total)

    @classmethod
    def record_trade(
        cls,
        asset: str,
        direction: str,
        engine: str,
        outcome: str = 'pending'
    ):
        """Record a new trade."""
        cls.trades_total.labels(
            asset=asset,
            direction=direction,
            engine=engine,
            outcome=outcome
        ).inc()

    @classmethod
    def set_price(cls, asset: str, price: float, change_24h: float = None):
        """Set asset price metrics."""
        cls.asset_price.labels(asset=asset).set(price)
        if change_24h is not None:
            cls.price_change_24h.labels(asset=asset).set(change_24h)

    @classmethod
    def set_engine_stats(
        cls,
        engine: str,
        rank: int,
        weight: float,
        points: int,
        win_rate: float,
        active: bool = True
    ):
        """Set engine tournament stats."""
        cls.engine_rank.labels(engine=engine).set(rank)
        cls.engine_weight.labels(engine=engine).set(weight)
        cls.engine_points.labels(engine=engine).set(points)
        cls.engine_win_rate.labels(engine=engine).set(win_rate)
        cls.engine_active.labels(engine=engine).set(1 if active else 0)

    # Alias for tournament_tracker compatibility
    set_engine_tournament_stats = set_engine_stats

    @classmethod
    def set_risk_metrics(
        cls,
        daily_dd: float,
        total_dd: float,
        kill_switch: bool,
        exposure: float
    ):
        """Set risk and safety metrics."""
        cls.daily_drawdown.set(daily_dd)
        cls.total_drawdown.set(total_dd)
        cls.kill_switch_active.set(1 if kill_switch else 0)
        cls.risk_exposure.set(exposure)

    @classmethod
    def record_api_call(cls, api: str, endpoint: str, duration: float):
        """Record API call latency."""
        cls.api_latency.labels(api=api, endpoint=endpoint).observe(duration)

    @classmethod
    def record_error(cls, error_type: str, component: str):
        """Record an error occurrence."""
        cls.errors_total.labels(type=error_type, component=component).inc()

    # ========== Bug Detection Convenience Methods ==========

    @classmethod
    def record_engine_error(cls, engine: str):
        """Record an engine error for bug detection."""
        cls.engine_errors_total.labels(engine=engine).inc()

    @classmethod
    def record_low_confidence(cls, engine: str):
        """Record a low confidence strategy generation."""
        cls.engine_low_confidence_total.labels(engine=engine).inc()

    @classmethod
    def record_data_feed_error(cls, source: str):
        """Record a data feed error."""
        cls.data_feed_errors_total.labels(source=source).inc()

    @classmethod
    def record_duplicate_blocked(cls):
        """Record a blocked duplicate order."""
        cls.duplicate_orders_blocked_total.inc()

    @classmethod
    def update_checkpoint_timestamp(cls):
        """Update the last checkpoint timestamp."""
        import time
        cls.last_checkpoint_timestamp.set(time.time())

    @classmethod
    def record_iteration(cls):
        """Record a completed trading iteration."""
        cls.iterations_total.inc()

    @classmethod
    def record_trade_closed(cls, outcome: str):
        """Record a closed trade with its outcome (win/loss)."""
        cls.trades_closed_total.labels(outcome=outcome).inc()

    @classmethod
    def record_cycle(cls, duration: float):
        """Record cycle completion."""
        cls.cycle_duration.observe(duration)
        cls.last_cycle_timestamp.set(time.time())
        cls.update_uptime()

    # ========== Phase 1 Convenience Methods ==========

    @classmethod
    def set_asset_stats(
        cls,
        asset: str,
        pnl_percent: float,
        win_rate: float,
        trade_count: int
    ):
        """Set per-asset performance metrics."""
        cls.asset_pnl.labels(asset=asset).set(pnl_percent)
        cls.asset_win_rate.labels(asset=asset).set(win_rate)
        cls.asset_trade_count.labels(asset=asset).set(trade_count)

    @classmethod
    def set_regime_info(
        cls,
        asset: str,
        regime: str,
        confidence: float,
        adx: float,
        atr: float,
        bb_width: float
    ):
        """Set regime and technical indicator metrics."""
        # Convert regime string to numeric
        regime_map = {
            'CHOPPY': 0,
            'RANGING': 1,
            'TRENDING_UP': 2,
            'TRENDING_DOWN': 3,
            'VOLATILE': 4
        }
        regime_num = regime_map.get(regime, 0)

        cls.regime_current.labels(asset=asset).set(regime_num)
        cls.regime_confidence.labels(asset=asset).set(confidence)
        cls.indicator_adx.labels(asset=asset).set(adx)
        cls.indicator_atr.labels(asset=asset).set(atr)
        cls.indicator_bb_width.labels(asset=asset).set(bb_width)

    @classmethod
    def set_guardian_state(
        cls,
        account_bal: float,
        peak_bal: float,
        daily_pnl: float,
        daily_pnl_pct: float,
        circuit_breaker: bool,
        emergency_shutdown: bool,
        trading_allowed: bool,
        position_multiplier: float
    ):
        """Set Guardian full state metrics."""
        cls.account_balance.set(account_bal)
        cls.peak_balance.set(peak_bal)
        cls.daily_pnl_usd.set(daily_pnl)
        cls.daily_pnl_percent.set(daily_pnl_pct)
        cls.circuit_breaker_active.set(1 if circuit_breaker else 0)
        cls.emergency_shutdown.set(1 if emergency_shutdown else 0)
        cls.trading_allowed.set(1 if trading_allowed else 0)
        cls.position_size_multiplier.set(position_multiplier)

    # ========== Phase 2 Convenience Methods ==========

    @classmethod
    def set_regime_stats(
        cls,
        regime: str,
        pnl_percent: float,
        win_rate: float,
        trade_count: int,
        avg_pnl: float
    ):
        """Set per-regime performance metrics."""
        cls.regime_pnl.labels(regime=regime).set(pnl_percent)
        cls.regime_win_rate.labels(regime=regime).set(win_rate)
        cls.regime_trade_count.labels(regime=regime).set(trade_count)
        cls.regime_avg_pnl.labels(regime=regime).set(avg_pnl)

    # ========== Phase 3 Convenience Methods ==========

    @classmethod
    def set_engine_asset_performance(
        cls,
        engine: str,
        asset: str,
        pnl_percent: float,
        accuracy: float
    ):
        """Set engine performance by asset."""
        cls.engine_asset_pnl.labels(engine=engine, asset=asset).set(pnl_percent)
        cls.engine_asset_accuracy.labels(engine=engine, asset=asset).set(accuracy)

    @classmethod
    def set_engine_votes(
        cls,
        engine: str,
        buy_votes: int,
        sell_votes: int,
        hold_votes: int,
        last_vote: str
    ):
        """Set engine voting statistics."""
        cls.engine_votes.labels(engine=engine, direction='BUY').set(buy_votes)
        cls.engine_votes.labels(engine=engine, direction='SELL').set(sell_votes)
        cls.engine_votes.labels(engine=engine, direction='HOLD').set(hold_votes)

        # Convert last vote to numeric
        vote_map = {'HOLD': 0, 'BUY': 1, 'SELL': -1}
        cls.engine_last_vote.labels(engine=engine).set(vote_map.get(last_vote, 0))

    @classmethod
    def set_engine_agreement(cls, rate: float):
        """Set engine agreement rate."""
        cls.engine_agreement.set(rate)

    # ========== Phase 4 Convenience Methods ==========

    @classmethod
    def set_advanced_stats(
        cls,
        sharpe: float,
        sortino: float,
        max_dd: float,
        calmar: float,
        profit_factor: float,
        expectancy: float,
        avg_rr: float,
        total_trades: int
    ):
        """Set advanced trading statistics."""
        cls.sharpe_ratio.set(sharpe)
        cls.sortino_ratio.set(sortino)
        cls.max_drawdown.set(max_dd)
        cls.calmar_ratio.set(calmar)
        cls.profit_factor.set(profit_factor)
        cls.expectancy.set(expectancy)
        cls.avg_risk_reward.set(avg_rr)
        cls.total_trades.set(total_trades)

    @classmethod
    def update_from_state(cls, state: Dict[str, Any]):
        """
        Bulk update metrics from a state dictionary.

        Expected state structure:
        {
            'pnl': {'total': float, 'daily': float},
            'win_rate': {'24h': float, 'total': float},
            'engines': {
                'A': {'rank': int, 'weight': float, 'points': int, 'win_rate': float},
                ...
            },
            'risk': {
                'daily_drawdown': float,
                'total_drawdown': float,
                'kill_switch': bool,
                'exposure': float
            },
            'prices': {
                'BTC-USD': {'price': float, 'change_24h': float},
                ...
            },
            'consecutive': {'wins': int, 'losses': int}
        }
        """
        # P&L
        if 'pnl' in state:
            cls.set_pnl(
                state['pnl'].get('total', 0),
                state['pnl'].get('daily', 0)
            )

        # Win rate
        if 'win_rate' in state:
            cls.set_win_rate(
                state['win_rate'].get('24h', 0),
                state['win_rate'].get('total', 0)
            )

        # Engine stats
        if 'engines' in state:
            for engine, stats in state['engines'].items():
                cls.set_engine_stats(
                    engine=engine,
                    rank=stats.get('rank', 0),
                    weight=stats.get('weight', 0),
                    points=stats.get('points', 0),
                    win_rate=stats.get('win_rate', 0),
                    active=stats.get('active', True)
                )

        # Risk metrics
        if 'risk' in state:
            cls.set_risk_metrics(
                daily_dd=state['risk'].get('daily_drawdown', 0),
                total_dd=state['risk'].get('total_drawdown', 0),
                kill_switch=state['risk'].get('kill_switch', False),
                exposure=state['risk'].get('exposure', 0)
            )

        # Prices
        if 'prices' in state:
            for asset, data in state['prices'].items():
                cls.set_price(
                    asset=asset,
                    price=data.get('price', 0),
                    change_24h=data.get('change_24h', 0)
                )

        # Consecutive
        if 'consecutive' in state:
            cls.consecutive_wins.set(state['consecutive'].get('wins', 0))
            cls.consecutive_losses.set(state['consecutive'].get('losses', 0))

        # Update uptime
        cls.update_uptime()

    # ========== 4-Engine Generation Methods ==========

    @classmethod
    def set_generation_stats(
        cls,
        total_strategies: int,
        by_engine: Dict[str, int],
        winning_engine: str,
        vote_breakdown: Dict[str, int],
        duration_ms: int,
        is_mock: bool = False
    ):
        """
        Set 4-engine strategy generation metrics.

        Args:
            total_strategies: Total strategies generated in cycle
            by_engine: Dict of engine -> strategy count
            winning_engine: Engine that won (A, B, C, D)
            vote_breakdown: Dict of engine -> vote count from top 100
            duration_ms: Generation cycle duration in milliseconds
            is_mock: Whether mock generation was used
        """
        import time

        cls.generation_strategies_total.set(total_strategies)

        # Per-engine stats
        for engine in ['A', 'B', 'C', 'D']:
            cls.generation_by_engine.labels(engine=engine).set(by_engine.get(engine, 0))
            cls.generation_vote_breakdown.labels(engine=engine).set(vote_breakdown.get(engine, 0))

        # Winning engine (1=A, 2=B, 3=C, 4=D)
        engine_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
        cls.generation_winning_engine.set(engine_map.get(winning_engine, 0))

        # Timing
        cls.generation_last_run.set(time.time())
        cls.generation_duration_ms.set(duration_ms)

        # Mode
        cls.generation_mode.set(1 if is_mock else 0)

        # Calculate next run (midnight UTC)
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        next_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        cls.generation_next_run.set(next_midnight.timestamp())

    # ========== Independent Trading Mode Methods ==========

    @classmethod
    def set_independent_mode(cls, active: bool):
        """Set whether independent trading mode is active."""
        cls.independent_mode_active.set(1 if active else 0)

    @classmethod
    def set_specialty_triggers(cls, triggers: Dict[str, bool]):
        """
        Set specialty trigger status for each engine.

        Args:
            triggers: Dict mapping engine name (A,B,C,D) to trigger status (True/False)
        """
        for engine in ['A', 'B', 'C', 'D']:
            cls.specialty_trigger.labels(engine=engine).set(
                1 if triggers.get(engine, False) else 0
            )

    @classmethod
    def set_engine_portfolio(
        cls,
        engine: str,
        balance: float,
        trades: int,
        wins: int,
        pnl: float
    ):
        """Set engine portfolio metrics for independent trading mode."""
        cls.engine_portfolio_balance.labels(engine=engine).set(balance)
        cls.engine_portfolio_trades.labels(engine=engine).set(trades)
        cls.engine_portfolio_wins.labels(engine=engine).set(wins)
        cls.engine_portfolio_pnl.labels(engine=engine).set(pnl)

    @classmethod
    def set_all_engine_portfolios(cls, portfolios: Dict[str, Dict[str, Any]]):
        """
        Set all engine portfolios at once.

        Args:
            portfolios: Dict mapping engine name to portfolio dict with keys:
                       balance, trades, wins, pnl
        """
        for engine, portfolio in portfolios.items():
            cls.set_engine_portfolio(
                engine=engine,
                balance=portfolio.get('balance', 3750.0),  # FTMO $15K / 4 engines
                trades=portfolio.get('trades', 0),
                wins=portfolio.get('wins', 0),
                pnl=portfolio.get('pnl', 0.0)
            )

    @classmethod
    def set_engine_specialties(cls):
        """Set engine specialty info."""
        cls.engine_specialty.info({
            'engine_a': 'liquidation_cascade',
            'engine_b': 'funding_extreme',
            'engine_c': 'orderbook_imbalance',
            'engine_d': 'regime_transition'
        })

    # ========== Turbo Batch Generation Methods ==========

    @classmethod
    def set_turbo_mode(cls, active: bool):
        """Set whether turbo batch generation mode is active."""
        cls.turbo_batch_active.set(1 if active else 0)

    @classmethod
    def set_turbo_generation_stats(
        cls,
        strategies_by_specialty: Dict[str, int],
        backtested_count: int,
        generation_time_ms: int
    ):
        """
        Set turbo generation statistics.

        Args:
            strategies_by_specialty: Dict mapping specialty name to strategy count
            backtested_count: Total strategies backtested
            generation_time_ms: Time taken for generation in milliseconds
        """
        specialty_map = {
            'liquidation_cascade': 'liquidation',
            'funding_extreme': 'funding',
            'orderbook_imbalance': 'orderbook',
            'regime_transition': 'regime'
        }

        for specialty, count in strategies_by_specialty.items():
            label = specialty_map.get(specialty, specialty)
            cls.turbo_strategies_per_specialty.labels(specialty=label).set(count)

        cls.turbo_backtested_count.set(backtested_count)
        cls.turbo_generation_time_ms.set(generation_time_ms)

    @classmethod
    def set_turbo_tournament_result(
        cls,
        top_sharpe: float,
        top_return: float,
        best_specialty: str,
        tournament_time_ms: int
    ):
        """
        Set turbo tournament results.

        Args:
            top_sharpe: Sharpe ratio of winning strategy
            top_return: Return percentage of winning strategy
            best_specialty: Specialty that produced winning strategy
            tournament_time_ms: Time taken for tournament in milliseconds
        """
        cls.turbo_top_sharpe.set(top_sharpe)
        cls.turbo_top_return.set(top_return)
        cls.turbo_tournament_time_ms.set(tournament_time_ms)

        # Convert specialty to numeric
        specialty_map = {
            'liquidation_cascade': 1,
            'funding_extreme': 2,
            'orderbook_imbalance': 3,
            'regime_transition': 4
        }
        cls.turbo_best_specialty.set(specialty_map.get(best_specialty, 0))

    @classmethod
    def initialize_hydra_40(cls):
        """Initialize HYDRA 4.0 specific metrics."""
        # Initialize independent mode metrics
        cls.independent_mode_active.set(0)
        # FTMO $15K challenge: $3,750 per engine ($15K / 4)
        for engine in ['A', 'B', 'C', 'D']:
            cls.specialty_trigger.labels(engine=engine).set(0)
            cls.engine_portfolio_balance.labels(engine=engine).set(3750.0)
            cls.engine_portfolio_trades.labels(engine=engine).set(0)
            cls.engine_portfolio_wins.labels(engine=engine).set(0)
            cls.engine_portfolio_pnl.labels(engine=engine).set(0.0)

        # Set engine specialties info
        cls.set_engine_specialties()

        # Initialize turbo mode metrics
        cls.turbo_batch_active.set(0)
        for specialty in ['liquidation', 'funding', 'orderbook', 'regime']:
            cls.turbo_strategies_per_specialty.labels(specialty=specialty).set(0)
        cls.turbo_backtested_count.set(0)
        cls.turbo_top_sharpe.set(0.0)
        cls.turbo_top_return.set(0.0)
        cls.turbo_generation_time_ms.set(0)
        cls.turbo_tournament_time_ms.set(0)
        cls.turbo_best_specialty.set(0)

        # Initialize FTMO multi-bot metrics
        cls.initialize_ftmo_bots()

    # ========== FTMO Multi-Bot Methods ==========

    # ========== FTMO Metalearning (L1/L2) Metrics ==========

    # Metalearning active flag
    ftmo_metalearning_active = Gauge(
        'ftmo_metalearning_active',
        'Whether metalearning is active (1) or disabled (0)'
    )

    # L1: Adaptive Position Sizing metrics
    ftmo_l1_position_multiplier = Gauge(
        'ftmo_l1_position_multiplier',
        'Current L1 adaptive position sizing multiplier (0.5-2.0)',
        ['bot']
    )

    ftmo_l1_risk_percent = Gauge(
        'ftmo_l1_risk_percent',
        'Current L1 risk percentage per trade',
        ['bot']
    )

    ftmo_l1_streak = Gauge(
        'ftmo_l1_streak',
        'Current win/loss streak (positive=wins, negative=losses)',
        ['bot']
    )

    ftmo_l1_kelly_fraction = Gauge(
        'ftmo_l1_kelly_fraction',
        'Current Kelly criterion fraction',
        ['bot']
    )

    # L2: Volatility Regime Detection metrics
    ftmo_l2_volatility_regime = Gauge(
        'ftmo_l2_volatility_regime',
        'Current volatility regime: 0=LOW, 1=MEDIUM, 2=HIGH, 3=EXTREME',
        ['symbol']
    )

    ftmo_l2_atr_multiplier = Gauge(
        'ftmo_l2_atr_multiplier',
        'Current ATR multiplier for stop-loss/take-profit',
        ['symbol']
    )

    ftmo_l2_regime_confidence = Gauge(
        'ftmo_l2_regime_confidence',
        'Confidence in current volatility regime detection (0-1)',
        ['symbol']
    )

    # ========== FTMO Turbo Mode Metrics ==========

    ftmo_turbo_mode_active = Gauge(
        'ftmo_turbo_mode_active',
        'Whether FTMO turbo mode is active (1) or disabled (0)'
    )

    ftmo_turbo_max_trades = Gauge(
        'ftmo_turbo_max_trades',
        'Maximum trades per day in turbo mode'
    )

    ftmo_turbo_threshold_multiplier = Gauge(
        'ftmo_turbo_threshold_multiplier',
        'Signal threshold multiplier in turbo mode (lower = more signals)'
    )

    @classmethod
    def initialize_ftmo_bots(cls):
        """Initialize FTMO multi-bot metrics."""
        ftmo_bots = [
            'GoldLondonReversal',
            'EURUSDBreakout',
            'US30ORB',
            'NAS100Gap',
            'GoldNYReversion',
            'EngineD_ATR',
            'HFScalper'  # Added HF Scalper bot
        ]

        for bot in ftmo_bots:
            cls.ftmo_bot_active.labels(bot=bot).set(0)
            cls.ftmo_bot_trades_today.labels(bot=bot).set(0)
            cls.ftmo_bot_win_rate.labels(bot=bot).set(0)
            cls.ftmo_bot_pnl_today.labels(bot=bot).set(0)
            cls.ftmo_bot_pnl_total.labels(bot=bot).set(0)
            cls.ftmo_bot_last_trade.labels(bot=bot).set(0)

        cls.ftmo_open_positions.set(0)
        cls.ftmo_max_positions.set(3)
        cls.ftmo_daily_drawdown.set(0)
        cls.ftmo_total_drawdown.set(0)
        cls.ftmo_kill_switch.set(0)
        cls.ftmo_account_balance.set(15000)
        cls.ftmo_paper_mode.set(1)
        cls.ftmo_daily_target_progress.set(0)

        # Initialize L1/L2 metalearning metrics
        cls.ftmo_metalearning_active.set(0)
        for bot in ftmo_bots:
            cls.ftmo_l1_position_multiplier.labels(bot=bot).set(1.0)
            cls.ftmo_l1_risk_percent.labels(bot=bot).set(1.5)
            cls.ftmo_l1_streak.labels(bot=bot).set(0)
            cls.ftmo_l1_kelly_fraction.labels(bot=bot).set(0.02)

        # Initialize L2 volatility regime metrics
        for symbol in ['XAUUSD', 'EURUSD', 'US30', 'NAS100', 'GBPUSD']:
            cls.ftmo_l2_volatility_regime.labels(symbol=symbol).set(1)  # MEDIUM
            cls.ftmo_l2_atr_multiplier.labels(symbol=symbol).set(1.0)
            cls.ftmo_l2_regime_confidence.labels(symbol=symbol).set(0.5)

        # Initialize turbo mode metrics
        cls.ftmo_turbo_mode_active.set(0)
        cls.ftmo_turbo_max_trades.set(9)  # 3x normal max
        cls.ftmo_turbo_threshold_multiplier.set(0.5)  # 50% thresholds

    @classmethod
    def set_ftmo_bot_status(
        cls,
        bot: str,
        active: bool,
        trades_today: int = 0,
        win_rate: float = 0,
        pnl_today: float = 0,
        pnl_total: float = 0
    ):
        """Set status for a single FTMO bot."""
        cls.ftmo_bot_active.labels(bot=bot).set(1 if active else 0)
        cls.ftmo_bot_trades_today.labels(bot=bot).set(trades_today)
        cls.ftmo_bot_win_rate.labels(bot=bot).set(win_rate)
        cls.ftmo_bot_pnl_today.labels(bot=bot).set(pnl_today)
        cls.ftmo_bot_pnl_total.labels(bot=bot).set(pnl_total)

    @classmethod
    def set_ftmo_orchestrator_status(
        cls,
        open_positions: int,
        max_positions: int,
        daily_drawdown: float,
        total_drawdown: float,
        kill_switch: bool,
        account_balance: float,
        paper_mode: bool
    ):
        """Set FTMO orchestrator status."""
        cls.ftmo_open_positions.set(open_positions)
        cls.ftmo_max_positions.set(max_positions)
        cls.ftmo_daily_drawdown.set(daily_drawdown)
        cls.ftmo_total_drawdown.set(total_drawdown)
        cls.ftmo_kill_switch.set(1 if kill_switch else 0)
        cls.ftmo_account_balance.set(account_balance)
        cls.ftmo_paper_mode.set(1 if paper_mode else 0)

    @classmethod
    def record_ftmo_signal(cls, bot: str, direction: str):
        """Record a new FTMO signal."""
        cls.ftmo_signals_total.labels(bot=bot, direction=direction).inc()
        cls.ftmo_bot_last_trade.labels(bot=bot).set(time.time())

    @classmethod
    def record_ftmo_trade_outcome(cls, bot: str, outcome: str):
        """Record FTMO trade outcome (win/loss)."""
        cls.ftmo_trade_outcomes.labels(bot=bot, outcome=outcome).inc()

    @classmethod
    def set_ftmo_daily_progress(cls, current_pnl: float, target: float = 300):
        """Set daily target progress percentage."""
        if target > 0:
            progress = min(100, (current_pnl / target) * 100)
            cls.ftmo_daily_target_progress.set(progress)

    @classmethod
    def update_ftmo_from_orchestrator(cls, orchestrator_status: Dict[str, Any]):
        """
        Update all FTMO metrics from orchestrator status.

        Expected status structure:
        {
            'running': bool,
            'paper_mode': bool,
            'kill_switch': bool,
            'open_positions': int,
            'max_positions': int,
            'daily_stats': {'trades': int, 'starting_balance': float, 'current_balance': float},
            'bots': {
                'gold_london': {'enabled': bool, 'trades_today': int, ...},
                ...
            }
        }
        """
        # Orchestrator-level metrics
        daily_stats = orchestrator_status.get('daily_stats', {})
        starting = daily_stats.get('starting_balance', 15000)
        current = daily_stats.get('current_balance', 15000)

        daily_dd = 0
        if starting > 0:
            daily_dd = max(0, (starting - current) / starting * 100)

        cls.set_ftmo_orchestrator_status(
            open_positions=orchestrator_status.get('open_positions', 0),
            max_positions=orchestrator_status.get('max_positions', 3),
            daily_drawdown=daily_dd,
            total_drawdown=orchestrator_status.get('total_drawdown', 0),
            kill_switch=orchestrator_status.get('kill_switch', False),
            account_balance=current,
            paper_mode=orchestrator_status.get('paper_mode', True)
        )

        # Bot-level metrics
        bots = orchestrator_status.get('bots', {})
        for bot_name, bot_status in bots.items():
            cls.set_ftmo_bot_status(
                bot=bot_status.get('name', bot_name),
                active=bot_status.get('enabled', False),
                trades_today=bot_status.get('trades_today', 0),
                win_rate=bot_status.get('win_rate', 0),
                pnl_today=bot_status.get('pnl_today', 0),
                pnl_total=bot_status.get('pnl_total', 0)
            )

        # Daily progress
        daily_pnl = current - starting
        cls.set_ftmo_daily_progress(daily_pnl)
