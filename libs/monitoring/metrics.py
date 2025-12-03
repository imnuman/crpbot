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

    @classmethod
    def record_cycle(cls, duration: float):
        """Record cycle completion."""
        cls.cycle_duration.observe(duration)
        cls.last_cycle_timestamp.set(time.time())
        cls.update_uptime()

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
