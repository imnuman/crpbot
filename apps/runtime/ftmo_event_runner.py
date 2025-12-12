#!/usr/bin/env python3
"""
FTMO Event-Driven Trading Runner

Real-time, event-driven FTMO trading with:
- ZMQ PUB/SUB streaming from Windows MT5
- Event bus for tick distribution
- Bot wrappers with session-aware triggers
- Prometheus metrics on :9100
- FTMO-compliant risk management

Architecture:
    Windows VPS          Linux Server
    ┌──────────┐        ┌───────────────────────────────────┐
    │ MT5      │        │  Event Bus (ZMQ SUB)              │
    │   │      │  SSH   │       │                           │
    │   v      │ Tunnel │       v                           │
    │ Streamer │───────>│  Bot Wrappers                     │
    │ (PUB)    │        │   - GoldLondon  (XAUUSD, session) │
    │          │        │   - EURUSDBreakout (session)      │
    │ Executor │<───────│   - US30ORB (session)             │
    │ (REQ/REP)│        │   - NAS100Gap (session)           │
    └──────────┘        │   - GoldNY (XAUUSD, session)      │
                        │   - HFScalper (multi, tick)       │
                        │                                   │
                        │  Risk Manager                     │
                        │  Prometheus Metrics :9100         │
                        └───────────────────────────────────┘

Usage:
    # Paper trading (default)
    python ftmo_event_runner.py

    # Live trading (requires explicit flag)
    python ftmo_event_runner.py --live

    # With custom metrics port
    python ftmo_event_runner.py --metrics-port 9101
"""

import os
import sys
import signal
import argparse
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from loguru import logger

# Phase 5: Position Manager for ATR trailing stops
from libs.hydra.position_manager import get_position_manager, ManagementConfig, ActionType
from libs.hydra.ftmo_bots.technical_utils import calculate_atr

# Phase 6: AGI Advisor for pre-trade intelligence
try:
    from libs.hydra.agi_advisor import get_agi_advisor, should_agi_approve, get_agi_size_multiplier
    AGI_AVAILABLE = True
except ImportError:
    AGI_AVAILABLE = False

# Phase 7: Trade Ledger for persistent trade tracking (2025-12-11)
from libs.hydra.ftmo_bots.trade_ledger import TradeLedger, get_ledger
from libs.hydra.ftmo_bots.reconciliation import PositionCloseMonitor, TradeReconciler, backfill_from_mt5
from libs.hydra.ftmo_bots.performance_gate import PerformanceGate, get_performance_gate
from libs.hydra.ftmo_bots.time_filter import check_trading_time, should_trade_now, get_time_size_multiplier

# Phase 8: Counter-Trade Intelligence + Certainty Engine (2025-12-11)
# CTI: Inverts signals when historical win rate < 40% (27% SELL → 73% BUY)
# CE: Blocks trades with < 80% confidence score
try:
    from libs.hydra.counter_trade_intelligence import get_counter_trade_intelligence
    CTI_AVAILABLE = True
except ImportError:
    CTI_AVAILABLE = False
    logger.warning("[Import] Counter-Trade Intelligence not available")

try:
    from libs.hydra.certainty_engine import get_certainty_engine
    CE_AVAILABLE = True
except ImportError:
    CE_AVAILABLE = False
    logger.warning("[Import] Certainty Engine not available")

try:
    from libs.hydra.edge_monitor import EdgeMonitor, EdgeAction, TradeContext
    EDGE_MONITOR_AVAILABLE = True
except ImportError:
    EDGE_MONITOR_AVAILABLE = False
    logger.warning("[Import] Edge Monitor not available")

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "/tmp/ftmo_event_{time:YYYYMMDD}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG"
)


class FTMOEventRunner:
    """
    Main event-driven FTMO trading orchestrator.

    Connects all components:
    - Event bus for price streaming
    - Bot wrappers for each trading strategy
    - Risk manager for FTMO compliance
    - Prometheus metrics for monitoring
    """

    # Symbols tracked by the system (using FTMO broker symbol names)
    ALL_SYMBOLS = ["XAUUSD", "EURUSD", "US30.cash", "US100.cash", "GBPUSD"]

    def __init__(
        self,
        paper_mode: bool = True,
        metrics_port: int = 9100,
        turbo_mode: bool = False
    ):
        self.paper_mode = paper_mode
        self.metrics_port = metrics_port
        self.turbo_mode = turbo_mode

        self._running = False
        self._event_bus = None
        self._bot_wrappers: Dict[str, Any] = {}
        self._zmq_client = None
        self._metrics = None

        # Risk state - TIGHTENED after $503 loss on 2025-12-10
        # Only $366 buffer remaining before challenge fails
        self._kill_switch = False
        self._daily_starting_balance: float = 15000.0
        self._max_daily_loss_percent = 2.0  # Was 4.5% - now 2% for safety
        self._max_total_drawdown_percent = 8.0  # Was 8.5% - trigger earlier

        # Per-bot P&L tracking (added 2025-12-11)
        # Kill individual bots at 1% loss instead of waiting for full system drawdown
        self._bot_daily_pnl: Dict[str, float] = {}  # bot_name -> daily P&L
        self._bot_disabled: Dict[str, bool] = {}  # bot_name -> is_disabled
        self._max_per_bot_loss_percent = 1.0  # 1% max loss per bot per day
        self._last_reset_date: Optional[datetime] = None

        # Signal queue for trade execution
        self._signal_queue: List[Any] = []
        self._signal_lock = threading.Lock()

        # Position time tracking for time-based exits (Phase 3 - 2025-12-11)
        # Maps ticket -> (open_time, bot_name, max_hold_hours)
        self._position_times: Dict[int, tuple] = {}
        self._default_max_hold_hours = 8.0  # Default max hold time

        # Bot-specific max hold times (from BotConfig in each bot)
        self._bot_max_hold_hours = {
            "gold_london": 2.0,
            "GoldLondonReversal": 2.0,
            "eurusd": 4.0,
            "EURUSDBreakout": 4.0,
            "us30": 2.0,
            "US30ORB": 2.0,
            "nas100": 2.0,
            "NAS100Gap": 2.0,
            "gold_ny": 3.0,
            "GoldNYReversion": 3.0,
            "hf_scalper": 1.0,
            "HFScalper": 1.0,
            "london_eur": 4.0,
            "LondonBreakout": 4.0,
        }

        # Weekend gap protection (Phase 4 - 2025-12-11)
        # Close all positions 30 min before Friday market close (21:30 UTC = 16:30 EST)
        self._friday_close_hour = 21  # 21:00 UTC
        self._friday_close_minute = 30  # Close at 21:30 UTC
        self._weekend_closeout_done = False

        # Trade clustering prevention (Phase 4 - 2025-12-11)
        # Prevent rapid-fire signals on same symbol
        self._last_trade_time: Dict[str, datetime] = {}  # symbol -> last trade time
        self._trade_cooldown_minutes = 5  # Minutes between trades on same symbol
        self._max_pending_signals = 3  # Max signals in queue before pause

        # Correlation-based position blocking (Phase 4 - 2025-12-11)
        # Prevent multiple positions in same direction on correlated assets
        self._correlation_groups = {
            "gold": ["XAUUSD"],
            "indices": ["US30.cash", "US100.cash"],
            "majors": ["EURUSD", "GBPUSD"],
        }

        # Conservative Risk Scaling (Phase 5 - 2025-12-11)
        # Expert insight: "Reduce risk 50% when down 5% or more"
        self._peak_balance: float = 15000.0  # Track peak for drawdown calculation
        self._current_balance: float = 15000.0  # Updated from account checks
        # Phase 9 (2025-12-11): Reduced from 1.5% to 0.75% per research
        self._base_risk_percent = 0.0075  # 0.75% risk per trade (was 1.5%)
        self._risk_scale_thresholds = {
            3.0: 0.50,   # At 3% DD: reduce to 50% of base risk (0.75%)
            5.0: 0.33,   # At 5% DD: reduce to 33% of base risk (0.5%)
            7.0: 0.25,   # At 7% DD: reduce to 25% of base risk (0.375%)
        }
        self._current_risk_multiplier = 1.0  # Updated based on drawdown

        # Daily Trade Limits (Phase 5 - 2025-12-11)
        # Expert insight: "Limit to 2-3 A+ trades per day, halt at 5% loss"
        self._daily_trade_count = 0
        self._max_daily_trades = 10  # Hard limit: 10 trades/day across all bots
        self._daily_loss_trade_limit = 3  # Soft limit: after 3% loss, max 3 more trades
        self._halt_trading_at_loss_percent = 5.0  # Halt all trading at 5% daily loss
        self._daily_halt_logged = False  # Track if halt was logged

        # ATR Trailing Stops (Phase 5 - 2025-12-11)
        # Expert insight: "Trail at 1.5x ATR behind price, only after breakeven hit"
        self._atr_values: Dict[str, float] = {}  # symbol -> current ATR
        self._atr_trail_multiplier = 1.5  # Trail distance = ATR * 1.5
        self._last_atr_update: Optional[datetime] = None
        self._atr_update_interval = 300  # Update ATR every 5 minutes
        self._position_manager = None  # Initialized in start()

        # Multi-Timeframe Confirmation (Phase 5 - 2025-12-11)
        # Expert insight: "Start analysis on higher TF, confirm on lower TF"
        self._h1_trends: Dict[str, str] = {}  # symbol -> "BULLISH", "BEARISH", "NEUTRAL"
        self._last_h1_update: Optional[datetime] = None
        self._h1_update_interval = 600  # Update H1 trend every 10 minutes
        self._mtf_enabled = True  # Enable/disable Multi-TF filter

        # AGI Advisor Integration (Phase 6 - 2025-12-11)
        # Uses Claude-AGI learned patterns for pre-trade validation
        self._agi_advisor = None
        self._agi_enabled = True  # Enable/disable AGI advisor
        self._agi_size_scaling = True  # Apply AGI size multipliers

        # Phase 7: Trade Ledger for persistent trade tracking (2025-12-11)
        # Ensures all trades are recorded in SQLite for reconciliation
        self._trade_ledger: Optional[TradeLedger] = None
        self._close_monitor: Optional[PositionCloseMonitor] = None
        self._reconciler: Optional[TradeReconciler] = None
        self._performance_gate: Optional[PerformanceGate] = None

        # Phase 8: Counter-Trade Intelligence + Certainty Engine + Edge Monitor (2025-12-11)
        # CTI: Inverts signals when historical WR < 40% (converts 27% losing to 73% winning)
        # CE: Requires 80% confidence score to take trades
        # EM: Exits trades when edge factors deteriorate
        self._counter_trade = None
        self._certainty_engine = None
        self._edge_monitor = None
        self._cti_enabled = True  # Enable Counter-Trade Intelligence
        self._ce_enabled = True   # Enable Certainty Engine (80% threshold)
        self._ce_threshold = 0.80  # Minimum confidence to trade
        self._edge_monitor_enabled = True  # Enable Edge Monitor for early exits

        # Track CTI fingerprints for outcome recording
        self._trade_fingerprints: Dict[int, str] = {}  # ticket -> CTI fingerprint

        # Phase 9: Losing Streak Detector (2025-12-11)
        # Source: GitHub GOLD_ORB - auto-pause after consecutive losses
        # When 3 consecutive losses hit, enter virtual mode (track but don't trade)
        # Resume when virtual trade would have been profitable
        self._consecutive_losses = 0  # Track consecutive losing trades
        self._virtual_mode = False  # When True, track signals but don't execute
        self._losing_streak_threshold = 3  # Pause after this many losses
        self._virtual_wins_to_resume = 1  # Virtual wins needed to resume live trading
        self._virtual_win_count = 0  # Track wins while in virtual mode

        # Phase 9: Daily Loss Auto-Stop (research-based)
        # FTMO allows 5% daily - we stop at 3% for safety buffer
        self._internal_daily_loss_limit = 3.0  # 3% daily loss = STOP trading

        logger.info(f"[EventRunner] Initialized (paper={paper_mode}, turbo={turbo_mode})")

    def start(self) -> bool:
        """Start the event-driven trading system."""
        print("=" * 70)
        print("    FTMO Event-Driven Trading System")
        print("=" * 70)
        print()
        print(f"  Mode: {'PAPER TRADING' if self.paper_mode else 'LIVE TRADING'}")
        print(f"  Metrics Port: {self.metrics_port}")
        print(f"  Turbo Mode: {'ENABLED' if self.turbo_mode else 'DISABLED'}")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Step 1: Initialize Prometheus metrics
        print("[1/5] Initializing Prometheus metrics...")
        if not self._init_metrics():
            return False

        # Step 2: Initialize ZMQ client for trade execution
        print("[2/5] Connecting to MT5 executor...")
        if not self._init_zmq_client():
            return False

        # Step 3: Initialize bots
        print("[3/5] Initializing trading bots...")
        if not self._init_bots():
            return False

        # Step 4: Initialize event bus
        print("[4/5] Starting event bus...")
        if not self._init_event_bus():
            return False

        # Step 5: Subscribe bots to price events
        print("[5/5] Subscribing bots to price streams...")
        self._subscribe_bots()

        # Phase 5: Initialize Position Manager with ATR-based trailing
        print("[6/7] Initializing Position Manager for ATR trailing...")
        self._init_position_manager()

        # Phase 6: Initialize AGI Advisor for pre-trade intelligence
        print("[7/8] Initializing AGI Advisor...")
        self._init_agi_advisor()

        # Phase 7: Initialize Trade Ledger for persistent tracking (2025-12-11)
        print("[8/9] Initializing Trade Ledger...")
        self._init_trade_ledger()

        # Phase 8: Initialize Profit Improvement Systems (2025-12-11)
        print("[9/9] Initializing Profit Systems (CTI, CE, EdgeMonitor)...")
        self._init_profit_systems()

        self._running = True

        # Start background threads
        self._start_background_threads()

        print()
        print("=" * 70)
        print("  System Active - Real-time event-driven trading")
        print("  Press Ctrl+C to stop")
        print("=" * 70)
        print()

        return True

    def _init_metrics(self) -> bool:
        """Initialize Prometheus metrics."""
        try:
            from libs.hydra.ftmo_bots.metrics import get_ftmo_metrics

            self._metrics = get_ftmo_metrics()
            self._metrics.start_server(self.metrics_port)
            self._metrics.set_system_info(
                version="2.0-event-driven",
                mode="paper" if self.paper_mode else "live",
                bots=",".join(self._get_bot_names())
            )

            print(f"  Prometheus metrics: http://localhost:{self.metrics_port}/metrics")
            return True

        except Exception as e:
            logger.error(f"[EventRunner] Failed to init metrics: {e}")
            print(f"  Warning: Metrics unavailable ({e})")
            return True  # Non-fatal

    def _init_zmq_client(self) -> bool:
        """Initialize ZMQ client for MT5 trade execution."""
        try:
            from libs.brokers.mt5_zmq_client import MT5ZMQClient, get_mt5_client

            self._zmq_client = get_mt5_client()

            if not self._zmq_client.connect():
                logger.error("[EventRunner] Failed to connect to MT5 executor")
                print("  Failed to connect to MT5 ZMQ executor!")
                return False

            # Test connection
            result = self._zmq_client.ping()
            if not result.get("success"):
                print("  MT5 ping failed!")
                return False

            # Get account info
            account = self._zmq_client.get_account()
            if account:
                self._daily_starting_balance = account.get("balance", 15000)
                print(f"  MT5 Connected - Balance: ${self._daily_starting_balance:,.2f}")

                if self._metrics:
                    self._metrics.update_account(
                        balance=account.get("balance", 0),
                        equity=account.get("equity", 0),
                        positions=len(self._zmq_client.get_positions()),
                        floating_pnl=account.get("profit", 0)
                    )
                    self._metrics.set_connection_status("mt5_executor", True)

            return True

        except Exception as e:
            logger.error(f"[EventRunner] ZMQ client error: {e}")
            print(f"  ZMQ Error: {e}")
            return False

    def _init_bots(self) -> bool:
        """Initialize all trading bots with event wrappers."""
        try:
            from libs.hydra.ftmo_bots import (
                get_gold_london_bot,
                get_eurusd_bot,
                get_us30_bot,
                get_nas100_bot,
                get_gold_ny_bot,
                get_hf_scalper,
                get_london_breakout_bot
            )
            from libs.hydra.ftmo_bots.event_bot_wrapper import (
                EventBotWrapper,
                MultiSymbolBotWrapper
            )

            # Create bots with event wrappers
            # PERFORMANCE-BASED BOT CONFIGURATION (2025-12-11)
            # Based on 46-trade backfill analysis:
            # - EURUSD: 0% win rate, -$941 -> DISABLED (all EURUSD bots forced to paper)
            # - XAUUSD: 20% WR, -$239 -> ENABLED with reduced size
            # - US30/NAS100: Limited data -> PAPER MODE
            bots_config = [
                ("gold_london", get_gold_london_bot(self.paper_mode, turbo_mode=self.turbo_mode), "XAUUSD"),
                # EURUSD DISABLED: 0% win rate across 6 trades, -$941 loss
                ("eurusd", get_eurusd_bot(paper_mode=True), "EURUSD"),  # FORCED PAPER: 0% WR
                ("us30", get_us30_bot(self.paper_mode), "US30.cash"),
                ("gold_ny", get_gold_ny_bot(self.paper_mode), "XAUUSD"),  # RE-ENABLED: $8.79/trade expectancy
                # PAPER MODE TESTING (2025-12-11):
                # - London Breakout v3: Bug fixes for SL placement + SMA filter
                # - NAS100 Gap v2: Stricter 0.6% threshold + ADR filter
                ("london_eur", get_london_breakout_bot("EURUSD", paper_mode=True), "EURUSD"),  # PAPER: 0% WR on EURUSD
                ("nas100", get_nas100_bot(paper_mode=True), "US100.cash"),  # PAPER: v2 improvements
            ]
            print("    - eurusd: PAPER MODE (0% WR, -$941 loss)")
            print("    - london_eur: PAPER MODE (0% WR on EURUSD)")
            print("    - nas100: PAPER MODE (v2 improvements)")

            for name, bot, symbol in bots_config:
                wrapper = EventBotWrapper(
                    bot=bot,
                    on_signal=self._handle_signal
                )
                self._bot_wrappers[name] = {
                    "wrapper": wrapper,
                    "symbol": symbol,
                    "multi": False
                }
                print(f"    - {name}: {symbol}")

            # HF Scalper - ENABLED IN PAPER MODE FOR TESTING (2025-12-11)
            # Lost $503 on 2025-12-10 due to over-trading without track record
            # Now running in PAPER MODE to validate strategy before live
            hf_bot = get_hf_scalper(paper_mode=True, turbo_mode=self.turbo_mode)  # FORCE paper mode
            hf_symbols = ["XAUUSD", "EURUSD"]  # Start with just 2 symbols for testing
            hf_wrapper = MultiSymbolBotWrapper(
                bot=hf_bot,
                symbols=hf_symbols,
                on_signal=self._handle_signal
            )
            self._bot_wrappers["hf_scalper"] = {
                "wrapper": hf_wrapper,
                "symbol": hf_symbols,
                "multi": True
            }
            print(f"    - hf_scalper: PAPER MODE (testing only)")

            print(f"  {len(self._bot_wrappers)} bots initialized")
            return True

        except Exception as e:
            logger.error(f"[EventRunner] Failed to init bots: {e}")
            print(f"  Bot init error: {e}")
            return False

    def _init_position_manager(self) -> bool:
        """Initialize Position Manager for ATR trailing stops (Phase 5)."""
        try:
            # Configure for ATR-based trailing
            config = ManagementConfig(
                # Breakeven: trigger at +0.5%, lock 2 pips profit
                breakeven_enabled=True,
                breakeven_trigger_percent=0.005,
                breakeven_buffer_pips=2.0,

                # Trailing: start at 1.5R, trail at 0.3% (will be overridden by ATR)
                trail_enabled=True,
                trail_trigger_r=1.5,  # Start trailing only after 1.5R profit
                trail_distance_percent=0.003,  # 0.3% (fallback if no ATR)
                trail_step_pips=5.0,

                # Partial profit: take 25% at 1R and 2R
                partial_enabled=True,
                partial_levels=[(1.0, 0.25), (2.0, 0.25)],

                # Pyramid disabled (too risky during challenge)
                pyramid_enabled=False,

                # Time exit: handled separately in existing code
                time_exit_enabled=False,
            )

            self._position_manager = get_position_manager(
                config=config,
                data_dir="/app/data/hydra" if os.path.exists("/app") else "data/hydra"
            )

            print(f"  Position Manager: BE=1.5R trigger, Trail=1.5x ATR")
            logger.info("[EventRunner] Position Manager initialized for ATR trailing")
            return True

        except Exception as e:
            logger.warning(f"[EventRunner] Position Manager init failed: {e}")
            print(f"  Warning: Position Manager unavailable ({e})")
            return True  # Non-fatal

    def _init_agi_advisor(self) -> bool:
        """Initialize AGI Advisor for pre-trade intelligence (Phase 6)."""
        try:
            if not AGI_AVAILABLE:
                print("  AGI Advisor: Not available (module not found)")
                logger.info("[EventRunner] AGI Advisor not available - running without AI guidance")
                return True

            self._agi_advisor = get_agi_advisor()

            if self._agi_advisor.is_active():
                print(f"  AGI Advisor: ACTIVE (recommendations loaded)")
                print(f"    Market: {self._agi_advisor.get_overall_market()}")
                alerts = self._agi_advisor.get_alerts()
                if alerts:
                    for alert in alerts[:2]:  # Show first 2 alerts
                        print(f"    Alert: {alert}")
            else:
                print("  AGI Advisor: STANDBY (no recommendations yet)")

            logger.info(f"[EventRunner] AGI Advisor initialized (active={self._agi_advisor.is_active()})")
            return True

        except Exception as e:
            logger.warning(f"[EventRunner] AGI Advisor init failed: {e}")
            print(f"  Warning: AGI Advisor unavailable ({e})")
            self._agi_enabled = False
            return True  # Non-fatal

    def _init_trade_ledger(self) -> bool:
        """Initialize Trade Ledger for persistent trade tracking (Phase 7 - 2025-12-11)."""
        try:
            from pathlib import Path

            # Use same data directory as other FTMO files
            db_path = Path("/app/data/hydra/ftmo/trade_ledger.db")
            self._trade_ledger = get_ledger(db_path)

            # Get stats for display
            stats = self._trade_ledger.get_stats(days=7)
            open_trades = len(self._trade_ledger.get_open_trades())

            print(f"  Trade Ledger: ACTIVE (DB at {db_path})")
            print(f"    Last 7 days: {stats['total_trades']} trades, ${stats['total_pnl']:.2f} P&L")
            print(f"    Open positions in ledger: {open_trades}")

            logger.info(f"[TradeLedger] Initialized: {stats['total_trades']} trades in last 7 days")

            # Start position close monitor (detects when MT5 positions close)
            if self._zmq_client and self._trade_ledger:
                self._close_monitor = PositionCloseMonitor(
                    ledger=self._trade_ledger,
                    zmq_client=self._zmq_client
                )
                self._close_monitor.start()
                print(f"  Close Monitor: ACTIVE (30s polling)")

            # Initialize performance gate (auto-disables underperforming bots)
            if self._trade_ledger:
                self._performance_gate = get_performance_gate(self._trade_ledger)
                disabled = self._performance_gate.get_disabled_bots()
                if disabled:
                    print(f"  Performance Gate: {len(disabled)} bots auto-disabled")
                    for bot in disabled:
                        print(f"    - {bot}: BLOCKED (poor performance)")
                else:
                    print(f"  Performance Gate: ACTIVE (no bots blocked)")
                logger.info(f"[PerformanceGate] Initialized, disabled={disabled}")

            return True

        except Exception as e:
            logger.warning(f"[EventRunner] Trade Ledger init failed: {e}")
            print(f"  Warning: Trade Ledger unavailable ({e})")
            return True  # Non-fatal, but trade tracking won't work

    def _init_profit_systems(self) -> bool:
        """
        Initialize Phase 8 profit improvement systems (2025-12-11):
        - Counter-Trade Intelligence: Inverts signals with < 40% historical WR
        - Certainty Engine: Requires 80% confidence score
        - Edge Monitor: Exits trades when edge deteriorates
        """
        try:
            systems_active = []

            # Initialize Counter-Trade Intelligence
            if CTI_AVAILABLE and self._cti_enabled:
                self._counter_trade = get_counter_trade_intelligence()
                conditions_tracked = len(self._counter_trade._condition_stats)
                systems_active.append(f"CTI ({conditions_tracked} conditions)")
                logger.info(f"[CTI] Initialized with {conditions_tracked} historical conditions")
            else:
                print(f"  Counter-Trade Intelligence: DISABLED")

            # Initialize Certainty Engine
            if CE_AVAILABLE and self._ce_enabled:
                self._certainty_engine = get_certainty_engine()
                systems_active.append(f"CE ({self._ce_threshold:.0%} threshold)")
                logger.info(f"[CertaintyEngine] Initialized with {self._ce_threshold:.0%} threshold")
            else:
                print(f"  Certainty Engine: DISABLED")

            # Initialize Edge Monitor
            if EDGE_MONITOR_AVAILABLE and self._edge_monitor_enabled:
                self._edge_monitor = EdgeMonitor()
                systems_active.append("EdgeMonitor")
                logger.info(f"[EdgeMonitor] Initialized for early exit on edge loss")
            else:
                print(f"  Edge Monitor: DISABLED")

            if systems_active:
                print(f"  Profit Systems: {', '.join(systems_active)}")
            else:
                print(f"  Profit Systems: NONE ACTIVE (check imports)")

            return True

        except Exception as e:
            logger.warning(f"[EventRunner] Profit systems init failed: {e}")
            print(f"  Warning: Profit systems unavailable ({e})")
            return True  # Non-fatal

    def _init_event_bus(self) -> bool:
        """Initialize the event bus for price streaming."""
        try:
            from libs.hydra.ftmo_bots.event_bus import FTMOEventBus

            # Use direct WireGuard connection (no SSH tunnel needed)
            use_tunnel = os.getenv("USE_SSH_TUNNEL", "false").lower() == "true"
            self._event_bus = FTMOEventBus(use_ssh_tunnel=use_tunnel)

            # Subscribe to heartbeats for monitoring
            self._event_bus.subscribe_heartbeat(self._handle_heartbeat)

            if not self._event_bus.start():
                logger.error("[EventRunner] Failed to start event bus")
                print("  Event bus failed to start!")
                return False

            if self._metrics:
                self._metrics.set_connection_status("event_bus", True)

            print("  Event bus connected to price stream")
            return True

        except Exception as e:
            logger.error(f"[EventRunner] Event bus error: {e}")
            print(f"  Event bus error: {e}")
            return False

    def _subscribe_bots(self):
        """Subscribe bot wrappers to their respective price streams."""
        for name, config in self._bot_wrappers.items():
            wrapper = config["wrapper"]
            symbols = config["symbol"]

            if config["multi"]:
                # Multi-symbol bot subscribes to all its symbols
                for symbol in symbols:
                    self._event_bus.subscribe_tick(symbol, wrapper.on_tick)
                    logger.info(f"[EventRunner] {name} subscribed to {symbol}")
            else:
                # Single-symbol bot
                self._event_bus.subscribe_tick(symbols, wrapper.on_tick)
                logger.info(f"[EventRunner] {name} subscribed to {symbols}")

        print(f"  Subscribed {len(self._bot_wrappers)} bots to price streams")

    def _start_background_threads(self):
        """Start background monitoring threads."""
        # Risk monitoring thread
        threading.Thread(
            target=self._risk_monitor_loop,
            daemon=True,
            name="risk_monitor"
        ).start()

        # Trade execution thread
        threading.Thread(
            target=self._trade_executor_loop,
            daemon=True,
            name="trade_executor"
        ).start()

        # Status reporting thread
        threading.Thread(
            target=self._status_reporter_loop,
            daemon=True,
            name="status_reporter"
        ).start()

    def _handle_signal(self, signal):
        """Handle trading signal from a bot."""
        if self._kill_switch:
            logger.warning(f"[EventRunner] Signal blocked - kill switch active: {signal}")
            return

        # Phase 9: Losing Streak Detector - enter virtual mode after 3 losses
        if self._virtual_mode:
            logger.info(
                f"[LosingStreak] VIRTUAL MODE: {signal.bot_name} {signal.direction} {signal.symbol} "
                f"(tracking only, {self._consecutive_losses} consecutive losses)"
            )
            # Record as virtual trade for later analysis
            self._record_virtual_trade(signal)
            return

        # Phase 9: Daily Loss Auto-Stop at 3% (FTMO allows 5%)
        if self._current_balance < self._daily_starting_balance:
            daily_loss_pct = (self._daily_starting_balance - self._current_balance) / self._daily_starting_balance * 100
            if daily_loss_pct >= self._internal_daily_loss_limit:
                logger.warning(
                    f"[DailyLoss] Signal BLOCKED: daily loss {daily_loss_pct:.2f}% >= "
                    f"{self._internal_daily_loss_limit}% internal limit"
                )
                return

        # Time filter check - block signals during historically bad hours
        time_allowed, time_multiplier, time_reason = check_trading_time()
        if not time_allowed:
            logger.warning(f"[TimeFilter] Signal BLOCKED: {time_reason}")
            return

        # Performance gate check - block signals from underperforming bots
        if self._performance_gate:
            allowed, reason = self._performance_gate.is_bot_allowed(signal.bot_name)
            if not allowed:
                logger.warning(f"[PerformanceGate] Signal BLOCKED from {signal.bot_name}: {reason}")
                return

        # Phase 8: Counter-Trade Intelligence - invert signals with < 40% historical WR
        # Example: SELL has 27% WR → invert to BUY → 73% WR
        cti_fingerprint = None
        if self._counter_trade and self._cti_enabled:
            try:
                # Get current session and RSI for fingerprinting
                current_session = self._get_current_session()
                current_rsi = self._get_rsi(signal.symbol) or 50.0
                current_regime = self._get_market_regime(signal.symbol) or "NEUTRAL"

                final_direction, cti_meta = self._counter_trade.get_smart_direction(
                    original_direction=signal.direction,
                    engine=signal.bot_name,
                    symbol=signal.symbol,
                    regime=current_regime,
                    rsi=current_rsi,
                    session=current_session,
                )

                if cti_meta.get("inverted"):
                    logger.info(f"[CTI] INVERTED {signal.direction} → {final_direction} for {signal.symbol} ({cti_meta.get('reason')})")
                    signal.direction = final_direction
                    cti_fingerprint = cti_meta.get("fingerprint")
                else:
                    cti_fingerprint = cti_meta.get("fingerprint")
                    logger.debug(f"[CTI] Signal unchanged: {signal.direction} (WR={cti_meta.get('historical_wr', 'N/A')})")

            except Exception as e:
                logger.warning(f"[CTI] Error processing signal: {e}")

        # Phase 8: Certainty Engine - require 80% confidence
        if self._certainty_engine and self._ce_enabled:
            try:
                # Get recent candles for certainty calculation
                candles = self._get_recent_candles(signal.symbol, 100)

                if candles and len(candles) >= 20:
                    certainty = self._certainty_engine.calculate_certainty(
                        symbol=signal.symbol,
                        direction=signal.direction,
                        candles=candles,
                        current_price=signal.entry_price or 0.0,
                    )

                    if not certainty.should_trade:
                        logger.warning(
                            f"[Certainty] Signal BLOCKED: {certainty.total_score:.0%} < {self._ce_threshold:.0%} "
                            f"(Tech={certainty.factors.technical_confluence:.0%}, "
                            f"Struct={certainty.factors.market_structure:.0%}, "
                            f"Sent={certainty.factors.sentiment_order_flow:.0%})"
                        )
                        return

                    logger.info(f"[Certainty] APPROVED: {certainty.total_score:.0%} confidence for {signal.direction} {signal.symbol}")
                else:
                    logger.debug(f"[Certainty] Skipped: insufficient candle data ({len(candles) if candles else 0})")

            except Exception as e:
                logger.warning(f"[Certainty] Error calculating: {e}")

        # Store CTI fingerprint for outcome recording when trade closes
        if cti_fingerprint:
            signal.cti_fingerprint = cti_fingerprint

        # Phase 8: R:R Validation - require minimum 1.5:1 risk/reward
        if hasattr(signal, 'stop_loss') and hasattr(signal, 'take_profit') and signal.entry_price:
            try:
                from libs.hydra.ftmo_bots.base_ftmo_bot import BaseFTMOBot
                # Use class-level constants
                min_rr = BaseFTMOBot.MIN_RR_RATIO

                if signal.direction.upper() == "BUY":
                    risk = signal.entry_price - signal.stop_loss
                    reward = signal.take_profit - signal.entry_price
                else:
                    risk = signal.stop_loss - signal.entry_price
                    reward = signal.entry_price - signal.take_profit

                if risk > 0:
                    rr_ratio = reward / risk
                    if rr_ratio < min_rr:
                        logger.warning(
                            f"[R:R] Signal BLOCKED: {rr_ratio:.2f} < {min_rr} min "
                            f"(Entry={signal.entry_price}, SL={signal.stop_loss}, TP={signal.take_profit})"
                        )
                        return
                    logger.debug(f"[R:R] APPROVED: {rr_ratio:.2f} >= {min_rr}")
            except Exception as e:
                logger.debug(f"[R:R] Validation skipped: {e}")

        # Store time multiplier for position sizing
        signal.time_size_multiplier = time_multiplier

        with self._signal_lock:
            self._signal_queue.append(signal)
            size_note = f" (size x{time_multiplier:.0%})" if time_multiplier != 1.0 else ""
            logger.info(f"[EventRunner] Signal queued: {signal.bot_name} {signal.direction} {signal.symbol}{size_note}")

    def _handle_heartbeat(self, heartbeat):
        """Handle heartbeat from price streamer."""
        if self._metrics:
            self._metrics.set_connection_status("price_stream", heartbeat.mt5_connected)

    # ==================== Phase 8 Helper Methods ====================

    def _get_current_session(self) -> str:
        """Get current trading session based on UTC time."""
        now = datetime.now(timezone.utc)
        hour = now.hour

        # Session definitions (UTC times)
        # Tokyo/Asia: 00:00-08:00 UTC
        # London: 08:00-16:00 UTC
        # New York: 13:00-21:00 UTC
        # Overlap (London+NY): 13:00-16:00 UTC

        if 0 <= hour < 8:
            return "asia"
        elif 8 <= hour < 13:
            return "london"
        elif 13 <= hour < 16:
            return "overlap"
        elif 16 <= hour < 21:
            return "new_york"
        else:
            return "off_hours"

    def _get_rsi(self, symbol: str, period: int = 14) -> Optional[float]:
        """Calculate RSI from recent candle data."""
        try:
            candles = self._get_recent_candles(symbol, period + 10)
            if not candles or len(candles) < period + 1:
                return None

            closes = [c.get("close", c.get("c", 0)) for c in candles]
            if len(closes) < period + 1:
                return None

            # Calculate price changes
            gains = []
            losses = []
            for i in range(1, len(closes)):
                change = closes[i] - closes[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))

            if len(gains) < period:
                return None

            # Simple moving average for initial RSI
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period

            if avg_loss == 0:
                return 100.0

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        except Exception as e:
            logger.debug(f"[RSI] Calculation error for {symbol}: {e}")
            return None

    def _get_market_regime(self, symbol: str) -> str:
        """Determine market regime from H1 trend data."""
        # Use H1 trends if available
        if symbol in self._h1_trends:
            trend = self._h1_trends[symbol]
            if trend == "BULLISH":
                return "TRENDING_UP"
            elif trend == "BEARISH":
                return "TRENDING_DOWN"
            else:
                return "RANGING"

        # Default to neutral if no data
        return "NEUTRAL"

    def _get_tick_trend(self, symbol: str, lookback: int = 100) -> str:
        """
        Calculate trend from tick buffer (no candles needed).

        This is a FALLBACK trend detection when H1 candles are unavailable.
        Uses price change over recent ticks to determine short-term direction.

        Args:
            symbol: Trading symbol
            lookback: Number of ticks to analyze

        Returns:
            "BULLISH", "BEARISH", or "NEUTRAL"
        """
        # Get tick buffer from event bus
        ticks = []
        if hasattr(self, "_event_bus") and self._event_bus:
            ticks = self._event_bus.get_tick_buffer(symbol)

        if not ticks or len(ticks) < lookback:
            return "NEUTRAL"

        recent = ticks[-lookback:]
        first_price = recent[0].bid if hasattr(recent[0], "bid") else recent[0].get("bid", 0)
        last_price = recent[-1].bid if hasattr(recent[-1], "bid") else recent[-1].get("bid", 0)

        if first_price <= 0 or last_price <= 0:
            return "NEUTRAL"

        change_pct = ((last_price - first_price) / first_price) * 100

        # Trend thresholds - require meaningful move
        # For XAUUSD at ~$2700, 0.1% = ~$2.70 move
        if change_pct > 0.10:  # Up 0.1%+ in lookback period
            return "BULLISH"
        elif change_pct < -0.10:  # Down 0.1%+
            return "BEARISH"

        return "NEUTRAL"

    def _get_recent_candles(self, symbol: str, count: int = 100) -> Optional[list]:
        """Get recent candles from ZMQ client or tick aggregator fallback."""
        try:
            # First try ZMQ client for proper candle data
            if self._zmq_client:
                try:
                    candles = self._zmq_client.get_candles(symbol, "M1", count)
                    if candles and len(candles) >= 10:
                        return candles
                except Exception:
                    pass  # Fall through to tick aggregator

            # Fallback: get candles from bot wrapper's tick aggregator
            # This uses real-time tick data aggregated into candles
            for name, config in self._bot_wrappers.items():
                wrapper = config.get("wrapper")
                if wrapper and hasattr(wrapper, "_tick_aggregator"):
                    # Check if this wrapper handles this symbol
                    wrapper_symbol = getattr(wrapper.bot.config, "symbol", None) if hasattr(wrapper, "bot") else None
                    if wrapper_symbol == symbol:
                        candles = wrapper._tick_aggregator.get_candles(count)
                        if candles and len(candles) >= 10:
                            return candles

            return None  # If no candle data available, return None

        except Exception as e:
            logger.debug(f"[Candles] Error getting candles for {symbol}: {e}")
            return None

    def _risk_monitor_loop(self):
        """Monitor risk limits and trigger kill switch if needed."""
        while self._running:
            try:
                self._check_risk_limits()
                self._check_time_exits()  # Time-based exit check (Phase 3 - 2025-12-11)
                self._check_weekend_closeout()  # Weekend gap protection (Phase 4 - 2025-12-11)
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"[EventRunner] Risk monitor error: {e}")
                time.sleep(5)

    def _check_risk_limits(self):
        """Check FTMO risk limits."""
        if not self._zmq_client:
            return

        try:
            account = self._zmq_client.get_account()
            if not account:
                return

            balance = account.get("balance", self._daily_starting_balance)
            equity = account.get("equity", balance)

            # Calculate drawdowns
            daily_dd = (self._daily_starting_balance - balance) / self._daily_starting_balance * 100
            total_dd = (self._daily_starting_balance - equity) / self._daily_starting_balance * 100

            # Update metrics
            if self._metrics:
                self._metrics.update_risk_metrics(daily_dd, total_dd, self._kill_switch)
                self._metrics.update_account(
                    balance=balance,
                    equity=equity,
                    positions=len(self._zmq_client.get_positions()),
                    floating_pnl=equity - balance
                )

            # Check limits
            if daily_dd >= self._max_daily_loss_percent:
                self._trigger_kill_switch(f"Daily loss limit hit: {daily_dd:.2f}%")

            if total_dd >= self._max_total_drawdown_percent:
                self._trigger_kill_switch(f"Total drawdown limit hit: {total_dd:.2f}%")

            # Phase 5: Update drawdown-based risk scaling
            self._update_risk_scaling(equity)

            # Phase 5: Check if daily trading should be halted
            if daily_dd >= self._halt_trading_at_loss_percent:
                if not hasattr(self, '_daily_halt_logged') or not self._daily_halt_logged:
                    logger.warning(f"[RiskScale] Daily halt triggered: {daily_dd:.2f}% loss >= {self._halt_trading_at_loss_percent}%")
                    self._daily_halt_logged = True

            # Phase 5: Update ATR values and trailing stops
            self._update_atr_values()
            self._update_trailing_stops()

            # Phase 5: Update H1 trends for multi-timeframe confirmation
            self._update_h1_trends()

        except Exception as e:
            logger.error(f"[EventRunner] Risk check error: {e}")

    def _update_risk_scaling(self, current_equity: float):
        """
        Update risk multiplier based on drawdown from peak (Phase 5 - 2025-12-11).

        Expert insight: "Reduce risk 50% when down 5% or more"

        Thresholds:
        - 0-3% DD: 100% of base risk (1.5%)
        - 3-5% DD: 50% of base risk (0.75%)
        - 5-7% DD: 33% of base risk (0.5%)
        - 7%+ DD: 25% of base risk (0.375%)
        """
        # Update peak if we have a new high
        if current_equity > self._peak_balance:
            self._peak_balance = current_equity
            # Reset risk multiplier at new equity high
            if self._current_risk_multiplier < 1.0:
                logger.info(f"[RiskScale] New equity high ${current_equity:.2f} - risk reset to 100%")
            self._current_risk_multiplier = 1.0
            return

        # Calculate drawdown from peak
        drawdown_pct = (self._peak_balance - current_equity) / self._peak_balance * 100

        # Determine risk multiplier based on drawdown
        old_multiplier = self._current_risk_multiplier
        new_multiplier = 1.0

        for dd_threshold, multiplier in sorted(self._risk_scale_thresholds.items()):
            if drawdown_pct >= dd_threshold:
                new_multiplier = multiplier

        self._current_risk_multiplier = new_multiplier
        self._current_balance = current_equity

        # Log if risk level changed
        if new_multiplier != old_multiplier:
            effective_risk = self._base_risk_percent * new_multiplier * 100
            logger.info(
                f"[RiskScale] Drawdown {drawdown_pct:.2f}% from peak ${self._peak_balance:.2f} - "
                f"risk scaled to {new_multiplier*100:.0f}% ({effective_risk:.2f}%)"
            )

    def get_scaled_risk_percent(self) -> float:
        """Get current risk percent after drawdown scaling (Phase 5)."""
        return self._base_risk_percent * self._current_risk_multiplier

    def _update_atr_values(self):
        """
        Update ATR values for all symbols (Phase 5 - ATR Trailing Stops).

        ATR is calculated from M1 candles every 5 minutes.
        Used for:
        - Dynamic trailing stop distance (1.5x ATR)
        - Volatility-aware position sizing
        """
        now = datetime.now(timezone.utc)

        # Only update every 5 minutes
        if self._last_atr_update:
            elapsed = (now - self._last_atr_update).total_seconds()
            if elapsed < self._atr_update_interval:
                return

        if not self._zmq_client:
            return

        for symbol in self.ALL_SYMBOLS:
            try:
                # Fetch last 20 candles for ATR calculation
                candles = self._zmq_client.get_candles(symbol, "M1", count=20)
                if not candles or len(candles) < 15:
                    continue

                atr = calculate_atr(candles, period=14)
                if atr and atr > 0:
                    old_atr = self._atr_values.get(symbol, 0)
                    self._atr_values[symbol] = atr

                    # Log significant ATR changes (>20%)
                    if old_atr > 0 and abs(atr - old_atr) / old_atr > 0.2:
                        logger.info(f"[ATR] {symbol} ATR changed: {old_atr:.4f} -> {atr:.4f}")

            except Exception as e:
                logger.debug(f"[ATR] Failed to calculate ATR for {symbol}: {e}")

        self._last_atr_update = now
        logger.debug(f"[ATR] Updated: {self._atr_values}")

    def _update_h1_trends(self):
        """
        Update H1 trend for all symbols (Phase 5 - Multi-TF Confirmation).

        Trend determined by SMA(10) vs SMA(30) on H1 timeframe:
        - BULLISH: Price > SMA10 > SMA30
        - BEARISH: Price < SMA10 < SMA30
        - NEUTRAL: Otherwise
        """
        now = datetime.now(timezone.utc)

        # Only update every 10 minutes
        if self._last_h1_update:
            elapsed = (now - self._last_h1_update).total_seconds()
            if elapsed < self._h1_update_interval:
                return

        if not self._zmq_client:
            return

        import numpy as np

        for symbol in self.ALL_SYMBOLS:
            try:
                # Fetch 50 H1 candles for SMA calculation
                candles = self._zmq_client.get_candles(symbol, "H1", count=50)
                if not candles or len(candles) < 35:
                    continue

                closes = [c.get("close", 0) for c in candles if c.get("close")]
                if len(closes) < 35:
                    continue

                # Calculate SMAs
                sma_10 = np.mean(closes[-10:])
                sma_30 = np.mean(closes[-30:])
                current = closes[-1]

                # Determine trend
                old_trend = self._h1_trends.get(symbol, "NEUTRAL")
                if current > sma_10 > sma_30:
                    new_trend = "BULLISH"
                elif current < sma_10 < sma_30:
                    new_trend = "BEARISH"
                else:
                    new_trend = "NEUTRAL"

                self._h1_trends[symbol] = new_trend

                # Log trend changes
                if new_trend != old_trend:
                    logger.info(f"[MTF] {symbol} H1 trend changed: {old_trend} -> {new_trend}")

            except Exception as e:
                logger.debug(f"[MTF] Failed to calculate H1 trend for {symbol}: {e}")

        self._last_h1_update = now
        logger.debug(f"[MTF] H1 trends updated: {self._h1_trends}")

    def _check_h1_alignment(self, symbol: str, direction: str) -> tuple[bool, str]:
        """
        Check if trade direction aligns with trend (Phase 5 + Tick Fallback).

        Uses H1 candle trend if available, falls back to tick-based trend detection.
        This ensures trend filtering even when CANDLES command is unavailable.

        Args:
            symbol: Trading symbol
            direction: Trade direction ("BUY" or "SELL")

        Returns:
            Tuple of (is_aligned, reason)
        """
        if not self._mtf_enabled:
            return True, "MTF disabled"

        h1_trend = self._h1_trends.get(symbol, "NEUTRAL")
        trend_source = "H1"

        # FALLBACK: If H1 trend is NEUTRAL (likely because CANDLES failed),
        # use tick-based trend detection
        if h1_trend == "NEUTRAL":
            tick_trend = self._get_tick_trend(symbol, lookback=100)
            if tick_trend != "NEUTRAL":
                h1_trend = tick_trend
                trend_source = "TICK"
                logger.debug(f"[TrendFilter] Using tick-based trend for {symbol}: {tick_trend}")

        # NEUTRAL allows both directions (no clear trend)
        if h1_trend == "NEUTRAL":
            return True, f"Trend neutral (no clear direction)"

        direction_upper = direction.upper()

        # Check alignment - CRITICAL: Block trades against trend
        if h1_trend == "BULLISH" and direction_upper == "BUY":
            return True, f"{trend_source} bullish + BUY aligned"
        elif h1_trend == "BEARISH" and direction_upper == "SELL":
            return True, f"{trend_source} bearish + SELL aligned"
        elif h1_trend == "BULLISH" and direction_upper == "SELL":
            return False, f"SELL blocked: {trend_source} trend is BULLISH"
        elif h1_trend == "BEARISH" and direction_upper == "BUY":
            logger.warning(f"[TrendFilter] BUY BLOCKED for {symbol} - {trend_source} trend is BEARISH")
            return False, f"BUY blocked: {trend_source} trend is BEARISH"

        return True, f"{trend_source} {h1_trend}"

    def _update_trailing_stops(self):
        """
        Update trailing stops for open positions using ATR (Phase 5).

        For each position registered with PositionManager:
        1. Get current price from latest tick
        2. Calculate ATR-based trail distance
        3. Check if stop should be moved
        4. Modify via ZMQ if needed
        """
        if not self._position_manager or not self._zmq_client:
            return

        if self.paper_mode:
            return  # Skip trailing in paper mode for now

        try:
            # Get all open positions from MT5
            positions_result = self._zmq_client.get_positions()
            if not positions_result or not positions_result.get("success"):
                return

            positions = positions_result.get("positions", [])
            if not positions:
                return

            for pos in positions:
                ticket = pos.get("ticket")
                symbol = pos.get("symbol", "")
                current_price = pos.get("price_current", 0)

                if not ticket or not current_price:
                    continue

                # Check if this position is managed
                state = self._position_manager.get_position_state(str(ticket))
                if not state:
                    continue

                # Get ATR for this symbol
                atr = self._atr_values.get(symbol, 0)
                if atr > 0:
                    # Override trail distance with ATR-based distance
                    trail_distance = atr * self._atr_trail_multiplier
                    self._position_manager.config.trail_distance_percent = trail_distance / current_price

                # Update position and check for actions
                action = self._position_manager.update_position(str(ticket), current_price)

                if action:
                    self._handle_position_action(action, ticket, pos)

        except Exception as e:
            logger.error(f"[TrailStop] Error updating trailing stops: {e}")

    def _handle_position_action(self, action, ticket: int, position: dict):
        """
        Handle position management actions (Phase 5).

        Actions: MODIFY_SL, TRAIL_UPDATE, PARTIAL_CLOSE, TIME_EXIT
        """
        try:
            if action.action_type in (ActionType.MODIFY_SL, ActionType.TRAIL_UPDATE):
                # Modify stop loss via ZMQ
                if action.new_sl:
                    result = self._zmq_client.modify_sl(ticket, action.new_sl)
                    if result and result.get("success"):
                        logger.info(
                            f"[TrailStop] Modified SL for #{ticket}: {action.new_sl:.5f} "
                            f"({action.reason})"
                        )
                    else:
                        logger.warning(
                            f"[TrailStop] Failed to modify SL for #{ticket}: "
                            f"{result.get('error', 'Unknown')}"
                        )

            elif action.action_type == ActionType.PARTIAL_CLOSE:
                # Partial close - calculate volume to close
                if action.partial_percent:
                    current_vol = position.get("volume", 0)
                    close_vol = round(current_vol * action.partial_percent, 2)
                    if close_vol >= 0.01:
                        result = self._zmq_client.close_partial(ticket, close_vol)
                        if result and result.get("success"):
                            logger.info(
                                f"[TrailStop] Partial close #{ticket}: {close_vol} lots "
                                f"({action.reason})"
                            )

            elif action.action_type == ActionType.TIME_EXIT:
                # Force close entire position
                result = self._zmq_client.close_position(ticket)
                if result and result.get("success"):
                    logger.info(f"[TrailStop] Force closed #{ticket}: {action.reason}")
                    self._position_manager.unregister_position(str(ticket))

        except Exception as e:
            logger.error(f"[TrailStop] Error handling action for #{ticket}: {e}")

    def _check_time_exits(self):
        """
        Check positions for time-based exits (Phase 3 - 2025-12-11).

        Closes positions that have been held longer than their max_hold_hours.
        This prevents:
        - Holding losers hoping they recover
        - Overnight/weekend gap risk
        - Capital being tied up in stale trades
        """
        if not self._zmq_client or self.paper_mode:
            return  # Only for live trading

        try:
            positions = self._zmq_client.get_positions()
            if not positions:
                return

            now = datetime.now(timezone.utc)

            for pos in positions:
                ticket = pos.get("ticket")
                if not ticket:
                    continue

                # Check if we're tracking this position
                if ticket in self._position_times:
                    open_time, bot_name, max_hold = self._position_times[ticket]
                    hold_hours = (now - open_time).total_seconds() / 3600

                    # Warning at 75% of max hold time
                    warning_threshold = max_hold * 0.75
                    if hold_hours >= warning_threshold and hold_hours < max_hold:
                        remaining = max_hold - hold_hours
                        logger.warning(
                            f"[TimeExit] Position {ticket} ({bot_name}) held {hold_hours:.1f}h, "
                            f"will auto-close in {remaining:.1f}h"
                        )

                    # Force close if exceeded
                    if hold_hours >= max_hold:
                        logger.warning(
                            f"[TimeExit] CLOSING position {ticket} ({bot_name}) - "
                            f"held {hold_hours:.1f}h > max {max_hold}h"
                        )
                        result = self._zmq_client.close(ticket)
                        if result.get("success"):
                            logger.info(f"[TimeExit] Position {ticket} closed successfully")
                            del self._position_times[ticket]
                        else:
                            logger.error(f"[TimeExit] Failed to close {ticket}: {result.get('error')}")

                else:
                    # New position we don't have a record for - track it now with default max hold
                    # This handles positions that existed before the runner started
                    self._position_times[ticket] = (
                        now,  # Use now as best estimate
                        pos.get("comment", "unknown"),
                        self._default_max_hold_hours
                    )
                    logger.debug(f"[TimeExit] Tracking new position {ticket}")

            # Clean up closed positions from tracking
            active_tickets = {p.get("ticket") for p in positions}
            closed_tickets = [t for t in self._position_times if t not in active_tickets]
            for ticket in closed_tickets:
                del self._position_times[ticket]

        except Exception as e:
            logger.error(f"[EventRunner] Time exit check error: {e}")

    def _check_weekend_closeout(self):
        """
        Close all positions before Friday market close (Phase 4 - 2025-12-11).

        Protects against weekend gap risk by closing positions
        30 minutes before Friday market close (21:30 UTC).
        """
        if not self._zmq_client or self.paper_mode:
            return  # Only for live trading

        now = datetime.now(timezone.utc)

        # Check if it's Friday
        if now.weekday() != 4:  # 0=Mon, 4=Fri
            self._weekend_closeout_done = False  # Reset for next week
            return

        # Check if we're past the closeout time
        closeout_time = now.replace(hour=self._friday_close_hour, minute=self._friday_close_minute, second=0)
        if now < closeout_time:
            return  # Not yet time

        # Already done for today?
        if self._weekend_closeout_done:
            return

        try:
            positions = self._zmq_client.get_positions()
            if not positions:
                self._weekend_closeout_done = True
                return

            logger.warning(
                f"[WeekendProtection] Friday {self._friday_close_hour}:{self._friday_close_minute:02d} UTC - "
                f"Closing {len(positions)} positions for weekend protection"
            )

            for pos in positions:
                ticket = pos.get("ticket")
                if not ticket:
                    continue

                symbol = pos.get("symbol", "unknown")
                profit = pos.get("profit", 0)

                logger.info(f"[WeekendProtection] Closing position {ticket} ({symbol}), P&L: ${profit:.2f}")
                result = self._zmq_client.close(ticket)

                if result.get("success"):
                    logger.info(f"[WeekendProtection] Position {ticket} closed")
                    # Clean up tracking
                    if ticket in self._position_times:
                        del self._position_times[ticket]
                else:
                    logger.error(f"[WeekendProtection] Failed to close {ticket}: {result.get('error')}")

            self._weekend_closeout_done = True

            # Send notification
            try:
                from libs.notifications.telegram_bot import send_telegram_message
                send_telegram_message(
                    f"[WeekendProtection] Closed {len(positions)} positions before weekend at {now.strftime('%H:%M')} UTC"
                )
            except Exception:
                pass

        except Exception as e:
            logger.error(f"[WeekendProtection] Error during closeout: {e}")

    def _trigger_kill_switch(self, reason: str):
        """Trigger emergency stop."""
        if self._kill_switch:
            return

        self._kill_switch = True
        logger.critical(f"[EventRunner] KILL SWITCH TRIGGERED: {reason}")

        if self._metrics:
            self._metrics.update_risk_metrics(0, 0, True)

        # Send alert
        try:
            from libs.notifications.telegram_bot import send_ftmo_kill_switch_alert
            account = self._zmq_client.get_account() if self._zmq_client else {}
            send_ftmo_kill_switch_alert(reason, account.get("balance", 0), 0)
        except Exception:
            pass

    def _trade_executor_loop(self):
        """Execute queued trades."""
        while self._running:
            try:
                signal = None

                with self._signal_lock:
                    if self._signal_queue:
                        signal = self._signal_queue.pop(0)

                if signal:
                    self._execute_signal(signal)

                time.sleep(0.1)  # Small delay

            except Exception as e:
                logger.error(f"[EventRunner] Trade executor error: {e}")
                time.sleep(1)

    def _execute_signal(self, signal):
        """Execute a trading signal."""
        logger.info(f"[EventRunner] Executing: {signal.bot_name} {signal.direction} {signal.symbol}")

        # Per-bot risk limit check (added 2025-12-11)
        # Reset at midnight UTC
        now = datetime.now(timezone.utc)
        if self._last_reset_date is None or now.date() != self._last_reset_date.date():
            self._bot_daily_pnl.clear()
            self._bot_disabled.clear()
            self._daily_trade_count = 0  # Phase 5: reset daily trade count
            self._daily_halt_logged = False  # Phase 5: reset halt log flag
            self._last_reset_date = now
            logger.info("[PerBotRisk] Daily P&L and trade count reset at midnight UTC")

        # Check if this bot is disabled for the day
        if self._bot_disabled.get(signal.bot_name, False):
            logger.warning(f"[PerBotRisk] Trade blocked: {signal.bot_name} disabled for day (hit loss limit)")
            return

        # Calculate max loss in dollars (1% of account balance)
        max_bot_loss = self._daily_starting_balance * (self._max_per_bot_loss_percent / 100)
        current_bot_pnl = self._bot_daily_pnl.get(signal.bot_name, 0)

        if current_bot_pnl <= -max_bot_loss:
            self._bot_disabled[signal.bot_name] = True
            logger.warning(
                f"[PerBotRisk] Bot {signal.bot_name} DISABLED for day: "
                f"P&L ${current_bot_pnl:.2f} <= -${max_bot_loss:.2f} limit"
            )
            return

        # DAILY TRADE LIMITS (Phase 5 - 2025-12-11)
        # Expert insight: "Limit to 2-3 A+ trades per day, halt at 5% loss"

        # Check if daily trading is halted due to loss
        daily_dd = (self._daily_starting_balance - self._current_balance) / self._daily_starting_balance * 100
        if daily_dd >= self._halt_trading_at_loss_percent:
            logger.warning(
                f"[TradeLimits] Trade blocked: daily loss {daily_dd:.2f}% >= "
                f"{self._halt_trading_at_loss_percent}% halt threshold"
            )
            return

        # Check hard limit on daily trades
        if self._daily_trade_count >= self._max_daily_trades:
            logger.warning(
                f"[TradeLimits] Trade blocked: {self._daily_trade_count} trades today >= "
                f"{self._max_daily_trades} daily max"
            )
            return

        # Soft limit: after 3% loss, reduce to 3 trades max
        if daily_dd >= 3.0 and self._daily_trade_count >= self._daily_loss_trade_limit:
            logger.warning(
                f"[TradeLimits] Trade blocked: {self._daily_trade_count} trades after "
                f"{daily_dd:.1f}% loss (soft limit: {self._daily_loss_trade_limit} trades)"
            )
            return

        # NEWS FILTER: Avoid trading around high-impact news events (Phase 3 - 2025-12-11)
        try:
            from libs.hydra.ftmo_bots.news_filter import should_avoid_news
            avoid, reason = should_avoid_news(signal.symbol)
            if avoid:
                logger.warning(f"[NewsFilter] Trade blocked: {signal.bot_name} {signal.symbol} - {reason}")
                return
        except ImportError:
            pass  # News filter not available

        # MULTI-TIMEFRAME FILTER: Only trade in direction of H1 trend (Phase 5 - 2025-12-11)
        aligned, mtf_reason = self._check_h1_alignment(signal.symbol, signal.direction)
        if not aligned:
            logger.warning(
                f"[MTFFilter] Trade blocked: {signal.bot_name} {signal.direction} {signal.symbol} - {mtf_reason}"
            )
            return

        # TRADE CLUSTERING PREVENTION: Avoid rapid-fire signals on same symbol (Phase 4 - 2025-12-11)
        now = datetime.now(timezone.utc)
        last_trade = self._last_trade_time.get(signal.symbol)
        if last_trade:
            minutes_since = (now - last_trade).total_seconds() / 60
            if minutes_since < self._trade_cooldown_minutes:
                logger.warning(
                    f"[ClusterFilter] Trade blocked: {signal.bot_name} {signal.symbol} - "
                    f"only {minutes_since:.1f}m since last trade (cooldown={self._trade_cooldown_minutes}m)"
                )
                return

        # Check max pending signals queue
        with self._signal_lock:
            pending_count = len(self._signal_queue)
        if pending_count >= self._max_pending_signals:
            logger.warning(
                f"[ClusterFilter] Trade blocked: {signal.bot_name} {signal.symbol} - "
                f"{pending_count} signals pending (max={self._max_pending_signals})"
            )
            return

        # CORRELATION-BASED POSITION BLOCKING: Prevent multiple positions in same direction (Phase 4 - 2025-12-11)
        try:
            # Get current open positions
            if self._zmq_client and not self.paper_mode:
                positions_result = self._zmq_client.get_positions()
                if positions_result.get("success"):
                    open_positions = positions_result.get("positions", [])

                    # Find which correlation group this symbol belongs to
                    signal_group = None
                    for group_name, symbols in self._correlation_groups.items():
                        if signal.symbol in symbols:
                            signal_group = group_name
                            break

                    if signal_group:
                        # Check if any correlated position exists in same direction
                        group_symbols = self._correlation_groups[signal_group]
                        for pos in open_positions:
                            pos_symbol = pos.get("symbol", "")
                            pos_type = pos.get("type", "")  # 0=BUY, 1=SELL

                            if pos_symbol in group_symbols:
                                # Map position type to direction
                                pos_direction = "BUY" if pos_type == 0 else "SELL"

                                if pos_direction == signal.direction.upper():
                                    logger.warning(
                                        f"[CorrelationFilter] Trade blocked: {signal.bot_name} {signal.direction} {signal.symbol} - "
                                        f"correlated position exists: {pos_direction} {pos_symbol} (group={signal_group})"
                                    )
                                    return
        except Exception as e:
            logger.debug(f"[CorrelationFilter] Check failed (non-blocking): {e}")

        # AGI ADVISOR FILTER: Check if AGI recommends this trade (Phase 6 - 2025-12-11)
        if self._agi_enabled and self._agi_advisor:
            # Determine current session for AGI check
            hour = datetime.now(timezone.utc).hour
            if 22 <= hour or hour < 6:
                current_session = "asian"
            elif 6 <= hour < 14:
                current_session = "london"
            else:
                current_session = "new_york"

            agi_approved, agi_reason = self._agi_advisor.should_trade(
                signal.bot_name, signal.symbol, current_session
            )
            if not agi_approved:
                logger.warning(f"[AGIFilter] Trade blocked: {signal.bot_name} {signal.symbol} - {agi_reason}")
                if self._metrics:
                    self._metrics.record_handler_error("agi_blocked", signal.bot_name)
                return

            # Log AGI approval
            if "risk_off" in agi_reason.lower():
                logger.info(f"[AGIFilter] Trade approved with caution: {agi_reason}")

        # MARKET INTELLIGENCE FILTER: Use contrarian signals from retail positioning
        # If retail is 70%+ long on a symbol, block BUY trades (fade the crowd)
        if self._agi_enabled and self._agi_advisor:
            try:
                contrarian_signal = self._agi_advisor.get_contrarian_signal(signal.symbol)
                if contrarian_signal:
                    direction_upper = signal.direction.upper()

                    # Block BUY when retail is heavily long (contrarian says SHORT)
                    if contrarian_signal == "CONTRARIAN_SHORT" and direction_upper == "BUY":
                        logger.warning(
                            f"[MarketIntel] BUY BLOCKED for {signal.symbol} - "
                            f"retail 70%+ long, contrarian signal is SHORT"
                        )
                        return

                    # Block SELL when retail is heavily short (contrarian says LONG)
                    if contrarian_signal == "CONTRARIAN_LONG" and direction_upper == "SELL":
                        logger.warning(
                            f"[MarketIntel] SELL BLOCKED for {signal.symbol} - "
                            f"retail 70%+ short, contrarian signal is LONG"
                        )
                        return

                    # Log when trading WITH contrarian signal
                    if contrarian_signal == "CONTRARIAN_SHORT" and direction_upper == "SELL":
                        logger.info(f"[MarketIntel] SELL aligned with contrarian signal for {signal.symbol}")
                    elif contrarian_signal == "CONTRARIAN_LONG" and direction_upper == "BUY":
                        logger.info(f"[MarketIntel] BUY aligned with contrarian signal for {signal.symbol}")
            except Exception as e:
                logger.debug(f"[MarketIntel] Check failed (non-blocking): {e}")

        # SELL FILTER: Block SELL for most bots, but allow proven performers
        # gold_london has 71% WR on SELL (7 trades) - allow it
        SELL_WHITELIST = ["GoldLondonReversal", "gold_london"]

        if signal.direction.upper() in ("SELL", "SHORT"):
            if signal.bot_name not in SELL_WHITELIST:
                logger.warning(f"[SellFilter] Trade blocked: {signal.bot_name} {signal.direction} {signal.symbol} - not in whitelist")
                return
            else:
                logger.info(f"[SellFilter] SELL allowed for {signal.bot_name} (whitelisted)")

        try:
            if self.paper_mode:
                # Paper trade - just log it
                self._record_paper_trade(signal)
            else:
                # Live trade via ZMQ
                self._execute_live_trade(signal)

            # Update last trade time for clustering prevention (Phase 4 - 2025-12-11)
            self._last_trade_time[signal.symbol] = datetime.now(timezone.utc)

            # Phase 5: Increment daily trade count
            self._daily_trade_count += 1
            logger.info(f"[TradeLimits] Trade {self._daily_trade_count}/{self._max_daily_trades} today")

            if self._metrics:
                self._metrics.record_trade_opened(signal.bot_name)

            # Send Telegram alert
            self._send_trade_alert(signal)

        except Exception as e:
            logger.error(f"[EventRunner] Trade execution failed: {e}")
            if self._metrics:
                self._metrics.record_handler_error("execution", signal.bot_name)

    def _record_paper_trade(self, signal):
        """Record a paper trade."""
        import json
        from pathlib import Path

        # Phase 5: Apply risk scaling to lot size for paper trades too
        scaled_lot = round(signal.lot_size * self._current_risk_multiplier, 2)

        # Phase 6: Apply AGI size multiplier on top of risk scaling
        agi_multiplier = 1.0
        if self._agi_size_scaling and self._agi_advisor:
            agi_multiplier = self._agi_advisor.get_size_multiplier(signal.bot_name)
            if agi_multiplier != 1.0:
                scaled_lot = round(scaled_lot * agi_multiplier, 2)
                logger.info(f"[AGIScale] Paper lot adjusted: {signal.lot_size} -> {scaled_lot} (AGI x{agi_multiplier})")

        scaled_lot = max(0.01, scaled_lot)
        if self._current_risk_multiplier < 1.0:
            logger.info(f"[RiskScale] Paper lot scaled: {signal.lot_size} -> {scaled_lot} ({self._current_risk_multiplier:.0%})")

        trade_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bot": signal.bot_name,
            "symbol": signal.symbol,
            "direction": signal.direction,
            "entry": signal.entry_price,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
            "lot": scaled_lot,
            "lot_original": signal.lot_size,
            "risk_scale": self._current_risk_multiplier,
            "reason": signal.reason,
            "status": "OPEN",
            "mode": "paper"
        }

        trade_file = Path("/app/data/hydra/ftmo/event_trades.jsonl")
        trade_file.parent.mkdir(parents=True, exist_ok=True)

        with open(trade_file, "a") as f:
            f.write(json.dumps(trade_record) + "\n")

        logger.info(f"[EventRunner] Paper trade recorded: {signal.symbol} {signal.direction}")

    def _execute_live_trade(self, signal):
        """Execute a live trade via ZMQ."""
        if not self._zmq_client:
            raise RuntimeError("ZMQ client not connected")

        # Phase 5: Apply risk scaling to lot size
        scaled_lot = round(signal.lot_size * self._current_risk_multiplier, 2)

        # Phase 6: Apply AGI size multiplier on top of risk scaling
        agi_multiplier = 1.0
        if self._agi_size_scaling and self._agi_advisor:
            agi_multiplier = self._agi_advisor.get_size_multiplier(signal.bot_name)
            if agi_multiplier != 1.0:
                scaled_lot = round(scaled_lot * agi_multiplier, 2)
                logger.info(f"[AGIScale] Lot adjusted: {signal.lot_size} -> {scaled_lot} (AGI x{agi_multiplier})")

        # Ensure minimum lot size of 0.01
        scaled_lot = max(0.01, scaled_lot)
        if self._current_risk_multiplier < 1.0:
            logger.info(f"[RiskScale] Lot scaled: {signal.lot_size} -> {scaled_lot} ({self._current_risk_multiplier:.0%})")

        result = self._zmq_client.trade(
            symbol=signal.symbol,
            direction=signal.direction.upper(),  # ZMQ expects uppercase BUY/SELL
            volume=scaled_lot,
            sl=signal.stop_loss,
            tp=signal.take_profit,
            comment=f"HYDRA_{signal.bot_name}"
        )

        if not result.get("success"):
            raise RuntimeError(f"Trade failed: {result.get('error', 'Unknown')}")

        ticket = result.get('ticket')
        logger.info(f"[EventRunner] Live trade executed: ticket={ticket}")

        # Track position time for time-based exits (Phase 3 - 2025-12-11)
        if ticket:
            max_hold = self._bot_max_hold_hours.get(signal.bot_name, self._default_max_hold_hours)
            self._position_times[ticket] = (
                datetime.now(timezone.utc),
                signal.bot_name,
                max_hold
            )
            logger.debug(f"[TimeExit] Tracking ticket {ticket} ({signal.bot_name}), max hold: {max_hold}h")

            # Phase 5: Register position with PositionManager for ATR trailing
            if self._position_manager:
                self._position_manager.register_position(
                    trade_id=str(ticket),
                    symbol=signal.symbol,
                    direction=signal.direction.upper(),
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    position_size=scaled_lot,
                )
                logger.debug(f"[TrailStop] Registered position #{ticket} for ATR trailing")

            # Phase 7: Record trade open in persistent ledger (2025-12-11)
            if self._trade_ledger:
                try:
                    self._trade_ledger.record_open(
                        mt5_ticket=ticket,
                        bot_name=signal.bot_name,
                        symbol=signal.symbol,
                        direction=signal.direction.upper(),
                        volume=scaled_lot,
                        entry_price=signal.entry_price,
                        entry_sl=signal.stop_loss,
                        entry_tp=signal.take_profit,
                        mode="LIVE" if not self.paper_mode else "PAPER"
                    )
                    logger.info(f"[TradeLedger] Recorded OPEN: #{ticket} {signal.direction} {scaled_lot} {signal.symbol}")
                except Exception as e:
                    logger.warning(f"[TradeLedger] Failed to record trade open: {e}")

    def _send_trade_alert(self, signal):
        """Send trade alert via Telegram."""
        try:
            from libs.notifications.telegram_bot import send_ftmo_trade_alert

            send_ftmo_trade_alert(
                bot_name=signal.bot_name,
                symbol=signal.symbol,
                direction=signal.direction,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                lot_size=signal.lot_size,
                reason=signal.reason,
                paper_mode=self.paper_mode
            )
        except Exception as e:
            logger.warning(f"[EventRunner] Telegram alert failed: {e}")

    def update_bot_pnl(self, bot_name: str, pnl: float):
        """
        Update bot P&L (called when trade closes).

        Args:
            bot_name: Name of the bot
            pnl: P&L from the closed trade (positive or negative)
        """
        current = self._bot_daily_pnl.get(bot_name, 0)
        self._bot_daily_pnl[bot_name] = current + pnl

        max_loss = self._daily_starting_balance * (self._max_per_bot_loss_percent / 100)

        logger.info(f"[PerBotRisk] {bot_name}: trade P&L ${pnl:.2f}, daily total ${current + pnl:.2f}")

        if self._bot_daily_pnl[bot_name] <= -max_loss:
            self._bot_disabled[bot_name] = True
            logger.warning(f"[PerBotRisk] Bot {bot_name} DISABLED: daily loss ${-self._bot_daily_pnl[bot_name]:.2f} >= limit ${max_loss:.2f}")

        # Phase 9: Losing Streak Detector - track consecutive losses
        if pnl < 0:
            # Loss - increment streak counter
            self._consecutive_losses += 1
            logger.info(f"[LosingStreak] Loss recorded. Consecutive losses: {self._consecutive_losses}")

            # Enter virtual mode after threshold
            if self._consecutive_losses >= self._losing_streak_threshold and not self._virtual_mode:
                self._virtual_mode = True
                self._virtual_win_count = 0
                logger.warning(
                    f"[LosingStreak] ENTERING VIRTUAL MODE after {self._consecutive_losses} consecutive losses. "
                    f"Trading paused until {self._virtual_wins_to_resume} virtual win(s)."
                )
                # Send notification
                try:
                    from libs.notifications.telegram_bot import send_telegram_message
                    send_telegram_message(
                        f"⚠️ [LosingStreak] Entered VIRTUAL MODE after {self._consecutive_losses} losses. "
                        f"Signals will be tracked but not executed."
                    )
                except Exception:
                    pass
        else:
            # Win - reset streak counter
            if self._consecutive_losses > 0:
                logger.info(f"[LosingStreak] Win recorded. Resetting streak from {self._consecutive_losses} to 0")
            self._consecutive_losses = 0

    def _record_virtual_trade(self, signal):
        """
        Record a virtual trade during losing streak pause (Phase 9).

        Used to track what trades WOULD have been taken.
        If virtual trades are profitable, resume live trading.
        """
        import json
        from pathlib import Path

        virtual_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bot": signal.bot_name,
            "symbol": signal.symbol,
            "direction": signal.direction,
            "entry": signal.entry_price,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
            "reason": signal.reason,
            "mode": "VIRTUAL",
            "consecutive_losses": self._consecutive_losses,
        }

        virtual_file = Path("/app/data/hydra/ftmo/virtual_trades.jsonl")
        virtual_file.parent.mkdir(parents=True, exist_ok=True)

        with open(virtual_file, "a") as f:
            f.write(json.dumps(virtual_record) + "\n")

        logger.debug(f"[LosingStreak] Virtual trade recorded: {signal.symbol} {signal.direction}")

    def record_virtual_outcome(self, pnl: float):
        """
        Record outcome of a virtual trade (Phase 9).

        Called by close monitor when tracking virtual trades.
        If virtual trade would have been profitable, increment win counter.
        Resume live trading after enough virtual wins.
        """
        if not self._virtual_mode:
            return

        if pnl > 0:
            self._virtual_win_count += 1
            logger.info(f"[LosingStreak] Virtual WIN (${pnl:.2f}). Win count: {self._virtual_win_count}/{self._virtual_wins_to_resume}")

            if self._virtual_win_count >= self._virtual_wins_to_resume:
                self._virtual_mode = False
                self._consecutive_losses = 0
                self._virtual_win_count = 0
                logger.warning(
                    f"[LosingStreak] RESUMING LIVE TRADING after {self._virtual_wins_to_resume} virtual win(s)"
                )
                try:
                    from libs.notifications.telegram_bot import send_telegram_message
                    send_telegram_message(
                        f"✅ [LosingStreak] Resuming LIVE TRADING after successful virtual trades"
                    )
                except Exception:
                    pass
        else:
            logger.info(f"[LosingStreak] Virtual LOSS (${pnl:.2f}). Still in virtual mode.")

    def _status_reporter_loop(self):
        """Print periodic status updates."""
        while self._running:
            try:
                time.sleep(60)  # Every minute
                self._print_status()
            except Exception:
                pass

    def _print_status(self):
        """Print system status."""
        if not self._event_bus:
            return

        bus_stats = self._event_bus.get_stats()

        status_lines = [
            "",
            f"[{datetime.now().strftime('%H:%M:%S')}] Event Bus Status:",
            f"  Ticks/sec: {bus_stats.get('ticks_per_second', 0):.1f}",
            f"  Total ticks: {bus_stats.get('ticks_received', 0):,}",
            f"  Heartbeat age: {bus_stats.get('last_heartbeat_age', 'N/A')}s",
            f"  Kill switch: {'ACTIVE' if self._kill_switch else 'inactive'}",
            f"  Virtual mode: {'ACTIVE' if self._virtual_mode else 'inactive'} (losses: {self._consecutive_losses})",
            f"  Daily trades: {self._daily_trade_count}/{self._max_daily_trades}",
        ]

        # Bot statuses
        for name, config in self._bot_wrappers.items():
            wrapper = config["wrapper"]
            ws = wrapper.get_status()
            status_lines.append(
                f"  {name}: ticks={ws['tick_count']}, session={'Y' if ws['in_session'] else 'N'}"
            )

        print("\n".join(status_lines))

    def _get_bot_names(self) -> List[str]:
        """Get list of bot names."""
        return [
            "gold_london", "eurusd", "us30", "nas100", "gold_ny", "hf_scalper"
        ]

    def run(self):
        """Run the event-driven trading system."""
        if not self.start():
            logger.error("[EventRunner] Failed to start")
            return

        # Main loop - just keep running
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.stop()

    def stop(self):
        """Stop the system."""
        self._running = False

        # Stop close monitor
        if self._close_monitor:
            self._close_monitor.stop()
            logger.info("[EventRunner] Close monitor stopped")

        # Stop bots
        for config in self._bot_wrappers.values():
            config["wrapper"].stop()

        # Stop event bus
        if self._event_bus:
            self._event_bus.stop()

        # Disconnect ZMQ
        if self._zmq_client:
            self._zmq_client.disconnect()

        logger.info("[EventRunner] Stopped")


def main():
    parser = argparse.ArgumentParser(description="FTMO Event-Driven Trading Runner")
    parser.add_argument("--live", action="store_true", help="Enable LIVE trading")
    parser.add_argument("--turbo", action="store_true", help="Enable TURBO mode")
    parser.add_argument("--metrics-port", type=int, default=9100, help="Prometheus metrics port")
    args = parser.parse_args()

    paper_mode = not args.live

    if not paper_mode:
        print("=" * 70)
        print("  WARNING: LIVE TRADING MODE")
        print("  Real money will be used!")
        print("  Press Ctrl+C within 10 seconds to abort...")
        print("=" * 70)
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)

    # Setup signal handlers
    runner = FTMOEventRunner(
        paper_mode=paper_mode,
        metrics_port=args.metrics_port,
        turbo_mode=args.turbo
    )

    def shutdown(signum, frame):
        print("\nShutdown signal received...")
        runner.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    runner.run()


if __name__ == "__main__":
    main()
