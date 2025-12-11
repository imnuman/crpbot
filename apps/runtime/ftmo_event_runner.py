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
            # Re-enabled gold_ny on 2025-12-10 after data analysis showed $8.79/trade expectancy
            bots_config = [
                ("gold_london", get_gold_london_bot(self.paper_mode, turbo_mode=self.turbo_mode), "XAUUSD"),
                ("eurusd", get_eurusd_bot(self.paper_mode), "EURUSD"),
                ("us30", get_us30_bot(self.paper_mode), "US30.cash"),
                ("gold_ny", get_gold_ny_bot(self.paper_mode), "XAUUSD"),  # RE-ENABLED: $8.79/trade expectancy
                # PAPER MODE TESTING (2025-12-11):
                # - London Breakout v3: Bug fixes for SL placement + SMA filter
                # - NAS100 Gap v2: Stricter 0.6% threshold + ADR filter
                ("london_eur", get_london_breakout_bot("EURUSD", paper_mode=True), "EURUSD"),  # PAPER: v3 bug fixes
                ("nas100", get_nas100_bot(paper_mode=True), "US100.cash"),  # PAPER: v2 improvements
            ]
            print("    - london_eur: PAPER MODE (v3 bug fixes)")
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

    def _init_event_bus(self) -> bool:
        """Initialize the event bus for price streaming."""
        try:
            from libs.hydra.ftmo_bots.event_bus import FTMOEventBus

            self._event_bus = FTMOEventBus(use_ssh_tunnel=True)

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

        with self._signal_lock:
            self._signal_queue.append(signal)
            logger.info(f"[EventRunner] Signal queued: {signal.bot_name} {signal.direction} {signal.symbol}")

    def _handle_heartbeat(self, heartbeat):
        """Handle heartbeat from price streamer."""
        if self._metrics:
            self._metrics.set_connection_status("price_stream", heartbeat.mt5_connected)

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

        except Exception as e:
            logger.error(f"[EventRunner] Risk check error: {e}")

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
            self._last_reset_date = now
            logger.info("[PerBotRisk] Daily P&L reset at midnight UTC")

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

        # NEWS FILTER: Avoid trading around high-impact news events (Phase 3 - 2025-12-11)
        try:
            from libs.hydra.ftmo_bots.news_filter import should_avoid_news
            avoid, reason = should_avoid_news(signal.symbol)
            if avoid:
                logger.warning(f"[NewsFilter] Trade blocked: {signal.bot_name} {signal.symbol} - {reason}")
                return
        except ImportError:
            pass  # News filter not available

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

        trade_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bot": signal.bot_name,
            "symbol": signal.symbol,
            "direction": signal.direction,
            "entry": signal.entry_price,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
            "lot": signal.lot_size,
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

        result = self._zmq_client.trade(
            symbol=signal.symbol,
            direction=signal.direction.upper(),  # ZMQ expects uppercase BUY/SELL
            volume=signal.lot_size,
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
