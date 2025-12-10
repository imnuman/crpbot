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

        # Signal queue for trade execution
        self._signal_queue: List[Any] = []
        self._signal_lock = threading.Lock()

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
                get_hf_scalper
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
                # DISABLED: nas100 (47.6% WR, only $0.30/trade - negligible P&L)
            ]
            print("    - nas100: DISABLED (47.6% WR, only $0.30/trade)")

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

            # HF Scalper - DISABLED until proven profitable
            # Lost $503 on 2025-12-10 due to over-trading without track record
            # TODO: Re-enable after paper testing shows positive expectancy
            # hf_bot = get_hf_scalper(self.paper_mode, turbo_mode=self.turbo_mode)
            # hf_symbols = ["XAUUSD", "EURUSD", "US30.cash", "US100.cash"]
            # hf_wrapper = MultiSymbolBotWrapper(
            #     bot=hf_bot,
            #     symbols=hf_symbols,
            #     on_signal=self._handle_signal
            # )
            # self._bot_wrappers["hf_scalper"] = {
            #     "wrapper": hf_wrapper,
            #     "symbol": hf_symbols,
            #     "multi": True
            # }
            print(f"    - hf_scalper: DISABLED (unproven)")

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

        logger.info(f"[EventRunner] Live trade executed: ticket={result.get('ticket')}")

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
