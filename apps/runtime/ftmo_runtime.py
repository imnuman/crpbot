"""
FTMO Multi-Bot Runtime

Standalone runtime for FTMO challenge bots with full monitoring integration.
Can run independently or be imported into main HYDRA runtime.

Usage:
    # Standalone
    python apps/runtime/ftmo_runtime.py --paper

    # Or integrated into hydra_runtime.py
    from apps.runtime.ftmo_runtime import FTMORuntime
    ftmo = FTMORuntime(paper_mode=True)
    ftmo.start_background()
"""

import os
import sys
import time
import signal
import argparse
import threading
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv()

from libs.hydra.ftmo_bots import (
    FTMOOrchestrator,
    get_ftmo_orchestrator,
    TradeSignal,
)
from libs.monitoring import MetricsExporter, HydraMetrics
from libs.notifications.telegram_bot import (
    get_telegram_notifier,
    send_telegram_message,
)


class FTMORuntime:
    """
    FTMO Multi-Bot Runtime Manager

    Features:
    - Runs 5 specialized FTMO bots + Engine D integration
    - Unified risk management (4.5% daily, 8.5% total DD limits)
    - Prometheus metrics integration
    - Telegram alerts for all trades
    - Graceful shutdown handling
    """

    def __init__(
        self,
        paper_mode: bool = True,
        cycle_interval: int = 60,
        enable_metrics: bool = True,
        metrics_port: int = 9101,  # Different port from main HYDRA
    ):
        self.paper_mode = paper_mode
        self.cycle_interval = cycle_interval
        self.enable_metrics = enable_metrics
        self.metrics_port = metrics_port

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._orchestrator: Optional[FTMOOrchestrator] = None
        self._metrics_exporter: Optional[MetricsExporter] = None
        self._shutdown_event = threading.Event()

        # Statistics
        self._start_time: Optional[datetime] = None
        self._total_signals = 0
        self._total_trades = 0

        logger.info(f"[FTMORuntime] Initializing (paper_mode={paper_mode})")

    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers."""
        def handler(signum, frame):
            logger.info(f"[FTMORuntime] Received signal {signum}, shutting down...")
            self.stop()

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _init_components(self):
        """Initialize all components."""
        # Initialize orchestrator
        self._orchestrator = get_ftmo_orchestrator(paper_mode=self.paper_mode)

        # Initialize metrics
        if self.enable_metrics:
            try:
                HydraMetrics.initialize()
                self._metrics_exporter = MetricsExporter(port=self.metrics_port)
                self._metrics_exporter.start()
                logger.info(f"[FTMORuntime] Metrics exporter started on port {self.metrics_port}")
            except Exception as e:
                logger.warning(f"[FTMORuntime] Metrics exporter failed: {e}")

        # Send startup notification
        self._send_startup_notification()

    def _send_startup_notification(self):
        """Send Telegram notification on startup."""
        mode = "PAPER" if self.paper_mode else "LIVE"
        bots = list(self._orchestrator.bots.keys())

        message = f"""
🚀 <b>FTMO RUNTIME STARTED</b>

<b>Mode:</b> {mode}
<b>Bots Active:</b> {len(bots)}
• {', '.join(bots)}

<b>Risk Limits:</b>
• Daily DD: {self._orchestrator.limits.max_daily_loss_percent}%
• Total DD: {self._orchestrator.limits.max_total_drawdown_percent}%
• Max Positions: {self._orchestrator.limits.max_concurrent_positions}

<b>Cycle Interval:</b> {self.cycle_interval}s
        """.strip()

        try:
            send_telegram_message(message)
        except Exception as e:
            logger.warning(f"[FTMORuntime] Failed to send startup notification: {e}")

    def _send_shutdown_notification(self):
        """Send Telegram notification on shutdown."""
        uptime = "N/A"
        if self._start_time:
            delta = datetime.now(timezone.utc) - self._start_time
            hours = delta.total_seconds() / 3600
            uptime = f"{hours:.1f} hours"

        message = f"""
🛑 <b>FTMO RUNTIME STOPPED</b>

<b>Uptime:</b> {uptime}
<b>Signals Generated:</b> {self._total_signals}
<b>Trades Executed:</b> {self._total_trades}
        """.strip()

        try:
            send_telegram_message(message)
        except Exception:
            pass

    def _update_metrics(self):
        """Update Prometheus metrics from orchestrator status."""
        if not self.enable_metrics:
            return

        try:
            status = self._orchestrator.get_status()
            HydraMetrics.update_ftmo_from_orchestrator(status)
        except Exception as e:
            logger.warning(f"[FTMORuntime] Failed to update metrics: {e}")

    def _run_cycle(self):
        """Run a single trading cycle."""
        try:
            signals = self._orchestrator.run_single_cycle()

            if signals:
                self._total_signals += len(signals)
                for signal in signals:
                    if signal.executed:
                        self._total_trades += 1
                        logger.info(
                            f"[FTMORuntime] Trade executed: {signal.bot_name} "
                            f"{signal.direction} {signal.symbol} @ {signal.entry_price}"
                        )

                        # Record metric
                        HydraMetrics.record_ftmo_signal(signal.bot_name, signal.direction)

            # Update metrics after each cycle
            self._update_metrics()

        except Exception as e:
            logger.error(f"[FTMORuntime] Cycle error: {e}")

    def _main_loop(self):
        """Main trading loop."""
        logger.info("[FTMORuntime] Starting main loop...")
        self._start_time = datetime.now(timezone.utc)

        while self._running and not self._shutdown_event.is_set():
            cycle_start = time.time()

            self._run_cycle()

            # Calculate sleep time
            elapsed = time.time() - cycle_start
            sleep_time = max(0, self.cycle_interval - elapsed)

            # Use event wait for responsive shutdown
            if sleep_time > 0:
                self._shutdown_event.wait(timeout=sleep_time)

        logger.info("[FTMORuntime] Main loop stopped")

    def start(self):
        """Start the FTMO runtime (blocking)."""
        if self._running:
            logger.warning("[FTMORuntime] Already running")
            return

        self._running = True
        self._setup_signal_handlers()
        self._init_components()

        try:
            self._main_loop()
        finally:
            self._cleanup()

    def start_background(self) -> threading.Thread:
        """Start the FTMO runtime in a background thread."""
        if self._running:
            logger.warning("[FTMORuntime] Already running")
            return self._thread

        self._running = True
        self._init_components()

        self._thread = threading.Thread(
            target=self._main_loop,
            daemon=True,
            name="FTMORuntime"
        )
        self._thread.start()

        logger.info("[FTMORuntime] Started in background thread")
        return self._thread

    def stop(self):
        """Stop the FTMO runtime."""
        if not self._running:
            return

        logger.info("[FTMORuntime] Stopping...")
        self._running = False
        self._shutdown_event.set()

        if self._orchestrator:
            self._orchestrator.stop()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)

        self._cleanup()

    def _cleanup(self):
        """Cleanup resources."""
        self._send_shutdown_notification()

        if self._metrics_exporter:
            self._metrics_exporter.stop()

        logger.info("[FTMORuntime] Cleanup complete")

    def get_status(self) -> dict:
        """Get current runtime status."""
        status = {
            "running": self._running,
            "paper_mode": self.paper_mode,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "total_signals": self._total_signals,
            "total_trades": self._total_trades,
        }

        if self._orchestrator:
            status["orchestrator"] = self._orchestrator.get_status()

        return status

    def set_live_mode(self, enable: bool = True):
        """Switch between paper and live mode."""
        self.paper_mode = not enable
        if self._orchestrator:
            self._orchestrator.set_paper_mode(self.paper_mode)
        logger.info(f"[FTMORuntime] Mode changed to: {'LIVE' if enable else 'PAPER'}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="FTMO Multi-Bot Runtime")
    parser.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Run in paper trading mode (default: True)"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in LIVE trading mode (CAUTION!)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Cycle interval in seconds (default: 60)"
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Disable Prometheus metrics"
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=9101,
        help="Prometheus metrics port (default: 9101)"
    )

    args = parser.parse_args()

    # Determine mode
    paper_mode = not args.live

    if not paper_mode:
        logger.warning("=" * 60)
        logger.warning("  LIVE TRADING MODE - REAL MONEY AT RISK!")
        logger.warning("=" * 60)
        confirm = input("Type 'CONFIRM LIVE' to proceed: ")
        if confirm != "CONFIRM LIVE":
            logger.info("Live mode not confirmed, exiting.")
            return

    # Create and start runtime
    runtime = FTMORuntime(
        paper_mode=paper_mode,
        cycle_interval=args.interval,
        enable_metrics=not args.no_metrics,
        metrics_port=args.metrics_port,
    )

    logger.info("=" * 60)
    logger.info(f"  FTMO MULTI-BOT RUNTIME")
    logger.info(f"  Mode: {'PAPER' if paper_mode else 'LIVE'}")
    logger.info(f"  Cycle: {args.interval}s")
    logger.info("=" * 60)

    runtime.start()


if __name__ == "__main__":
    main()
