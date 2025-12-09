#!/usr/bin/env python3
"""
FTMO Multi-Bot Paper Trading Runner

Runs the FTMO orchestrator in paper trading mode with ZMQ MT5 connection.
Uses real MT5 prices but simulates trades locally.

Usage:
    python ftmo_runner.py [--live]  # --live for real trading (requires explicit flag)
"""

import sys
import os
import signal
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "/tmp/ftmo_runner_{time:YYYYMMDD}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG"
)


def main():
    parser = argparse.ArgumentParser(description="FTMO Multi-Bot Runner")
    parser.add_argument("--live", action="store_true", help="Enable LIVE trading (real money!)")
    parser.add_argument("--interval", type=int, default=60, help="Cycle interval in seconds")
    parser.add_argument("--turbo", action="store_true", help="Enable TURBO mode (more trades, lower thresholds)")
    args = parser.parse_args()

    paper_mode = not args.live
    turbo_mode = args.turbo

    print("=" * 60)
    print("    FTMO Multi-Bot Trading System")
    print("=" * 60)
    print()
    print(f"  Mode: {'📋 PAPER TRADING (Simulation)' if paper_mode else '💰 LIVE TRADING (Real Money!)'}")
    if turbo_mode:
        print("  ⚡ TURBO MODE ENABLED - More trades, lower thresholds")
    print(f"  Cycle Interval: {args.interval}s")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    if not paper_mode:
        print("⚠️  WARNING: LIVE TRADING MODE - Real money will be used!")
        print("    Press Ctrl+C within 10 seconds to abort...")
        import time
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            print("\n❌ Aborted.")
            sys.exit(0)

    # Test ZMQ connection first
    print("[1/3] Testing MT5 ZMQ connection...")
    try:
        from libs.brokers.mt5_zmq_client import MT5ZMQClient
        client = MT5ZMQClient()
        if not client.connect():
            logger.error("Failed to connect to MT5 via ZMQ")
            print("❌ MT5 connection failed! Make sure ZMQ executor is running on Windows.")
            sys.exit(1)

        result = client.ping()
        if not result.get("success"):
            logger.error(f"MT5 ping failed: {result}")
            print("❌ MT5 not responding!")
            sys.exit(1)

        account = client.get_account()
        if account:
            print(f"✅ MT5 Connected - Balance: ${account.get('balance', 0):,.2f} {account.get('currency', 'USD')}")

        client.disconnect()
    except Exception as e:
        logger.error(f"ZMQ connection error: {e}")
        print(f"❌ ZMQ Error: {e}")
        sys.exit(1)

    # Initialize orchestrator
    print("[2/3] Initializing FTMO Orchestrator...")
    try:
        from libs.hydra.ftmo_bots import FTMOOrchestrator
        orchestrator = FTMOOrchestrator(paper_mode=paper_mode, enable_metalearning=True, use_zmq=True, turbo_mode=turbo_mode)

        print(f"✅ Orchestrator ready - {len(orchestrator.bots)} bots loaded")
        for bot_name in orchestrator.bots.keys():
            print(f"   - {bot_name}")
    except Exception as e:
        logger.error(f"Orchestrator init error: {e}")
        print(f"❌ Orchestrator Error: {e}")
        sys.exit(1)

    # Setup shutdown handler
    def shutdown(signum, frame):
        print("\n🛑 Shutting down...")
        orchestrator.running = False
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Start trading
    print("[3/3] Starting trading loop...")
    print()
    print("=" * 60)
    print("  Trading Active - Press Ctrl+C to stop")
    print("=" * 60)
    print()

    try:
        orchestrator.run(cycle_interval_seconds=args.interval)
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        logger.exception(f"Runtime error: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
