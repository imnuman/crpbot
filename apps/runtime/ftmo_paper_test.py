#!/usr/bin/env python3
"""
FTMO Paper Test Runner

Tests disabled bots (gold_ny, nas100, hf_scalper) in paper mode
to build track record before re-enabling in live.

Usage:
    python apps/runtime/ftmo_paper_test.py

Results saved to: data/hydra/ftmo/paper_test_results.jsonl
"""

import json
import os
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:<8} | {message}", level="INFO")


class PaperTestRunner:
    """Run disabled bots in paper mode to build track record."""

    def __init__(self):
        self.results_file = Path("data/hydra/ftmo/paper_test_results.jsonl")
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        self._running = False
        self._signals = []
        self._lock = threading.Lock()

    def _handle_signal(self, signal):
        """Handle paper signal from bot."""
        with self._lock:
            result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "bot": signal.bot_name if hasattr(signal, 'bot_name') else "unknown",
                "symbol": signal.symbol if hasattr(signal, 'symbol') else "unknown",
                "direction": signal.direction if hasattr(signal, 'direction') else "unknown",
                "entry_price": signal.entry_price if hasattr(signal, 'entry_price') else 0,
                "stop_loss": signal.stop_loss if hasattr(signal, 'stop_loss') else 0,
                "take_profit": signal.take_profit if hasattr(signal, 'take_profit') else 0,
                "confidence": signal.confidence if hasattr(signal, 'confidence') else 0,
                "mode": "PAPER_TEST"
            }
            self._signals.append(result)

            # Log signal
            logger.info(
                f"[PAPER] {result['bot']}: {result['direction']} {result['symbol']} "
                f"@ {result['entry_price']:.2f} (conf: {result['confidence']:.2f})"
            )

            # Save to file
            with open(self.results_file, 'a') as f:
                f.write(json.dumps(result) + '\n')

    def start(self):
        """Start paper testing disabled bots."""
        print("=" * 60)
        print("  FTMO Paper Test Runner - Testing Disabled Bots")
        print("=" * 60)
        print()
        print("  Testing: gold_ny, nas100, hf_scalper")
        print("  Mode: PAPER (no real trades)")
        print(f"  Results: {self.results_file}")
        print()

        try:
            from libs.hydra.ftmo_bots import (
                get_gold_ny_bot,
                get_nas100_bot,
                get_hf_scalper
            )
            from libs.hydra.ftmo_bots.event_bot_wrapper import (
                EventBotWrapper,
                MultiSymbolBotWrapper
            )
            from libs.hydra.ftmo_bots.event_bus import FTMOEventBus

            # Initialize bots in PAPER mode
            paper_mode = True

            bots = {}

            # Gold NY
            gold_ny = get_gold_ny_bot(paper_mode)
            bots["gold_ny"] = {
                "wrapper": EventBotWrapper(bot=gold_ny, on_signal=self._handle_signal),
                "symbol": "XAUUSD",
                "multi": False
            }
            print(f"  - gold_ny: XAUUSD")

            # NAS100
            nas100 = get_nas100_bot(paper_mode)
            bots["nas100"] = {
                "wrapper": EventBotWrapper(bot=nas100, on_signal=self._handle_signal),
                "symbol": "US100.cash",
                "multi": False
            }
            print(f"  - nas100: US100.cash")

            # HF Scalper (multi-symbol)
            hf_bot = get_hf_scalper(paper_mode, turbo_mode=True)
            hf_symbols = ["XAUUSD", "EURUSD"]  # Only test 2 symbols
            bots["hf_scalper"] = {
                "wrapper": MultiSymbolBotWrapper(
                    bot=hf_bot,
                    symbols=hf_symbols,
                    on_signal=self._handle_signal
                ),
                "symbol": hf_symbols,
                "multi": True
            }
            print(f"  - hf_scalper: {hf_symbols}")

            print(f"\n  {len(bots)} bots initialized for paper testing")

            # Initialize event bus
            event_bus = FTMOEventBus(
                host="45.82.167.195",
                port=5556,
                use_tunnel=True
            )

            if not event_bus.start():
                logger.error("Failed to start event bus")
                return False

            print("  Event bus connected")

            # Subscribe bots
            for name, bot_info in bots.items():
                symbols = bot_info["symbol"] if isinstance(bot_info["symbol"], list) else [bot_info["symbol"]]
                for symbol in symbols:
                    event_bus.subscribe(symbol, bot_info["wrapper"].on_tick)
                print(f"  Subscribed {name}")

            print()
            print("=" * 60)
            print("  Paper Testing Active - Press Ctrl+C to stop")
            print("=" * 60)

            self._running = True

            # Main loop with status reports
            last_report = time.time()
            while self._running:
                time.sleep(1)

                # Status report every 60 seconds
                if time.time() - last_report > 60:
                    with self._lock:
                        signal_count = len(self._signals)
                    logger.info(f"[STATUS] Paper signals generated: {signal_count}")
                    last_report = time.time()

        except KeyboardInterrupt:
            print("\n\nStopping paper test...")
            self._running = False
        except Exception as e:
            logger.error(f"Paper test error: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Print summary
        print()
        print("=" * 60)
        print("  Paper Test Summary")
        print("=" * 60)
        with self._lock:
            print(f"  Total signals: {len(self._signals)}")
            by_bot = {}
            for s in self._signals:
                bot = s.get('bot', 'unknown')
                by_bot[bot] = by_bot.get(bot, 0) + 1
            for bot, count in sorted(by_bot.items()):
                print(f"    - {bot}: {count} signals")
        print(f"  Results saved to: {self.results_file}")

        return True


if __name__ == "__main__":
    runner = PaperTestRunner()
    runner.start()
