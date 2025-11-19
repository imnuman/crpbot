#!/usr/bin/env python3
"""
V7 Ultimate - Telegram Bot Command Listener

Runs the Telegram bot in standalone mode to listen for V7 control commands.

Commands supported:
- /v7_status - Show V7 runtime status
- /v7_start - Start V7 runtime
- /v7_stop - Stop V7 runtime
- /v7_stats - Show detailed V7 statistics
- /v7_config - Adjust V7 parameters

Usage:
    python apps/runtime/v7_telegram_bot_runner.py

Or run in background:
    nohup python apps/runtime/v7_telegram_bot_runner.py > /tmp/v7_telegram_bot.log 2>&1 &
"""

import asyncio
import signal
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger
from libs.config.config import Settings
from apps.runtime.telegram_bot import TelegramBot


class TelegramBotRunner:
    """Runner for V7 Telegram bot command listener"""

    def __init__(self):
        """Initialize bot runner"""
        # Load config
        self.config = Settings()

        # Validate Telegram credentials
        if not self.config.telegram_token or not self.config.telegram_chat_id:
            logger.error("Telegram token or chat_id not configured in .env")
            logger.error("Please set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID")
            sys.exit(1)

        # Initialize bot
        self.bot = TelegramBot(self.config)

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.running = False

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    async def run(self):
        """Run the bot"""
        try:
            logger.info("=" * 80)
            logger.info("V7 ULTIMATE - TELEGRAM BOT COMMAND LISTENER")
            logger.info("=" * 80)
            logger.info(f"Chat ID: {self.config.telegram_chat_id}")
            logger.info("Listening for commands...")
            logger.info("")
            logger.info("Available commands:")
            logger.info("  /v7_status  - Show V7 runtime status")
            logger.info("  /v7_start   - Start V7 runtime")
            logger.info("  /v7_stop    - Stop V7 runtime")
            logger.info("  /v7_stats   - Show detailed V7 statistics")
            logger.info("  /v7_config  - Adjust V7 parameters")
            logger.info("=" * 80)

            # Start bot
            await self.bot.start()
            self.running = True

            # Send startup notification
            startup_msg = (
                "🤖 <b>V7 Telegram Bot - ONLINE</b>\n\n"
                "Command listener is now active.\n\n"
                "Available commands:\n"
                "• /v7_status - Runtime status\n"
                "• /v7_start - Start V7\n"
                "• /v7_stop - Stop V7\n"
                "• /v7_stats - Statistics\n"
                "• /v7_config - Configuration\n"
                "• /help - Full command list"
            )
            await self.bot.application.bot.send_message(
                chat_id=self.config.telegram_chat_id,
                text=startup_msg,
                parse_mode='HTML'
            )

            # Keep running until shutdown signal
            while self.running:
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error running Telegram bot: {e}")
            raise

        finally:
            # Shutdown bot
            logger.info("Shutting down Telegram bot...")
            await self.bot.stop()
            logger.info("Telegram bot stopped")


def main():
    """Main entry point"""
    runner = TelegramBotRunner()

    try:
        # Run bot in event loop
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, exiting...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
