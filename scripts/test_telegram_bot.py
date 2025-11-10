#!/usr/bin/env python3
"""Test Telegram bot connection and functionality."""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from apps.runtime.telegram_bot import init_bot
from libs.config.config import Settings


async def test_telegram_bot():
    """Test Telegram bot connection and message sending."""
    logger.info("Testing Telegram bot...")

    # Load config
    config = Settings()
    config.validate()

    # Check if Telegram is configured
    if not config.telegram_token or not config.telegram_chat_id:
        logger.error("❌ Telegram bot not configured!")
        logger.error("   Please set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in .env file")
        logger.error("   See docs/CREDENTIALS_CHECKLIST.md for setup instructions")
        return False

    logger.info(f"✅ Telegram token configured: {config.telegram_token[:10]}...")
    logger.info(f"✅ Telegram chat ID configured: {config.telegram_chat_id}")

    # Note about chat ID format
    if not config.telegram_chat_id.isdigit() and not config.telegram_chat_id.startswith("-"):
        logger.warning(
            "⚠️  Chat ID appears to be a username, not a numeric ID.\n"
            "   Telegram chat IDs should be numeric (e.g., '123456789' or '-1001234567890' for groups).\n"
            "   To get your chat ID:\n"
            "   1. Start a chat with your bot\n"
            "   2. Visit: https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates\n"
            "   3. Look for 'chat': {'id': ...} in the response\n"
            "   4. Use that numeric ID as TELEGRAM_CHAT_ID"
        )

    # Initialize bot
    try:
        bot = init_bot(config)
        logger.info("✅ Bot initialized successfully")

        # Start bot
        await bot.start()
        logger.info("✅ Bot started successfully")

        # Test message sending
        test_message = (
            "🧪 Test message from Trading AI Bot\n\n"
            "This is a test to verify Telegram bot connectivity.\n"
            "If you receive this message, your bot is configured correctly!"
        )
        success = await bot.send_message(test_message, mode="dryrun")

        if success:
            logger.info("✅ Test message sent successfully!")
            logger.info("   Check your Telegram chat to confirm receipt")
        else:
            logger.error("❌ Failed to send test message")
            return False

        # Wait a bit for message to be sent
        await asyncio.sleep(2)

        # Test command handlers (get bot info)
        try:
            bot_info = await bot.application.bot.get_me()
            logger.info(f"✅ Bot info retrieved: @{bot_info.username} ({bot_info.first_name})")
        except Exception as e:
            logger.warning(f"⚠️  Could not retrieve bot info: {e}")

        # Stop bot
        await bot.stop()
        logger.info("✅ Bot stopped successfully")

        logger.info("\n" + "=" * 60)
        logger.info("✅ Telegram bot test PASSED!")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"❌ Telegram bot test FAILED: {e}")
        logger.exception(e)

        # Provide troubleshooting tips
        logger.error("\n💡 Troubleshooting:")
        logger.error("1. Verify TELEGRAM_TOKEN is correct (from @BotFather)")
        logger.error("2. Verify TELEGRAM_CHAT_ID is numeric (not username)")
        logger.error("3. Start a chat with your bot first")
        logger.error("4. Get your chat ID from: https://api.telegram.org/bot<TOKEN>/getUpdates")
        logger.error("5. Check bot permissions and that it's not blocked")

        return False


def main():
    """Main entry point."""
    try:
        success = asyncio.run(test_telegram_bot())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
