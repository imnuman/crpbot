#!/usr/bin/env python3
"""Test Telegram bot connectivity and send a test message."""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from libs.config.config import Settings
from apps.runtime.telegram_bot import init_bot, send_message


async def main():
    """Test Telegram bot."""
    print("=" * 60)
    print("Testing Telegram Bot")
    print("=" * 60)

    # Initialize config and bot
    config = Settings()
    print(f"\nTelegram Token: {config.telegram_token[:20]}..." if config.telegram_token else "Not configured")
    print(f"Telegram Chat ID: {config.telegram_chat_id}")

    # Initialize bot
    bot = init_bot(config)

    if not bot or not bot.token:
        print("\n❌ Telegram bot not configured!")
        return

    print("\n✅ Bot initialized")

    # Start bot
    await bot.start()
    print("✅ Bot started")

    # Send test message
    print("\n📤 Sending test message...")
    success = await send_message(
        "🧪 **Test Message**\n\n"
        "This is a test notification to verify Telegram integration is working correctly.\n\n"
        "If you see this message, notifications are functioning! ✅",
        mode="live"
    )

    if success:
        print("✅ Test message sent successfully!")
    else:
        print("❌ Failed to send test message")

    # Send a mock signal notification
    print("\n📤 Sending mock signal notification...")
    signal_message = (
        "🚨 **TRADING SIGNAL** 🚨\n\n"
        "**Symbol:** TEST-USD\n"
        "**Direction:** LONG\n"
        "**Confidence:** 95.50%\n"
        "**Tier:** HIGH\n"
        "**Entry Price:** $1000.00\n\n"
        "**Model Predictions:**\n"
        "• LSTM: 0.955\n"
        "• Transformer: 0.960\n"
        "• RL: 0.950\n\n"
        "⏰ 2025-11-16 18:05:00"
    )

    success = await send_message(signal_message, mode="live")

    if success:
        print("✅ Mock signal sent successfully!")
    else:
        print("❌ Failed to send mock signal")

    # Stop bot
    await bot.stop()
    print("\n✅ Bot stopped")
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
