#!/usr/bin/env python3
"""Diagnose Telegram bot connection issues."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import requests
from loguru import logger

from libs.config.config import Settings


def diagnose_telegram():
    """Diagnose Telegram bot connection."""
    config = Settings()
    
    print("🔍 Telegram Bot Diagnosis")
    print("=" * 70)
    print()
    
    # Check config
    if not config.telegram_token:
        print("❌ TELEGRAM_TOKEN not set in .env")
        return False
    if not config.telegram_chat_id:
        print("❌ TELEGRAM_CHAT_ID not set in .env")
        return False
    
    print(f"✅ Token: {config.telegram_token[:15]}...")
    print(f"✅ Chat ID: {config.telegram_chat_id}")
    print()
    
    # Test bot API directly
    try:
        # Get bot info
        bot_info_url = f"https://api.telegram.org/bot{config.telegram_token}/getMe"
        resp = requests.get(bot_info_url, timeout=10)
        if resp.status_code == 200:
            bot_info = resp.json()
            if bot_info.get("ok"):
                bot_data = bot_info["result"]
                print(f"✅ Bot verified: @{bot_data.get('username')} ({bot_data.get('first_name')})")
            else:
                print(f"❌ Bot API error: {bot_info.get('description')}")
                return False
        else:
            print(f"❌ HTTP error: {resp.status_code}")
            return False
        
        # Get updates (to find chat ID)
        updates_url = f"https://api.telegram.org/bot{config.telegram_token}/getUpdates"
        resp = requests.get(updates_url, timeout=10)
        if resp.status_code == 200:
            updates = resp.json()
            if updates.get("ok"):
                results = updates.get("result", [])
                if results:
                    print(f"\n📋 Found {len(results)} update(s) in bot's history")
                    print("\nChat IDs found in updates:")
                    chat_ids = set()
                    for update in results:
                        if "message" in update:
                            chat = update["message"].get("chat", {})
                            chat_id = chat.get("id")
                            chat_type = chat.get("type", "unknown")
                            username = chat.get("username", "N/A")
                            first_name = chat.get("first_name", "N/A")
                            if chat_id:
                                chat_ids.add(chat_id)
                                print(f"   • Chat ID: {chat_id} (type: {chat_type}, user: {first_name} @{username})")
                    
                    print(f"\n📊 Your configured chat ID: {config.telegram_chat_id}")
                    if str(config.telegram_chat_id) in [str(cid) for cid in chat_ids]:
                        print("✅ Your chat ID matches one found in updates!")
                    else:
                        print("⚠️  Your chat ID does NOT match any found in updates")
                        print("\n💡 Possible issues:")
                        print("   1. You haven't sent a message to the bot yet")
                        print("   2. The chat ID is incorrect")
                        print("   3. You need to start a chat with the bot first")
                        if chat_ids:
                            print(f"\n💡 Try using one of these chat IDs: {list(chat_ids)}")
                else:
                    print("\n⚠️  No updates found. This means:")
                    print("   1. You haven't sent any messages to the bot yet")
                    print("   2. Start a chat with your bot (@your_bot_username)")
                    print("   3. Send any message (e.g., '/start') to the bot")
                    print("   4. Then rerun this diagnostic")
        
        # Try to send a message to test
        send_url = f"https://api.telegram.org/bot{config.telegram_token}/sendMessage"
        payload = {
            "chat_id": config.telegram_chat_id,
            "text": "🧪 Test message from diagnostic script"
        }
        resp = requests.post(send_url, json=payload, timeout=10)
        if resp.status_code == 200:
            result = resp.json()
            if result.get("ok"):
                print("\n✅ Successfully sent test message!")
                print("   Check your Telegram chat to confirm receipt")
                return True
            else:
                error_desc = result.get("description", "Unknown error")
                error_code = result.get("error_code", "N/A")
                print(f"\n❌ Failed to send message: {error_desc} (code: {error_code})")
                if "chat not found" in error_desc.lower():
                    print("\n💡 'Chat not found' usually means:")
                    print("   1. You haven't started a chat with the bot")
                    print("   2. The chat ID is incorrect")
                    print("   3. The bot is blocked")
                    print("\n   Solution:")
                    print("   1. Open Telegram")
                    print("   2. Search for your bot by username")
                    print("   3. Start a chat and send '/start'")
                    print("   4. Rerun this diagnostic to get the correct chat ID")
        else:
            print(f"\n❌ HTTP error when sending: {resp.status_code}")
            try:
                result = resp.json()
                error_desc = result.get("description", "Unknown error")
                print(f"   Error: {error_desc}")
            except:
                print(f"   Response: {resp.text[:200]}")
                    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return False


if __name__ == "__main__":
    try:
        success = diagnose_telegram()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        sys.exit(1)
