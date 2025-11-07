# Telegram Bot Setup Guide

## Quick Setup

### Step 1: Create a Bot
1. Open Telegram and search for **@BotFather**
2. Send `/newbot` command
3. Follow instructions to name your bot
4. Copy the **bot token** (looks like: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

### Step 2: Get Your Chat ID
You need a **numeric chat ID**, not a username.

#### Option A: Get Updates (Recommended)
1. Start a chat with your bot (search for your bot by username)
2. Send any message to the bot
3. Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
4. Look for `"chat":{"id":123456789}` in the response
5. Use that number as your `TELEGRAM_CHAT_ID`

#### Option B: Use @userinfobot
1. Start a chat with **@userinfobot**
2. It will reply with your user ID
3. Use that number as your `TELEGRAM_CHAT_ID`

### Step 3: Configure Environment Variables
Add to your `.env` file:
```bash
TELEGRAM_TOKEN=8425324139:AAGXmo2h3_4xTbkMW-TiASELOlWtMryN5ho
TELEGRAM_CHAT_ID=123456789  # Must be numeric, not username
```

**Important**: 
- Chat ID must be **numeric** (e.g., `123456789` or `-1001234567890` for groups)
- If you see a username like `trading_47_bot`, that's the bot's username, not your chat ID
- Use the numeric ID from `getUpdates` API response

### Step 4: Test the Bot
```bash
# Test Telegram bot connection
python scripts/test_telegram_bot.py
```

You should receive a test message in your Telegram chat.

## Troubleshooting

### Error: "Chat not found"
- **Cause**: Chat ID is incorrect or bot hasn't been started
- **Solution**: 
  1. Start a chat with your bot first
  2. Send a message to the bot
  3. Get your chat ID from `getUpdates` API
  4. Ensure chat ID is numeric

### Error: "Unauthorized"
- **Cause**: Bot token is incorrect
- **Solution**: 
  1. Verify token from @BotFather
  2. Check for typos or extra spaces
  3. Ensure token starts with numbers and colon (e.g., `123456789:ABC...`)

### Error: "Forbidden"
- **Cause**: Bot is blocked or doesn't have permission
- **Solution**: 
  1. Unblock the bot in Telegram
  2. Start a new chat with the bot
  3. Ensure bot is active (not deleted)

### Chat ID is a Username
- **Problem**: Using `@trading_47_bot` instead of numeric ID
- **Solution**: 
  1. Get numeric chat ID from `getUpdates` API
  2. Use numeric ID (e.g., `123456789`)
  3. Usernames don't work for direct messages

## Verification

### Check Bot Status
```bash
# Test connection
python scripts/test_telegram_bot.py
```

Expected output:
```
✅ Telegram token configured: 8425324139...
✅ Telegram chat ID configured: 123456789
✅ Bot initialized successfully
✅ Bot started successfully
✅ Test message sent successfully!
✅ Bot info retrieved: @your_bot (Your Bot Name)
✅ Telegram bot test PASSED!
```

### Check Bot Info via API
Visit in browser:
```
https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getMe
```

Should return:
```json
{
  "ok": true,
  "result": {
    "id": 8425324139,
    "is_bot": true,
    "first_name": "Your Bot Name",
    "username": "your_bot_username"
  }
}
```

## Bot Commands

Once set up, your bot supports these commands:

- `/start` - Start bot and show help
- `/check` - System status
- `/stats` - Performance metrics
- `/ftmo_status` - FTMO account status
- `/threshold <0.0-1.0>` - Adjust confidence threshold
- `/kill_switch <on|off>` - Emergency stop
- `/help` - Show help

## Runtime Integration

The Telegram bot is automatically integrated with the runtime:

- **Dry-run mode**: All messages tagged with 🔵 [DRY-RUN]
- **Live mode**: All messages tagged with 🟢 [LIVE]
- **Signal notifications**: Formatted trading signals
- **Status updates**: System status and alerts

## Security Notes

- **Never commit** `.env` file to Git (already in `.gitignore`)
- Keep bot token secret
- Use different bots for development and production
- Regularly rotate tokens if compromised

## Next Steps

After setting up Telegram bot:
1. Test with `python scripts/test_telegram_bot.py`
2. Run runtime in dry-run mode: `RUNTIME_MODE=dryrun python apps/runtime/main.py`
3. Check Telegram for notifications
4. Test commands: `/start`, `/check`, `/stats`

## References

- [Telegram Bot API Documentation](https://core.telegram.org/bots/api)
- [python-telegram-bot Documentation](https://python-telegram-bot.org/)
- [Get Updates API](https://core.telegram.org/bots/api#getupdates)

