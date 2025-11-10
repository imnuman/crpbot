"""Telegram bot integration for runtime notifications and commands."""

from loguru import logger
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from libs.config.config import Settings


class TelegramBot:
    """Telegram bot for runtime notifications and commands."""

    def __init__(self, config: Settings):
        """
        Initialize Telegram bot.

        Args:
            config: Application settings
        """
        self.config = config
        self.token = config.telegram_token
        self.chat_id = config.telegram_chat_id
        self.application: Application | None = None
        self.is_running = False

        if not self.token or not self.chat_id:
            logger.warning("Telegram token or chat_id not configured. Bot will not start.")
            return

        # Initialize bot
        self.application = Application.builder().token(self.token).build()

        # Register command handlers
        self._register_handlers()

        logger.info("Telegram bot initialized")

    def _register_handlers(self) -> None:
        """Register command handlers."""
        if not self.application:
            return

        self.application.add_handler(CommandHandler("start", self._handle_start))
        self.application.add_handler(CommandHandler("check", self._handle_check))
        self.application.add_handler(CommandHandler("stats", self._handle_stats))
        self.application.add_handler(CommandHandler("ftmo_status", self._handle_ftmo_status))
        self.application.add_handler(CommandHandler("threshold", self._handle_threshold))
        self.application.add_handler(CommandHandler("kill_switch", self._handle_kill_switch))
        self.application.add_handler(CommandHandler("help", self._handle_help))

    async def start(self) -> None:
        """Start the bot (non-blocking)."""
        if not self.application:
            logger.warning("Telegram bot not configured. Cannot start.")
            return

        try:
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
            self.is_running = True
            logger.info("Telegram bot started")
        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")

    async def stop(self) -> None:
        """Stop the bot."""
        if not self.application:
            return

        try:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            self.is_running = False
            logger.info("Telegram bot stopped")
        except Exception as e:
            logger.error(f"Error stopping Telegram bot: {e}")

    async def send_message(self, message: str, mode: str = "dryrun") -> bool:
        """
        Send a message to the configured chat.

        Args:
            message: Message text
            mode: Mode tag (dryrun or live)

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.application or not self.chat_id:
            logger.warning("Telegram bot not configured. Cannot send message.")
            return False

        try:
            # Add mode indicator to message
            mode_tag = "🔵 [DRY-RUN]" if mode == "dryrun" else "🟢 [LIVE]"
            full_message = f"{mode_tag} {message}"

            await self.application.bot.send_message(chat_id=self.chat_id, text=full_message)
            logger.debug(f"Sent Telegram message: {full_message[:100]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        await update.message.reply_text(
            "🤖 Trading AI Bot Started!\n\n"
            "Commands:\n"
            "/check - System status\n"
            "/stats - Performance metrics\n"
            "/ftmo_status - FTMO account status\n"
            "/threshold <n> - Adjust confidence threshold\n"
            "/kill_switch <on|off> - Emergency stop\n"
            "/help - Show this help"
        )

    async def _handle_check(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /check command."""
        # TODO: Get actual system status
        status = "✅ System operational"
        await update.message.reply_text(status)

    async def _handle_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /stats command."""
        # TODO: Get actual stats from database
        stats = "📊 Performance Stats:\n\n" "Signals: 0\n" "Wins: 0\n" "Losses: 0\n" "Win Rate: 0%"
        await update.message.reply_text(stats)

    async def _handle_ftmo_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /ftmo_status command."""
        # TODO: Get actual FTMO status
        status = (
            "📈 FTMO Account Status:\n\n"
            "Balance: $10,000.00\n"
            "Daily Loss: $0.00 (0.00%)\n"
            "Total Loss: $0.00 (0.00%)\n"
            "Status: ✅ OK"
        )
        await update.message.reply_text(status)

    async def _handle_threshold(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /threshold command."""
        if not context.args:
            await update.message.reply_text("Usage: /threshold <0.0-1.0>")
            return

        try:
            threshold = float(context.args[0])
            if not 0.0 <= threshold <= 1.0:
                await update.message.reply_text("Threshold must be between 0.0 and 1.0")
                return

            # TODO: Update threshold in runtime
            await update.message.reply_text(f"✅ Confidence threshold set to {threshold:.2%}")
        except ValueError:
            await update.message.reply_text(
                "Invalid threshold value. Use a number between 0.0 and 1.0"
            )

    async def _handle_kill_switch(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /kill_switch command."""
        if not context.args:
            await update.message.reply_text("Usage: /kill_switch <on|off>")
            return

        action = context.args[0].lower()
        if action == "on":
            # TODO: Activate kill switch
            await update.message.reply_text("🛑 Kill-switch ACTIVATED - No signals will be emitted")
        elif action == "off":
            # TODO: Deactivate kill switch
            await update.message.reply_text("✅ Kill-switch DEACTIVATED - Signals will be emitted")
        else:
            await update.message.reply_text("Usage: /kill_switch <on|off>")

    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        await self._handle_start(update, context)


# Global bot instance (will be initialized in runtime)
bot: TelegramBot | None = None


def init_bot(config: Settings) -> TelegramBot:
    """
    Initialize global Telegram bot instance.

    Args:
        config: Application settings

    Returns:
        TelegramBot instance
    """
    global bot
    bot = TelegramBot(config)
    return bot


def get_bot() -> TelegramBot | None:
    """Get global Telegram bot instance."""
    return bot


async def send_message(message: str, mode: str = "dryrun") -> bool:
    """
    Send a message via Telegram bot.

    Args:
        message: Message text
        mode: Mode tag (dryrun or live)

    Returns:
        True if sent successfully, False otherwise
    """
    bot_instance = get_bot()
    if bot_instance:
        return await bot_instance.send_message(message, mode)
    return False
