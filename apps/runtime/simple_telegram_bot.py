#!/usr/bin/env python3
"""
Simple V7 Telegram Bot with Always-Visible Menu

Ultra-simple bot with custom keyboard that's always visible.
No complex commands - just tap buttons.
"""

import asyncio
import signal
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loguru import logger
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from libs.config.config import Settings
from libs.db.models import Signal, get_session
from sqlalchemy import desc, and_


class SimpleV7Bot:
    """Simple V7 bot with always-visible buttons"""

    def __init__(self, config: Settings):
        self.config = config
        self.token = config.telegram_token
        self.chat_id = config.telegram_chat_id
        self.application = None
        self.running = False

        if not self.token or not self.chat_id:
            logger.error("Telegram token or chat_id not configured")
            sys.exit(1)

        # Build application
        self.application = Application.builder().token(self.token).build()

        # Register handlers
        self.application.add_handler(CommandHandler("start", self.cmd_start))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_button))

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def get_main_keyboard(self):
        """Get the main menu keyboard (always visible)"""
        keyboard = [
            [KeyboardButton("📊 Status"), KeyboardButton("💰 Latest Price")],
            [KeyboardButton("🔔 Recent Signals"), KeyboardButton("📈 Stats")],
            [KeyboardButton("📝 Open Trades"), KeyboardButton("📊 Performance")],
            [KeyboardButton("⏸️ Stop Bot"), KeyboardButton("▶️ Start Bot")],
        ]
        return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        await update.message.reply_text(
            "🚀 <b>V7 Ultimate Bot - Simple Mode</b>\n\n"
            "Tap any button below to get info.\n"
            "Menu buttons stay visible always!",
            parse_mode='HTML',
            reply_markup=self.get_main_keyboard()
        )

    async def handle_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button presses"""
        text = update.message.text

        if text == "📊 Status":
            await self.show_status(update)
        elif text == "💰 Latest Price":
            await self.show_latest_price(update)
        elif text == "🔔 Recent Signals":
            await self.show_recent_signals(update)
        elif text == "📈 Stats":
            await self.show_stats(update)
        elif text == "📝 Open Trades":
            await self.show_open_trades(update)
        elif text == "📊 Performance":
            await self.show_performance(update)
        elif text == "⏸️ Stop Bot":
            await self.stop_v7_runtime(update)
        elif text == "▶️ Start Bot":
            await self.start_v7_runtime(update)
        else:
            await update.message.reply_text(
                "Unknown command. Use the buttons below.",
                reply_markup=self.get_main_keyboard()
            )

    async def show_status(self, update: Update):
        """Show V7 runtime status"""
        import subprocess

        # Check if V7 runtime is running
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        v7_running = 'v7_runtime.py' in result.stdout

        if v7_running:
            msg = "🟢 <b>V7 Runtime: RUNNING</b>\n\n"
        else:
            msg = "🔴 <b>V7 Runtime: STOPPED</b>\n\n"

        # Get latest signal
        session = get_session(self.config.db_url)
        try:
            latest = session.query(Signal).filter(
                Signal.model_version == 'v7_ultimate'
            ).order_by(desc(Signal.timestamp)).first()

            if latest:
                time_ago = datetime.utcnow() - latest.timestamp
                mins = int(time_ago.total_seconds() / 60)
                msg += f"Last Signal: {mins}m ago\n"
                msg += f"Symbol: {latest.symbol}\n"
                msg += f"Signal: {latest.direction.upper()}\n"
                msg += f"Confidence: {latest.confidence*100:.0f}%"
            else:
                msg += "No signals generated yet"
        finally:
            session.close()

        await update.message.reply_text(msg, parse_mode='HTML', reply_markup=self.get_main_keyboard())

    async def show_latest_price(self, update: Update):
        """Show latest market prices"""
        from apps.runtime.data_fetcher import MarketDataFetcher

        fetcher = MarketDataFetcher()
        msg = "💰 <b>Live Market Prices</b>\n\n"

        for symbol in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
            try:
                df = fetcher.fetch_latest_candles(symbol, limit=1)
                if not df.empty:
                    price = df.iloc[-1]['close']
                    msg += f"{symbol}: ${price:,.2f}\n"
            except:
                msg += f"{symbol}: Error fetching\n"

        await update.message.reply_text(msg, parse_mode='HTML', reply_markup=self.get_main_keyboard())

    async def show_recent_signals(self, update: Update):
        """Show recent V7 signals"""
        session = get_session(self.config.db_url)
        try:
            signals = session.query(Signal).filter(
                Signal.model_version == 'v7_ultimate'
            ).order_by(desc(Signal.timestamp)).limit(5).all()

            if not signals:
                msg = "No V7 signals found"
            else:
                msg = f"🔔 <b>Recent Signals ({len(signals)})</b>\n\n"
                for sig in signals:
                    time_str = sig.timestamp.strftime("%H:%M")
                    msg += f"<b>{sig.symbol}</b> {sig.direction.upper()}\n"
                    msg += f"  {time_str} | {sig.confidence*100:.0f}% | {sig.tier}\n\n"

            await update.message.reply_text(msg, parse_mode='HTML', reply_markup=self.get_main_keyboard())
        finally:
            session.close()

    async def show_stats(self, update: Update):
        """Show 24h statistics"""
        session = get_session(self.config.db_url)
        try:
            day_ago = datetime.utcnow() - timedelta(days=1)
            signals = session.query(Signal).filter(
                Signal.model_version == 'v7_ultimate',
                Signal.timestamp > day_ago
            ).all()

            buy_count = sum(1 for s in signals if s.direction == 'long')
            sell_count = sum(1 for s in signals if s.direction == 'short')
            hold_count = sum(1 for s in signals if s.direction == 'hold')

            msg = "📈 <b>Stats (24 Hours)</b>\n\n"
            msg += f"Total Signals: {len(signals)}\n"
            msg += f"BUY: {buy_count}\n"
            msg += f"SELL: {sell_count}\n"
            msg += f"HOLD: {hold_count}\n"

            if signals:
                avg_conf = sum(s.confidence for s in signals) / len(signals)
                msg += f"\nAvg Confidence: {avg_conf*100:.0f}%"

            await update.message.reply_text(msg, parse_mode='HTML', reply_markup=self.get_main_keyboard())
        finally:
            session.close()

    async def stop_v7_runtime(self, update: Update):
        """Stop V7 runtime"""
        import subprocess
        subprocess.run(['pkill', '-f', 'v7_runtime.py'])
        await update.message.reply_text(
            "⏸️ V7 Runtime stopped",
            reply_markup=self.get_main_keyboard()
        )

    async def start_v7_runtime(self, update: Update):
        """Start V7 runtime"""
        import subprocess

        # Check if already running
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'v7_runtime.py' in result.stdout:
            await update.message.reply_text(
                "ℹ️ V7 Runtime already running",
                reply_markup=self.get_main_keyboard()
            )
            return

        # Start runtime
        subprocess.Popen(
            ['/root/crpbot/.venv/bin/python3', 'apps/runtime/v7_runtime.py', '--iterations', '-1', '--sleep-seconds', '300'],
            cwd='/root/crpbot',
            stdout=open('/tmp/v7_runtime.log', 'w'),
            stderr=subprocess.STDOUT,
            start_new_session=True
        )

        await update.message.reply_text(
            "▶️ V7 Runtime started",
            reply_markup=self.get_main_keyboard()
        )

    async def show_open_trades(self, update: Update):
        """Show open paper trades"""
        from apps.runtime.paper_trader import PaperTrader

        try:
            trader = PaperTrader(self.config.db_url)
            open_trades = trader.get_open_trades()

            if not open_trades:
                msg = "📝 <b>No Open Trades</b>\n\nNo paper trades currently open."
            else:
                msg = f"📝 <b>Open Trades ({len(open_trades)})</b>\n\n"
                for trade in open_trades:
                    duration = datetime.utcnow() - trade.execution_time
                    hours = int(duration.total_seconds() / 3600)
                    minutes = int((duration.total_seconds() % 3600) / 60)

                    msg += f"<b>Signal #{trade.signal_id}</b> - {trade.symbol} {trade.direction.upper()}\n"
                    msg += f"  Entry: ${trade.entry_price:.2f}\n"
                    msg += f"  Confidence: {trade.confidence*100:.1f}% ({trade.tier})\n"
                    msg += f"  Duration: {hours}h {minutes}m\n\n"

            await update.message.reply_text(msg, parse_mode='HTML', reply_markup=self.get_main_keyboard())
        except Exception as e:
            await update.message.reply_text(
                f"Error getting open trades: {e}",
                reply_markup=self.get_main_keyboard()
            )

    async def show_performance(self, update: Update):
        """Show paper trading performance"""
        from apps.runtime.paper_trader import PaperTrader

        try:
            trader = PaperTrader(self.config.db_url)
            stats = trader.get_performance_stats(days=30)

            if stats.total_trades == 0:
                msg = "📊 <b>No Performance Data</b>\n\nNo closed trades yet."
            else:
                msg = f"📊 <b>Performance (30 Days)</b>\n\n"
                msg += f"<b>Overall:</b>\n"
                msg += f"  Trades: {stats.total_trades} ({stats.wins}W / {stats.losses}L)\n"
                msg += f"  Win Rate: {stats.win_rate:.1f}%\n"
                msg += f"  Total PnL: {stats.total_pnl:+.2f}%\n"
                msg += f"  Avg Win: +{stats.avg_win:.2f}%\n"
                msg += f"  Avg Loss: {stats.avg_loss:.2f}%\n\n"

                if stats.profit_factor:
                    msg += f"  Profit Factor: {stats.profit_factor:.2f}\n"

            await update.message.reply_text(msg, parse_mode='HTML', reply_markup=self.get_main_keyboard())
        except Exception as e:
            await update.message.reply_text(
                f"Error getting performance: {e}",
                reply_markup=self.get_main_keyboard()
            )

    async def run(self):
        """Run the bot"""
        try:
            logger.info("=" * 60)
            logger.info("SIMPLE V7 TELEGRAM BOT - STARTED")
            logger.info("=" * 60)
            logger.info(f"Chat ID: {self.chat_id}")
            logger.info("Tap buttons in Telegram to control bot")
            logger.info("=" * 60)

            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
            self.running = True

            # Send startup message with keyboard
            await self.application.bot.send_message(
                chat_id=self.chat_id,
                text="✅ <b>V7 Bot Started</b>\n\nTap buttons below to control.",
                parse_mode='HTML',
                reply_markup=self.get_main_keyboard()
            )

            # Keep running
            while self.running:
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error running bot: {e}")
            raise

        finally:
            logger.info("Shutting down bot...")
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            logger.info("Bot stopped")


def main():
    """Main entry point"""
    config = Settings()
    bot = SimpleV7Bot(config)

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
