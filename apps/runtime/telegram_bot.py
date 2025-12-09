"""Telegram bot integration for runtime notifications and commands."""
import asyncio
from datetime import datetime
from typing import Any

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

        # Original V6 commands
        self.application.add_handler(CommandHandler("start", self._handle_start))
        self.application.add_handler(CommandHandler("check", self._handle_check))
        self.application.add_handler(CommandHandler("stats", self._handle_stats))
        self.application.add_handler(CommandHandler("ftmo_status", self._handle_ftmo_status))
        self.application.add_handler(CommandHandler("threshold", self._handle_threshold))
        self.application.add_handler(CommandHandler("kill_switch", self._handle_kill_switch))
        self.application.add_handler(CommandHandler("help", self._handle_help))

        # V7 Ultimate commands
        self.application.add_handler(CommandHandler("v7_status", self._handle_v7_status))
        self.application.add_handler(CommandHandler("v7_stop", self._handle_v7_stop))
        self.application.add_handler(CommandHandler("v7_start", self._handle_v7_start))
        self.application.add_handler(CommandHandler("v7_stats", self._handle_v7_stats))
        self.application.add_handler(CommandHandler("v7_config", self._handle_v7_config))

        # Performance tracking commands
        self.application.add_handler(CommandHandler("v7_performance", self._handle_v7_performance))
        self.application.add_handler(CommandHandler("v7_recent_trades", self._handle_v7_recent_trades))

        # Bayesian learning commands
        self.application.add_handler(CommandHandler("v7_learning", self._handle_v7_learning))

        # Paper trading commands
        self.application.add_handler(CommandHandler("execute", self._handle_execute))
        self.application.add_handler(CommandHandler("close", self._handle_close))
        self.application.add_handler(CommandHandler("trades", self._handle_trades))

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
        help_text = (
            "🤖 <b>Trading AI Bot - V7 Ultimate</b>\n\n"
            "<b>V7 RUNTIME COMMANDS</b>\n"
            "/v7_status - V7 runtime status\n"
            "/v7_start - Start V7 runtime\n"
            "/v7_stop - Stop V7 runtime\n"
            "/v7_stats - Signal statistics\n"
            "/v7_config - Adjust parameters\n\n"
            "<b>PAPER TRADING</b>\n"
            "/execute &lt;signal_id&gt; [price] - Execute a signal\n"
            "/close &lt;signal_id&gt; win|loss [price] - Close a trade\n"
            "/trades - View open paper trades\n\n"
            "<b>PERFORMANCE TRACKING</b>\n"
            "/v7_performance - Win rate & PnL\n"
            "/v7_recent_trades - Recent trades\n"
            "/v7_learning - Bayesian learning metrics\n\n"
            "<b>LEGACY COMMANDS</b>\n"
            "/check - System status\n"
            "/stats - Performance metrics\n"
            "/ftmo_status - FTMO account status\n"
            "/threshold <n> - Adjust confidence threshold\n"
            "/kill_switch <on|off> - Emergency stop\n"
            "/help - Show this help"
        )
        await update.message.reply_text(help_text, parse_mode='HTML')

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
            await update.message.reply_text("Invalid threshold value. Use a number between 0.0 and 1.0")

    async def _handle_kill_switch(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /kill_switch command."""
        if not context.args:
            await update.message.reply_text("Usage: /kill_switch <on|off>")
            return

        action = context.args[0].lower()
        try:
            from libs.hydra.guardian import get_guardian
            guardian = get_guardian()

            if action == "on":
                guardian.trigger_kill_switch(reason="Manual activation via Telegram")
                await update.message.reply_text("🛑 Kill-switch ACTIVATED - All trading suspended")
            elif action == "off":
                guardian.reset_kill_switch()
                await update.message.reply_text("✅ Kill-switch DEACTIVATED - Trading resumed")
            else:
                await update.message.reply_text("Usage: /kill_switch <on|off>")
        except Exception as e:
            logger.error(f"Failed to toggle kill switch: {e}")
            await update.message.reply_text(f"⚠️ Error toggling kill switch: {str(e)}")

    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        await self._handle_start(update, context)

    # ========== V7 ULTIMATE COMMAND HANDLERS ==========

    async def _handle_v7_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /v7_status command - Show V7 runtime status."""
        try:
            import subprocess
            import json
            from libs.db.models import Signal, get_session
            from sqlalchemy import desc

            # Check if V7 runtime process is running
            result = subprocess.run(
                ['ps', 'aux'],
                capture_output=True,
                text=True
            )
            v7_running = 'v7_runtime.py' in result.stdout

            if not v7_running:
                await update.message.reply_text(
                    "🔴 <b>V7 RUNTIME - STOPPED</b>\n\n"
                    "The V7 runtime is not currently running.\n\n"
                    "Use /v7_start to start it.",
                    parse_mode='HTML'
                )
                return

            # Get V7 process info
            lines = [line for line in result.stdout.split('\n') if 'v7_runtime.py' in line and 'grep' not in line]
            pid = lines[0].split()[1] if lines else "Unknown"

            # Get latest V7 signal from database
            session = get_session(self.config.db_url)
            try:
                latest_signal = session.query(Signal).filter(
                    Signal.model_version == 'v7_ultimate'
                ).order_by(desc(Signal.timestamp)).first()

                if latest_signal:
                    # Parse V7 data from notes
                    v7_data = {}
                    try:
                        v7_data = json.loads(latest_signal.notes) if latest_signal.notes else {}
                    except:
                        pass

                    last_signal_time = latest_signal.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
                    signal_type = latest_signal.direction.upper()
                    confidence = latest_signal.confidence * 100

                    # Count recent signals (24h)
                    from datetime import datetime, timedelta
                    count_24h = session.query(Signal).filter(
                        Signal.model_version == 'v7_ultimate',
                        Signal.timestamp > datetime.utcnow() - timedelta(days=1)
                    ).count()

                    # Get cost data
                    total_cost = v7_data.get('llm_cost_usd', 0.0)

                    status_msg = (
                        f"🟢 <b>V7 RUNTIME - RUNNING</b>\n\n"
                        f"<b>Process ID:</b> {pid}\n"
                        f"<b>Status:</b> Active\n\n"
                        f"<b>LATEST SIGNAL</b>\n"
                        f"Time: {last_signal_time}\n"
                        f"Symbol: {latest_signal.symbol}\n"
                        f"Signal: {signal_type}\n"
                        f"Confidence: {confidence:.1f}%\n\n"
                        f"<b>STATISTICS (24h)</b>\n"
                        f"Signals Generated: {count_24h}\n"
                        f"Last Cost: ${total_cost:.6f}\n\n"
                        f"Use /v7_stats for detailed statistics."
                    )
                else:
                    status_msg = (
                        f"🟢 <b>V7 RUNTIME - RUNNING</b>\n\n"
                        f"<b>Process ID:</b> {pid}\n"
                        f"<b>Status:</b> Active (no signals yet)\n\n"
                        f"Waiting for first signal generation..."
                    )

                await update.message.reply_text(status_msg, parse_mode='HTML')

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error in /v7_status: {e}")
            await update.message.reply_text(
                f"⚠️ Error checking V7 status: {str(e)}",
                parse_mode='HTML'
            )

    async def _handle_v7_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /v7_stop command - Stop V7 runtime."""
        try:
            import subprocess

            # Find V7 runtime process
            result = subprocess.run(
                ['ps', 'aux'],
                capture_output=True,
                text=True
            )

            lines = [line for line in result.stdout.split('\n') if 'v7_runtime.py' in line and 'grep' not in line]

            if not lines:
                await update.message.reply_text(
                    "ℹ️ V7 runtime is not currently running.",
                    parse_mode='HTML'
                )
                return

            # Kill the process
            subprocess.run(['pkill', '-f', 'v7_runtime.py'])

            await update.message.reply_text(
                "🛑 <b>V7 RUNTIME STOPPED</b>\n\n"
                "The V7 runtime has been stopped.\n\n"
                "Use /v7_start to restart it.",
                parse_mode='HTML'
            )

        except Exception as e:
            logger.error(f"Error in /v7_stop: {e}")
            await update.message.reply_text(
                f"⚠️ Error stopping V7: {str(e)}",
                parse_mode='HTML'
            )

    async def _handle_v7_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /v7_start command - Start V7 runtime."""
        try:
            import subprocess
            import os

            # Check if already running
            result = subprocess.run(
                ['ps', 'aux'],
                capture_output=True,
                text=True
            )

            if 'v7_runtime.py' in result.stdout:
                await update.message.reply_text(
                    "ℹ️ V7 runtime is already running.\n\n"
                    "Use /v7_status to check status or /v7_stop to stop it first.",
                    parse_mode='HTML'
                )
                return

            # Start V7 runtime in background - use dynamic path detection
            from libs.hydra.config import PROJECT_ROOT
            cmd = [
                str(PROJECT_ROOT / '.venv/bin/python3'),
                'apps/runtime/v7_runtime.py',
                '--iterations', '-1',
                '--sleep-seconds', '300'
            ]

            # Start process in background
            subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=open('/tmp/v7_runtime_telegram.log', 'w'),
                stderr=subprocess.STDOUT,
                start_new_session=True
            )

            await update.message.reply_text(
                "🟢 <b>V7 RUNTIME STARTED</b>\n\n"
                "The V7 runtime has been started in background mode.\n\n"
                "Configuration:\n"
                "• Scan interval: 300 seconds (5 minutes)\n"
                "• Rate limit: 6 signals/hour\n"
                "• Symbols: BTC-USD, ETH-USD, SOL-USD\n\n"
                "Use /v7_status to check status.",
                parse_mode='HTML'
            )

        except Exception as e:
            logger.error(f"Error in /v7_start: {e}")
            await update.message.reply_text(
                f"⚠️ Error starting V7: {str(e)}",
                parse_mode='HTML'
            )

    async def _handle_v7_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /v7_stats command - Show detailed V7 statistics."""
        try:
            import json
            from libs.db.models import Signal, get_session
            from sqlalchemy import func, desc
            from datetime import datetime, timedelta

            session = get_session(self.config.db_url)
            try:
                # Get signals from last 24h and last 7 days
                now = datetime.utcnow()
                day_ago = now - timedelta(days=1)
                week_ago = now - timedelta(days=7)

                signals_24h = session.query(Signal).filter(
                    Signal.model_version == 'v7_ultimate',
                    Signal.timestamp > day_ago
                ).all()

                signals_7d = session.query(Signal).filter(
                    Signal.model_version == 'v7_ultimate',
                    Signal.timestamp > week_ago
                ).all()

                # Count by signal type (24h)
                buy_count_24h = sum(1 for s in signals_24h if s.direction == 'long')
                sell_count_24h = sum(1 for s in signals_24h if s.direction == 'short')
                hold_count_24h = sum(1 for s in signals_24h if s.direction == 'hold')

                # Count by signal type (7d)
                buy_count_7d = sum(1 for s in signals_7d if s.direction == 'long')
                sell_count_7d = sum(1 for s in signals_7d if s.direction == 'short')
                hold_count_7d = sum(1 for s in signals_7d if s.direction == 'hold')

                # Calculate total costs
                total_cost_24h = 0.0
                total_cost_7d = 0.0

                for signal in signals_24h:
                    try:
                        if signal.notes:
                            v7_data = json.loads(signal.notes)
                            total_cost_24h += v7_data.get('llm_cost_usd', 0.0)
                    except:
                        pass

                for signal in signals_7d:
                    try:
                        if signal.notes:
                            v7_data = json.loads(signal.notes)
                            total_cost_7d += v7_data.get('llm_cost_usd', 0.0)
                    except:
                        pass

                # Average confidence
                avg_conf_24h = sum(s.confidence for s in signals_24h) / len(signals_24h) if signals_24h else 0
                avg_conf_7d = sum(s.confidence for s in signals_7d) / len(signals_7d) if signals_7d else 0

                # Count by symbol (24h)
                btc_count = sum(1 for s in signals_24h if 'BTC' in s.symbol)
                eth_count = sum(1 for s in signals_24h if 'ETH' in s.symbol)
                sol_count = sum(1 for s in signals_24h if 'SOL' in s.symbol)

                stats_msg = (
                    f"📊 <b>V7 ULTIMATE STATISTICS</b>\n\n"
                    f"<b>SIGNALS (Last 24 Hours)</b>\n"
                    f"Total: {len(signals_24h)}\n"
                    f"BUY: {buy_count_24h} | SELL: {sell_count_24h} | HOLD: {hold_count_24h}\n"
                    f"Avg Confidence: {avg_conf_24h*100:.1f}%\n"
                    f"Cost: ${total_cost_24h:.6f}\n\n"
                    f"<b>SIGNALS (Last 7 Days)</b>\n"
                    f"Total: {len(signals_7d)}\n"
                    f"BUY: {buy_count_7d} | SELL: {sell_count_7d} | HOLD: {hold_count_7d}\n"
                    f"Avg Confidence: {avg_conf_7d*100:.1f}%\n"
                    f"Cost: ${total_cost_7d:.6f}\n\n"
                    f"<b>BY SYMBOL (24h)</b>\n"
                    f"BTC: {btc_count} | ETH: {eth_count} | SOL: {sol_count}\n\n"
                    f"<b>COST PROJECTIONS</b>\n"
                    f"Daily Rate: ${total_cost_24h:.4f}/day\n"
                    f"Monthly Est: ${total_cost_24h * 30:.2f}/month\n"
                    f"Budget: $3.00/day, $100.00/month"
                )

                await update.message.reply_text(stats_msg, parse_mode='HTML')

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error in /v7_stats: {e}")
            await update.message.reply_text(
                f"⚠️ Error getting V7 stats: {str(e)}",
                parse_mode='HTML'
            )

    async def _handle_v7_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /v7_config command - Adjust V7 parameters."""
        if not context.args:
            config_help = (
                "⚙️ <b>V7 CONFIGURATION</b>\n\n"
                "Usage:\n"
                "/v7_config show - Show current config\n"
                "/v7_config rate_limit &lt;n&gt; - Set signals/hour (1-10)\n"
                "/v7_config confidence &lt;n&gt; - Set min confidence (0.0-1.0)\n\n"
                "Example:\n"
                "/v7_config rate_limit 10\n"
                "/v7_config confidence 0.70"
            )
            await update.message.reply_text(config_help, parse_mode='HTML')
            return

        action = context.args[0].lower()

        if action == "show":
            # Show current configuration
            config_msg = (
                "⚙️ <b>V7 CURRENT CONFIGURATION</b>\n\n"
                f"<b>Rate Limit:</b> 6 signals/hour\n"
                f"<b>Min Confidence:</b> {self.config.confidence_threshold:.1%}\n"
                f"<b>Scan Interval:</b> 300 seconds (5 min)\n"
                f"<b>Daily Budget:</b> $3.00\n"
                f"<b>Monthly Budget:</b> $100.00\n"
                f"<b>Conservative Mode:</b> Enabled\n\n"
                "Note: Config changes require V7 restart to take effect.\n"
                "Use /v7_stop then /v7_start after changes."
            )
            await update.message.reply_text(config_msg, parse_mode='HTML')

        elif action == "rate_limit":
            if len(context.args) < 2:
                await update.message.reply_text("Usage: /v7_config rate_limit <1-10>", parse_mode='HTML')
                return

            try:
                rate = int(context.args[1])
                if not 1 <= rate <= 10:
                    await update.message.reply_text("Rate limit must be between 1 and 10 signals/hour", parse_mode='HTML')
                    return

                # Note: This would require modifying .env or runtime config
                await update.message.reply_text(
                    f"⚠️ <b>CONFIGURATION CHANGE</b>\n\n"
                    f"Rate limit setting noted: {rate} signals/hour\n\n"
                    f"To apply:\n"
                    f"1. Update MAX_SIGNALS_PER_HOUR in .env\n"
                    f"2. Run /v7_stop\n"
                    f"3. Run /v7_start\n\n"
                    f"This feature requires manual config update for safety.",
                    parse_mode='HTML'
                )

            except ValueError:
                await update.message.reply_text("Invalid rate limit value. Use a number between 1 and 10.", parse_mode='HTML')

        elif action == "confidence":
            if len(context.args) < 2:
                await update.message.reply_text("Usage: /v7_config confidence <0.0-1.0>", parse_mode='HTML')
                return

            try:
                conf = float(context.args[1])
                if not 0.0 <= conf <= 1.0:
                    await update.message.reply_text("Confidence must be between 0.0 and 1.0", parse_mode='HTML')
                    return

                await update.message.reply_text(
                    f"⚠️ <b>CONFIGURATION CHANGE</b>\n\n"
                    f"Confidence threshold noted: {conf:.1%}\n\n"
                    f"To apply:\n"
                    f"1. Update CONFIDENCE_THRESHOLD in .env\n"
                    f"2. Run /v7_stop\n"
                    f"3. Run /v7_start\n\n"
                    f"This feature requires manual config update for safety.",
                    parse_mode='HTML'
                )

            except ValueError:
                await update.message.reply_text("Invalid confidence value. Use a number between 0.0 and 1.0.", parse_mode='HTML')

        else:
            await update.message.reply_text(
                "Unknown config action. Use:\n"
                "/v7_config show\n"
                "/v7_config rate_limit <n>\n"
                "/v7_config confidence <n>",
                parse_mode='HTML'
            )

    async def _handle_v7_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /v7_performance command - Show performance metrics."""
        try:
            from libs.performance.trade_tracker import TradeTracker

            tracker = TradeTracker(self.config.db_url)

            # Get performance for different periods
            perf_7d = tracker.get_performance_stats(days=7)
            perf_30d = tracker.get_performance_stats(days=30)

            if perf_30d.total_trades == 0:
                await update.message.reply_text(
                    "📊 <b>V7 PERFORMANCE TRACKING</b>\n\n"
                    "No executed trades yet.\n\n"
                    "To start tracking:\n"
                    "1. When you execute a V7 signal, use the dashboard or API to mark it as executed\n"
                    "2. When the trade closes, record the outcome (win/loss)\n\n"
                    "Performance tracking will then show win rate, PnL, and other metrics.",
                    parse_mode='HTML'
                )
                return

            # Build performance message
            perf_msg = f"📊 <b>V7 PERFORMANCE METRICS</b>\n\n"

            # 7-day stats
            perf_msg += f"<b>LAST 7 DAYS</b>\n"
            perf_msg += f"Total Trades: {perf_7d.total_trades}\n"
            perf_msg += f"Wins: {perf_7d.wins} | Losses: {perf_7d.losses} | Pending: {perf_7d.pending}\n"
            perf_msg += f"Win Rate: {perf_7d.win_rate:.1f}%\n"
            perf_msg += f"Total PnL: {perf_7d.total_pnl:+.2f}%\n"
            if perf_7d.total_trades > 0:
                perf_msg += f"Avg Win: +{perf_7d.avg_win:.2f}% | Avg Loss: {perf_7d.avg_loss:.2f}%\n"
                perf_msg += f"Max Win: +{perf_7d.max_win:.2f}% | Max Loss: {perf_7d.max_loss:.2f}%\n"
                perf_msg += f"Profit Factor: {perf_7d.profit_factor:.2f}\n"
            perf_msg += "\n"

            # 30-day stats
            perf_msg += f"<b>LAST 30 DAYS</b>\n"
            perf_msg += f"Total Trades: {perf_30d.total_trades}\n"
            perf_msg += f"Wins: {perf_30d.wins} | Losses: {perf_30d.losses}\n"
            perf_msg += f"Win Rate: {perf_30d.win_rate:.1f}%\n"
            perf_msg += f"Total PnL: {perf_30d.total_pnl:+.2f}%\n"
            if perf_30d.total_trades > 0:
                perf_msg += f"Avg Win: +{perf_30d.avg_win:.2f}% | Avg Loss: {perf_30d.avg_loss:.2f}%\n"
                perf_msg += f"Profit Factor: {perf_30d.profit_factor:.2f}\n"
                if perf_30d.sharpe_ratio:
                    perf_msg += f"Sharpe Ratio: {perf_30d.sharpe_ratio:.2f}\n"
                perf_msg += f"Max Drawdown: {perf_30d.max_drawdown:.2f}% ({perf_30d.max_drawdown_pct:.1f}% of peak)\n"
            perf_msg += "\n"

            # By signal type
            if perf_30d.total_trades > 0:
                perf_msg += f"<b>BY SIGNAL TYPE (30d)</b>\n"
                perf_msg += f"BUY Win Rate: {perf_30d.buy_win_rate:.1f}%\n"
                perf_msg += f"SELL Win Rate: {perf_30d.sell_win_rate:.1f}%\n"
                perf_msg += f"HOLD Skipped: {perf_30d.hold_skipped}\n"

            await update.message.reply_text(perf_msg, parse_mode='HTML')

        except Exception as e:
            logger.error(f"Error in /v7_performance: {e}")
            await update.message.reply_text(
                f"⚠️ Error getting performance metrics: {str(e)}",
                parse_mode='HTML'
            )

    async def _handle_v7_recent_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /v7_recent_trades command - Show recent executed trades."""
        try:
            from libs.performance.trade_tracker import TradeTracker

            tracker = TradeTracker(self.config.db_url)
            trades = tracker.get_recent_trades(limit=5)

            if not trades:
                await update.message.reply_text(
                    "📝 <b>RECENT TRADES</b>\n\n"
                    "No executed trades found.\n\n"
                    "Trades appear here after you manually execute V7 signals.",
                    parse_mode='HTML'
                )
                return

            msg = "📝 <b>RECENT TRADES (Last 5)</b>\n\n"

            for trade in trades:
                # Format timestamp
                exec_time = trade.execution_time.strftime("%m/%d %H:%M") if trade.execution_time else "N/A"

                # Result emoji
                if trade.result == 'win':
                    result_emoji = "✅"
                elif trade.result == 'loss':
                    result_emoji = "❌"
                elif trade.result == 'pending':
                    result_emoji = "⏳"
                else:
                    result_emoji = "➖"

                # Direction
                direction = "BUY" if trade.direction == "long" else ("SELL" if trade.direction == "short" else "HOLD")

                msg += f"{result_emoji} <b>{trade.symbol}</b> {direction}\n"
                msg += f"  Time: {exec_time}\n"
                msg += f"  Entry: ${trade.execution_price:.2f}\n"

                if trade.exit_price:
                    msg += f"  Exit: ${trade.exit_price:.2f}\n"

                if trade.pnl is not None:
                    msg += f"  PnL: {trade.pnl:+.2f}%\n"

                msg += f"  Status: {trade.result or 'N/A'}\n"
                msg += "\n"

            msg += "Use /v7_performance for detailed metrics."

            await update.message.reply_text(msg, parse_mode='HTML')

        except Exception as e:
            logger.error(f"Error in /v7_recent_trades: {e}")
            await update.message.reply_text(
                f"⚠️ Error getting recent trades: {str(e)}",
                parse_mode='HTML'
            )

    async def _handle_v7_learning(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /v7_learning command - Show Bayesian learning metrics."""
        try:
            from libs.bayesian import BayesianLearner

            learner = BayesianLearner(self.config.db_url)
            summary = learner.get_learning_summary(days=30)

            if not summary or summary['overall']['sample_size'] == 0:
                await update.message.reply_text(
                    "📊 <b>V7 Bayesian Learning</b>\n\n"
                    "No trade data yet. Bayesian learning will activate once you start recording trade outcomes.\n\n"
                    "To record trades:\n"
                    "1. Execute a V7 signal manually\n"
                    "2. Use TradeTracker to record execution\n"
                    "3. Close the trade and record outcome (win/loss)\n"
                    "4. V7 will continuously improve from your results!",
                    parse_mode='HTML'
                )
                return

            overall = summary['overall']
            by_type = summary['by_signal_type']
            by_symbol = summary['by_symbol']
            calibration = summary.get('calibration', {})

            msg = "📊 <b>V7 Bayesian Learning (30 days)</b>\n\n"

            # Overall learning
            msg += f"<b>Overall Win Rate Estimate:</b>\n"
            msg += f"• Mean: {overall['win_rate']:.1%}\n"
            msg += f"• Uncertainty: ±{overall['uncertainty']:.1%}\n"
            msg += f"• Confidence: {overall['confidence']:.1%}\n"
            msg += f"• Sample Size: {overall['sample_size']} trades\n\n"

            # By signal type
            msg += "<b>Win Rate by Signal Type:</b>\n"
            if by_type['long']['sample_size'] > 0:
                msg += f"• LONG: {by_type['long']['win_rate']:.1%} ± {by_type['long']['uncertainty']:.1%} ({by_type['long']['sample_size']} trades)\n"
            else:
                msg += "• LONG: No data\n"

            if by_type['short']['sample_size'] > 0:
                msg += f"• SHORT: {by_type['short']['win_rate']:.1%} ± {by_type['short']['uncertainty']:.1%} ({by_type['short']['sample_size']} trades)\n\n"
            else:
                msg += "• SHORT: No data\n\n"

            # By symbol
            msg += "<b>Win Rate by Symbol:</b>\n"
            for symbol_name in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
                symbol_data = by_symbol.get(symbol_name, {})
                if symbol_data.get('sample_size', 0) > 0:
                    msg += f"• {symbol_name}: {symbol_data['win_rate']:.1%} ± {symbol_data['uncertainty']:.1%} ({symbol_data['sample_size']} trades)\n"
                else:
                    msg += f"• {symbol_name}: No data\n"

            # Confidence calibration
            if calibration:
                msg += "\n<b>Confidence Calibration:</b>\n"
                for bucket_name, bucket_data in calibration.items():
                    msg += f"• {bucket_name.title()}: "
                    msg += f"Predicted {bucket_data['predicted_win_rate']:.1%}, "
                    msg += f"Actual {bucket_data['actual_win_rate']:.1%} "
                    msg += f"(Error: {bucket_data['calibration_error']:.1%}, n={bucket_data['sample_size']})\n"

            msg += "\n💡 V7 automatically adjusts confidence based on these learned win rates!"

            await update.message.reply_text(msg, parse_mode='HTML')

        except Exception as e:
            logger.error(f"Error in /v7_learning: {e}")
            await update.message.reply_text(
                f"⚠️ Error getting Bayesian learning metrics: {str(e)}",
                parse_mode='HTML'
            )

    # ========== PAPER TRADING COMMAND HANDLERS ==========

    async def _handle_execute(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /execute command - Mark a signal as executed."""
        if not context.args:
            await update.message.reply_text(
                "📝 <b>Execute Signal</b>\n\n"
                "Usage: /execute &lt;signal_id&gt; [price]\n\n"
                "Mark a V7 signal as executed (paper trade entry).\n\n"
                "Examples:\n"
                "• /execute 123 - Execute signal #123 at signal's entry price\n"
                "• /execute 123 92500.00 - Execute at specific price",
                parse_mode='HTML'
            )
            return

        try:
            from apps.runtime.paper_trader import PaperTrader

            # Parse arguments
            signal_id = int(context.args[0])
            execution_price = float(context.args[1]) if len(context.args) > 1 else None

            # Execute signal
            trader = PaperTrader(self.config.db_url)
            result = trader.execute_signal(signal_id, execution_price)

            if result['success']:
                msg = (
                    f"✅ <b>TRADE EXECUTED</b>\n\n"
                    f"<b>Signal #{result['signal_id']}</b>\n"
                    f"Symbol: {result['symbol']}\n"
                    f"Direction: {result['direction'].upper()}\n"
                    f"Entry Price: ${result['entry_price']:.2f}\n"
                    f"Confidence: {result['confidence']*100:.1f}% ({result['tier']})\n"
                    f"Time: {result['execution_time']}\n\n"
                    f"Use /close {signal_id} win|loss to record outcome when trade closes."
                )
            else:
                msg = f"❌ <b>ERROR</b>\n\n{result['error']}"

            await update.message.reply_text(msg, parse_mode='HTML')

        except ValueError:
            await update.message.reply_text(
                "❌ Invalid signal_id or price. Use numbers only.",
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"Error in /execute: {e}")
            await update.message.reply_text(
                f"⚠️ Error executing signal: {str(e)}",
                parse_mode='HTML'
            )

    async def _handle_close(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /close command - Close a paper trade."""
        if not context.args or len(context.args) < 2:
            await update.message.reply_text(
                "📝 <b>Close Trade</b>\n\n"
                "Usage: /close &lt;signal_id&gt; &lt;result&gt; [exit_price OR pnl_pct]\n\n"
                "Close a paper trade and record outcome.\n\n"
                "Examples:\n"
                "• /close 123 win 93500.00 - Close as win at specific price\n"
                "• /close 123 loss - Close as loss (uses current price)\n"
                "• /close 123 win pnl:2.5 - Close as win with +2.5% PnL",
                parse_mode='HTML'
            )
            return

        try:
            from apps.runtime.paper_trader import PaperTrader

            # Parse arguments
            signal_id = int(context.args[0])
            result = context.args[1].lower()

            if result not in ['win', 'loss']:
                await update.message.reply_text(
                    "❌ Result must be 'win' or 'loss'",
                    parse_mode='HTML'
                )
                return

            # Parse optional exit price or pnl_pct
            exit_price = None
            pnl_pct = None

            if len(context.args) > 2:
                arg3 = context.args[2]
                if arg3.startswith('pnl:'):
                    pnl_pct = float(arg3.replace('pnl:', ''))
                else:
                    exit_price = float(arg3)

            # Close trade
            trader = PaperTrader(self.config.db_url)
            close_result = trader.close_trade(signal_id, result, exit_price, pnl_pct)

            if close_result['success']:
                pnl_emoji = "📈" if close_result['pnl_pct'] > 0 else "📉"
                msg = (
                    f"{'✅' if result == 'win' else '❌'} <b>TRADE CLOSED - {result.upper()}</b>\n\n"
                    f"<b>Signal #{close_result['signal_id']}</b>\n"
                    f"Symbol: {close_result['symbol']}\n"
                    f"Direction: {close_result['direction'].upper()}\n"
                    f"Entry: ${close_result['entry_price']:.2f}\n"
                    f"Exit: ${close_result['exit_price']:.2f}\n"
                    f"{pnl_emoji} PnL: {close_result['pnl_pct']:+.2f}%\n"
                    f"Exit Time: {close_result['exit_time']}\n\n"
                    f"Bayesian learner updated with this outcome!"
                )
            else:
                msg = f"❌ <b>ERROR</b>\n\n{close_result['error']}"

            await update.message.reply_text(msg, parse_mode='HTML')

        except ValueError as e:
            await update.message.reply_text(
                f"❌ Invalid input: {str(e)}",
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"Error in /close: {e}")
            await update.message.reply_text(
                f"⚠️ Error closing trade: {str(e)}",
                parse_mode='HTML'
            )

    async def _handle_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /trades command - Show open paper trades."""
        try:
            from apps.runtime.paper_trader import PaperTrader, format_open_trades_message

            trader = PaperTrader(self.config.db_url)
            open_trades = trader.get_open_trades()

            msg = format_open_trades_message(open_trades)
            await update.message.reply_text(msg, parse_mode='Markdown')

        except Exception as e:
            logger.error(f"Error in /trades: {e}")
            await update.message.reply_text(
                f"⚠️ Error getting open trades: {str(e)}",
                parse_mode='HTML'
            )


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

