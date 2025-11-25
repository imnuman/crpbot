"""
Guardian: Automated V7 Monitoring & Protection System

Continuously monitors V7 performance and automatically responds to failures.

Features:
1. Real-time performance monitoring (every 5 minutes)
2. Automatic kill switch on critical failures
3. Alert generation (Telegram + logs)
4. Root cause analysis
5. Auto-remediation suggestions

Critical Thresholds:
- Win rate < 40% after 10+ trades → KILL SWITCH
- Avg loss hold time < 30 min → ENTRY TIMING ALERT
- P&L degrading > 10% → PERFORMANCE DEGRADATION ALERT
- Stop loss rate > 80% → STOP LOSS TOO TIGHT ALERT
"""

import sys
sys.path.insert(0, '/root/crpbot')

import os
import time
import sqlite3
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Guardian:
    """Automated monitoring and protection for V7 Runtime"""

    # Critical thresholds
    MIN_WIN_RATE = 40.0  # Kill switch if < 40% after 10 trades
    MIN_TRADES_FOR_KILL = 10  # Need at least 10 trades before kill switch
    MAX_STOP_LOSS_RATE = 80.0  # Alert if > 80% losses are stop_loss
    MAX_QUICK_LOSS_RATE = 70.0  # Alert if > 70% losses in < 1 hour
    MIN_AVG_HOLD_MINUTES = 30  # Alert if avg loss hold < 30 min
    MAX_PNL_DEGRADATION = -10.0  # Alert if total P&L < -10%

    def __init__(
        self,
        db_path: str = 'tradingai.db',
        check_interval: int = 300,  # 5 minutes
        telegram_enabled: bool = True
    ):
        self.db_path = db_path
        self.check_interval = check_interval
        self.telegram_enabled = telegram_enabled
        self.telegram_token = os.getenv('TELEGRAM_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')

        # State tracking
        self.last_alert_time = {}
        self.alert_cooldown = 3600  # 1 hour between same alerts

        logger.info(f"Guardian initialized | Check interval: {check_interval}s")

    def get_performance_metrics(self) -> Optional[Dict]:
        """Get current V7 performance metrics"""

        try:
            conn = sqlite3.connect(self.db_path)

            # Overall stats
            stats = pd.read_sql_query("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
                    ROUND(AVG(CASE WHEN outcome IN ('win', 'loss') THEN
                        (CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END)
                        ELSE NULL END) * 100, 2) as win_rate,
                    ROUND(SUM(pnl_percent), 2) as total_pnl,
                    ROUND(AVG(pnl_percent), 2) as avg_pnl
                FROM signal_results
                WHERE outcome IN ('win', 'loss')
            """, conn)

            if stats['total_trades'].iloc[0] == 0:
                conn.close()
                return None

            # Loss analysis
            loss_stats = pd.read_sql_query("""
                SELECT
                    exit_reason,
                    ROUND((julianday(exit_timestamp) - julianday(entry_timestamp)) * 24 * 60, 0) as hold_minutes
                FROM signal_results
                WHERE outcome = 'loss'
            """, conn)

            # Recent performance trend (last 10 trades)
            recent_trades = pd.read_sql_query("""
                SELECT
                    outcome,
                    pnl_percent,
                    exit_timestamp
                FROM signal_results
                WHERE outcome IN ('win', 'loss')
                ORDER BY exit_timestamp DESC
                LIMIT 10
            """, conn)

            conn.close()

            # Calculate metrics
            metrics = {
                'total_trades': int(stats['total_trades'].iloc[0]),
                'wins': int(stats['wins'].iloc[0]),
                'losses': int(stats['losses'].iloc[0]),
                'win_rate': float(stats['win_rate'].iloc[0]),
                'total_pnl': float(stats['total_pnl'].iloc[0]),
                'avg_pnl': float(stats['avg_pnl'].iloc[0]),
            }

            # Loss analysis
            if len(loss_stats) > 0:
                stop_loss_count = len(loss_stats[loss_stats['exit_reason'].str.contains('stop_loss|sl_hit', na=False)])
                stop_loss_rate = (stop_loss_count / len(loss_stats)) * 100
                avg_loss_hold = loss_stats['hold_minutes'].mean()
                quick_losses = len(loss_stats[loss_stats['hold_minutes'] < 60])
                quick_loss_rate = (quick_losses / len(loss_stats)) * 100

                metrics['stop_loss_rate'] = stop_loss_rate
                metrics['avg_loss_hold_minutes'] = avg_loss_hold
                metrics['quick_loss_rate'] = quick_loss_rate
            else:
                metrics['stop_loss_rate'] = 0
                metrics['avg_loss_hold_minutes'] = 0
                metrics['quick_loss_rate'] = 0

            # Recent trend
            if len(recent_trades) >= 5:
                recent_wins = len(recent_trades[recent_trades['outcome'] == 'win'])
                metrics['recent_win_rate'] = (recent_wins / len(recent_trades)) * 100
                metrics['recent_pnl'] = recent_trades['pnl_percent'].sum()
            else:
                metrics['recent_win_rate'] = metrics['win_rate']
                metrics['recent_pnl'] = metrics['total_pnl']

            return metrics

        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return None

    def check_critical_failures(self, metrics: Dict) -> List[Dict]:
        """Check for critical failures that require immediate action"""

        alerts = []

        # CRITICAL: Low win rate after sufficient trades
        if metrics['total_trades'] >= self.MIN_TRADES_FOR_KILL:
            if metrics['win_rate'] < self.MIN_WIN_RATE:
                alerts.append({
                    'severity': 'CRITICAL',
                    'type': 'LOW_WIN_RATE',
                    'message': f"Win rate {metrics['win_rate']:.1f}% < {self.MIN_WIN_RATE}% after {metrics['total_trades']} trades",
                    'action': 'KILL_SWITCH',
                    'root_cause': 'Entry timing or stop loss configuration',
                    'solution': 'Stop V7, widen stop losses, add entry confirmation'
                })

        # CRITICAL: Negative P&L degradation
        if metrics['total_pnl'] < self.MAX_PNL_DEGRADATION:
            alerts.append({
                'severity': 'CRITICAL',
                'type': 'NEGATIVE_PNL',
                'message': f"Total P&L {metrics['total_pnl']:.2f}% (losing money)",
                'action': 'KILL_SWITCH',
                'root_cause': 'Strategy is not profitable',
                'solution': 'Review entry timing and risk management'
            })

        # WARNING: High stop loss rate
        if metrics['stop_loss_rate'] > self.MAX_STOP_LOSS_RATE:
            alerts.append({
                'severity': 'WARNING',
                'type': 'HIGH_STOP_LOSS_RATE',
                'message': f"Stop loss rate {metrics['stop_loss_rate']:.1f}% > {self.MAX_STOP_LOSS_RATE}%",
                'action': 'WIDEN_STOP_LOSS',
                'root_cause': 'Stop losses too tight or bad entry timing',
                'solution': 'Increase stop loss from 2% to 3-4%'
            })

        # WARNING: Quick losses (getting stopped out immediately)
        if metrics['quick_loss_rate'] > self.MAX_QUICK_LOSS_RATE:
            alerts.append({
                'severity': 'WARNING',
                'type': 'QUICK_LOSSES',
                'message': f"{metrics['quick_loss_rate']:.1f}% losses stopped out in < 60 min",
                'action': 'FIX_ENTRY_TIMING',
                'root_cause': 'Entering at local highs/lows (chasing)',
                'solution': 'Add pullback waiting logic, entry confirmation'
            })

        # WARNING: Average loss hold time too short
        if metrics['avg_loss_hold_minutes'] < self.MIN_AVG_HOLD_MINUTES:
            alerts.append({
                'severity': 'WARNING',
                'type': 'SHORT_LOSS_HOLDS',
                'message': f"Avg loss hold {metrics['avg_loss_hold_minutes']:.0f} min < {self.MIN_AVG_HOLD_MINUTES} min",
                'action': 'FIX_ENTRY_TIMING',
                'root_cause': 'Entering too early, not waiting for setup',
                'solution': 'Wait for price confirmation before entry'
            })

        return alerts

    def send_telegram_alert(self, alert: Dict, metrics: Dict):
        """Send alert via Telegram"""

        if not self.telegram_enabled or not self.telegram_token:
            return

        # Check cooldown
        alert_key = alert['type']
        if alert_key in self.last_alert_time:
            time_since_last = time.time() - self.last_alert_time[alert_key]
            if time_since_last < self.alert_cooldown:
                return

        severity_emoji = {
            'CRITICAL': '🚨',
            'WARNING': '⚠️',
            'INFO': 'ℹ️'
        }

        message = f"""
{severity_emoji.get(alert['severity'], '❗')} **V7 GUARDIAN ALERT**

**Severity:** {alert['severity']}
**Type:** {alert['type']}

**Problem:**
{alert['message']}

**Root Cause:**
{alert['root_cause']}

**Recommended Action:**
{alert['action']}

**Solution:**
{alert['solution']}

**Current Metrics:**
• Win Rate: {metrics['win_rate']:.1f}% ({metrics['wins']}/{metrics['total_trades']})
• Total P&L: {metrics['total_pnl']:.2f}%
• Stop Loss Rate: {metrics['stop_loss_rate']:.1f}%
• Avg Loss Hold: {metrics['avg_loss_hold_minutes']:.0f} min

**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            response = requests.post(url, json=payload, timeout=10)

            if response.status_code == 200:
                self.last_alert_time[alert_key] = time.time()
                logger.info(f"Telegram alert sent: {alert['type']}")
            else:
                logger.error(f"Telegram alert failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")

    def execute_kill_switch(self):
        """Execute emergency kill switch - stop V7 runtime"""

        logger.critical("🚨 EXECUTING KILL SWITCH - STOPPING V7 RUNTIME 🚨")

        try:
            import subprocess

            # Kill V7 runtime
            subprocess.run(['pkill', '-f', 'v7_runtime.py'], check=False)

            # Create kill switch marker
            Path('/tmp/v7_kill_switch_active').touch()

            logger.critical("✅ V7 Runtime stopped successfully")

            # Send final Telegram notification
            if self.telegram_enabled and self.telegram_token:
                message = """
🚨 **EMERGENCY KILL SWITCH ACTIVATED**

V7 Runtime has been **AUTOMATICALLY STOPPED** due to critical performance failures.

**DO NOT RESTART** until issues are resolved.

Check logs for details.
"""
                try:
                    url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                    requests.post(url, json={'chat_id': self.telegram_chat_id, 'text': message}, timeout=10)
                except:
                    pass

        except Exception as e:
            logger.error(f"Failed to execute kill switch: {e}")

    def run_check(self):
        """Run single monitoring check"""

        logger.info("Running Guardian check...")

        # Get metrics
        metrics = self.get_performance_metrics()

        if not metrics:
            logger.info("No trades yet, skipping check")
            return

        # Check for failures
        alerts = self.check_critical_failures(metrics)

        if not alerts:
            logger.info(f"✅ All metrics healthy | Win rate: {metrics['win_rate']:.1f}% | P&L: {metrics['total_pnl']:.2f}%")
            return

        # Process alerts
        for alert in alerts:
            logger.warning(f"{alert['severity']}: {alert['message']}")

            # Send Telegram alert
            self.send_telegram_alert(alert, metrics)

            # Execute kill switch for critical alerts
            if alert['severity'] == 'CRITICAL' and alert['action'] == 'KILL_SWITCH':
                self.execute_kill_switch()
                return  # Stop after kill switch

    def run_forever(self):
        """Run monitoring loop forever"""

        logger.info("🛡️  Guardian monitoring started")
        logger.info(f"Check interval: {self.check_interval}s")
        logger.info(f"Telegram alerts: {'ENABLED' if self.telegram_enabled else 'DISABLED'}")

        while True:
            try:
                self.run_check()
            except Exception as e:
                logger.error(f"Guardian check failed: {e}")

            time.sleep(self.check_interval)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='V7 Guardian Monitoring System')
    parser.add_argument('--check-interval', type=int, default=300, help='Check interval in seconds (default: 300)')
    parser.add_argument('--no-telegram', action='store_true', help='Disable Telegram alerts')
    parser.add_argument('--once', action='store_true', help='Run once and exit (for testing)')

    args = parser.parse_args()

    guardian = Guardian(
        check_interval=args.check_interval,
        telegram_enabled=not args.no_telegram
    )

    if args.once:
        guardian.run_check()
    else:
        guardian.run_forever()
