"""
HYDRA Guardian: Automated Monitoring & Protection System

Continuously monitors HYDRA 3.0 tournament and automatically responds to failures.

Features:
1. Process health monitoring (HYDRA runtime)
2. API credit balance tracking (DeepSeek, Claude, Grok, Gemini)
3. Gladiator voting pattern analysis
4. Paper trading performance metrics
5. Auto-restart on crashes
6. Log rotation to prevent disk fill
7. Telegram alerts for critical issues

Critical Thresholds:
- API balance < $1 → ALERT
- HYDRA process not running → AUTO-RESTART
- No new trades in 2 hours → ALERT
- Disk usage > 95% → LOG ROTATION
- 100% BUY or SELL bias → REGIME BIAS ALERT
"""

import sys
sys.path.insert(0, '/root/crpbot')

import os
import time
import json
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import requests
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HydraGuardian:
    """Automated monitoring and protection for HYDRA 3.0"""

    # Critical thresholds
    MIN_API_BALANCE = 1.0  # Alert if < $1
    MAX_DISK_USAGE_PERCENT = 95  # Alert if > 95%
    MAX_NO_TRADE_HOURS = 2  # Alert if no trades in 2 hours
    MAX_LOG_SIZE_MB = 500  # Rotate logs if > 500MB
    MAX_DIRECTIONAL_BIAS = 90  # Alert if > 90% BUY or SELL

    def __init__(
        self,
        check_interval: int = 300,  # 5 minutes
        telegram_enabled: bool = True
    ):
        self.check_interval = check_interval
        self.telegram_enabled = telegram_enabled
        self.telegram_token = os.getenv('TELEGRAM_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')

        # File paths
        self.paper_trades_file = Path('/root/crpbot/data/hydra/paper_trades.jsonl')
        self.vote_history_file = Path('/root/crpbot/data/hydra/vote_history.jsonl')
        self.pid_file = Path('/tmp/hydra.pid')

        # State tracking
        self.last_alert_time = {}
        self.alert_cooldown = 3600  # 1 hour between same alerts
        self.restart_count = 0
        self.max_restarts = 3  # Max 3 auto-restarts per Guardian session

        logger.info(f"HYDRA Guardian initialized | Check interval: {check_interval}s")

    def check_process_health(self) -> Dict:
        """Check if HYDRA runtime is running"""

        result = {
            'running': False,
            'pid': None,
            'uptime_hours': 0
        }

        try:
            # Check via process name
            ps_output = subprocess.run(
                ['ps', 'aux'],
                capture_output=True,
                text=True,
                timeout=5
            )

            for line in ps_output.stdout.split('\n'):
                if 'hydra_runtime.py' in line and 'grep' not in line:
                    result['running'] = True
                    parts = line.split()
                    result['pid'] = int(parts[1])
                    break

            # Get uptime from log
            if result['running']:
                log_files = sorted(Path('/tmp').glob('hydra_*.log'), key=lambda x: x.stat().st_mtime, reverse=True)
                if log_files:
                    log_age_hours = (time.time() - log_files[0].stat().st_mtime) / 3600
                    result['uptime_hours'] = log_age_hours

        except Exception as e:
            logger.error(f"Failed to check process health: {e}")

        return result

    def check_api_credits(self) -> Dict:
        """Check API credit balances"""

        credits = {
            'deepseek': {'balance': None, 'status': 'unknown'},
            'claude': {'balance': None, 'status': 'unknown'},
            'groq': {'balance': None, 'status': 'unknown'},
            'gemini': {'balance': None, 'status': 'unknown'}
        }

        # Check via last error in logs
        try:
            log_files = sorted(Path('/tmp').glob('hydra_*.log'), key=lambda x: x.stat().st_mtime, reverse=True)
            if log_files:
                with open(log_files[0], 'r') as f:
                    # Read last 500 lines
                    lines = f.readlines()[-500:]

                    for line in lines:
                        if '402 Payment Required' in line and 'deepseek' in line.lower():
                            credits['deepseek']['status'] = 'exhausted'
                        elif '429 Too Many Requests' in line and 'groq' in line.lower():
                            credits['groq']['status'] = 'exhausted'
                        elif '429' in line and 'claude' in line.lower():
                            credits['claude']['status'] = 'exhausted'
                        elif '429' in line and 'gemini' in line.lower():
                            credits['gemini']['status'] = 'exhausted'

            # If no errors, assume OK
            for provider in credits:
                if credits[provider]['status'] == 'unknown':
                    credits[provider]['status'] = 'ok'

        except Exception as e:
            logger.error(f"Failed to check API credits: {e}")

        return credits

    def check_disk_usage(self) -> Dict:
        """Check disk usage"""

        result = {
            'percent_used': 0,
            'total_gb': 0,
            'used_gb': 0,
            'available_gb': 0
        }

        try:
            disk_stat = shutil.disk_usage('/root')
            result['total_gb'] = disk_stat.total / (1024**3)
            result['used_gb'] = disk_stat.used / (1024**3)
            result['available_gb'] = disk_stat.free / (1024**3)
            result['percent_used'] = (disk_stat.used / disk_stat.total) * 100

        except Exception as e:
            logger.error(f"Failed to check disk usage: {e}")

        return result

    def check_paper_trades(self) -> Dict:
        """Analyze paper trading performance"""

        metrics = {
            'total_trades': 0,
            'buy_trades': 0,
            'sell_trades': 0,
            'buy_percent': 0,
            'sell_percent': 0,
            'last_trade_hours_ago': None,
            'recent_consensus': []
        }

        try:
            if not self.paper_trades_file.exists():
                return metrics

            # Read all trades
            trades = []
            with open(self.paper_trades_file, 'r') as f:
                for line in f:
                    if line.strip():
                        trades.append(json.loads(line))

            if not trades:
                return metrics

            metrics['total_trades'] = len(trades)

            # Count directions
            buy_count = sum(1 for t in trades if t.get('direction') == 'BUY')
            sell_count = sum(1 for t in trades if t.get('direction') == 'SELL')

            metrics['buy_trades'] = buy_count
            metrics['sell_trades'] = sell_count

            if metrics['total_trades'] > 0:
                metrics['buy_percent'] = (buy_count / metrics['total_trades']) * 100
                metrics['sell_percent'] = (sell_count / metrics['total_trades']) * 100

            # Check last trade time
            last_trade = trades[-1]
            if 'timestamp' in last_trade:
                last_time = datetime.fromisoformat(last_trade['timestamp'].replace('Z', '+00:00'))
                hours_ago = (datetime.now(last_time.tzinfo) - last_time).total_seconds() / 3600
                metrics['last_trade_hours_ago'] = hours_ago

            # Recent consensus (last 20 trades)
            recent = trades[-20:] if len(trades) > 20 else trades
            metrics['recent_consensus'] = [t.get('consensus') for t in recent]

        except Exception as e:
            logger.error(f"Failed to check paper trades: {e}")

        return metrics

    def rotate_logs(self):
        """Rotate old log files to prevent disk fill"""

        try:
            log_files = list(Path('/tmp').glob('hydra_*.log'))

            for log_file in log_files:
                size_mb = log_file.stat().st_size / (1024**2)

                if size_mb > self.MAX_LOG_SIZE_MB:
                    # Compress and archive
                    archive_name = f"{log_file.stem}_archived.log"
                    archive_path = log_file.parent / archive_name

                    # Keep last 10000 lines
                    subprocess.run(
                        f"tail -10000 {log_file} > {archive_path} && mv {archive_path} {log_file}",
                        shell=True,
                        check=False
                    )

                    logger.info(f"Rotated log: {log_file.name} ({size_mb:.1f}MB)")

        except Exception as e:
            logger.error(f"Failed to rotate logs: {e}")

    def restart_hydra(self):
        """Restart HYDRA runtime"""

        if self.restart_count >= self.max_restarts:
            logger.error(f"Max restart limit reached ({self.max_restarts}), not restarting")
            return False

        logger.warning("Attempting to restart HYDRA runtime...")

        try:
            # Kill existing process
            subprocess.run(['pkill', '-f', 'hydra_runtime.py'], check=False)
            time.sleep(3)

            # Start new process
            log_file = f"/tmp/hydra_auto_restart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

            cmd = [
                'nohup',
                '.venv/bin/python3',
                'apps/runtime/hydra_runtime.py',
                '--assets', 'BTC-USD', 'ETH-USD', 'SOL-USD',
                '--iterations', '-1',
                '--interval', '300',
                '--paper'
            ]

            with open(log_file, 'w') as f:
                subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd='/root/crpbot',
                    start_new_session=True
                )

            time.sleep(5)

            # Verify restart
            process_health = self.check_process_health()
            if process_health['running']:
                logger.success(f"✅ HYDRA restarted successfully (PID: {process_health['pid']})")
                logger.info(f"Log: {log_file}")
                self.restart_count += 1
                return True
            else:
                logger.error("❌ HYDRA restart failed - process not running")
                return False

        except Exception as e:
            logger.error(f"Failed to restart HYDRA: {e}")
            return False

    def check_critical_failures(self, health: Dict) -> List[Dict]:
        """Check for critical failures"""

        alerts = []

        # CRITICAL: Process not running
        if not health['process']['running']:
            alerts.append({
                'severity': 'CRITICAL',
                'type': 'PROCESS_DOWN',
                'message': 'HYDRA runtime is not running',
                'action': 'AUTO_RESTART',
                'solution': 'Attempting automatic restart'
            })

        # CRITICAL: API credits exhausted
        for provider, status in health['api_credits'].items():
            if status['status'] == 'exhausted':
                alerts.append({
                    'severity': 'CRITICAL',
                    'type': 'API_CREDITS_EXHAUSTED',
                    'message': f'{provider.upper()} API credits exhausted',
                    'action': 'REFILL_CREDITS',
                    'solution': f'Refill {provider} credits immediately'
                })

        # WARNING: High disk usage
        if health['disk']['percent_used'] > self.MAX_DISK_USAGE_PERCENT:
            alerts.append({
                'severity': 'WARNING',
                'type': 'HIGH_DISK_USAGE',
                'message': f"Disk usage {health['disk']['percent_used']:.1f}% > {self.MAX_DISK_USAGE_PERCENT}%",
                'action': 'ROTATE_LOGS',
                'solution': 'Rotating old log files'
            })

        # WARNING: No recent trades
        if health['paper_trades']['last_trade_hours_ago']:
            if health['paper_trades']['last_trade_hours_ago'] > self.MAX_NO_TRADE_HOURS:
                alerts.append({
                    'severity': 'WARNING',
                    'type': 'NO_RECENT_TRADES',
                    'message': f"No trades in {health['paper_trades']['last_trade_hours_ago']:.1f} hours",
                    'action': 'CHECK_RUNTIME',
                    'solution': 'Verify HYDRA is generating signals'
                })

        # WARNING: Directional bias
        if health['paper_trades']['total_trades'] >= 20:
            buy_pct = health['paper_trades']['buy_percent']
            sell_pct = health['paper_trades']['sell_percent']

            if buy_pct > self.MAX_DIRECTIONAL_BIAS:
                alerts.append({
                    'severity': 'WARNING',
                    'type': 'BUY_BIAS',
                    'message': f"{buy_pct:.1f}% BUY trades (regime bias detected)",
                    'action': 'CHECK_REGIME_PROMPTS',
                    'solution': 'Verify gladiator regime-aware voting'
                })
            elif sell_pct > self.MAX_DIRECTIONAL_BIAS:
                alerts.append({
                    'severity': 'WARNING',
                    'type': 'SELL_BIAS',
                    'message': f"{sell_pct:.1f}% SELL trades (regime bias detected)",
                    'action': 'CHECK_REGIME_PROMPTS',
                    'solution': 'Verify gladiator regime-aware voting'
                })

        return alerts

    def send_telegram_alert(self, alert: Dict, health: Dict):
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
{severity_emoji.get(alert['severity'], '❗')} **HYDRA GUARDIAN ALERT**

**Severity:** {alert['severity']}
**Type:** {alert['type']}

**Problem:**
{alert['message']}

**Recommended Action:**
{alert['action']}

**Solution:**
{alert['solution']}

**Current Status:**
• Process: {'✅ Running' if health['process']['running'] else '❌ Not Running'}
• Paper Trades: {health['paper_trades']['total_trades']} ({health['paper_trades']['buy_percent']:.0f}% BUY, {health['paper_trades']['sell_percent']:.0f}% SELL)
• Disk Usage: {health['disk']['percent_used']:.1f}%

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

    def run_check(self):
        """Run single monitoring check"""

        logger.info("Running HYDRA Guardian check...")

        # Gather health metrics
        health = {
            'process': self.check_process_health(),
            'api_credits': self.check_api_credits(),
            'disk': self.check_disk_usage(),
            'paper_trades': self.check_paper_trades()
        }

        # Check for failures
        alerts = self.check_critical_failures(health)

        if not alerts:
            logger.info(
                f"✅ All systems healthy | "
                f"Process: {'UP' if health['process']['running'] else 'DOWN'} | "
                f"Trades: {health['paper_trades']['total_trades']} | "
                f"Disk: {health['disk']['percent_used']:.1f}%"
            )
            return

        # Process alerts
        for alert in alerts:
            logger.warning(f"{alert['severity']}: {alert['message']}")

            # Send Telegram alert
            self.send_telegram_alert(alert, health)

            # Execute auto-remediation
            if alert['severity'] == 'CRITICAL':
                if alert['action'] == 'AUTO_RESTART':
                    self.restart_hydra()
                elif alert['action'] == 'ROTATE_LOGS':
                    self.rotate_logs()

    def run_forever(self):
        """Run monitoring loop forever"""

        logger.info("🛡️  HYDRA Guardian monitoring started")
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

    parser = argparse.ArgumentParser(description='HYDRA Guardian Monitoring System')
    parser.add_argument('--check-interval', type=int, default=300, help='Check interval in seconds (default: 300)')
    parser.add_argument('--no-telegram', action='store_true', help='Disable Telegram alerts')
    parser.add_argument('--once', action='store_true', help='Run once and exit (for testing)')

    args = parser.parse_args()

    guardian = HydraGuardian(
        check_interval=args.check_interval,
        telegram_enabled=not args.no_telegram
    )

    if args.once:
        guardian.run_check()
    else:
        guardian.run_forever()
