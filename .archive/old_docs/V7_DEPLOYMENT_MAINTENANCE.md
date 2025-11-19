# V7 Ultimate - Deployment & Maintenance Guide

**Version**: 1.0
**Last Updated**: 2025-11-19
**For**: System Administrators and DevOps

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Initial Deployment](#initial-deployment)
3. [Configuration](#configuration)
4. [Starting Services](#starting-services)
5. [Monitoring](#monitoring)
6. [Maintenance Tasks](#maintenance-tasks)
7. [Troubleshooting](#troubleshooting)
8. [Backup & Recovery](#backup--recovery)
9. [Scaling](#scaling)
10. [Security](#security)

---

## System Requirements

### Hardware

**Minimum**:
- CPU: 2 cores
- RAM: 4GB
- Disk: 10GB
- Network: Stable internet connection

**Recommended** (Current Production):
- CPU: 4 cores
- RAM: 8GB
- Disk: 20GB SSD
- Network: 100Mbps+ with low latency to exchanges

### Software

**Operating System**:
- Ubuntu 20.04+ (tested)
- Debian 11+ (should work)
- Other Linux (untested)

**Python**:
- Python 3.10+
- pip or uv package manager

**Database**:
- SQLite 3.35+ (included)
- Optional: PostgreSQL 13+ (for scaling)

**Optional**:
- systemd (for service management)
- nginx (for dashboard reverse proxy)
- fail2ban (security)

---

## Initial Deployment

### 1. Clone Repository

```bash
cd /root  # or your preferred location
git clone https://github.com/your-org/crpbot.git
cd crpbot
git checkout feature/v7-ultimate
```

### 2. Install Dependencies

**Using uv** (recommended):
```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

**Using pip**:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Initialize Database

```bash
.venv/bin/python3 -c "
from libs.config.config import Settings
from libs.db.models import create_tables

config = Settings()
create_tables(config.db_url)
print('Database initialized')
"
```

---

## Configuration

### Environment Variables (`.env`)

Create `.env` file in project root:

```bash
# Data Provider
DATA_PROVIDER=coinbase

# Coinbase API (JWT auth)
COINBASE_API_KEY_NAME=organizations/.../apiKeys/...
COINBASE_API_PRIVATE_KEY=-----BEGIN EC PRIVATE KEY-----
...
-----END EC PRIVATE KEY-----

# DeepSeek LLM API
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# CoinGecko Premium (optional but recommended)
COINGECKO_API_KEY=CG-xxxxxxxxxxxxxxxxxxxxxxxx

# Telegram Notifications
TELEGRAM_TOKEN=1234567890:ABCdefGHIjklMNOpqrSTUvwxYZ
TELEGRAM_CHAT_ID=8302332448

# Runtime Settings
CONFIDENCE_THRESHOLD=0.65
RUNTIME_MODE=live
KILL_SWITCH=false

# Database
DB_URL=sqlite:///tradingai.db
# Or PostgreSQL:
# DB_URL=postgresql+psycopg://user:pass@localhost:5432/tradingai
```

### File Permissions

```bash
chmod 600 .env  # Protect secrets
chmod 755 apps/runtime/v7_runtime.py
chmod 755 apps/dashboard/app.py
```

---

## Starting Services

### V7 Runtime

**Foreground** (testing):
```bash
.venv/bin/python3 apps/runtime/v7_runtime.py \
  --iterations -1 \
  --sleep-seconds 120
```

**Background** (production):
```bash
nohup .venv/bin/python3 apps/runtime/v7_runtime.py \
  --iterations -1 \
  --sleep-seconds 120 \
  > /tmp/v7_runtime.log 2>&1 &

echo $! > /tmp/v7_runtime.pid
```

**With systemd** (recommended):
```bash
# Create service file
sudo tee /etc/systemd/system/v7-runtime.service << 'SYSTEMD'
[Unit]
Description=V7 Ultimate Trading Runtime
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/crpbot
Environment=PATH=/root/crpbot/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/root/crpbot/.venv/bin/python3 apps/runtime/v7_runtime.py --iterations -1 --sleep-seconds 120
Restart=always
RestartSec=10
StandardOutput=append:/var/log/v7-runtime.log
StandardError=append:/var/log/v7-runtime-error.log

[Install]
WantedBy=multi-user.target
SYSTEMD

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable v7-runtime
sudo systemctl start v7-runtime
```

### Dashboard

**Foreground** (testing):
```bash
.venv/bin/python3 -m apps.dashboard.app
```

**Background** (production):
```bash
nohup .venv/bin/python3 -m apps.dashboard.app \
  > /tmp/dashboard.log 2>&1 &

echo $! > /tmp/dashboard.pid
```

**With systemd**:
```bash
sudo tee /etc/systemd/system/v7-dashboard.service << 'SYSTEMD'
[Unit]
Description=V7 Dashboard Web Interface
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/crpbot
Environment=PATH=/root/crpbot/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/root/crpbot/.venv/bin/python3 -m apps.dashboard.app
Restart=always
RestartSec=10
StandardOutput=append:/var/log/v7-dashboard.log
StandardError=append:/var/log/v7-dashboard-error.log

[Service]
WantedBy=multi-user.target
SYSTEMD

sudo systemctl daemon-reload
sudo systemctl enable v7-dashboard
sudo systemctl start v7-dashboard
```

### Verify Services

```bash
# Check runtime
ps aux | grep v7_runtime | grep -v grep

# Check dashboard
ps aux | grep "dashboard/app.py" | grep -v grep

# Test dashboard
curl http://localhost:5000/api/status

# Check logs
tail -f /tmp/v7_runtime.log
tail -f /tmp/dashboard.log
```

---

## Monitoring

### Health Checks

**Automated health check script**:

```bash
#!/bin/bash
# /root/crpbot/healthcheck.sh

check_runtime() {
    pgrep -f "v7_runtime.py" > /dev/null
    return $?
}

check_dashboard() {
    curl -s http://localhost:5000/api/status | grep -q '"status":"live"'
    return $?
}

check_database() {
    sqlite3 /root/crpbot/tradingai.db "SELECT COUNT(*) FROM signals" > /dev/null 2>&1
    return $?
}

if ! check_runtime; then
    echo "ALERT: V7 Runtime is down!"
    # Restart or send alert
fi

if ! check_dashboard; then
    echo "ALERT: Dashboard is down or unhealthy!"
fi

if ! check_database; then
    echo "ALERT: Database error!"
fi
```

**Run via cron** (every 5 minutes):
```bash
# crontab -e
*/5 * * * * /root/crpbot/healthcheck.sh >> /var/log/v7-healthcheck.log 2>&1
```

### Metrics to Monitor

**V7 Runtime**:
- Process uptime
- Signal generation rate (should be ~6/hour)
- Error count in logs
- API costs (daily/monthly)

**Dashboard**:
- HTTP response time
- Active connections
- Error rate

**System**:
- CPU usage
- Memory usage
- Disk space
- Network connectivity

### Logging

**Log locations**:
- V7 Runtime: `/tmp/v7_runtime.log` or `/var/log/v7-runtime.log`
- Dashboard: `/tmp/dashboard.log` or `/var/log/v7-dashboard.log`

**Log rotation** (recommended):
```bash
# /etc/logrotate.d/v7
/var/log/v7-*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
}
```

---

## Maintenance Tasks

### Daily

**Check status**:
```bash
# Quick status
systemctl status v7-runtime
systemctl status v7-dashboard

# Or via processes
ps aux | grep "v7_runtime\|dashboard"

# Check costs
curl -s http://localhost:5000/api/v7/costs | python3 -m json.tool
```

**Review logs**:
```bash
tail -100 /tmp/v7_runtime.log | grep -i "error\|warning"
```

### Weekly

**Database maintenance**:
```bash
# Vacuum (optimize)
sqlite3 tradingai.db "VACUUM"

# Check integrity
sqlite3 tradingai.db "PRAGMA integrity_check"

# Backup
cp tradingai.db tradingai.db.backup.$(date +%Y%m%d)
```

**Review performance**:
```bash
# Signal statistics
curl -s http://localhost:5000/api/v7/statistics | python3 -m json.tool

# Win rate (if tracking trades)
curl -s http://localhost:5000/api/v7/performance | python3 -m json.tool
```

### Monthly

**Update dependencies**:
```bash
cd /root/crpbot
git pull origin feature/v7-ultimate
uv sync --upgrade
```

**Cost analysis**:
```bash
# Check monthly spend
curl -s http://localhost:5000/api/v7/costs | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Monthly cost: \${data['month']['cost']:.2f} / \${data['month']['budget']:.2f}\")
print(f\"Percent used: {data['month']['percent_used']:.1f}%\")
"
```

**Security updates**:
```bash
sudo apt update
sudo apt upgrade -y
sudo reboot  # If kernel updated
```

---

## Troubleshooting

### V7 Not Generating Signals

**Diagnostic steps**:

1. **Check if running**:
```bash
ps aux | grep v7_runtime | grep -v grep
```

2. **Check logs for errors**:
```bash
tail -100 /tmp/v7_runtime.log | grep -i error
```

3. **Test API keys**:
```bash
# DeepSeek
curl -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
  https://api.deepseek.com/v1/models

# Coinbase
# (requires JWT signature - use test script)
.venv/bin/python3 test_coinbase_api.py
```

4. **Check rate limiting**:
```bash
# Should see signal every 10 minutes (6/hour)
grep "Signal generated" /tmp/v7_runtime.log | tail -10
```

### Dashboard Not Loading

1. **Check if running**:
```bash
ps aux | grep "app.py" | grep -v grep
lsof -i :5000
```

2. **Test API**:
```bash
curl http://localhost:5000/api/status
```

3. **Check browser**:
- Hard refresh: `Ctrl + Shift + R`
- Check console for JavaScript errors
- Verify Chart.js CDN is reachable

### High CPU/Memory Usage

**Normal usage**:
- V7 Runtime: 5-10% CPU, 200-300MB RAM
- Dashboard: 2-5% CPU, 100-150MB RAM

**If higher**:
```bash
# Check for memory leaks
top -p $(pgrep -f v7_runtime)

# Restart services
systemctl restart v7-runtime
systemctl restart v7-dashboard
```

### Database Locked

```bash
# Check for stale locks
fuser tradingai.db

# Kill processes holding lock
fuser -k tradingai.db

# Restart services
systemctl restart v7-runtime v7-dashboard
```

---

## Backup & Recovery

### Automated Backup

```bash
#!/bin/bash
# /root/crpbot/backup.sh

BACKUP_DIR="/root/crpbot/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup database
cp tradingai.db $BACKUP_DIR/tradingai_$DATE.db

# Backup .env (contains secrets)
cp .env $BACKUP_DIR/env_$DATE

# Backup models (if custom trained)
tar -czf $BACKUP_DIR/models_$DATE.tar.gz models/promoted/

# Keep last 30 days
find $BACKUP_DIR -name "*.db" -mtime +30 -delete
find $BACKUP_DIR -name "env_*" -mtime +30 -delete
find $BACKUP_DIR -name "models_*.tar.gz" -mtime +30 -delete

echo "Backup completed: $DATE"
```

**Schedule via cron**:
```bash
# Daily at 2 AM
0 2 * * * /root/crpbot/backup.sh >> /var/log/v7-backup.log 2>&1
```

### Recovery

**Restore database**:
```bash
# Stop services
systemctl stop v7-runtime v7-dashboard

# Restore
cp backups/tradingai_20251119_020000.db tradingai.db

# Restart
systemctl start v7-runtime v7-dashboard
```

**Disaster recovery** (complete reinstall):
```bash
# 1. Clone repo
git clone https://github.com/your-org/crpbot.git /root/crpbot-new
cd /root/crpbot-new

# 2. Restore .env
cp /root/crpbot/backups/env_latest .env

# 3. Install dependencies
uv sync

# 4. Restore database
cp /root/crpbot/backups/tradingai_latest.db tradingai.db

# 5. Start services
# (see Starting Services section)
```

---

## Scaling

### Vertical Scaling

**Upgrade resources**:
- More CPU: Handle more concurrent calculations
- More RAM: Larger data caches
- Faster disk: Quicker database queries

### Horizontal Scaling

**Multiple instances** (different symbols):
- Instance 1: BTC-USD only
- Instance 2: ETH-USD only
- Instance 3: SOL-USD only

**Load balancing** (dashboard):
```
nginx → Dashboard 1 (port 5000)
     → Dashboard 2 (port 5001)
     → Dashboard 3 (port 5002)
```

### Database Scaling

**Migrate to PostgreSQL**:

```bash
# 1. Export SQLite data
sqlite3 tradingai.db .dump > dump.sql

# 2. Set up PostgreSQL
sudo apt install postgresql
sudo -u postgres createdb tradingai

# 3. Import data
psql -U postgres tradingai < dump.sql

# 4. Update .env
DB_URL=postgresql+psycopg://postgres:password@localhost:5432/tradingai

# 5. Restart services
```

---

## Security

### API Keys

**Store securely**:
```bash
chmod 600 .env
chown root:root .env
```

**Rotate regularly**:
- DeepSeek API key: Every 90 days
- Coinbase API key: Every 180 days
- Telegram token: On compromise

### Firewall

**Allow only necessary ports**:
```bash
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 5000/tcp  # Dashboard (or use reverse proxy)
sudo ufw enable
```

**Better: Use reverse proxy**:
```nginx
# /etc/nginx/sites-available/v7-dashboard
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Add SSL with Let's Encrypt
    # listen 443 ssl;
    # ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    # ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
}
```

### Rate Limiting

**Prevent abuse**:
```python
# Add to apps/dashboard/app.py
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["60 per minute"]
)

@app.route('/api/v7/signals/<int:signal_id>/result', methods=['POST'])
@limiter.limit("10 per minute")
def api_v7_update_result(signal_id):
    # ... existing code
```

### Monitoring for Intrusions

```bash
# Install fail2ban
sudo apt install fail2ban

# Monitor dashboard access
sudo tee /etc/fail2ban/filter.d/v7-dashboard.conf << 'EOF'
[Definition]
failregex = ^<HOST> .* "POST /api/v7/signals/.*/result HTTP/.*" 400
            ^<HOST> .* "GET /api/v7/.* HTTP/.*" 500
ignoreregex =
