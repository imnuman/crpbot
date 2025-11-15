# Quick Server Commands Reference

One-page reference for working on your cloud server with Claude Code CLI.

---

## 🚀 First Time Setup

### On Your Local Machine (One Time)
```bash
cd /home/numan/crpbot

# Sync credentials to cloud
./scripts/sync_credentials.sh user@your-server-ip

# Or with SSH key:
./scripts/sync_credentials.sh -k ~/.ssh/your_key user@your-server-ip
```

### On Cloud Server (One Time)
```bash
# Follow the complete setup guide
cat ~/crpbot/CLOUD_SERVER_SETUP.md

# Or quick setup:
# 1. Update system
sudo apt update && sudo apt upgrade -y && sudo apt install -y build-essential curl wget git python3.10 python3.10-venv python3-pip

# 2. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.cargo/env

# 3. Install AWS CLI
cd /tmp && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && unzip awscliv2.zip && sudo ./aws/install && cd ~

# 4. Clone repo
cd ~ && git clone https://github.com/YOUR_USERNAME/crpbot.git

# 5. Setup Python
cd ~/crpbot && python3 -m venv .venv && source .venv/bin/activate && pip install uv && uv pip install -e .

# 6. Download data
aws s3 sync s3://crpbot-market-data-dev/data/raw/ data/raw/
aws s3 sync s3://crpbot-market-data-dev/models/ models/

# 7. Test
make test
```

---

## 📝 Daily Commands

### Connect & Activate
```bash
# Connect to server
ssh crpbot-cloud  # or: ssh user@server-ip

# Go to project and activate environment
cd ~/crpbot && source .venv/bin/activate

# Or if you added alias (see CLOUD_SERVER_SETUP.md Step 16):
crpbot
```

### Pull Latest Code
```bash
# If changes were made locally and pushed
git pull origin main

# Reinstall if dependencies changed
uv pip install -e .
```

### Run Tests
```bash
# All tests
make test

# Specific test
pytest tests/unit/test_ensemble.py -v

# With coverage
pytest --cov=apps tests/
```

### Start Runtime
```bash
# Dry-run mode (testing)
make run-dry

# Or manually with iterations
python apps/runtime/main.py --mode dryrun --iterations 10 --sleep-seconds 60

# Live mode (production)
python apps/runtime/main.py --mode live --iterations -1
```

### Claude Code CLI
```bash
# Start Claude Code in project
cd ~/crpbot
source .venv/bin/activate
claude-code .

# Or with specific task
claude-code "Help me debug the failing tests"
claude-code "Review the runtime code"
claude-code "Add a new feature to track metrics"
```

---

## 🔄 Sync Between Local & Cloud

### After Working on Cloud
```bash
# On cloud server - commit and push
cd ~/crpbot
git add .
git commit -m "Changes made on cloud"
git push origin main

# On local machine - pull changes
cd /home/numan/crpbot
git pull origin main
```

### After Working Locally
```bash
# On local machine - commit and push
cd /home/numan/crpbot
git add .
git commit -m "Changes made locally"
git push origin main

# On cloud server - pull changes
cd ~/crpbot
git pull origin main
```

### Sync Models
```bash
# Upload models to S3 (from either machine)
aws s3 sync models/ s3://crpbot-market-data-dev/models/

# Download models from S3 (on other machine)
aws s3 sync s3://crpbot-market-data-dev/models/ models/
```

---

## 📊 Monitoring & Logs

### View Logs
```bash
# Runtime logs
tail -f ~/crpbot/logs/runtime.log

# Error logs
tail -f ~/crpbot/logs/runtime.error.log

# Systemd logs (if using service)
sudo journalctl -u crpbot.service -f

# Last 100 lines
sudo journalctl -u crpbot.service -n 100
```

### Check Processes
```bash
# Find Python processes
ps aux | grep python

# Check if runtime is running
ps aux | grep "apps/runtime/main.py"

# System resources
htop
```

### System Status
```bash
# Disk usage
df -h

# Memory
free -h

# CPU load
uptime
```

---

## 🛠️ Common Tasks

### Train Models
```bash
cd ~/crpbot
source .venv/bin/activate

# Train LSTM
make train COIN=BTC EPOCHS=15

# Train Transformer
python apps/trainer/main.py --task transformer --epochs 15

# Evaluate model
python scripts/evaluate_model.py --model models/lstm_BTC_USD_1m_*.pt --symbol BTC-USD
```

### Fetch New Data
```bash
cd ~/crpbot
source .venv/bin/activate

# Fetch latest data
python scripts/fetch_data.py --symbol BTC-USD --interval 1m --start 2025-01-01

# Engineer features
python scripts/engineer_features.py --input data/raw/BTC-USD_*.parquet --symbol BTC-USD
```

### Database Operations
```bash
# Connect to PostgreSQL
PGPASSWORD="$(cat .db_password)" psql \
  -h crpbot-rds-postgres-db.cyjcoys82evx.us-east-1.rds.amazonaws.com \
  -p 5432 \
  -U crpbot_admin \
  -d crpbot

# Query signals
PGPASSWORD="$(cat .db_password)" psql \
  -h crpbot-rds-postgres-db.cyjcoys82evx.us-east-1.rds.amazonaws.com \
  -p 5432 \
  -U crpbot_admin \
  -d crpbot \
  -c "SELECT * FROM trading.signals ORDER BY timestamp DESC LIMIT 10;"
```

### AWS Operations
```bash
# Check AWS identity
aws sts get-caller-identity

# List S3 buckets
aws s3 ls

# List files in bucket
aws s3 ls s3://crpbot-market-data-dev/ --recursive

# Download specific file
aws s3 cp s3://crpbot-market-data-dev/path/to/file.parquet ./
```

---

## 🔧 Troubleshooting

### Environment Not Working
```bash
# Recreate virtual environment
cd ~/crpbot
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install uv
uv pip install -e .
```

### Database Connection Failed
```bash
# Check password file
cat ~/crpbot/.db_password

# Test connection
PGPASSWORD="$(cat .db_password)" psql \
  -h crpbot-rds-postgres-db.cyjcoys82evx.us-east-1.rds.amazonaws.com \
  -p 5432 \
  -U crpbot_admin \
  -d crpbot \
  -c "SELECT 1;"

# Check network connectivity
nc -zv crpbot-rds-postgres-db.cyjcoys82evx.us-east-1.rds.amazonaws.com 5432
```

### Git Issues
```bash
# Discard local changes
git reset --hard HEAD

# Pull latest (force)
git fetch origin
git reset --hard origin/main

# View commit history
git log --oneline -10
```

### Disk Space
```bash
# Check usage
df -h
du -sh ~/crpbot/*

# Clean up
rm -rf ~/.cache/pip
find ~/crpbot -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find ~/crpbot -name "*.pyc" -delete
```

---

## 🚦 Systemd Service (If Configured)

```bash
# Start service
sudo systemctl start crpbot

# Stop service
sudo systemctl stop crpbot

# Restart service
sudo systemctl restart crpbot

# Check status
sudo systemctl status crpbot

# Enable on boot
sudo systemctl enable crpbot

# Disable on boot
sudo systemctl disable crpbot

# View logs
sudo journalctl -u crpbot.service -f
```

---

## 📋 Environment Variables

Check current configuration:
```bash
cd ~/crpbot
grep -E "RUNTIME_MODE|CONFIDENCE_THRESHOLD|DB_URL" .env
```

Common variables:
```bash
RUNTIME_MODE=dryrun          # or 'live'
CONFIDENCE_THRESHOLD=0.75
DB_URL=postgresql://...
AWS_REGION=us-east-1
KILL_SWITCH=false
```

---

## 🔐 Security Checks

```bash
# Check file permissions
ls -la ~/crpbot/.env ~/crpbot/.db_password ~/.aws/credentials

# Should all be 600 (rw-------)

# Fix if needed
chmod 600 ~/crpbot/.env ~/crpbot/.db_password
chmod 700 ~/.aws
chmod 600 ~/.aws/credentials
```

---

## 📞 Quick Help

| Issue | Command |
|-------|---------|
| Can't connect to DB | Check `.db_password` exists and RDS security group |
| Tests failing | `git pull && uv pip install -e .` |
| Out of sync | `git status` on both machines |
| Need latest models | `aws s3 sync s3://crpbot-market-data-dev/models/ models/` |
| Python import errors | Reinstall: `uv pip install -e . --force-reinstall` |
| Disk full | Clean cache: `rm -rf ~/.cache/pip` |

---

## 📚 Documentation

- Full setup: `cat ~/crpbot/CLOUD_SERVER_SETUP.md`
- Daily ops: `cat ~/crpbot/CLOUD_SERVER_QUICKSTART.md`
- Migration: `cat ~/crpbot/MIGRATION_GUIDE.md`
- Architecture: `cat ~/crpbot/CLAUDE.md`

---

## 💡 Pro Tips

```bash
# Add aliases to ~/.bashrc (see CLOUD_SERVER_SETUP.md Step 16)
alias crpbot='cd ~/crpbot && source .venv/bin/activate'
alias crpbot-test='cd ~/crpbot && source .venv/bin/activate && make test'
alias crpbot-run='cd ~/crpbot && source .venv/bin/activate && make run-dry'

# Then just use:
crpbot          # Go to project and activate
crpbot-test     # Run tests
crpbot-run      # Start runtime
```

---

**Print this page or keep it handy for quick reference!** 📄
