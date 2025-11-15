# Cloud Server Quick Start Guide

Quick reference for working with CRPBot on your Ubuntu cloud server after migration.

## Connect to Server (Terminus SSH)

```bash
# Basic SSH connection
ssh user@your-cloud-server-ip

# With specific key (if needed)
ssh -i ~/.ssh/your-key.pem user@your-cloud-server-ip
```

## Essential Daily Commands

### Activate Environment
```bash
cd ~/crpbot
source .venv/bin/activate
```

### Check Runtime Status
```bash
# If running as systemd service
sudo systemctl status crpbot.service

# View live logs
sudo journalctl -u crpbot.service -f

# Or check process manually
ps aux | grep python
```

### Start/Stop Runtime
```bash
# Manual run (dry-run mode)
cd ~/crpbot
source .venv/bin/activate
python apps/runtime/main.py --mode dryrun --iterations -1 --sleep-seconds 120

# As systemd service
sudo systemctl start crpbot.service    # Start
sudo systemctl stop crpbot.service     # Stop
sudo systemctl restart crpbot.service  # Restart
sudo systemctl status crpbot.service   # Check status
```

### View Logs
```bash
# Application logs
tail -f ~/crpbot/logs/runtime.log

# System logs (if using systemd)
sudo journalctl -u crpbot.service -f

# Last 100 lines
sudo journalctl -u crpbot.service -n 100
```

## Common Operations

### Update Code from Git
```bash
cd ~/crpbot
git pull origin main
source .venv/bin/activate
uv pip install -e .  # Reinstall if dependencies changed
sudo systemctl restart crpbot.service  # If using service
```

### Run Tests
```bash
cd ~/crpbot
source .venv/bin/activate

# All tests
make test

# Specific test
pytest tests/unit/test_ensemble.py -v

# Smoke test
make smoke
```

### Check AWS Connectivity
```bash
# Test AWS credentials
aws sts get-caller-identity

# List S3 buckets
aws s3 ls s3://crpbot-market-data-dev/

# Test RDS connection
PGPASSWORD="$(cat .db_password)" psql \
  -h crpbot-rds-postgres-db.cyjcoys82evx.us-east-1.rds.amazonaws.com \
  -p 5432 \
  -U crpbot_admin \
  -d crpbot \
  -c "SELECT COUNT(*) FROM trading.signals;"
```

### Download Latest Data/Models from S3
```bash
cd ~/crpbot
source .venv/bin/activate

# Download new data
aws s3 sync s3://crpbot-market-data-dev/data/raw/ data/raw/

# Download new models
aws s3 sync s3://crpbot-market-data-dev/models/ models/
```

### Train Models on Server
```bash
cd ~/crpbot
source .venv/bin/activate

# Train LSTM for BTC
make train COIN=BTC EPOCHS=15

# Or directly
python apps/trainer/main.py --task lstm --coin BTC --epochs 15
```

### Evaluate Models
```bash
cd ~/crpbot
source .venv/bin/activate

python scripts/evaluate_model.py \
  --model models/lstm_BTC_USD_1m_a7aff5c4.pt \
  --symbol BTC-USD \
  --model-type lstm
```

## System Monitoring

### Check System Resources
```bash
# CPU, Memory, Disk usage
htop

# Disk space
df -h

# Memory usage
free -h

# Network traffic
sudo nethogs

# I/O usage
sudo iotop
```

### Check Python Process
```bash
# Find Python processes
ps aux | grep python

# Monitor specific process
top -p $(pgrep -f "apps/runtime/main.py")
```

## Troubleshooting

### Runtime Won't Start
```bash
# Check logs for errors
tail -n 50 ~/crpbot/logs/runtime.log

# Check systemd logs
sudo journalctl -u crpbot.service -n 50

# Try manual run to see errors
cd ~/crpbot
source .venv/bin/activate
python apps/runtime/main.py --mode dryrun --iterations 1
```

### Database Connection Failed
```bash
# Verify credentials
cat ~/crpbot/.db_password

# Test connection
PGPASSWORD="$(cat ~/crpbot/.db_password)" psql \
  -h crpbot-rds-postgres-db.cyjcoys82evx.us-east-1.rds.amazonaws.com \
  -p 5432 \
  -U crpbot_admin \
  -d crpbot \
  -c "\dt trading.*"

# Check security group allows your server IP
# AWS Console → RDS → Security Groups
```

### AWS S3 Access Denied
```bash
# Check AWS credentials
aws configure list

# Verify identity
aws sts get-caller-identity

# Reconfigure if needed
aws configure
```

### Disk Space Running Low
```bash
# Check large files
du -h ~/ | sort -rh | head -20

# Clean up
rm -rf ~/.cache/pip
rm -rf ~/crpbot/__pycache__
find ~/crpbot -name "*.pyc" -delete

# Clean old logs
find ~/crpbot/logs -name "*.log" -mtime +7 -delete
```

### Dependencies Broken
```bash
cd ~/crpbot
source .venv/bin/activate

# Reinstall everything
pip install --upgrade pip
uv pip install -e . --force-reinstall
uv pip install -e ".[dev]" --force-reinstall
```

## Security Maintenance

### Update System Packages
```bash
sudo apt update
sudo apt upgrade -y
sudo apt autoremove -y
```

### Check Firewall
```bash
# View firewall status
sudo ufw status verbose

# Allow SSH if not already
sudo ufw allow 22/tcp

# Enable firewall
sudo ufw enable
```

### Review File Permissions
```bash
cd ~/crpbot

# Secure sensitive files
chmod 600 .env .db_password .rds_connection_info
chmod 700 ~/.aws

# Check permissions
ls -la .env .db_password .rds_connection_info
```

## Backup & Recovery

### Manual Backup
```bash
# Backup configuration
tar -czf ~/backups/crpbot-config-$(date +%Y%m%d).tar.gz \
  ~/crpbot/.env \
  ~/crpbot/.db_password \
  ~/crpbot/.rds_connection_info

# Backup models to S3
cd ~/crpbot
aws s3 sync models/ s3://crpbot-market-data-dev/models-backup-$(date +%Y%m%d)/
```

### Restore from Backup
```bash
# Restore configuration
cd ~
tar -xzf backups/crpbot-config-YYYYMMDD.tar.gz

# Restore models from S3
cd ~/crpbot
aws s3 sync s3://crpbot-market-data-dev/models-backup-YYYYMMDD/ models/
```

## Performance Optimization

### Monitor Runtime Performance
```bash
# Watch resource usage
watch -n 5 'ps aux | grep python | grep -v grep'

# Profile memory usage
cd ~/crpbot
source .venv/bin/activate
python -m memory_profiler apps/runtime/main.py --mode dryrun --iterations 1
```

### Optimize for Production
```bash
# Use production mode instead of dry-run
cd ~/crpbot
source .venv/bin/activate

# Edit .env file
nano .env
# Set: RUNTIME_MODE=live

# Restart
sudo systemctl restart crpbot.service
```

## Scheduled Tasks (Cron)

### Setup Daily Tasks
```bash
# Edit crontab
crontab -e

# Add these lines:

# Daily backup at 2 AM
0 2 * * * cd ~/crpbot && ./infra/scripts/backup_db.sh

# Update data at 6 AM
0 6 * * * cd ~/crpbot && source .venv/bin/activate && python scripts/fetch_data.py --symbol BTC-USD --interval 1m

# Check runtime health every hour
0 * * * * systemctl is-active --quiet crpbot.service || systemctl start crpbot.service
```

## Quick Reference Card

| Task | Command |
|------|---------|
| Connect | `ssh user@server-ip` |
| Activate env | `cd ~/crpbot && source .venv/bin/activate` |
| Start runtime | `sudo systemctl start crpbot.service` |
| Stop runtime | `sudo systemctl stop crpbot.service` |
| View logs | `sudo journalctl -u crpbot.service -f` |
| Run tests | `make test` |
| Update code | `git pull origin main` |
| Check AWS | `aws sts get-caller-identity` |
| Test DB | `python test_runtime_connection.py` |
| Download models | `aws s3 sync s3://crpbot-market-data-dev/models/ models/` |

## Getting Help

- **Full Documentation**: See `MIGRATION_GUIDE.md`
- **Architecture**: See `CLAUDE.md`
- **Phase Status**: See `PHASE1_COMPLETE_NEXT_STEPS.md`
- **V3 System**: See `v3_ultimate/README.md`
- **Logs**: Check `~/crpbot/logs/` for detailed error messages

## Emergency Procedures

### Kill Switch (Stop All Trading)
```bash
# Stop runtime immediately
sudo systemctl stop crpbot.service

# Or set kill switch
cd ~/crpbot
nano .env
# Set: KILL_SWITCH=true

# Restart
sudo systemctl restart crpbot.service
```

### Rollback to Previous Version
```bash
cd ~/crpbot
git log --oneline -10  # Find commit hash
git reset --hard COMMIT_HASH
source .venv/bin/activate
uv pip install -e .
sudo systemctl restart crpbot.service
```

### Complete Reset
```bash
# Backup first!
tar -czf ~/crpbot-backup-$(date +%Y%m%d).tar.gz ~/crpbot/

# Remove and redeploy
rm -rf ~/crpbot
# Then run deployment script from local machine
```
