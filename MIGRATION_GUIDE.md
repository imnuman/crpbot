# CRPBot Migration Guide - Moving to Cloud Server

This guide walks you through migrating your CRPBot project from your local machine to a new Ubuntu cloud server.

## Prerequisites

**On Your Local Machine**:
- Git repository with latest code committed
- Data files (764MB in `data/`)
- Model files (3.2MB in `models/`)
- Environment configuration (`.env`, `.db_password`, etc.)
- AWS credentials configured

**On Cloud Server**:
- Ubuntu 20.04+ (recommended 22.04 LTS)
- Minimum 2 CPU cores, 4GB RAM, 20GB disk
- SSH access via Terminus app
- Internet connectivity

## Migration Overview

```
Local Machine                          Cloud Server
├── Code (git)         ──────────────> Git clone
├── Models (3.2MB)     ──────────────> SCP/rsync transfer
├── Data (764MB)       ──────────────> S3 download (recommended)
├── .env config        ──────────────> Secure copy
└── AWS credentials    ──────────────> AWS CLI configure
```

## Quick Migration (Recommended)

Use the automated deployment script:

```bash
# On local machine
./scripts/deploy_to_cloud.sh user@your-cloud-server-ip
```

## Step-by-Step Manual Migration

### Step 1: Prepare Local Machine (5 minutes)

```bash
# 1. Ensure all code is committed
cd /home/numan/crpbot
git status
git add .
git commit -m "Pre-migration commit"
git push origin main  # Push to GitHub/GitLab

# 2. Create migration package (code + config only, no data/models)
tar -czf ~/crpbot-config.tar.gz \
  --exclude='data/*' \
  --exclude='models/*' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  .env .db_password .rds_connection_info 2>/dev/null || true

# 3. Optional: Upload data to S3 (recommended for 764MB)
./scripts/upload_to_s3.sh data/raw/*.parquet
./scripts/upload_to_s3.sh data/features/*.parquet

# 4. Optional: Upload models to S3
aws s3 sync models/ s3://crpbot-market-data-dev/models/ \
  --exclude "*.gitkeep" \
  --exclude "__pycache__/*"
```

### Step 2: Setup Cloud Server (10 minutes)

```bash
# SSH into your cloud server using Terminus
ssh user@your-cloud-server-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y \
  python3.10 \
  python3.10-venv \
  python3-pip \
  git \
  postgresql-client \
  redis-tools \
  build-essential \
  curl \
  wget \
  unzip

# Install uv (ultra-fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Verify installations
python3 --version  # Should be 3.10+
uv --version
psql --version
```

### Step 3: Clone Repository (2 minutes)

```bash
# Clone from Git (recommended)
cd ~
git clone https://github.com/YOUR_USERNAME/crpbot.git
cd crpbot

# Or if using private repo with SSH
# git clone git@github.com:YOUR_USERNAME/crpbot.git

# Verify repository
ls -la
```

### Step 4: Transfer Configuration Files (3 minutes)

**Option A: Using SCP** (from local machine):
```bash
# Copy configuration archive
scp ~/crpbot-config.tar.gz user@your-cloud-server-ip:~/crpbot/

# On cloud server, extract
cd ~/crpbot
tar -xzf crpbot-config.tar.gz
rm crpbot-config.tar.gz

# Verify files
ls -la .env .db_password .rds_connection_info
```

**Option B: Manual Copy** (if tar method doesn't work):
```bash
# From local machine, copy individual files
scp /home/numan/crpbot/.env user@your-cloud-server-ip:~/crpbot/
scp /home/numan/crpbot/.db_password user@your-cloud-server-ip:~/crpbot/
scp /home/numan/crpbot/.rds_connection_info user@your-cloud-server-ip:~/crpbot/

# Set proper permissions on cloud server
chmod 600 ~/crpbot/.env ~/crpbot/.db_password ~/crpbot/.rds_connection_info
```

### Step 5: Install Dependencies (5 minutes)

```bash
cd ~/crpbot

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies with uv (faster)
pip install uv
uv pip install -e .
uv pip install -e ".[dev]"

# Verify installation
python -c "import torch; import pandas; print('Dependencies OK')"
```

### Step 6: Setup AWS CLI (3 minutes)

```bash
# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf aws awscliv2.zip

# Configure AWS credentials
aws configure
# Enter your:
#   AWS Access Key ID: [from your AWS account]
#   AWS Secret Access Key: [from your AWS account]
#   Default region: us-east-1
#   Default output format: json

# Verify AWS access
aws s3 ls s3://crpbot-market-data-dev/
aws secretsmanager list-secrets --region us-east-1
```

### Step 7: Download Data from S3 (10 minutes)

```bash
cd ~/crpbot

# Create directories
mkdir -p data/raw data/features models

# Download data from S3
aws s3 sync s3://crpbot-market-data-dev/data/raw/ data/raw/ \
  --exclude "*.gitkeep"

aws s3 sync s3://crpbot-market-data-dev/data/features/ data/features/ \
  --exclude "*.gitkeep"

# Verify data
ls -lh data/raw/
ls -lh data/features/
```

### Step 8: Download Models (2 minutes)

**Option A: From S3** (recommended):
```bash
# Download all models from S3
aws s3 sync s3://crpbot-market-data-dev/models/ models/ \
  --exclude "*.gitkeep" \
  --exclude ".DS_Store"

# Verify models
ls -lh models/
ls -lh models/gpu_trained_proper/
```

**Option B: Transfer from Local Machine**:
```bash
# From local machine
rsync -avz --progress /home/numan/crpbot/models/ \
  user@your-cloud-server-ip:~/crpbot/models/

# Or using tar + scp for faster transfer
cd /home/numan/crpbot
tar -czf ~/crpbot-models.tar.gz models/
scp ~/crpbot-models.tar.gz user@your-cloud-server-ip:~/crpbot/

# On cloud server
cd ~/crpbot
tar -xzf crpbot-models.tar.gz
rm crpbot-models.tar.gz
```

### Step 9: Test Database Connection (2 minutes)

```bash
cd ~/crpbot
source .venv/bin/activate

# Test RDS PostgreSQL connection
PGPASSWORD="$(cat .db_password)" psql \
  -h crpbot-rds-postgres-db.cyjcoys82evx.us-east-1.rds.amazonaws.com \
  -p 5432 \
  -U crpbot_admin \
  -d crpbot \
  -c "SELECT version();"

# Should show: PostgreSQL 16.10

# Test with Python script
python test_runtime_connection.py
```

### Step 10: Run Tests (5 minutes)

```bash
cd ~/crpbot
source .venv/bin/activate

# Run unit tests
make test

# Run smoke test (5-minute backtest)
make smoke

# Test runtime in dry-run mode
python apps/runtime/main.py --mode dryrun --iterations 3 --sleep-seconds 10
```

### Step 11: Setup as System Service (Optional, 10 minutes)

```bash
# Create systemd service for runtime
sudo tee /etc/systemd/system/crpbot.service > /dev/null <<EOF
[Unit]
Description=CRPBot Trading Runtime
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$HOME/crpbot
Environment="PATH=$HOME/crpbot/.venv/bin:$PATH"
ExecStart=$HOME/crpbot/.venv/bin/python apps/runtime/main.py --mode dryrun --iterations -1
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable crpbot.service
sudo systemctl start crpbot.service

# Check status
sudo systemctl status crpbot.service

# View logs
sudo journalctl -u crpbot.service -f
```

## Verification Checklist

After migration, verify all components:

- [ ] Code repository cloned successfully
- [ ] Python 3.10+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`uv pip install -e .`)
- [ ] AWS CLI configured with correct credentials
- [ ] `.env` file present with correct values
- [ ] `.db_password` file present
- [ ] Data directory populated (764MB)
- [ ] Models directory populated (3.2MB)
- [ ] RDS PostgreSQL connection working
- [ ] S3 access working (`aws s3 ls s3://crpbot-market-data-dev/`)
- [ ] Unit tests passing (`make test`)
- [ ] Smoke test passing (`make smoke`)
- [ ] Runtime starts in dry-run mode

## Estimated Transfer Times

| Component | Size | Transfer Method | Time |
|-----------|------|----------------|------|
| Code | ~10MB | git clone | 1 min |
| Config | <1KB | scp | <1 min |
| Models | 3.2MB | S3 or scp | 2 min |
| Data | 764MB | S3 (recommended) | 5-10 min |
| Data | 764MB | scp/rsync | 15-30 min |

**Total Time**: 30-60 minutes (depending on internet speed)

## Ongoing Maintenance

### Daily Operations

```bash
# Activate environment
cd ~/crpbot
source .venv/bin/activate

# Check runtime status (if using systemd)
sudo systemctl status crpbot.service

# View logs
tail -f logs/runtime.log

# Export metrics
make export-metrics
```

### Weekly Tasks

```bash
# Update dependencies
cd ~/crpbot
source .venv/bin/activate
uv lock --upgrade
uv pip sync

# Pull latest code
git pull origin main

# Restart service if changes made
sudo systemctl restart crpbot.service
```

### Backup Strategy

```bash
# Backup database (already in infra/scripts/backup_db.sh)
./infra/scripts/backup_db.sh

# Backup models to S3
aws s3 sync models/ s3://crpbot-market-data-dev/models-backup-$(date +%Y%m%d)/

# Backup configuration
tar -czf ~/backups/crpbot-config-$(date +%Y%m%d).tar.gz \
  .env .db_password .rds_connection_info
```

## Troubleshooting

### Issue: "Cannot connect to RDS"
```bash
# Check security group allows your cloud server IP
# Verify credentials
cat .db_password
# Test connectivity
nc -zv crpbot-rds-postgres-db.cyjcoys82evx.us-east-1.rds.amazonaws.com 5432
```

### Issue: "AWS S3 access denied"
```bash
# Verify AWS credentials
aws sts get-caller-identity
# Reconfigure if needed
aws configure
```

### Issue: "Models not found"
```bash
# List available models
ls -la models/
# Re-download from S3
aws s3 sync s3://crpbot-market-data-dev/models/ models/
```

### Issue: "Module import errors"
```bash
# Reinstall dependencies
source .venv/bin/activate
pip install --upgrade pip
uv pip install -e . --force-reinstall
```

## Security Best Practices

1. **Firewall Configuration**:
   ```bash
   sudo ufw enable
   sudo ufw allow 22/tcp   # SSH
   sudo ufw allow from 10.0.0.0/8 to any port 5432  # RDS (if needed)
   ```

2. **File Permissions**:
   ```bash
   chmod 600 .env .db_password .rds_connection_info
   chmod 700 ~/.aws
   ```

3. **SSH Hardening** (optional):
   ```bash
   # Disable password auth, use SSH keys only
   sudo nano /etc/ssh/sshd_config
   # Set: PasswordAuthentication no
   sudo systemctl restart sshd
   ```

4. **Monitoring**:
   ```bash
   # Install monitoring tools
   sudo apt install htop nethogs iotop
   ```

## Next Steps After Migration

1. **Validate GPU Models**: Run evaluation on GPU models
   ```bash
   python scripts/evaluate_model.py --model models/gpu_trained_proper/BTC_lstm_model.pt
   ```

2. **Start Dry-Run Observation**: Monitor for 3-5 days
   ```bash
   make run-dry
   ```

3. **Paper Trading**: If models validate, start paper trading

4. **Production Deployment**: After successful paper trading

## Support

- **Documentation**: See `PHASE1_COMPLETE_NEXT_STEPS.md`, `MASTER_SUMMARY.md`
- **Issues**: Check logs in `logs/` directory
- **AWS Console**: Monitor RDS, S3, Secrets Manager for issues
