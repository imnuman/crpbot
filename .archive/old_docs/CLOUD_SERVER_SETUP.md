# Cloud Server Setup Instructions
## For Claude Code CLI Usage

Follow these steps **while connected to your cloud server via SSH**.

---

## 📋 Overview

You'll set up:
1. ✅ System dependencies (Python, Git, AWS CLI, etc.)
2. ✅ Clone CRPBot repository
3. ✅ Transfer credentials from local machine
4. ✅ Install Python dependencies
5. ✅ Install & configure Claude Code CLI
6. ✅ Verify everything works

**Estimated Time**: 30-40 minutes

---

## Part 1: System Setup (On Cloud Server)

### Step 1: Update System (2 minutes)

```bash
# Update package lists
sudo apt update

# Upgrade existing packages
sudo apt upgrade -y

# Install essential build tools
sudo apt install -y \
  build-essential \
  curl \
  wget \
  git \
  unzip \
  htop \
  tree
```

### Step 2: Install Python 3.10+ (3 minutes)

```bash
# Install Python 3.10
sudo apt install -y \
  python3.10 \
  python3.10-venv \
  python3.10-dev \
  python3-pip

# Verify installation
python3 --version  # Should show 3.10 or higher

# Set Python 3.10 as default (optional)
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
```

### Step 3: Install uv Package Manager (1 minute)

```bash
# Install uv (ultra-fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to current shell
source $HOME/.cargo/env

# Verify
uv --version
```

### Step 4: Install AWS CLI (2 minutes)

```bash
# Download AWS CLI v2
cd /tmp
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"

# Extract and install
unzip awscliv2.zip
sudo ./aws/install

# Clean up
rm -rf aws awscliv2.zip

# Verify
aws --version
```

### Step 5: Install PostgreSQL Client & Redis Tools (1 minute)

```bash
# Install database clients
sudo apt install -y postgresql-client redis-tools

# Verify
psql --version
redis-cli --version
```

---

## Part 2: Clone Repository (On Cloud Server)

### Step 6: Clone CRPBot Repository (2 minutes)

```bash
# Go to home directory
cd ~

# Clone repository
git clone https://github.com/YOUR_USERNAME/crpbot.git

# Or if using SSH:
# git clone git@github.com:YOUR_USERNAME/crpbot.git

# Enter directory
cd crpbot

# Verify
ls -la
git log --oneline -5
```

**Note**: If you don't have a git remote, we'll transfer files from your local machine in the next step.

---

## Part 3: Transfer Credentials (From Local Machine)

### Step 7: Transfer Credentials from Local to Cloud

**⚠️ IMPORTANT**: Run these commands **on your LOCAL machine** (not on the cloud server).

Open a **new terminal** on your local machine:

```bash
# Set your server details
SERVER="user@your-cloud-server-ip"  # Replace with your actual server
# Or if you have SSH config: SERVER="crpbot-cloud"

# Transfer .env file
scp /home/numan/crpbot/.env $SERVER:~/crpbot/

# Transfer database password
scp /home/numan/crpbot/.db_password $SERVER:~/crpbot/

# Transfer RDS connection info (if exists)
scp /home/numan/crpbot/.rds_connection_info $SERVER:~/crpbot/ 2>/dev/null || true

# Transfer AWS credentials directory
scp -r ~/.aws $SERVER:~/

# Set proper permissions on cloud server
ssh $SERVER "chmod 600 ~/crpbot/.env ~/crpbot/.db_password ~/crpbot/.rds_connection_info 2>/dev/null && chmod 700 ~/.aws && chmod 600 ~/.aws/credentials ~/.aws/config"
```

**If using SSH key**:
```bash
# Add -i flag with your key
scp -i ~/.ssh/your_key /home/numan/crpbot/.env $SERVER:~/crpbot/
scp -i ~/.ssh/your_key /home/numan/crpbot/.db_password $SERVER:~/crpbot/
scp -i ~/.ssh/your_key /home/numan/crpbot/.rds_connection_info $SERVER:~/crpbot/ 2>/dev/null || true
scp -i ~/.ssh/your_key -r ~/.aws $SERVER:~/
```

**Back on Cloud Server**: Verify credentials were transferred

```bash
# Check files exist
ls -la ~/crpbot/.env ~/crpbot/.db_password ~/.aws/credentials

# Should show all files with 600 permissions

# Verify .env content (first few lines, don't expose secrets)
head -5 ~/crpbot/.env
```

---

## Part 4: Python Environment Setup (On Cloud Server)

### Step 8: Create Virtual Environment (1 minute)

```bash
cd ~/crpbot

# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Verify
which python  # Should show ~/crpbot/.venv/bin/python
python --version
```

### Step 9: Install Dependencies (5 minutes)

```bash
# Still in ~/crpbot with .venv activated
cd ~/crpbot
source .venv/bin/activate

# Install package manager
pip install -U pip uv

# Install project dependencies
uv pip install -e .

# Install dev dependencies
uv pip install -e ".[dev]"

# Verify installation
python -c "import torch; import pandas; import numpy; print('✅ Core dependencies OK')"
```

---

## Part 5: Download Data & Models (On Cloud Server)

### Step 10: Download from S3 (10 minutes)

```bash
cd ~/crpbot
source .venv/bin/activate

# Test AWS credentials
aws sts get-caller-identity

# If that works, download data and models:

# Create directories
mkdir -p data/raw data/features models

# Download raw data
echo "Downloading raw data from S3..."
aws s3 sync s3://crpbot-market-data-dev/data/raw/ data/raw/ \
  --exclude "*.gitkeep" \
  --exclude ".DS_Store"

# Download features
echo "Downloading features from S3..."
aws s3 sync s3://crpbot-market-data-dev/data/features/ data/features/ \
  --exclude "*.gitkeep"

# Download models
echo "Downloading models from S3..."
aws s3 sync s3://crpbot-market-data-dev/models/ models/ \
  --exclude "*.gitkeep"

# Verify
echo ""
echo "Data size: $(du -sh data/ | cut -f1)"
echo "Models size: $(du -sh models/ | cut -f1)"
echo ""
ls -lh data/raw/ | head -5
ls -lh models/
```

**If AWS credentials not configured yet**:
```bash
# Configure AWS CLI
aws configure

# Enter when prompted:
# AWS Access Key ID: [paste from local ~/.aws/credentials]
# AWS Secret Access Key: [paste from local ~/.aws/credentials]
# Default region: us-east-1
# Default output format: json

# Then run the download commands above
```

---

## Part 6: Install Claude Code CLI (On Cloud Server)

### Step 11: Install Claude Code CLI (3 minutes)

```bash
# Install Claude Code CLI (if available via npm/pip)
# Check the official Claude Code documentation for the latest installation method

# Example installation methods (use the one that applies):

# Method 1: If available via npm
# curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
# sudo apt install -y nodejs
# npm install -g @anthropic/claude-code-cli

# Method 2: If available via pip
# pip install claude-code-cli

# Method 3: If available as a binary download
# wget https://example.com/claude-code-cli -O ~/bin/claude-code
# chmod +x ~/bin/claude-code

# Verify installation
claude-code --version  # or whatever the command is
```

**Note**: Replace with actual Claude Code CLI installation commands. Check:
- https://docs.anthropic.com/claude/docs/claude-code
- Or the specific CLI installation guide

### Step 12: Configure Claude Code CLI (2 minutes)

```bash
# Initialize Claude Code in the project
cd ~/crpbot

# Set up Claude Code for this project
claude-code init  # or equivalent command

# Configure API key (if needed)
# This might be stored in ~/.config/claude/ or similar
# Follow the prompts from Claude Code CLI
```

---

## Part 7: Verification & Testing (On Cloud Server)

### Step 13: Test Database Connection (2 minutes)

```bash
cd ~/crpbot
source .venv/bin/activate

# Test PostgreSQL connection
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

### Step 14: Run Tests (5 minutes)

```bash
cd ~/crpbot
source .venv/bin/activate

# Run unit tests
pytest tests/unit/ -v

# Or use make
make test

# Run smoke test
make smoke
```

### Step 15: Test Runtime (2 minutes)

```bash
cd ~/crpbot
source .venv/bin/activate

# Test runtime in dry-run mode (3 iterations)
python apps/runtime/main.py --mode dryrun --iterations 3 --sleep-seconds 10

# Should complete without errors
```

---

## Part 8: Environment Configuration (On Cloud Server)

### Step 16: Add Convenience Aliases (1 minute)

```bash
# Add to ~/.bashrc for convenience
cat >> ~/.bashrc <<'EOF'

# CRPBot shortcuts
alias crpbot='cd ~/crpbot && source .venv/bin/activate'
alias crpbot-test='cd ~/crpbot && source .venv/bin/activate && make test'
alias crpbot-run='cd ~/crpbot && source .venv/bin/activate && make run-dry'
alias crpbot-logs='tail -f ~/crpbot/logs/*.log'

# Auto-activate when entering directory
cd() {
    builtin cd "$@"
    if [ -f .venv/bin/activate ]; then
        source .venv/bin/activate
    fi
}
EOF

# Reload bashrc
source ~/.bashrc

# Now you can use:
# crpbot              -> cd to project and activate venv
# crpbot-test         -> run tests
# crpbot-run          -> run in dry-run mode
```

---

## Part 9: Optional - Set Up as Systemd Service (On Cloud Server)

### Step 17: Create Systemd Service (5 minutes)

```bash
# Create service file
sudo tee /etc/systemd/system/crpbot.service > /dev/null <<EOF
[Unit]
Description=CRPBot Trading Runtime
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/crpbot
Environment="PATH=$HOME/crpbot/.venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$HOME/crpbot/.venv/bin/python apps/runtime/main.py --mode dryrun --iterations -1
Restart=always
RestartSec=10
StandardOutput=append:$HOME/crpbot/logs/runtime.log
StandardError=append:$HOME/crpbot/logs/runtime.error.log

[Install]
WantedBy=multi-user.target
EOF

# Create logs directory
mkdir -p ~/crpbot/logs

# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable crpbot.service

# Start service
sudo systemctl start crpbot.service

# Check status
sudo systemctl status crpbot.service

# View logs
sudo journalctl -u crpbot.service -f
```

**Service commands**:
```bash
sudo systemctl start crpbot      # Start
sudo systemctl stop crpbot       # Stop
sudo systemctl restart crpbot    # Restart
sudo systemctl status crpbot     # Check status
sudo journalctl -u crpbot -f     # Follow logs
```

---

## ✅ Verification Checklist

After completing all steps, verify:

```bash
# Run this verification script on cloud server
cd ~/crpbot
source .venv/bin/activate

echo "=== Verification Checklist ==="
echo ""

# 1. Git repository
echo "1. Git repository:"
git log --oneline -1 && echo "✅ Git OK" || echo "❌ Git failed"
echo ""

# 2. Python environment
echo "2. Python environment:"
python --version && echo "✅ Python OK" || echo "❌ Python failed"
echo ""

# 3. Dependencies
echo "3. Dependencies:"
python -c "import torch; import pandas; import sqlalchemy" && echo "✅ Dependencies OK" || echo "❌ Dependencies failed"
echo ""

# 4. Credentials
echo "4. Credentials:"
[ -f .env ] && echo "✅ .env exists" || echo "❌ .env missing"
[ -f .db_password ] && echo "✅ .db_password exists" || echo "❌ .db_password missing"
[ -f ~/.aws/credentials ] && echo "✅ AWS credentials exist" || echo "❌ AWS credentials missing"
echo ""

# 5. Data & Models
echo "5. Data & Models:"
[ -d data/raw ] && echo "✅ Raw data: $(ls data/raw/ | wc -l) files" || echo "❌ Raw data missing"
[ -d models ] && echo "✅ Models: $(ls models/*.pt 2>/dev/null | wc -l) files" || echo "❌ Models missing"
echo ""

# 6. AWS access
echo "6. AWS access:"
aws sts get-caller-identity > /dev/null 2>&1 && echo "✅ AWS access OK" || echo "❌ AWS access failed"
echo ""

# 7. Database connection
echo "7. Database connection:"
PGPASSWORD="$(cat .db_password)" psql \
  -h crpbot-rds-postgres-db.cyjcoys82evx.us-east-1.rds.amazonaws.com \
  -p 5432 \
  -U crpbot_admin \
  -d crpbot \
  -c "SELECT 1" > /dev/null 2>&1 && echo "✅ Database OK" || echo "❌ Database failed"
echo ""

# 8. Tests
echo "8. Running tests (this may take a minute):"
pytest tests/unit/ -q > /dev/null 2>&1 && echo "✅ Tests passed" || echo "❌ Tests failed"
echo ""

echo "=== Verification Complete ==="
```

---

## 🚀 Usage - Claude Code CLI

### Basic Usage

```bash
# Activate environment
cd ~/crpbot
source .venv/bin/activate

# Or just use the alias
crpbot

# Start Claude Code CLI
claude-code .

# Or with specific task
claude-code "Help me debug the runtime issues"
```

### Working with Claude Code

```bash
# Example sessions:

# 1. Code review
cd ~/crpbot
claude-code "Review the runtime code and suggest improvements"

# 2. Debug issues
claude-code "The tests are failing in test_ensemble.py, help me fix it"

# 3. Add features
claude-code "Add a new feature to track win rate per hour"

# 4. Check logs
claude-code "Analyze the runtime logs and identify any issues"
```

---

## 📊 Quick Reference Commands

### Daily Commands
```bash
# Connect to server
ssh crpbot-cloud  # or: ssh user@server-ip

# Activate environment
cd ~/crpbot && source .venv/bin/activate
# Or just: crpbot

# Pull latest code
git pull origin main

# Run tests
make test

# Start runtime
make run-dry

# Check logs
tail -f logs/*.log

# Claude Code
claude-code .
```

### Sync Commands (Run on LOCAL machine)
```bash
# Push local changes to cloud
cd /home/numan/crpbot
git push origin main

# Then on cloud:
ssh crpbot-cloud "cd ~/crpbot && git pull origin main"

# Or use sync script (from local machine):
./scripts/sync_to_cloud.sh
```

---

## 🔧 Troubleshooting

### Issue: AWS credentials not working
```bash
# Reconfigure AWS
aws configure

# Or copy from local machine:
# On local: scp -r ~/.aws user@server:~/
```

### Issue: Database connection failed
```bash
# Check password file exists
cat ~/crpbot/.db_password

# Test connection manually
PGPASSWORD="$(cat .db_password)" psql \
  -h crpbot-rds-postgres-db.cyjcoys82evx.us-east-1.rds.amazonaws.com \
  -p 5432 \
  -U crpbot_admin \
  -d crpbot

# Check security group allows your server IP
```

### Issue: Dependencies import errors
```bash
cd ~/crpbot
source .venv/bin/activate

# Reinstall dependencies
pip install --upgrade pip
uv pip install -e . --force-reinstall
```

### Issue: Out of disk space
```bash
# Check disk usage
df -h

# Clean up
rm -rf ~/.cache/pip
find ~/crpbot -name "__pycache__" -type d -exec rm -rf {} +
find ~/crpbot -name "*.pyc" -delete
```

---

## 🎉 You're All Set!

Your cloud server now has:
- ✅ All system dependencies installed
- ✅ CRPBot repository cloned
- ✅ All credentials synced (.env, .db_password, AWS)
- ✅ Python environment with all dependencies
- ✅ Data and models downloaded from S3
- ✅ Claude Code CLI ready to use
- ✅ Optional systemd service for runtime

### Next Steps

1. **Test everything**: Run the verification checklist above
2. **Start Claude Code**: `cd ~/crpbot && claude-code .`
3. **Keep environments synced**: Use the sync scripts from your local machine
4. **Daily workflow**: Connect → activate → use Claude Code → sync back

### Staying in Sync

**After making changes on cloud**:
```bash
# On cloud server
cd ~/crpbot
git add .
git commit -m "Changes made on cloud"
git push origin main

# On local machine
cd /home/numan/crpbot
git pull origin main
```

**After making changes locally**:
```bash
# On local machine
cd /home/numan/crpbot
git add .
git commit -m "Changes made locally"
git push origin main

# On cloud server
cd ~/crpbot
git pull origin main
```

Enjoy working with Claude Code on both environments! 🚀
