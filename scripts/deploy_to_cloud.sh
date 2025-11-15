#!/bin/bash
set -e

# CRPBot Cloud Deployment Script
# Usage: ./scripts/deploy_to_cloud.sh [-k SSH_KEY] user@cloud-server-ip
# Or:    ./scripts/deploy_to_cloud.sh [-k SSH_KEY] ssh-config-host

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   CRPBot Cloud Deployment Script          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo ""

# Parse arguments
SSH_KEY=""
REMOTE_SERVER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -k|--key)
            SSH_KEY="$2"
            shift 2
            ;;
        -i)
            SSH_KEY="$2"
            shift 2
            ;;
        *)
            REMOTE_SERVER="$1"
            shift
            ;;
    esac
done

# Check arguments
if [ -z "$REMOTE_SERVER" ]; then
    echo -e "${RED}❌ Error: Remote server address required${NC}"
    echo ""
    echo "Usage: $0 [-k SSH_KEY] user@cloud-server-ip"
    echo "   Or: $0 [-k SSH_KEY] ssh-config-host"
    echo ""
    echo "Examples:"
    echo "  $0 ubuntu@52.23.45.67"
    echo "  $0 -k ~/.ssh/id_ed25519 ubuntu@52.23.45.67"
    echo "  $0 crpbot-cloud  (if configured in ~/.ssh/config)"
    exit 1
fi

REMOTE_DIR="crpbot"

# Build SSH command with key if provided
SSH_CMD="ssh"
SCP_CMD="scp"
RSYNC_CMD="rsync"
if [ -n "$SSH_KEY" ]; then
    if [ ! -f "$SSH_KEY" ]; then
        echo -e "${RED}❌ Error: SSH key not found: $SSH_KEY${NC}"
        exit 1
    fi
    echo -e "${YELLOW}📝 Using SSH key: $SSH_KEY${NC}"
    SSH_CMD="ssh -i $SSH_KEY"
    SCP_CMD="scp -i $SSH_KEY"
    RSYNC_CMD="rsync -e \"ssh -i $SSH_KEY\""
fi

echo -e "${YELLOW}📋 Deployment Configuration:${NC}"
echo "  Remote Server: $REMOTE_SERVER"
echo "  Remote Directory: ~/$REMOTE_DIR"
echo "  Project Root: $PROJECT_ROOT"
echo ""

# Verify SSH connection
echo -e "${BLUE}🔐 Step 1/8: Verifying SSH connection...${NC}"
if ! $SSH_CMD -o ConnectTimeout=10 "$REMOTE_SERVER" "echo 'SSH connection successful'" > /dev/null 2>&1; then
    echo -e "${RED}❌ Cannot connect to $REMOTE_SERVER${NC}"
    echo "Please check:"
    echo "  - Server IP/hostname is correct"
    echo "  - SSH key has correct permissions (chmod 600)"
    echo "  - SSH key is added to server's authorized_keys"
    echo "  - Server is running and accessible"
    echo ""
    echo "Debug: Try manual connection:"
    if [ -n "$SSH_KEY" ]; then
        echo "  ssh -i $SSH_KEY -v $REMOTE_SERVER"
    else
        echo "  ssh -v $REMOTE_SERVER"
    fi
    exit 1
fi
echo -e "${GREEN}✅ SSH connection verified${NC}"
echo ""

# Prepare local files
echo -e "${BLUE}📦 Step 2/8: Preparing local files...${NC}"
cd "$PROJECT_ROOT"

# Ensure code is committed
if ! git diff-index --quiet HEAD 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Uncommitted changes detected${NC}"
    read -p "Commit changes before deployment? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add .
        git commit -m "Pre-deployment commit $(date +%Y-%m-%d_%H-%M-%S)"
        echo -e "${GREEN}✅ Changes committed${NC}"
    fi
fi

# Create config archive
echo "Creating configuration archive..."
TEMP_DIR=$(mktemp -d)
tar -czf "$TEMP_DIR/crpbot-config.tar.gz" \
    --exclude='data/*' \
    --exclude='models/*' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    .env .db_password .rds_connection_info 2>/dev/null || true

echo -e "${GREEN}✅ Configuration prepared${NC}"
echo ""

# Setup remote server
echo -e "${BLUE}🖥️  Step 3/8: Setting up remote server...${NC}"
$SSH_CMD "$REMOTE_SERVER" bash <<'ENDSSH'
set -e

echo "Updating system packages..."
sudo apt update -qq

echo "Installing essential packages..."
sudo apt install -y -qq \
  python3.10 \
  python3.10-venv \
  python3-pip \
  git \
  postgresql-client \
  redis-tools \
  build-essential \
  curl \
  wget \
  unzip \
  htop

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Install AWS CLI if not present
if ! command -v aws &> /dev/null; then
    echo "Installing AWS CLI v2..."
    cd /tmp
    curl -sS "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip -q awscliv2.zip
    sudo ./aws/install
    rm -rf aws awscliv2.zip
fi

echo "Remote server setup complete"
ENDSSH
echo -e "${GREEN}✅ Remote server configured${NC}"
echo ""

# Clone/update repository
echo -e "${BLUE}📥 Step 4/8: Deploying code...${NC}"

# Get git remote URL
GIT_URL=$(git config --get remote.origin.url || echo "")
if [ -z "$GIT_URL" ]; then
    echo -e "${YELLOW}⚠️  No git remote found, using rsync for code transfer${NC}"

    # Use rsync to transfer code
    if [ -n "$SSH_KEY" ]; then
        rsync -avz --progress -e "ssh -i $SSH_KEY" \
            --exclude='.venv' \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='data/' \
            --exclude='models/' \
            --exclude='.git' \
            --exclude='.env' \
            --exclude='.db_password' \
            "$PROJECT_ROOT/" "$REMOTE_SERVER:~/$REMOTE_DIR/"
    else
        rsync -avz --progress \
            --exclude='.venv' \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='data/' \
            --exclude='models/' \
            --exclude='.git' \
            --exclude='.env' \
            --exclude='.db_password' \
            "$PROJECT_ROOT/" "$REMOTE_SERVER:~/$REMOTE_DIR/"
    fi
else
    echo "Git remote found: $GIT_URL"
    $SSH_CMD "$REMOTE_SERVER" bash <<ENDSSH
set -e
if [ -d "$REMOTE_DIR" ]; then
    echo "Updating existing repository..."
    cd $REMOTE_DIR
    git fetch origin
    git reset --hard origin/main || git reset --hard origin/master
else
    echo "Cloning repository..."
    git clone "$GIT_URL" $REMOTE_DIR
    cd $REMOTE_DIR
fi
ENDSSH
fi
echo -e "${GREEN}✅ Code deployed${NC}"
echo ""

# Transfer configuration
echo -e "${BLUE}🔑 Step 5/8: Transferring configuration...${NC}"
$SCP_CMD "$TEMP_DIR/crpbot-config.tar.gz" "$REMOTE_SERVER:~/$REMOTE_DIR/" > /dev/null 2>&1
$SSH_CMD "$REMOTE_SERVER" bash <<ENDSSH
cd $REMOTE_DIR
tar -xzf crpbot-config.tar.gz 2>/dev/null || true
rm -f crpbot-config.tar.gz
chmod 600 .env .db_password .rds_connection_info 2>/dev/null || true
echo "Configuration files transferred"
ENDSSH
rm -rf "$TEMP_DIR"
echo -e "${GREEN}✅ Configuration transferred${NC}"
echo ""

# Install dependencies
echo -e "${BLUE}📚 Step 6/8: Installing dependencies...${NC}"
$SSH_CMD "$REMOTE_SERVER" bash <<'ENDSSH'
set -e
cd crpbot
source $HOME/.cargo/env

# Create virtual environment
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# Install dependencies with uv
pip install -q uv
uv pip install -e . -q
uv pip install -e ".[dev]" -q

# Verify installation
python -c "import torch; import pandas; print('✅ Dependencies installed successfully')"
ENDSSH
echo -e "${GREEN}✅ Dependencies installed${NC}"
echo ""

# Prompt for data/models transfer
echo -e "${BLUE}💾 Step 7/8: Data and models transfer...${NC}"
echo ""
echo "How would you like to transfer data and models?"
echo "  1) Download from S3 (recommended - fast, 10 min)"
echo "  2) Transfer from local machine (slow, 30+ min for 764MB)"
echo "  3) Skip (already on server or will do manually)"
read -p "Select option (1/2/3): " -n 1 -r
echo ""

case $REPLY in
    1)
        echo "Downloading data and models from S3..."
        $SSH_CMD "$REMOTE_SERVER" bash <<'ENDSSH'
set -e
cd crpbot
source .venv/bin/activate

# Check AWS credentials
if ! aws sts get-caller-identity &>/dev/null; then
    echo "⚠️  AWS credentials not configured"
    echo "Please run: aws configure"
    exit 1
fi

# Create directories
mkdir -p data/raw data/features models

# Download from S3
echo "Downloading raw data..."
aws s3 sync s3://crpbot-market-data-dev/data/raw/ data/raw/ --exclude "*.gitkeep" --quiet

echo "Downloading features..."
aws s3 sync s3://crpbot-market-data-dev/data/features/ data/features/ --exclude "*.gitkeep" --quiet

echo "Downloading models..."
aws s3 sync s3://crpbot-market-data-dev/models/ models/ --exclude "*.gitkeep" --quiet

echo "✅ Data and models downloaded from S3"
ENDSSH
        ;;
    2)
        echo "Transferring data and models from local machine..."
        cd "$PROJECT_ROOT"

        echo "Creating archive..."
        tar -czf /tmp/crpbot-data-models.tar.gz data/ models/

        echo "Uploading to server (this may take 10-30 minutes)..."
        $SCP_CMD -C /tmp/crpbot-data-models.tar.gz "$REMOTE_SERVER:~/$REMOTE_DIR/"

        echo "Extracting on server..."
        $SSH_CMD "$REMOTE_SERVER" bash <<ENDSSH
cd $REMOTE_DIR
tar -xzf crpbot-data-models.tar.gz
rm crpbot-data-models.tar.gz
echo "✅ Data and models extracted"
ENDSSH
        rm /tmp/crpbot-data-models.tar.gz
        ;;
    3)
        echo "Skipping data/models transfer"
        ;;
    *)
        echo -e "${YELLOW}Invalid option, skipping${NC}"
        ;;
esac
echo -e "${GREEN}✅ Data/models step complete${NC}"
echo ""

# Test deployment
echo -e "${BLUE}🧪 Step 8/8: Testing deployment...${NC}"
$SSH_CMD "$REMOTE_SERVER" bash <<'ENDSSH'
set -e
cd crpbot
source .venv/bin/activate

echo "Testing database connection..."
if [ -f ".db_password" ]; then
    PGPASSWORD="$(cat .db_password)" psql \
      -h crpbot-rds-postgres-db.cyjcoys82evx.us-east-1.rds.amazonaws.com \
      -p 5432 \
      -U crpbot_admin \
      -d crpbot \
      -c "SELECT version();" -t | head -1
else
    echo "⚠️  .db_password not found, skipping DB test"
fi

echo "Running quick test..."
python -c "
import sys
sys.path.insert(0, '.')
from apps.trainer.models import lstm
print('✅ Module imports working')
"

echo "✅ Deployment tests passed"
ENDSSH
echo -e "${GREEN}✅ Tests completed${NC}"
echo ""

# Summary
echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║        Deployment Complete! 🎉             ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. SSH into server: ssh $REMOTE_SERVER"
echo "  2. Activate environment: cd crpbot && source .venv/bin/activate"
echo "  3. Configure AWS if needed: aws configure"
echo "  4. Run tests: make test"
echo "  5. Start runtime: make run-dry"
echo ""
echo -e "${YELLOW}📖 See MIGRATION_GUIDE.md for detailed information${NC}"
echo ""
