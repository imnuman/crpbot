#!/bin/bash
set -e

# Sync credentials from local machine to cloud server
# Usage: ./scripts/sync_credentials.sh [-k SSH_KEY] user@server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Credential Sync: Local → Cloud      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Parse arguments
SSH_KEY=""
REMOTE_SERVER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -k|--key|-i)
            SSH_KEY="$2"
            shift 2
            ;;
        *)
            REMOTE_SERVER="$1"
            shift
            ;;
    esac
done

# Load remote server from config if not provided
if [ -z "$REMOTE_SERVER" ]; then
    if [ -f "$PROJECT_ROOT/.cloud_server" ]; then
        REMOTE_SERVER=$(cat "$PROJECT_ROOT/.cloud_server")
        echo -e "${BLUE}Using saved server: $REMOTE_SERVER${NC}"
    else
        echo -e "${RED}❌ Error: Remote server required${NC}"
        echo ""
        echo "Usage: $0 [-k SSH_KEY] user@server"
        echo "Or save server: echo 'user@server' > .cloud_server"
        exit 1
    fi
fi

# Load SSH key from config if not provided
if [ -z "$SSH_KEY" ] && [ -f "$PROJECT_ROOT/.ssh_key" ]; then
    SSH_KEY=$(cat "$PROJECT_ROOT/.ssh_key")
    echo -e "${BLUE}Using saved SSH key: $SSH_KEY${NC}"
fi

# Build SCP command
SCP_CMD="scp"
SSH_CMD="ssh"
if [ -n "$SSH_KEY" ]; then
    SCP_CMD="scp -i $SSH_KEY"
    SSH_CMD="ssh -i $SSH_KEY"
fi

echo ""
echo -e "${YELLOW}Files to sync:${NC}"
echo "  📄 .env"
echo "  🔑 .db_password"
echo "  🔗 .rds_connection_info (if exists)"
echo "  ☁️  AWS credentials (~/.aws)"
echo ""

# Check files exist locally
MISSING=0

if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo -e "${RED}❌ .env not found${NC}"
    MISSING=1
else
    echo -e "${GREEN}✅ .env found${NC}"
fi

if [ ! -f "$PROJECT_ROOT/.db_password" ]; then
    echo -e "${RED}❌ .db_password not found${NC}"
    MISSING=1
else
    echo -e "${GREEN}✅ .db_password found${NC}"
fi

if [ ! -f "$HOME/.aws/credentials" ]; then
    echo -e "${YELLOW}⚠️  AWS credentials not found (will skip)${NC}"
else
    echo -e "${GREEN}✅ AWS credentials found${NC}"
fi

if [ $MISSING -eq 1 ]; then
    echo ""
    echo -e "${RED}❌ Missing required files${NC}"
    exit 1
fi

echo ""
read -p "Proceed with sync? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Sync cancelled"
    exit 0
fi

# Create remote directory if needed
echo -e "${BLUE}📁 Preparing remote directory...${NC}"
$SSH_CMD "$REMOTE_SERVER" "mkdir -p ~/crpbot"

# Transfer .env
echo -e "${BLUE}📤 Transferring .env...${NC}"
$SCP_CMD "$PROJECT_ROOT/.env" "$REMOTE_SERVER:~/crpbot/.env"
echo -e "${GREEN}✅ .env transferred${NC}"

# Transfer .db_password
echo -e "${BLUE}📤 Transferring .db_password...${NC}"
$SCP_CMD "$PROJECT_ROOT/.db_password" "$REMOTE_SERVER:~/crpbot/.db_password"
echo -e "${GREEN}✅ .db_password transferred${NC}"

# Transfer .rds_connection_info if exists
if [ -f "$PROJECT_ROOT/.rds_connection_info" ]; then
    echo -e "${BLUE}📤 Transferring .rds_connection_info...${NC}"
    $SCP_CMD "$PROJECT_ROOT/.rds_connection_info" "$REMOTE_SERVER:~/crpbot/.rds_connection_info"
    echo -e "${GREEN}✅ .rds_connection_info transferred${NC}"
fi

# Transfer AWS credentials if exist
if [ -f "$HOME/.aws/credentials" ]; then
    echo -e "${BLUE}📤 Transferring AWS credentials...${NC}"

    # Create .aws directory on remote
    $SSH_CMD "$REMOTE_SERVER" "mkdir -p ~/.aws"

    # Transfer credentials and config
    $SCP_CMD "$HOME/.aws/credentials" "$REMOTE_SERVER:~/.aws/credentials"
    if [ -f "$HOME/.aws/config" ]; then
        $SCP_CMD "$HOME/.aws/config" "$REMOTE_SERVER:~/.aws/config"
    fi

    echo -e "${GREEN}✅ AWS credentials transferred${NC}"
fi

# Set proper permissions on remote
echo -e "${BLUE}🔒 Setting secure permissions...${NC}"
$SSH_CMD "$REMOTE_SERVER" bash <<'ENDSSH'
# Secure project credentials
chmod 600 ~/crpbot/.env 2>/dev/null || true
chmod 600 ~/crpbot/.db_password 2>/dev/null || true
chmod 600 ~/crpbot/.rds_connection_info 2>/dev/null || true

# Secure AWS credentials
chmod 700 ~/.aws 2>/dev/null || true
chmod 600 ~/.aws/credentials 2>/dev/null || true
chmod 600 ~/.aws/config 2>/dev/null || true

echo "✅ Permissions set (600 for files, 700 for directories)"
ENDSSH

# Verify on remote
echo ""
echo -e "${BLUE}🔍 Verifying on remote server...${NC}"
$SSH_CMD "$REMOTE_SERVER" bash <<'ENDSSH'
cd ~/crpbot 2>/dev/null || { echo "❌ ~/crpbot directory not found"; exit 1; }

echo "Checking files:"
[ -f .env ] && echo "  ✅ .env ($(wc -l < .env) lines)" || echo "  ❌ .env missing"
[ -f .db_password ] && echo "  ✅ .db_password" || echo "  ❌ .db_password missing"
[ -f .rds_connection_info ] && echo "  ✅ .rds_connection_info" || echo "  ⚠️  .rds_connection_info not present"
[ -f ~/.aws/credentials ] && echo "  ✅ AWS credentials" || echo "  ⚠️  AWS credentials not present"

echo ""
echo "Permissions:"
ls -la .env .db_password 2>/dev/null | tail -n +2
ENDSSH

# Summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Credential Sync Complete! ✅         ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
echo "All credentials synced to: $REMOTE_SERVER"
echo ""
echo -e "${YELLOW}Next steps on cloud server:${NC}"
echo "  1. ssh $REMOTE_SERVER"
echo "  2. cd ~/crpbot"
echo "  3. Follow CLOUD_SERVER_SETUP.md"
echo ""
