#!/bin/bash
set -e

# Sync local changes to cloud server
# Usage: ./scripts/sync_to_cloud.sh [-k SSH_KEY] [user@server or ssh-config-host]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

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
        echo "Usage: $0 [-k SSH_KEY] user@server"
        echo "Or save server: echo 'user@server' > .cloud_server"
        exit 1
    fi
fi

# Build SSH command
SSH_CMD="ssh"
if [ -n "$SSH_KEY" ]; then
    SSH_CMD="ssh -i $SSH_KEY"
fi

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Syncing Local → Cloud               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

cd "$PROJECT_ROOT"

# Check for uncommitted changes
if ! git diff-index --quiet HEAD 2>/dev/null; then
    echo -e "${YELLOW}⚠️  You have uncommitted changes${NC}"
    git status -s
    echo ""
    read -p "Commit these changes? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add .
        read -p "Commit message: " commit_msg
        git commit -m "$commit_msg"
        echo -e "${GREEN}✅ Changes committed${NC}"
    else
        echo -e "${YELLOW}⚠️  Proceeding with uncommitted changes${NC}"
    fi
fi

# Push to git
echo -e "${BLUE}📤 Pushing to git remote...${NC}"
if git push origin $(git branch --show-current) 2>/dev/null; then
    echo -e "${GREEN}✅ Pushed to git${NC}"
else
    echo -e "${YELLOW}⚠️  No git remote or push failed${NC}"
fi

# Pull on cloud server
echo -e "${BLUE}📥 Pulling on cloud server...${NC}"
$SSH_CMD "$REMOTE_SERVER" bash <<ENDSSH
cd ~/crpbot
git fetch origin
BRANCH=\$(git branch --show-current)
git pull origin \$BRANCH
echo "✅ Cloud server updated from git"
ENDSSH

# Sync dependencies if needed
echo -e "${BLUE}📚 Checking dependencies...${NC}"
if git diff HEAD~1 HEAD --name-only | grep -q "pyproject.toml\|uv.lock"; then
    echo "Dependencies changed, updating on cloud..."
    $SSH_CMD "$REMOTE_SERVER" bash <<'ENDSSH'
cd ~/crpbot
source .venv/bin/activate
source $HOME/.cargo/env 2>/dev/null || true
uv pip install -e . -q
echo "✅ Dependencies updated"
ENDSSH
    echo -e "${GREEN}✅ Dependencies synced${NC}"
else
    echo "Dependencies unchanged, skipping"
fi

# Summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Sync Complete! ✅                    ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
echo "Local and cloud code are now in sync"
echo ""
