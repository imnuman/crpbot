#!/bin/bash
set -e

# Sync cloud changes to local machine
# Usage: ./scripts/sync_from_cloud.sh [-k SSH_KEY] [user@server or ssh-config-host]

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
echo -e "${BLUE}║   Syncing Cloud → Local               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

cd "$PROJECT_ROOT"

# Check for uncommitted changes locally
if ! git diff-index --quiet HEAD 2>/dev/null; then
    echo -e "${YELLOW}⚠️  You have uncommitted local changes${NC}"
    git status -s
    echo ""
    echo "Please commit or stash your changes before syncing from cloud"
    exit 1
fi

# Push from cloud to git
echo -e "${BLUE}📤 Pushing from cloud to git...${NC}"
$SSH_CMD "$REMOTE_SERVER" bash <<ENDSSH
cd ~/crpbot
BRANCH=\$(git branch --show-current)
git push origin \$BRANCH
echo "✅ Cloud changes pushed to git"
ENDSSH

# Pull locally
echo -e "${BLUE}📥 Pulling changes locally...${NC}"
git fetch origin
BRANCH=$(git branch --show-current)
git pull origin $BRANCH
echo -e "${GREEN}✅ Local repository updated${NC}"

# Sync dependencies if needed
echo -e "${BLUE}📚 Checking dependencies...${NC}"
if git diff HEAD~1 HEAD --name-only | grep -q "pyproject.toml\|uv.lock"; then
    echo "Dependencies changed, updating locally..."
    source .venv/bin/activate
    uv pip install -e . -q
    echo -e "${GREEN}✅ Dependencies updated${NC}"
else
    echo "Dependencies unchanged, skipping"
fi

# Summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Sync Complete! ✅                    ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
echo "Cloud and local code are now in sync"
echo ""
