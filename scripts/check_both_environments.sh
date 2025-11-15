#!/bin/bash

# Check status of both local and cloud environments
# Usage: ./scripts/check_both_environments.sh [-k SSH_KEY] [user@server]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
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

echo -e "${CYAN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║       Dual Environment Status Check              ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# ===== LOCAL ENVIRONMENT =====
echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   LOCAL ENVIRONMENT                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

cd "$PROJECT_ROOT"

echo -e "${GREEN}Git Status:${NC}"
echo "  Branch: $(git branch --show-current)"
echo "  Commit: $(git log --oneline -1)"
echo ""

if ! git diff-index --quiet HEAD 2>/dev/null; then
    echo -e "${YELLOW}  Uncommitted changes:${NC}"
    git status -s | head -10
    CHANGE_COUNT=$(git status -s | wc -l)
    if [ "$CHANGE_COUNT" -gt 10 ]; then
        echo "  ... and $((CHANGE_COUNT - 10)) more"
    fi
else
    echo -e "${GREEN}  ✅ Working directory clean${NC}"
fi
echo ""

echo -e "${GREEN}Python Environment:${NC}"
if [ -f ".venv/bin/python" ]; then
    source .venv/bin/activate
    echo "  Version: $(python --version)"
    echo "  Location: $(which python)"
else
    echo -e "${YELLOW}  ⚠️  Virtual environment not found${NC}"
fi
echo ""

echo -e "${GREEN}Data & Models:${NC}"
echo "  Data: $(du -sh data/ 2>/dev/null | cut -f1 || echo 'N/A')"
echo "  Models: $(du -sh models/ 2>/dev/null | cut -f1 || echo 'N/A')"
echo ""

echo -e "${GREEN}Processes:${NC}"
RUNTIME_PROC=$(ps aux | grep "apps/runtime/main.py" | grep -v grep | wc -l)
if [ "$RUNTIME_PROC" -gt 0 ]; then
    echo -e "  ${GREEN}✅ Runtime is running ($RUNTIME_PROC process)${NC}"
else
    echo "  ⏸️  Runtime not running"
fi
echo ""

# ===== CLOUD ENVIRONMENT =====
echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   CLOUD ENVIRONMENT                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

echo "Connecting to: $REMOTE_SERVER"
echo ""

$SSH_CMD "$REMOTE_SERVER" bash <<'ENDSSH'
cd ~/crpbot 2>/dev/null || { echo "❌ crpbot directory not found"; exit 1; }

echo -e "\033[0;32mGit Status:\033[0m"
echo "  Branch: $(git branch --show-current)"
echo "  Commit: $(git log --oneline -1)"
echo ""

if ! git diff-index --quiet HEAD 2>/dev/null; then
    echo -e "\033[1;33m  Uncommitted changes:\033[0m"
    git status -s | head -10
    CHANGE_COUNT=$(git status -s | wc -l)
    if [ "$CHANGE_COUNT" -gt 10 ]; then
        echo "  ... and $((CHANGE_COUNT - 10)) more"
    fi
else
    echo -e "\033[0;32m  ✅ Working directory clean\033[0m"
fi
echo ""

echo -e "\033[0;32mPython Environment:\033[0m"
if [ -f ".venv/bin/python" ]; then
    source .venv/bin/activate
    echo "  Version: $(python --version)"
    echo "  Location: $(which python)"
else
    echo -e "\033[1;33m  ⚠️  Virtual environment not found\033[0m"
fi
echo ""

echo -e "\033[0;32mData & Models:\033[0m"
echo "  Data: $(du -sh data/ 2>/dev/null | cut -f1 || echo 'N/A')"
echo "  Models: $(du -sh models/ 2>/dev/null | cut -f1 || echo 'N/A')"
echo ""

echo -e "\033[0;32mProcesses:\033[0m"
RUNTIME_PROC=$(ps aux | grep "apps/runtime/main.py" | grep -v grep | wc -l)
if [ "$RUNTIME_PROC" -gt 0 ]; then
    echo -e "  \033[0;32m✅ Runtime is running ($RUNTIME_PROC process)\033[0m"
else
    echo "  ⏸️  Runtime not running"
fi

# Check systemd service if exists
if systemctl is-active --quiet crpbot.service 2>/dev/null; then
    echo -e "  \033[0;32m✅ Systemd service active\033[0m"
fi
echo ""

echo -e "\033[0;32mSystem Resources:\033[0m"
echo "  CPU Load: $(uptime | awk -F'load average:' '{print $2}')"
echo "  Memory: $(free -h | awk '/^Mem:/ {print $3 "/" $2}')"
echo "  Disk: $(df -h . | awk 'NR==2 {print $3 "/" $2 " (" $5 " used)"}')"
echo ""
ENDSSH

# ===== COMPARISON =====
echo -e "${CYAN}╔════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   SYNC STATUS                          ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════╝${NC}"
echo ""

LOCAL_COMMIT=$(git log --oneline -1 | awk '{print $1}')
REMOTE_COMMIT=$($SSH_CMD "$REMOTE_SERVER" "cd ~/crpbot && git log --oneline -1" | awk '{print $1}')

echo "Local commit:  $LOCAL_COMMIT"
echo "Remote commit: $REMOTE_COMMIT"
echo ""

if [ "$LOCAL_COMMIT" = "$REMOTE_COMMIT" ]; then
    echo -e "${GREEN}✅ Environments are in sync${NC}"
else
    echo -e "${YELLOW}⚠️  Environments are OUT OF SYNC${NC}"
    echo ""
    echo "To sync:"
    echo "  Local → Cloud: ./scripts/sync_to_cloud.sh"
    echo "  Cloud → Local: ./scripts/sync_from_cloud.sh"
fi
echo ""
