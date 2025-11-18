#!/bin/bash
# Sync Status Check Script
# Checks if local repo is in sync with GitHub

echo "🔍 Sync Status Check"
echo "===================="
echo ""

# Environment
HOSTNAME=$(hostname)
PWD=$(pwd)
echo "🖥️  Environment: $HOSTNAME"
echo "📂 Directory: $PWD"
echo ""

# Current branch
BRANCH=$(git branch --show-current)
echo "🌿 Branch: $BRANCH"
echo ""

# Fetch latest
echo "📡 Fetching from GitHub..."
git fetch origin -q

# Local vs Remote
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/$BRANCH 2>/dev/null)

if [ -z "$REMOTE" ]; then
    echo "⚠️  Branch '$BRANCH' not found on remote"
    echo "   This is a local-only branch"
    echo ""
else
    if [ "$LOCAL" = "$REMOTE" ]; then
        echo "✅ SYNCED: Local and remote are identical"
    else
        echo "⚠️  OUT OF SYNC: Local and remote differ"

        AHEAD=$(git rev-list --count origin/$BRANCH..HEAD 2>/dev/null || echo "0")
        BEHIND=$(git rev-list --count HEAD..origin/$BRANCH 2>/dev/null || echo "0")

        if [ "$AHEAD" -gt 0 ]; then
            echo "   📤 Local is $AHEAD commits AHEAD (need to push)"
            echo ""
            echo "   Unpushed commits:"
            git log origin/$BRANCH..HEAD --pretty=format:"   %C(yellow)%h%Creset %s %C(cyan)(%ar)%Creset" --abbrev-commit
            echo ""
        fi

        if [ "$BEHIND" -gt 0 ]; then
            echo "   📥 Local is $BEHIND commits BEHIND (need to pull)"
            echo ""
            echo "   Commits not in local:"
            git log HEAD..origin/$BRANCH --pretty=format:"   %C(yellow)%h%Creset %s %C(cyan)(%ar)%Creset %C(green)<%an>%Creset" --abbrev-commit
            echo ""
        fi
    fi
fi

echo ""

# Uncommitted changes
UNCOMMITTED=$(git status --porcelain | wc -l)
if [ "$UNCOMMITTED" -eq 0 ]; then
    echo "✅ No uncommitted changes"
else
    echo "⚠️  $UNCOMMITTED uncommitted file(s):"
    git status --short
fi

echo ""
echo "===================="
