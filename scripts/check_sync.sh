#!/bin/bash
# Check sync status between local repository and GitHub

set -e

echo "🔍 Checking Sync Status"
echo "======================"

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Not in a git repository!"
    exit 1
fi

# Check current branch
CURRENT_BRANCH=$(git branch --show-current)
echo ""
echo "📍 Current Branch: $CURRENT_BRANCH"

# Check git status
echo ""
echo "📊 Git Status:"
git status --short || echo "   (no changes)"

# Check remote
REMOTE=$(git remote get-url origin 2>/dev/null || echo "none")
echo ""
echo "🔗 Remote: $REMOTE"

# Fetch latest (without merging)
echo ""
echo "🔄 Fetching latest from remote..."
git fetch origin --quiet

# Check if up to date
LOCAL=$(git rev-parse HEAD)
REMOTE_HEAD=$(git rev-parse origin/$CURRENT_BRANCH 2>/dev/null || echo "")

if [ -z "$REMOTE_HEAD" ]; then
    echo "⚠️  Remote branch 'origin/$CURRENT_BRANCH' not found"
    echo "   This might be a new branch that hasn't been pushed yet"
else
    echo ""
    echo "🔄 Sync Status:"
    
    if [ "$LOCAL" = "$REMOTE_HEAD" ]; then
        echo "✅ Local and remote are in sync"
    else
        echo "⚠️  Local and remote differ"
        echo "   Local:  $(git rev-parse --short HEAD)"
        echo "   Remote: $(git rev-parse --short origin/$CURRENT_BRANCH)"
        
        # Check commits ahead/behind
        AHEAD=$(git rev-list --count HEAD ^origin/$CURRENT_BRANCH 2>/dev/null || echo "0")
        BEHIND=$(git rev-list --count origin/$CURRENT_BRANCH ^HEAD 2>/dev/null || echo "0")
        
        if [ "$AHEAD" -gt 0 ]; then
            echo "   📤 Local is $AHEAD commits ahead (need to push)"
            echo ""
            echo "   Local commits not on remote:"
            git log --oneline HEAD ^origin/$CURRENT_BRANCH | head -5 | sed 's/^/      /'
        fi
        
        if [ "$BEHIND" -gt 0 ]; then
            echo "   📥 Local is $BEHIND commits behind (need to pull)"
            echo ""
            echo "   Remote commits not in local:"
            git log --oneline origin/$CURRENT_BRANCH ^HEAD | head -5 | sed 's/^/      /'
        fi
    fi
fi

# Check uncommitted changes
echo ""
echo "📝 Uncommitted Changes:"
UNCOMMITTED=$(git status --porcelain | wc -l)
if [ "$UNCOMMITTED" -eq 0 ]; then
    echo "✅ No uncommitted changes"
else
    echo "⚠️  $UNCOMMITTED uncommitted file(s):"
    git status --short | sed 's/^/   /'
fi

# Check untracked files
echo ""
echo "📦 Untracked Files:"
UNTRACKED=$(git ls-files --others --exclude-standard | wc -l)
if [ "$UNTRACKED" -eq 0 ]; then
    echo "✅ No untracked files"
else
    echo "ℹ️  $UNTRACKED untracked file(s) (not shown - run 'git status' for details)"
fi

# Recent commits
echo ""
echo "📜 Recent Commits:"
git log --oneline -5 | sed 's/^/   /'

echo ""
echo "======================"
echo ""
echo "💡 Tips:"

# Only show tips if we have remote branch info
if [ -n "$REMOTE_HEAD" ]; then
    if [ "${AHEAD:-0}" -gt 0 ] && [ "${BEHIND:-0}" -gt 0 ]; then
        echo "   • Run: git pull --rebase origin $CURRENT_BRANCH (to sync)"
        echo "   • Then: git push origin $CURRENT_BRANCH"
    elif [ "${AHEAD:-0}" -gt 0 ]; then
        echo "   • Run: git push origin $CURRENT_BRANCH (to push local commits)"
    elif [ "${BEHIND:-0}" -gt 0 ]; then
        echo "   • Run: git pull origin $CURRENT_BRANCH (to pull remote commits)"
    fi
fi

if [ "$UNCOMMITTED" -gt 0 ]; then
    echo "   • Run: git status (to see uncommitted changes)"
    echo "   • Run: git add . && git commit -m 'your message' (to commit)"
fi

