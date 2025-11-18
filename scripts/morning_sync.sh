#!/bin/bash
# Morning Sync Script
# Run this at the start of each session

echo "🌅 Morning Sync Starting..."
echo ""

# Fetch latest
echo "📥 Fetching from GitHub..."
git fetch origin

# Check status
BRANCH=$(git branch --show-current)
echo "📍 Current branch: $BRANCH"

# Check if behind
BEHIND=$(git rev-list --count HEAD..origin/$BRANCH 2>/dev/null || echo "0")
if [ "$BEHIND" -gt 0 ]; then
    echo "⚠️  You are $BEHIND commits behind origin/$BRANCH"
    echo "📥 Pulling latest changes..."
    git pull origin $BRANCH
else
    echo "✅ You are up to date with origin/$BRANCH"
fi

# Check for uncommitted changes
UNCOMMITTED=$(git status --porcelain | wc -l)
if [ "$UNCOMMITTED" -gt 0 ]; then
    echo "⚠️  You have $UNCOMMITTED uncommitted changes"
    git status --short
else
    echo "✅ No uncommitted changes"
fi

# Show recent commits
echo ""
echo "📝 Recent commits:"
git log -5 --pretty=format:"%C(yellow)%h%Creset %s %C(cyan)(%ar)%Creset %C(green)<%an>%Creset" --abbrev-commit

echo ""
echo ""
echo "✅ Morning sync complete!"
