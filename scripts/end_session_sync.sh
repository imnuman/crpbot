#!/bin/bash
# End of Session Sync Script
# Run this at the end of each session

echo "🌙 End of Session Sync..."
echo ""

# Check for uncommitted changes
UNCOMMITTED=$(git status --porcelain | wc -l)
if [ "$UNCOMMITTED" -gt 0 ]; then
    echo "⚠️  You have $UNCOMMITTED uncommitted changes"
    git status --short
    echo ""
    echo "Please commit these changes before ending session:"
    echo "  git add <files>"
    echo "  git commit -m 'type: description'"
    echo "  git push origin \$(git branch --show-current)"
    exit 1
fi

# Check if ahead of remote
BRANCH=$(git branch --show-current)
AHEAD=$(git rev-list --count origin/$BRANCH..HEAD 2>/dev/null || echo "0")
if [ "$AHEAD" -gt 0 ]; then
    echo "📤 You are $AHEAD commits ahead of origin/$BRANCH"
    echo "🚀 Pushing to GitHub..."
    git push origin $BRANCH
    if [ $? -eq 0 ]; then
        echo "✅ Pushed successfully"
    else
        echo "❌ Push failed - please resolve manually"
        exit 1
    fi
else
    echo "✅ Already synced with GitHub"
fi

echo ""
echo "✅ End of session sync complete!"
echo ""
echo "Summary:"
echo "  - All changes committed: ✅"
echo "  - Synced with GitHub: ✅"
echo "  - Ready for next session: ✅"
