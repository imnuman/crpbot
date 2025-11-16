#!/bin/bash
# Sync local and cloud environments for V6 rebuild
# Ensures both machines have identical code, configs, and dependencies

set -e  # Exit on error

CLOUD_HOST="root@178.156.136.185"
CLOUD_DIR="~/crpbot"
LOCAL_DIR="/home/numan/crpbot"

echo "============================================================"
echo "Environment Sync Tool - V6 Rebuild"
echo "============================================================"
echo "Local:  $LOCAL_DIR"
echo "Cloud:  $CLOUD_HOST:$CLOUD_DIR"
echo "============================================================"
echo ""

# Function to sync code via git
sync_code() {
    echo "📦 Syncing code via Git..."

    # Ensure local changes are committed
    if [[ -n $(git status -s) ]]; then
        echo "⚠️  Local changes detected. Committing..."
        git add .
        git commit -m "Auto-sync: $(date '+%Y-%m-%d %H:%M:%S')" || echo "Nothing to commit"
    fi

    # Push to GitHub
    echo "Pushing to GitHub..."
    git push origin main || echo "Push failed or nothing to push"

    # Pull on cloud
    echo "Pulling on cloud machine..."
    ssh $CLOUD_HOST "cd $CLOUD_DIR && git pull origin main"

    echo "✅ Code synced via Git"
}

# Function to sync .env file
sync_env() {
    echo ""
    echo "🔐 Syncing .env file..."

    # Check if .env exists locally
    if [[ ! -f "$LOCAL_DIR/.env" ]]; then
        echo "❌ Local .env not found!"
        return 1
    fi

    # Copy to cloud
    scp "$LOCAL_DIR/.env" "$CLOUD_HOST:$CLOUD_DIR/.env"

    echo "✅ .env synced to cloud"
}

# Function to sync dependencies
sync_dependencies() {
    echo ""
    echo "📚 Syncing dependencies..."

    # Copy uv.lock
    scp "$LOCAL_DIR/uv.lock" "$CLOUD_HOST:$CLOUD_DIR/uv.lock" || echo "uv.lock not found"

    # Copy pyproject.toml
    scp "$LOCAL_DIR/pyproject.toml" "$CLOUD_HOST:$CLOUD_DIR/pyproject.toml" || echo "pyproject.toml not found"

    # Install on cloud (if uv is available)
    echo "Installing dependencies on cloud..."
    ssh $CLOUD_HOST "cd $CLOUD_DIR && if command -v uv &> /dev/null; then uv sync; else echo 'uv not installed on cloud, using pip'; fi"

    echo "✅ Dependencies synced"
}

# Function to sync data sources config
sync_data_sources() {
    echo ""
    echo "📊 Syncing data source configurations..."

    # Sync test scripts
    scp "$LOCAL_DIR/test_kraken_connection.py" "$CLOUD_HOST:$CLOUD_DIR/" || echo "Kraken test not found"
    scp "$LOCAL_DIR/test_kraken_auth.py" "$CLOUD_HOST:$CLOUD_DIR/" || echo "Kraken auth test not found"
    scp "$LOCAL_DIR/test_coingecko_integration.py" "$CLOUD_HOST:$CLOUD_DIR/" || echo "CoinGecko test not found"

    # Sync collection scripts
    scp "$LOCAL_DIR/scripts/collect_multi_source_data.py" "$CLOUD_HOST:$CLOUD_DIR/scripts/" || echo "Collection script not found"

    echo "✅ Data source configs synced"
}

# Function to sync V6 rebuild files
sync_v6_files() {
    echo ""
    echo "🚀 Syncing V6 rebuild files..."

    # Sync documentation
    scp "$LOCAL_DIR/V6_REBUILD_STATUS.md" "$CLOUD_HOST:$CLOUD_DIR/" || echo "V6 status not found"
    scp "$LOCAL_DIR/V6_DATA_SOURCE_STRATEGY.md" "$CLOUD_HOST:$CLOUD_DIR/" || echo "V6 strategy not found"

    # Sync enhanced features module
    scp "$LOCAL_DIR/apps/trainer/enhanced_features.py" "$CLOUD_HOST:$CLOUD_DIR/apps/trainer/" || echo "Enhanced features not found"

    echo "✅ V6 files synced"
}

# Function to verify sync
verify_sync() {
    echo ""
    echo "🔍 Verifying sync..."

    # Check Git status on both
    echo "Local Git status:"
    git status -s | head -5

    echo ""
    echo "Cloud Git status:"
    ssh $CLOUD_HOST "cd $CLOUD_DIR && git status -s" | head -5

    # Check .env exists on both
    echo ""
    if [[ -f "$LOCAL_DIR/.env" ]]; then
        echo "✅ Local .env exists"
    else
        echo "❌ Local .env missing"
    fi

    ssh $CLOUD_HOST "test -f $CLOUD_DIR/.env && echo '✅ Cloud .env exists' || echo '❌ Cloud .env missing'"

    # Check data source files
    echo ""
    echo "Data source files on cloud:"
    ssh $CLOUD_HOST "cd $CLOUD_DIR && ls -lh test_*_*.py 2>/dev/null | wc -l" | while read count; do
        echo "  Found $count test files"
    done
}

# Function to show sync status
show_status() {
    echo ""
    echo "============================================================"
    echo "Current Environment Status"
    echo "============================================================"

    # Git branches
    echo ""
    echo "📍 Git Branches:"
    echo "Local:  $(git branch --show-current)"
    ssh $CLOUD_HOST "cd $CLOUD_DIR && echo \"Cloud:  \$(git branch --show-current)\""

    # Last commits
    echo ""
    echo "📝 Last Commits:"
    echo "Local:  $(git log -1 --oneline)"
    ssh $CLOUD_HOST "cd $CLOUD_DIR && echo \"Cloud:  \$(git log -1 --oneline)\""

    # Data sources
    echo ""
    echo "🔌 Data Sources Configured:"

    # Check Coinbase
    if grep -q "COINBASE_API_KEY_NAME" "$LOCAL_DIR/.env" 2>/dev/null; then
        echo "  ✅ Coinbase (local)"
    else
        echo "  ❌ Coinbase (local)"
    fi

    ssh $CLOUD_HOST "grep -q 'COINBASE_API_KEY_NAME' $CLOUD_DIR/.env 2>/dev/null && echo '  ✅ Coinbase (cloud)' || echo '  ❌ Coinbase (cloud)'"

    # Check Kraken
    if grep -q "KRAKEN_API_KEY" "$LOCAL_DIR/.env" 2>/dev/null; then
        echo "  ✅ Kraken (local)"
    else
        echo "  ❌ Kraken (local)"
    fi

    ssh $CLOUD_HOST "grep -q 'KRAKEN_API_KEY' $CLOUD_DIR/.env 2>/dev/null && echo '  ✅ Kraken (cloud)' || echo '  ❌ Kraken (cloud)'"

    # Check CoinGecko
    if grep -q "COINGECKO_API_KEY" "$LOCAL_DIR/.env" 2>/dev/null; then
        echo "  ✅ CoinGecko (local)"
    else
        echo "  ❌ CoinGecko (local)"
    fi

    ssh $CLOUD_HOST "grep -q 'COINGECKO_API_KEY' $CLOUD_DIR/.env 2>/dev/null && echo '  ✅ CoinGecko (cloud)' || echo '  ❌ CoinGecko (cloud)'"

    echo ""
    echo "============================================================"
}

# Main menu
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  all          - Sync everything (code, env, deps, V6 files)"
    echo "  code         - Sync code via Git"
    echo "  env          - Sync .env file only"
    echo "  deps         - Sync dependencies"
    echo "  v6           - Sync V6 rebuild files"
    echo "  status       - Show sync status"
    echo "  verify       - Verify sync integrity"
    echo ""
    echo "Example: $0 all"
    exit 1
fi

COMMAND=$1

case $COMMAND in
    all)
        sync_code
        sync_env
        sync_dependencies
        sync_data_sources
        sync_v6_files
        verify_sync
        show_status
        ;;
    code)
        sync_code
        ;;
    env)
        sync_env
        ;;
    deps)
        sync_dependencies
        ;;
    v6)
        sync_v6_files
        ;;
    status)
        show_status
        ;;
    verify)
        verify_sync
        ;;
    *)
        echo "❌ Unknown command: $COMMAND"
        exit 1
        ;;
esac

echo ""
echo "✅ Sync complete!"
echo ""
