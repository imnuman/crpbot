#!/bin/bash
# Comprehensive Project Cleanup Script
# Removes old files, caches, logs, and organizes structure

set -e

echo "🧹 CRPBot Project Cleanup"
echo "========================="
echo ""

# Safety check
read -p "This will delete cache files, old logs, and reorganize files. Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 1
fi

CLEANUP_LOG="cleanup_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date +%H:%M:%S)] $1" | tee -a "$CLEANUP_LOG"
}

log "🚀 Starting cleanup..."

# ============================================================================
# 1. Remove Python Cache Files
# ============================================================================
log ""
log "📦 Cleaning Python cache..."

find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true

log "✅ Removed Python cache files"

# ============================================================================
# 2. Clean Old Log Files
# ============================================================================
log ""
log "📝 Cleaning old log files..."

# Remove logs older than 7 days from /tmp
find /tmp -name "*.log" -mtime +7 -type f -delete 2>/dev/null || true

# Clean specific old logs
rm -f /tmp/fetch_bnb.log 2>/dev/null || true
rm -f /tmp/features_btc.log 2>/dev/null || true

log "✅ Cleaned old log files"

# ============================================================================
# 3. Archive Old Documentation
# ============================================================================
log ""
log "📚 Archiving old documentation..."

mkdir -p docs/archive

# Move old/deprecated docs
DEPRECATED_DOCS=(
    "URGENT_STOP_CPU_TRAINING.md"
    "COLAB_GPU_TRAINING.md"
    "COLAB_GPU_TRAINING_INSTRUCTIONS.md"
    "COLAB_TROUBLESHOOTING.md"
    "COLAB_TRAINING_SCRIPT.py"
    "COLAB_TRAINING_SCRIPT_V2.py"
    "PHASE1_COMPLETE_NEXT_STEPS.md"
    "INVESTIGATION_GUIDE_50FEAT.md"
    "NEXT_ACTION_INVESTIGATION.md"
    "RETRAINING_PLAN_31FEAT.md"
    "MIGRATION_GUIDE.md"
    "PRE_MIGRATION_CHECKLIST.md"
)

for doc in "${DEPRECATED_DOCS[@]}"; do
    if [ -f "$doc" ]; then
        mv "$doc" docs/archive/ 2>/dev/null || true
        log "  Archived: $doc"
    fi
done

log "✅ Archived deprecated documentation"

# ============================================================================
# 4. Clean Old Model Directories
# ============================================================================
log ""
log "🤖 Cleaning old model directories..."

# Remove old GPU training attempts
rm -rf models/gpu_trained 2>/dev/null || true
rm -rf models/gpu_trained_new 2>/dev/null || true
rm -rf models/gpu_trained_proper 2>/dev/null || true
rm -rf models/v6_statistical 2>/dev/null || true

# Remove individual old model files (keeping only latest versions)
# Keep: V5 FIXED (in promoted/), V6 models when ready
# Remove: Old a7aff5c4, c5a1b96f versions

rm -f models/lstm_*_a7aff5c4.pt 2>/dev/null || true
rm -f models/lstm_*_c5a1b96f.pt 2>/dev/null || true

log "✅ Cleaned old model directories"

# ============================================================================
# 5. Organize Root Directory Scripts
# ============================================================================
log ""
log "📁 Organizing root directory scripts..."

# Create archive for old test scripts
mkdir -p scripts/archive

# Move test/experimental scripts to archive
TEST_SCRIPTS=(
    "backtest_gpu_models.py"
    "backtest_gpu_proper.py"
    "colab_find_and_download_models.py"
    "compare_jwts.py"
    "create_v6_training_data.py"
    "deploy_runtime.py"
    "extract_v6_features.py"
    "test_aws_connections.py"
    "test_coinbase_auth.py"
    "test_gpu_runtime.py"
    "test_provider.py"
    "test_runtime_connection.py"
    "test_s3_integration.py"
    "test_s3_simple.py"
    "test_v5_integration.py"
    "train_gpu_proper_clean.py"
    "train_gpu_proper.py"
    "train_v6_real.py"
    "train_v6_runtime_features.py"
    "v6_monte_carlo_models.py"
    "v6_simple_models.py"
    "validate_gpu_models.py"
    "verify_colab_files.py"
    "verify_models.py"
)

for script in "${TEST_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        mv "$script" scripts/archive/ 2>/dev/null || true
        log "  Moved to archive: $script"
    fi
done

log "✅ Organized root directory"

# ============================================================================
# 6. Clean Temporary Data Files
# ============================================================================
log ""
log "🗑️  Cleaning temporary files..."

# Remove specific temp files
rm -f gpu_backtest_results.json 2>/dev/null || true
rm -f gpu_runtime_test_results.json 2>/dev/null || true
rm -f runtime_config.json 2>/dev/null || true
rm -f investigation_results_*.txt 2>/dev/null || true
rm -f performance_summary.md 2>/dev/null || true

# Remove old zip files
rm -f gpu_models.zip 2>/dev/null || true

log "✅ Cleaned temporary files"

# ============================================================================
# 7. Clean Old Data Artifacts
# ============================================================================
log ""
log "💾 Checking data directories..."

# Don't delete data, just report
DATA_SIZE=$(du -sh data/ 2>/dev/null | cut -f1)
RAW_SIZE=$(du -sh data/raw 2>/dev/null | cut -f1)
FEATURES_SIZE=$(du -sh data/features 2>/dev/null | cut -f1)
TRAINING_SIZE=$(du -sh data/training 2>/dev/null | cut -f1)

log "  Total data: $DATA_SIZE"
log "  - Raw: $RAW_SIZE"
log "  - Features: $FEATURES_SIZE"
log "  - Training: $TRAINING_SIZE"
log "  (Data preserved - no cleanup needed)"

# ============================================================================
# 8. Update .gitignore
# ============================================================================
log ""
log "📄 Updating .gitignore..."

# Add common patterns if not already there
GITIGNORE_ADDITIONS=(
    "*.log"
    "__pycache__/"
    "*.pyc"
    "*.egg-info/"
    ".pytest_cache/"
    ".ruff_cache/"
    ".mypy_cache/"
    ".env"
    "*.swp"
    "*.swo"
    ".DS_Store"
    "gpu_*.json"
    "runtime_config.json"
    "investigation_results_*.txt"
)

for pattern in "${GITIGNORE_ADDITIONS[@]}"; do
    if ! grep -q "^$pattern$" .gitignore 2>/dev/null; then
        echo "$pattern" >> .gitignore
    fi
done

log "✅ Updated .gitignore"

# ============================================================================
# 9. Summary
# ============================================================================
log ""
log "📊 Cleanup Summary"
log "=================="

# Count remaining files
PYTHON_FILES=$(find . -name "*.py" -type f | wc -l)
MD_FILES=$(find . -name "*.md" -type f | wc -l)
MODEL_FILES=$(find models/ -name "*.pt" -type f 2>/dev/null | wc -l)
CACHE_DIRS=$(find . -name "__pycache__" -type d 2>/dev/null | wc -l)

log "  Python files: $PYTHON_FILES"
log "  Documentation: $MD_FILES"
log "  Model files: $MODEL_FILES"
log "  Cache directories: $CACHE_DIRS"

# Disk usage
TOTAL_SIZE=$(du -sh . 2>/dev/null | cut -f1)
log "  Total project size: $TOTAL_SIZE"

log ""
log "✅ Cleanup complete!"
log "📋 Full log: $CLEANUP_LOG"

echo ""
echo "Next steps:"
echo "  1. Review cleanup log: cat $CLEANUP_LOG"
echo "  2. Check git status: git status"
echo "  3. Commit cleanup: git add . && git commit -m 'chore: project cleanup'"
