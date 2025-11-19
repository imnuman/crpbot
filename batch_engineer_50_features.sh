#!/bin/bash
# Batch engineer 50-feature datasets for all symbols

set -e

export PATH="/root/.local/bin:$PATH"
source .venv/bin/activate

echo "=================================================="
echo "Batch 50-Feature Engineering (Multi-Timeframe)"
echo "=================================================="
echo ""

SYMBOLS=("BTC-USD" "ETH-USD" "SOL-USD")

for symbol in "${SYMBOLS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Engineering 50 features for $symbol"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python scripts/engineer_50_features.py \
        --symbol "$symbol" \
        --intervals 1m 5m 15m 1h \
        --data-dir data/raw \
        --output-dir data/features \
        --start-date 2023-11-10

    echo ""
done

echo "=================================================="
echo "✅ All 50-feature datasets complete!"
echo "=================================================="
echo ""
echo "Output files:"
ls -lh data/features/*_50feat.parquet

echo ""
echo "Symlinks (latest):"
ls -lh data/features/features_*_1m_latest.parquet
