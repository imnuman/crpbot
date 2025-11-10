#!/bin/bash
# Batch feature engineering for all symbols
# Phase 6.5 - CRPBot Training Pipeline

set -e  # Exit on error

SYMBOLS=("BTC-USD" "ETH-USD" "SOL-USD")
INTERVAL="1m"
START="2023-11-10"
END="2025-11-10"

echo "========================================="
echo "Feature Engineering - Phase 6.5"
echo "========================================="
echo "Dataset: $START to $END"
echo "Interval: $INTERVAL"
echo "Symbols: ${SYMBOLS[@]}"
echo "========================================="
echo ""

for SYMBOL in "${SYMBOLS[@]}"; do
    echo "========================================="
    echo "Processing $SYMBOL..."
    echo "========================================="

    INPUT_FILE="data/raw/${SYMBOL}_${INTERVAL}_${START}_${END}.parquet"

    if [ ! -f "$INPUT_FILE" ]; then
        echo "❌ Error: $INPUT_FILE not found!"
        echo "   Please run data fetch first:"
        echo "   uv run python scripts/fetch_data.py --symbol $SYMBOL --start $START --interval $INTERVAL"
        continue
    fi

    # Show file size
    FILE_SIZE=$(ls -lh "$INPUT_FILE" | awk '{print $5}')
    echo "Input file: $INPUT_FILE ($FILE_SIZE)"

    # Run feature engineering
    echo "Generating features..."
    uv run python scripts/engineer_features.py \
        --input "$INPUT_FILE" \
        --symbol "$SYMBOL" \
        --interval "$INTERVAL"

    if [ $? -eq 0 ]; then
        # Show output file
        OUTPUT_FILE="data/features/features_${SYMBOL}_${INTERVAL}_latest.parquet"
        if [ -f "$OUTPUT_FILE" ]; then
            OUTPUT_SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
            echo "✅ $SYMBOL features completed"
            echo "   Output: $OUTPUT_FILE ($OUTPUT_SIZE)"
        else
            echo "⚠️  $SYMBOL features completed but output file not found"
        fi
    else
        echo "❌ $SYMBOL features failed"
        exit 1
    fi

    echo ""
done

echo "========================================="
echo "✅ All feature engineering complete!"
echo "========================================="
echo ""
echo "Generated files:"
ls -lh data/features/*_latest.parquet 2>/dev/null || echo "No feature files found"
echo ""
echo "Next step: Train models"
echo "  make train COIN=BTC EPOCHS=15"
