#!/bin/bash
# Sequential multi-timeframe data fetch (rate-limit friendly)
# Fetches 5m, 15m, 1h data one at a time to avoid API rate limits

set -e  # Exit on error

SYMBOLS=("BTC-USD" "ETH-USD" "SOL-USD")
INTERVALS=("5m" "15m" "1h")  # Skip 1m (already have it)
START="2023-11-10"
PROVIDER="coinbase"

echo "========================================="
echo "Sequential Multi-TF Data Fetch (Phase 3.5)"
echo "========================================="
echo "Symbols: ${SYMBOLS[@]}"
echo "Intervals: ${INTERVALS[@]}"
echo "Start date: $START"
echo "Provider: $PROVIDER"
echo "Fetching SEQUENTIALLY to avoid rate limits"
echo "========================================="
echo ""

TOTAL_FETCHES=$((${#SYMBOLS[@]} * ${#INTERVALS[@]}))
CURRENT=0

for SYMBOL in "${SYMBOLS[@]}"; do
    for INTERVAL in "${INTERVALS[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "========================================="
        echo "[$CURRENT/$TOTAL_FETCHES] Fetching $SYMBOL $INTERVAL data..."
        echo "========================================="

        LOG_FILE="/tmp/fetch_${SYMBOL}_${INTERVAL}_sequential.log"

        # Fetch data SYNCHRONOUSLY (not in background)
        uv run python scripts/fetch_data.py \
            --symbol "$SYMBOL" \
            --start "$START" \
            --interval "$INTERVAL" \
            --output data/raw \
            2>&1 | tee "$LOG_FILE"

        # Check if successful
        if [ $? -eq 0 ]; then
            echo "✅ Success: $SYMBOL $INTERVAL"
        else
            echo "❌ Failed: $SYMBOL $INTERVAL (see $LOG_FILE)"
        fi

        # Rate limit delay (10 seconds between fetches)
        if [ $CURRENT -lt $TOTAL_FETCHES ]; then
            echo "Waiting 10 seconds before next fetch..."
            sleep 10
        fi
    done
done

echo ""
echo "========================================="
echo "All fetches complete!"
echo "========================================="
echo ""
echo "Expected output files in data/raw/:"
for SYMBOL in "${SYMBOLS[@]}"; do
    for INTERVAL in "${INTERVALS[@]}"; do
        echo "  ${SYMBOL}_${INTERVAL}_${START}_*.parquet"
    done
done
echo ""
