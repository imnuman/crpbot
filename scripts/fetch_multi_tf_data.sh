#!/bin/bash
# Batch fetch multi-timeframe data for Phase 3.5 (V2)
# Fetches 1m, 5m, 15m, 1h data for all symbols

set -e  # Exit on error

SYMBOLS=("BTC-USD" "ETH-USD" "SOL-USD")
INTERVALS=("1m" "5m" "15m" "1h")
START="2023-11-10"
PROVIDER="coinbase"  # Using Coinbase Advanced Trade API

echo "========================================="
echo "Multi-Timeframe Data Fetch - Phase 3.5"
echo "========================================="
echo "Symbols: ${SYMBOLS[@]}"
echo "Intervals: ${INTERVALS[@]}"
echo "Start date: $START"
echo "Provider: $PROVIDER"
echo "========================================="
echo ""

# Track total fetches
TOTAL_FETCHES=$((${#SYMBOLS[@]} * ${#INTERVALS[@]}))
CURRENT=0

for SYMBOL in "${SYMBOLS[@]}"; do
    for INTERVAL in "${INTERVALS[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo "[$CURRENT/$TOTAL_FETCHES] Fetching $SYMBOL $INTERVAL data..."

        LOG_FILE="/tmp/fetch_${SYMBOL}_${INTERVAL}.log"

        # Fetch data in background
        uv run python scripts/fetch_data.py \
            --symbol "$SYMBOL" \
            --start "$START" \
            --interval "$INTERVAL" \
            --output data/raw \
            > "$LOG_FILE" 2>&1 &

        PID=$!
        echo "  Started (PID: $PID, log: $LOG_FILE)"

        # Small delay to avoid overwhelming API
        sleep 2
    done
done

echo ""
echo "========================================="
echo "All fetch jobs started!"
echo "========================================="
echo ""
echo "Monitor progress:"
echo "  watch -n 10 'ps aux | grep fetch_data.py'"
echo ""
echo "Check logs:"
for SYMBOL in "${SYMBOLS[@]}"; do
    for INTERVAL in "${INTERVALS[@]}"; do
        echo "  tail -f /tmp/fetch_${SYMBOL}_${INTERVAL}.log"
    done
done
echo ""
echo "Expected output files in data/raw/:"
for SYMBOL in "${SYMBOLS[@]}"; do
    for INTERVAL in "${INTERVALS[@]}"; do
        echo "  ${SYMBOL}_${INTERVAL}_${START}_*.parquet"
    done
done
echo ""
