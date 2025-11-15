#!/bin/bash
# Batch download all CoinGecko data for V5
# Run this AFTER Coinbase downloads complete

set -e  # Exit on error

echo "=================================================="
echo " CoinGecko Batch Download - V5 Ultimate Dataset"
echo "=================================================="
echo ""
echo "This will download:"
echo "  - On-chain 1m data (2021-2025, 4 years)"
echo "  - Hourly OHLCV (2018-2025, 7 years)"
echo "  - Daily metadata (2023-2025, 2 years)"
echo ""
echo "For symbols: BTC-USD, ETH-USD, SOL-USD"
echo ""
echo "Estimated time: ~55 minutes total"
echo "  - On-chain: ~30 min (3 symbols)"
echo "  - Hourly: ~10 min (3 symbols)"
echo "  - Daily: ~15 min (3 symbols)"
echo ""
echo "=================================================="
echo ""

# Check if API key is set
if [ -z "$COINGECKO_API_KEY" ]; then
    echo "❌ ERROR: COINGECKO_API_KEY not set"
    echo "Please set it in .env file or export it"
    exit 1
fi

echo "✅ API Key found: ${COINGECKO_API_KEY:0:10}..."
echo ""

# Symbols to download
SYMBOLS=("BTC-USD" "ETH-USD" "SOL-USD")

# ================================================
# Phase 1: On-Chain Data (1-minute, 2021-2025)
# ================================================
echo "=================================================="
echo " Phase 1: On-Chain 1m Data (Highest Priority)"
echo "=================================================="
echo ""
echo "Downloading whale signals & network metrics..."
echo "Timeframe: 2021-01-01 to 2025-11-15 (4 years)"
echo ""

for symbol in "${SYMBOLS[@]}"; do
    echo "---"
    echo "Downloading on-chain data for $symbol..."
    echo "Start time: $(date '+%H:%M:%S')"

    python scripts/fetch_coingecko_onchain.py \
        --symbol "$symbol" \
        --start 2021-01-01 \
        --end 2025-11-15 \
        --chunk-days 30

    if [ $? -eq 0 ]; then
        echo "✅ $symbol on-chain data complete"
    else
        echo "⚠️  $symbol on-chain data failed (continuing...)"
    fi
    echo ""
done

echo "✅ Phase 1 complete: On-chain data downloaded"
echo ""

# ================================================
# Phase 2: Hourly OHLCV (2018-2025)
# ================================================
echo "=================================================="
echo " Phase 2: Hourly OHLCV (7 Years)"
echo "=================================================="
echo ""
echo "Downloading multi-timeframe trend data..."
echo "Timeframe: 2018-01-01 to 2025-11-15 (7 years)"
echo ""

for symbol in "${SYMBOLS[@]}"; do
    echo "---"
    echo "Downloading hourly OHLCV for $symbol..."
    echo "Start time: $(date '+%H:%M:%S')"

    python scripts/fetch_coingecko_hourly.py \
        --symbol "$symbol" \
        --start 2018-01-01 \
        --end 2025-11-15 \
        --chunk-days 365

    if [ $? -eq 0 ]; then
        echo "✅ $symbol hourly data complete"
    else
        echo "⚠️  $symbol hourly data failed (continuing...)"
    fi
    echo ""
done

echo "✅ Phase 2 complete: Hourly OHLCV downloaded"
echo ""

# ================================================
# Phase 3: Daily Metadata (2023-2025)
# ================================================
echo "=================================================="
echo " Phase 3: Daily Market Metadata"
echo "=================================================="
echo ""
echo "Downloading market context & sentiment..."
echo "Timeframe: 2023-11-15 to 2025-11-15 (2 years)"
echo ""

for symbol in "${SYMBOLS[@]}"; do
    echo "---"
    echo "Downloading daily metadata for $symbol..."
    echo "Start time: $(date '+%H:%M:%S')"

    python scripts/fetch_coingecko_metadata.py \
        --symbol "$symbol" \
        --start 2023-11-15 \
        --end 2025-11-15

    if [ $? -eq 0 ]; then
        echo "✅ $symbol daily metadata complete"
    else
        echo "⚠️  $symbol daily metadata failed (continuing...)"
    fi
    echo ""
done

echo "✅ Phase 3 complete: Daily metadata downloaded"
echo ""

# ================================================
# Summary
# ================================================
echo "=================================================="
echo " Download Summary"
echo "=================================================="
echo ""
echo "Checking downloaded files..."
echo ""

echo "Coinbase 1m OHLCV:"
ls -lh data/raw/coinbase/*.parquet 2>/dev/null || echo "  (none found)"
echo ""

echo "CoinGecko On-Chain 1m:"
ls -lh data/raw/coingecko_onchain/*.parquet 2>/dev/null || echo "  (none found)"
echo ""

echo "CoinGecko Hourly OHLCV:"
ls -lh data/raw/coingecko_hourly/*.parquet 2>/dev/null || echo "  (none found)"
echo ""

echo "CoinGecko Daily Metadata:"
ls -lh data/raw/coingecko_daily/*.parquet 2>/dev/null || echo "  (none found)"
echo ""

echo "=================================================="
echo " ✅ ALL COINGECKO DOWNLOADS COMPLETE!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Validate data quality"
echo "  2. Engineer features from all sources"
echo "  3. Merge into hybrid dataset (65-82 features)"
echo "  4. Begin model training (Week 2)"
echo ""
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="
