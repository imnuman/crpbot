#!/bin/bash
# Monitor data fetch progress
# Run this script to check status of ongoing fetches

echo "======================================"
echo "Data Fetch Monitor"
echo "======================================"
echo "Time: $(date)"
echo ""

# Check running processes
PROCESSES=$(ps aux | grep "fetch_data.py" | grep -v grep | wc -l)
echo "Active fetch processes: $PROCESSES"

if [ "$PROCESSES" -eq 0 ]; then
    echo "⚠️  No fetch processes running!"
fi

echo ""
echo "Expected files:"
echo "  - BTC-USD_1m_2023-11-10_2025-11-13.parquet (~35 MB, 1M+ rows)"
echo "  - ETH-USD_1m_2023-11-10_2025-11-13.parquet (~32 MB, 1M+ rows)"
echo "  - SOL-USD_1m_2023-11-10_2025-11-13.parquet (~23 MB, 1M+ rows)"
echo ""

# Check for output files
echo "Current files in data/raw/:"
if ls data/raw/*_2025-*.parquet 1> /dev/null 2>&1; then
    ls -lh data/raw/*_2025-*.parquet
    echo ""
    echo "✅ PARQUET FILES FOUND - FETCH COMPLETE!"
    echo ""
    echo "Next step: Run feature engineering"
    echo "  ./batch_engineer_features.sh"
else
    echo "  (No parquet files yet - still fetching...)"
fi

echo ""
echo "To check detailed progress:"
echo "  tail -f /tmp/fetch_*.log"
echo ""
echo "======================================"
