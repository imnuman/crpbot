#!/bin/bash
# Quick script to check download progress for all 3 symbols

echo "=================================================="
echo " Data Download Progress Check"
echo "=================================================="
echo ""
echo "Started: 2025-11-15 15:43 EST"
echo "Target: 730 days (2023-11-15 to 2025-11-15)"
echo ""

# Check if data directory exists
if [ -d "data/raw/coinbase" ]; then
    echo "Files in data/raw/coinbase:"
    ls -lh data/raw/coinbase/ 2>/dev/null || echo "  (none yet)"
    echo ""
fi

echo "Background processes:"
echo ""

# BTC-USD
echo "1. BTC-USD (process 9ce74e):"
ps aux | grep "BTC-USD" | grep -v grep | head -1 || echo "   Not running"
echo ""

# ETH-USD
echo "2. ETH-USD (process 85e38d):"
ps aux | grep "ETH-USD" | grep -v grep | head -1 || echo "   Not running"
echo ""

# SOL-USD
echo "3. SOL-USD (process 7c6d4a):"
ps aux | grep "SOL-USD" | grep -v grep | head -1 || echo "   Not running"
echo ""

echo "=================================================="
echo "To check detailed output:"
echo "  - Use BashOutput tool with process IDs above"
echo "  - Or check: ls -lh data/raw/coinbase/"
echo "=================================================="
