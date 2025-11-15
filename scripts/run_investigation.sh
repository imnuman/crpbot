#!/bin/bash
# Run comprehensive investigation on all 3 symbols

set -e

cd "$(dirname "$0")/.."

echo "======================================================================"
echo "🔍 INVESTIGATING 50-FEATURE MODEL FAILURE"
echo "======================================================================"
echo ""
echo "This will check:"
echo "  1. Data quality (NaN, Inf, outliers)"
echo "  2. Target distribution"
echo "  3. Feature statistics"
echo "  4. Data leakage"
echo "  5. Simple baseline model"
echo ""
echo "Running investigation on all 3 symbols..."
echo "Estimated time: ~15 minutes"
echo ""

OUTPUT_FILE="investigation_results_$(date +%Y%m%d_%H%M%S).txt"

echo "Output will be saved to: $OUTPUT_FILE"
echo ""

# Run for each symbol
for symbol in BTC-USD ETH-USD SOL-USD; do
    echo "======================================================================"
    echo "Investigating $symbol..."
    echo "======================================================================"

    {
        echo ""
        echo "======================================================================"
        echo "=== $symbol ==="
        echo "======================================================================"
        uv run python scripts/investigate_50feat_failure.py --symbol "$symbol"
        echo ""
        echo ""
    } | tee -a "$OUTPUT_FILE"

    echo ""
    echo "✅ $symbol complete"
    echo ""
done

echo "======================================================================"
echo "✅ ALL INVESTIGATIONS COMPLETE"
echo "======================================================================"
echo ""
echo "Results saved to: $OUTPUT_FILE"
echo ""
echo "Next steps:"
echo "1. Check the recommendation at the end of each section"
echo "2. Look for baseline accuracy:"
echo "   - <52%: DATA/TARGET issue"
echo "   - 52-55%: HARD PROBLEM"
echo "   - 55-65%: TRAINABLE"
echo "   - ≥65%: TRAINING ISSUE"
echo ""
echo "Share the results with QC Claude for next steps!"
echo ""
