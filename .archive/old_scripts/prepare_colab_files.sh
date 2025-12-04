#!/bin/bash
# Prepare files for Google Colab upload

echo "=================================================="
echo "Preparing Files for Google Colab Upload"
echo "=================================================="
echo ""

# Set working directory
cd /root/crpbot

# Create upload directory
UPLOAD_DIR="/tmp/colab_upload"
mkdir -p "$UPLOAD_DIR"/{models,features}

echo "📦 Copying model files..."
cp /root/crpbot/models/new/lstm_*_7b5f0829.pt "$UPLOAD_DIR/models/"
echo "  ✅ Copied 3 model files (3.9 MB each)"

echo ""
echo "📦 Copying feature datasets..."
cp /root/crpbot/data/features/features_*_50feat.parquet "$UPLOAD_DIR/features/"
echo "  ✅ Copied 3 feature files (~200-230 MB each)"

echo ""
echo "📊 Upload Summary:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ls -lh "$UPLOAD_DIR/models/"
echo ""
ls -lh "$UPLOAD_DIR/features/"
echo ""

TOTAL_SIZE=$(du -sh "$UPLOAD_DIR" | cut -f1)
echo "📦 Total upload size: $TOTAL_SIZE"
echo ""

echo "=================================================="
echo "✅ Files ready in: $UPLOAD_DIR"
echo "=================================================="
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Open the Colab notebook:"
echo "   File: colab_evaluate_50feat_models.ipynb"
echo "   URL: https://colab.research.google.com/"
echo ""
echo "2. Upload this notebook to Colab"
echo ""
echo "3. Set runtime to GPU:"
echo "   Runtime → Change runtime type → GPU (T4)"
echo ""
echo "4. Upload files to Colab:"
echo "   Option A: Direct upload via Colab file browser"
echo "     - Upload models/*.pt to models/new/"
echo "     - Upload features/*.parquet to data/features/"
echo ""
echo "   Option B: Use Google Drive"
echo "     - Upload $UPLOAD_DIR to Google Drive"
echo "     - Mount Drive in Colab and copy files"
echo ""
echo "5. Run all cells in the notebook"
echo ""
echo "6. Download evaluation_results.csv when complete"
echo ""
echo "⏱️  Expected runtime: 5-10 minutes (vs 60+ minutes on CPU)"
echo ""
