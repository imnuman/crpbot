#!/bin/bash
# Upload current data to S3

set -e

# Get bucket name
if [ -f .s3_bucket_name ]; then
  BUCKET_NAME=$(cat .s3_bucket_name)
else
  echo "Error: Bucket name not found. Run setup_s3_storage.sh first."
  exit 1
fi

echo "=== Uploading Data to S3 ==="
echo "Bucket: s3://${BUCKET_NAME}"
echo ""

# Show current data size
echo "Current data size:"
du -sh data/raw/ data/features/ models/
echo ""

# Upload with progress
echo "[1/3] Uploading raw data..."
aws s3 sync data/raw/ "s3://${BUCKET_NAME}/raw/" \
  --exclude "*.log" \
  --exclude "*.tmp" \
  --storage-class STANDARD

echo "[2/3] Uploading features..."
aws s3 sync data/features/ "s3://${BUCKET_NAME}/features/" \
  --exclude "*.log" \
  --exclude "*.tmp" \
  --storage-class STANDARD

echo "[3/3] Uploading models..."
aws s3 sync models/ "s3://${BUCKET_NAME}/models/production/" \
  --exclude "*.log" \
  --exclude "*.tmp" \
  --storage-class STANDARD

echo ""
echo "✅ Data uploaded successfully!"
echo ""
echo "Verify upload:"
echo "  aws s3 ls s3://${BUCKET_NAME} --recursive --human-readable --summarize"
echo ""

# Show storage cost estimate
TOTAL_SIZE=$(aws s3 ls s3://${BUCKET_NAME} --recursive --summarize | grep "Total Size" | awk '{print $3}')
echo "Total size in S3: ${TOTAL_SIZE} bytes"
COST=$(echo "scale=2; ${TOTAL_SIZE} / 1073741824 * 0.023" | bc 2>/dev/null || echo "~$2.50")
echo "Estimated monthly cost: \$${COST}"
