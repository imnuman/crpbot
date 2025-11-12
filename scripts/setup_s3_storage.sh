#!/bin/bash
# Setup S3 storage for crpbot data
# Cost: $2.50/month for 3GB storage

set -e

PROJECT_NAME="crpbot"
BUCKET_NAME="crpbot-ml-data-$(date +%Y%m%d)"
REGION="us-east-1"

echo "=== CryptoBot S3 Storage Setup ==="
echo "Bucket: $BUCKET_NAME"
echo "Region: $REGION"
echo ""

# 1. Create S3 bucket
echo "[1/5] Creating S3 bucket..."
aws s3 mb "s3://${BUCKET_NAME}" --region "${REGION}" 2>&1 || echo "Bucket may already exist"

# 2. Enable versioning
echo "[2/5] Enabling versioning..."
aws s3api put-bucket-versioning \
  --bucket "${BUCKET_NAME}" \
  --versioning-configuration Status=Enabled

# 3. Add lifecycle policy (optional: move old versions to cheaper storage)
echo "[3/5] Setting up lifecycle policy..."
cat > /tmp/lifecycle-policy.json << 'EOF'
{
  "Rules": [
    {
      "Id": "ArchiveOldVersions",
      "Status": "Enabled",
      "NoncurrentVersionTransitions": [
        {
          "NoncurrentDays": 30,
          "StorageClass": "STANDARD_IA"
        },
        {
          "NoncurrentDays": 90,
          "StorageClass": "GLACIER"
        }
      ],
      "NoncurrentVersionExpiration": {
        "NoncurrentDays": 365
      }
    }
  ]
}
EOF

aws s3api put-bucket-lifecycle-configuration \
  --bucket "${BUCKET_NAME}" \
  --lifecycle-configuration file:///tmp/lifecycle-policy.json

# 4. Add tags
echo "[4/5] Adding tags..."
aws s3api put-bucket-tagging \
  --bucket "${BUCKET_NAME}" \
  --tagging 'TagSet=[{Key=Project,Value=CryptoBot},{Key=Environment,Value=Production},{Key=CostCenter,Value=ML}]'

# 5. Create folder structure
echo "[5/5] Creating folder structure..."
aws s3api put-object --bucket "${BUCKET_NAME}" --key raw/
aws s3api put-object --bucket "${BUCKET_NAME}" --key features/
aws s3api put-object --bucket "${BUCKET_NAME}" --key models/production/
aws s3api put-object --bucket "${BUCKET_NAME}" --key models/experiments/
aws s3api put-object --bucket "${BUCKET_NAME}" --key sentiment/
aws s3api put-object --bucket "${BUCKET_NAME}" --key backups/

echo ""
echo "✅ S3 bucket created successfully!"
echo ""
echo "Bucket: s3://${BUCKET_NAME}"
echo "Region: ${REGION}"
echo "Versioning: Enabled"
echo "Lifecycle: Old versions archived after 30 days"
echo ""
echo "Estimated cost: $2.50/month for 3GB storage"
echo ""
echo "Next steps:"
echo "  1. Upload data: ./scripts/upload_to_s3.sh"
echo "  2. Setup GPU training: ./scripts/setup_gpu_training.sh"
echo ""

# Save bucket name for other scripts
echo "${BUCKET_NAME}" > .s3_bucket_name
echo "Bucket name saved to .s3_bucket_name"
