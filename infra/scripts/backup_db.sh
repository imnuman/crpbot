#!/bin/bash
# Backup database to S3 or local storage

set -e

# Configuration
DB_PATH="${DB_PATH:-tradingai.db}"
BACKUP_DIR="${BACKUP_DIR:-backups}"
S3_BUCKET="${S3_BUCKET:-}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Generate backup filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/tradingai_${TIMESTAMP}.db"

echo "📦 Backing up database..."
echo "   Source: $DB_PATH"
echo "   Destination: $BACKUP_FILE"

# Copy database file
if [ -f "$DB_PATH" ]; then
    cp "$DB_PATH" "$BACKUP_FILE"
    echo "✅ Database backed up to $BACKUP_FILE"
    
    # Compress backup
    gzip "$BACKUP_FILE"
    echo "✅ Backup compressed: ${BACKUP_FILE}.gz"
    
    # Upload to S3 if configured
    if [ -n "$S3_BUCKET" ] && command -v aws &> /dev/null; then
        echo "📤 Uploading to S3: s3://$S3_BUCKET/backups/"
        aws s3 cp "${BACKUP_FILE}.gz" "s3://$S3_BUCKET/backups/" --region "$AWS_REGION"
        echo "✅ Backup uploaded to S3"
    else
        echo "⚠️  S3 upload skipped (S3_BUCKET not set or AWS CLI not installed)"
    fi
    
    # Clean up old backups (keep last 30 days)
    find "$BACKUP_DIR" -name "tradingai_*.db.gz" -mtime +30 -delete
    echo "✅ Old backups cleaned up (kept last 30 days)"
else
    echo "❌ Database file not found: $DB_PATH"
    exit 1
fi

echo "✅ Backup complete!"

