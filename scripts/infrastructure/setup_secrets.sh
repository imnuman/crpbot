#!/bin/bash
set -e

# Setup AWS Secrets Manager for CryptoBot credentials
# Migrates credentials from .env files to Secrets Manager

REGION="us-east-1"
PROJECT="crpbot"

echo "=== Setting up AWS Secrets Manager ==="
echo "Region: $REGION"
echo "Project: $PROJECT"
echo ""

# Function to create or update secret
create_or_update_secret() {
    local secret_name=$1
    local secret_value=$2
    local description=$3

    echo "Processing secret: $secret_name"

    # Check if secret exists
    if aws secretsmanager describe-secret \
        --secret-id "$secret_name" \
        --region "$REGION" &>/dev/null; then

        echo "  → Secret exists, updating value..."
        aws secretsmanager put-secret-value \
            --secret-id "$secret_name" \
            --secret-string "$secret_value" \
            --region "$REGION" \
            --output text > /dev/null
    else
        echo "  → Creating new secret..."
        aws secretsmanager create-secret \
            --name "$secret_name" \
            --description "$description" \
            --secret-string "$secret_value" \
            --region "$REGION" \
            --tags Key=Project,Value=CryptoBot Key=Environment,Value=Production \
            --output text > /dev/null
    fi

    echo "  ✓ Done"
}

# 1. RDS Database credentials
if [ -f .rds_connection_info ]; then
    echo "Found RDS connection info..."
    source .rds_connection_info

    RDS_SECRET=$(cat <<EOF
{
  "host": "$DB_HOST",
  "port": $DB_PORT,
  "database": "$DB_NAME",
  "username": "$DB_USER",
  "password": "$DB_PASSWORD"
}
EOF
)

    create_or_update_secret \
        "${PROJECT}/rds/credentials" \
        "$RDS_SECRET" \
        "RDS PostgreSQL database credentials for CryptoBot"
else
    echo "⚠ RDS connection info not found (skip if RDS not deployed yet)"
fi

# 2. Redis credentials
if [ -f .redis_connection_info ]; then
    echo "Found Redis connection info..."
    source .redis_connection_info

    REDIS_SECRET=$(cat <<EOF
{
  "host": "$REDIS_HOST",
  "port": $REDIS_PORT,
  "url": "$REDIS_URL"
}
EOF
)

    create_or_update_secret \
        "${PROJECT}/redis/credentials" \
        "$REDIS_SECRET" \
        "ElastiCache Redis credentials for CryptoBot"
else
    echo "⚠ Redis connection info not found (skip if Redis not deployed yet)"
fi

# 3. Coinbase API credentials
if [ -f .env ]; then
    echo "Found .env file, extracting Coinbase credentials..."

    # Source .env
    export $(grep -v '^#' .env | xargs)

    if [ -n "$COINBASE_API_KEY" ]; then
        COINBASE_SECRET=$(cat <<EOF
{
  "api_key": "$COINBASE_API_KEY",
  "api_secret": "$COINBASE_API_SECRET"
}
EOF
)

        create_or_update_secret \
            "${PROJECT}/coinbase/api" \
            "$COINBASE_SECRET" \
            "Coinbase Advanced Trade API credentials"
    fi

    # 4. Reddit API credentials (if exists)
    if [ -n "$REDDIT_CLIENT_ID" ]; then
        REDDIT_SECRET=$(cat <<EOF
{
  "client_id": "$REDDIT_CLIENT_ID",
  "client_secret": "$REDDIT_CLIENT_SECRET",
  "user_agent": "$REDDIT_USER_AGENT"
}
EOF
)

        create_or_update_secret \
            "${PROJECT}/reddit/api" \
            "$REDDIT_SECRET" \
            "Reddit API credentials for sentiment analysis"
    fi

    # 5. Other API keys
    if [ -n "$CRYPTOCOMPARE_API_KEY" ]; then
        create_or_update_secret \
            "${PROJECT}/cryptocompare/api-key" \
            "$CRYPTOCOMPARE_API_KEY" \
            "CryptoCompare API key"
    fi

    if [ -n "$SANTIMENT_API_KEY" ]; then
        create_or_update_secret \
            "${PROJECT}/santiment/api-key" \
            "$SANTIMENT_API_KEY" \
            "Santiment API key"
    fi
else
    echo "⚠ .env file not found"
fi

# 6. MLflow tracking credentials (if needed)
MLFLOW_SECRET=$(cat <<EOF
{
  "tracking_uri": "http://localhost:5000",
  "experiment_name": "crpbot-training"
}
EOF
)

create_or_update_secret \
    "${PROJECT}/mlflow/config" \
    "$MLFLOW_SECRET" \
    "MLflow tracking server configuration"

echo ""
echo "=== Secrets Manager Setup Complete ==="
echo ""
echo "Stored secrets:"
aws secretsmanager list-secrets \
    --region "$REGION" \
    --filters Key=tag-key,Values=Project Key=tag-value,Values=CryptoBot \
    --query 'SecretList[*].[Name,Description]' \
    --output table

echo ""
echo "To retrieve a secret:"
echo "  aws secretsmanager get-secret-value --secret-id ${PROJECT}/rds/credentials --region $REGION"
echo ""
echo "Cost: \$0.40 per secret per month + \$0.05 per 10,000 API calls"
echo "Estimated: ~\$2.40/month for 6 secrets"
echo ""
echo "Next steps:"
echo "  1. Update application code to use Secrets Manager SDK"
echo "  2. Remove credentials from .env files (keep .env.example)"
echo "  3. Update IAM roles to grant secretsmanager:GetSecretValue permission"
echo "  4. Test secret retrieval from applications"
