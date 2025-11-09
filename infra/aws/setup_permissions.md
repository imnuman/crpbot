# AWS Permissions Setup

Your AWS user `ncldev` needs additional permissions to create RDS and Secrets Manager resources.

## Required IAM Policies

Add these managed policies to your user:

1. **AmazonRDSFullAccess** - For RDS database creation
2. **SecretsManagerReadWrite** - For Secrets Manager
3. **AmazonVPCFullAccess** - For VPC/networking (RDS requirement)

## Manual Setup Commands

Since CloudFormation requires these permissions, you can create resources manually:

### 1. Create Secrets (if you get permissions)

```bash
# Coinbase API
aws secretsmanager create-secret \
  --name "crpbot/coinbase-api/dev" \
  --description "Coinbase API credentials" \
  --secret-string '{"api_key":"YOUR_KEY","api_secret":"YOUR_SECRET","api_passphrase":"YOUR_PASSPHRASE"}'

# Telegram Bot
aws secretsmanager create-secret \
  --name "crpbot/telegram-bot/dev" \
  --description "Telegram bot credentials" \
  --secret-string '{"bot_token":"YOUR_TOKEN","chat_id":"YOUR_CHAT_ID"}'
```

### 2. Create RDS (if you get permissions)

```bash
# Create default VPC first (if needed)
aws ec2 create-default-vpc

# Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier crpbot-dev \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --master-username crpbot_admin \
  --master-user-password TempPassword123! \
  --allocated-storage 20 \
  --storage-type gp3 \
  --storage-encrypted \
  --backup-retention-period 7 \
  --no-deletion-protection
```

## Current Status

✅ **S3 Buckets**: Created successfully
- crpbot-market-data-dev
- crpbot-backups-dev  
- crpbot-logs-dev

❌ **RDS**: Requires additional permissions
❌ **Secrets Manager**: Requires additional permissions

## Workaround

For now, use environment variables in `.env` file:
- Database: SQLite (existing setup)
- API Keys: Environment variables (existing setup)
- S3: Working with current permissions