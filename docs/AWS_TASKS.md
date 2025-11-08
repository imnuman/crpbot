# AWS Tasks and Infrastructure Plan

This document outlines AWS-related tasks for the CRPBot trading system. **Amazon Q** should be used for all tasks in this document.

## 🎯 Overview

CRPBot will use AWS infrastructure for:
- **Data Storage**: S3 for historical market data and backups
- **Database**: RDS PostgreSQL for trading signals, patterns, and risk snapshots
- **Compute**: Lambda functions for signal processing and alerts
- **Notifications**: SNS/SQS for Telegram bot integration
- **Monitoring**: CloudWatch for system health and trading metrics

## 📋 AWS Infrastructure Roadmap

### Phase 1: Foundation (Priority: HIGH)

#### Task 1.1: S3 Bucket Setup
**Branch**: `aws/s3-setup`

Create S3 buckets for:
- `crpbot-market-data-{env}` - Historical OHLCV data from Coinbase
- `crpbot-backups-{env}` - Database backups and model checkpoints
- `crpbot-logs-{env}` - Application logs and trading history

Requirements:
- Enable versioning for backups bucket
- Lifecycle policies: Move to Glacier after 90 days
- Server-side encryption (AES-256)
- CORS configuration for web access if needed
- Bucket policies for least-privilege access

**Deliverables**:
- CloudFormation/CDK template for S3 buckets
- IAM roles and policies for S3 access
- Environment variables for bucket names
- Cost estimate for storage

#### Task 1.2: RDS PostgreSQL Database
**Branch**: `aws/rds-setup`

Create RDS PostgreSQL instance for production database:
- Instance class: db.t3.micro (free tier) or db.t3.small
- PostgreSQL 15+
- Multi-AZ for production (optional for dev)
- Automated backups (7-day retention)
- Storage: 20GB GP3 with autoscaling to 100GB

Requirements:
- Security group: Only allow access from Lambda/EC2
- VPC setup with private subnets
- Parameter group for PostgreSQL tuning
- Read replica for analytics (optional)

**Deliverables**:
- CloudFormation/CDK template for RDS
- Database connection string in Secrets Manager
- Migration scripts to move from SQLite
- Cost estimate for database

#### Task 1.3: Secrets Management
**Branch**: `aws/secrets-manager`

Store sensitive credentials in AWS Secrets Manager:
- Coinbase API key and private key
- Database connection strings
- Telegram bot token (future)
- MT5 credentials (future)

Requirements:
- Automatic rotation for database credentials
- IAM policies for secret access
- Encryption with KMS
- Secret versioning enabled

**Deliverables**:
- Secrets created in AWS Secrets Manager
- IAM roles for secret access
- Python code to retrieve secrets with boto3
- Cost estimate for secrets storage

### Phase 2: Compute and Processing (Priority: MEDIUM)

#### Task 2.1: Lambda Function - Signal Processing
**Branch**: `aws/lambda-signal-processing`

Create Lambda function to process trading signals:
- Runtime: Python 3.11
- Memory: 512MB
- Timeout: 30s
- Trigger: EventBridge schedule (every 5 minutes)

Function responsibilities:
- Fetch latest market data from Coinbase
- Run trained models (LSTM, Transformer, RL)
- Generate trading signals
- Store signals in RDS
- Check FTMO rules and rate limits
- Send high-confidence signals to SNS

Requirements:
- Lambda Layer with dependencies (torch, pandas, sqlalchemy, etc.)
- Environment variables for configuration
- CloudWatch Logs for monitoring
- X-Ray tracing for debugging

**Deliverables**:
- Lambda function code
- Lambda Layer with dependencies
- EventBridge schedule rule
- IAM role with necessary permissions
- Cost estimate for Lambda execution

#### Task 2.2: Lambda Function - Risk Monitoring
**Branch**: `aws/lambda-risk-monitor`

Create Lambda function to monitor account risk:
- Runtime: Python 3.11
- Memory: 256MB
- Timeout: 15s
- Trigger: EventBridge schedule (every hour)

Function responsibilities:
- Query RDS for recent trades and PnL
- Calculate daily/total loss percentages
- Check FTMO rule compliance
- Create RiskBookSnapshot records
- Send alerts if approaching limits

**Deliverables**:
- Lambda function code
- EventBridge schedule rule
- SNS topic for risk alerts
- CloudWatch alarms for critical thresholds
- Cost estimate

#### Task 2.3: Lambda Function - Telegram Notifications
**Branch**: `aws/lambda-telegram-bot`

Create Lambda function for Telegram bot:
- Runtime: Python 3.11
- Memory: 256MB
- Timeout: 10s
- Trigger: SNS topic subscription

Function responsibilities:
- Receive signal notifications from SNS
- Format messages with signal details
- Send to Telegram channel/group
- Handle user commands (future: /status, /stats)

**Deliverables**:
- Lambda function code
- SNS topic for signals
- Telegram bot setup guide
- Cost estimate

### Phase 3: Monitoring and Alerts (Priority: MEDIUM)

#### Task 3.1: CloudWatch Dashboards
**Branch**: `aws/cloudwatch-dashboards`

Create CloudWatch dashboards for monitoring:

**Dashboard 1: Trading Metrics**
- Total signals generated (24h, 7d, 30d)
- Win rate by tier (high/medium/low)
- Daily PnL chart
- FTMO rule compliance percentage
- Active trades count

**Dashboard 2: System Health**
- Lambda invocation counts and errors
- RDS CPU and memory utilization
- S3 storage usage and costs
- API latency (Coinbase)
- Error rates by component

**Deliverables**:
- CloudFormation template for dashboards
- Custom CloudWatch metrics from application
- Dashboard screenshots for documentation

#### Task 3.2: CloudWatch Alarms
**Branch**: `aws/cloudwatch-alarms`

Create CloudWatch alarms for critical events:

**Trading Alarms**:
- Daily loss approaching 5% limit (warning at 4%)
- Total loss approaching 10% limit (warning at 8%)
- Win rate drops below 55% (7-day rolling)
- No signals generated in 2 hours (system health)

**System Alarms**:
- Lambda errors > 5% of invocations
- RDS CPU > 80% for 15 minutes
- RDS storage < 20% free space
- API errors > 10 in 5 minutes

**Deliverables**:
- CloudFormation template for alarms
- SNS topic for alarm notifications
- Email/SMS subscription for critical alarms
- Runbook for alarm responses

### Phase 4: Data Pipeline (Priority: LOW)

#### Task 4.1: EventBridge Orchestration
**Branch**: `aws/eventbridge-pipeline`

Create EventBridge rules to orchestrate data pipeline:

**Schedule Rules**:
- Fetch market data every 1 minute (9:30 AM - 4:00 PM ET)
- Generate signals every 5 minutes
- Update risk snapshot every hour
- Backup database daily at 2 AM ET
- Weekly model retraining (Sunday 12 AM ET)

**Event-Driven Rules**:
- S3 upload → Trigger data processing
- High-confidence signal → Trigger notification
- FTMO rule violation → Trigger alert
- Trade execution → Update risk book

**Deliverables**:
- EventBridge rules and schedules
- Event pattern definitions
- Target configurations (Lambda, SNS, etc.)

#### Task 4.2: S3 Data Lake
**Branch**: `aws/s3-data-lake`

Organize S3 as a data lake for analytics:

**Bucket Structure**:
```
crpbot-market-data-prod/
├── raw/
│   └── coinbase/
│       └── YYYY/MM/DD/
│           └── {symbol}_{timestamp}.parquet
├── processed/
│   └── features/
│       └── YYYY/MM/DD/
│           └── {symbol}_features.parquet
└── models/
    └── checkpoints/
        └── {model_type}_{version}.pth
```

**Deliverables**:
- S3 bucket structure and naming convention
- Lambda for data partitioning
- Athena table definitions for SQL queries
- Glue crawler for schema discovery (optional)

### Phase 5: Deployment and CI/CD (Priority: LOW)

#### Task 5.1: AWS CodePipeline
**Branch**: `aws/codepipeline-setup`

Create CI/CD pipeline for automated deployments:

**Pipeline Stages**:
1. Source: GitHub repository
2. Build: Run tests, lint, type check
3. Deploy to Dev: Lambda, update CloudFormation stacks
4. Integration Tests: Run smoke tests against dev
5. Deploy to Prod: Update production Lambda functions

**Deliverables**:
- CodePipeline definition
- CodeBuild buildspec.yml
- Deployment scripts
- Blue/green deployment strategy for Lambda

#### Task 5.2: Infrastructure as Code
**Branch**: `aws/iac-consolidation`

Consolidate all infrastructure into reusable IaC:

**Options**:
- AWS CDK (Python) - Recommended for Python developers
- CloudFormation templates
- Terraform (if preferred)

**Stack Organization**:
- `storage-stack`: S3, RDS, Secrets Manager
- `compute-stack`: Lambda functions, layers
- `monitoring-stack`: CloudWatch, SNS, alarms
- `network-stack`: VPC, security groups, subnets

**Deliverables**:
- Complete IaC codebase
- Deployment instructions
- Multi-environment support (dev/staging/prod)
- Cost estimates for each stack

## 💰 Cost Estimates

### Free Tier Eligible (First 12 Months)
- EC2: 750 hours/month of t2.micro or t3.micro
- RDS: 750 hours/month of db.t2.micro or db.t3.micro
- S3: 5 GB storage
- Lambda: 1M requests + 400,000 GB-seconds compute
- CloudWatch: 10 custom metrics + 10 alarms

### Estimated Monthly Costs (After Free Tier)

**Development Environment**:
- RDS db.t3.micro: $15/month
- S3 storage (10GB): $0.23/month
- Lambda (10K invocations): $0.20/month
- Secrets Manager (3 secrets): $1.20/month
- Data transfer: $5/month
- **Total Dev**: ~$22/month

**Production Environment**:
- RDS db.t3.small: $30/month
- S3 storage (100GB): $2.30/month
- Lambda (100K invocations): $2.00/month
- Secrets Manager (5 secrets): $2.00/month
- CloudWatch alarms (20): $0.20/month
- SNS notifications: $1.00/month
- Data transfer: $20/month
- **Total Prod**: ~$58/month

## 📚 Reference Documents

- [AWS Lambda Python Documentation](https://docs.aws.amazon.com/lambda/latest/dg/lambda-python.html)
- [Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [AWS CDK Python Examples](https://github.com/aws-samples/aws-cdk-examples/tree/master/python)
- [RDS PostgreSQL Best Practices](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/CHAP_BestPractices.html)
- [Lambda Layers](https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html)

## 🔄 Integration with Trading System

### Environment Variables for AWS Integration

```python
# .env (local) - DO NOT commit real values
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=123456789012

# S3
S3_MARKET_DATA_BUCKET=crpbot-market-data-dev
S3_BACKUPS_BUCKET=crpbot-backups-dev
S3_LOGS_BUCKET=crpbot-logs-dev

# RDS
DB_SECRET_ARN=arn:aws:secretsmanager:us-east-1:123456789012:secret:crpbot-db-dev
DB_HOST=crpbot-dev.xxxxxxxxxx.us-east-1.rds.amazonaws.com
DB_NAME=crpbot_dev
DB_PORT=5432

# Secrets Manager
COINBASE_SECRET_ARN=arn:aws:secretsmanager:us-east-1:123456789012:secret:coinbase-api
TELEGRAM_SECRET_ARN=arn:aws:secretsmanager:us-east-1:123456789012:secret:telegram-bot

# Lambda
SIGNAL_PROCESSOR_ARN=arn:aws:lambda:us-east-1:123456789012:function:crpbot-signal-processor
RISK_MONITOR_ARN=arn:aws:lambda:us-east-1:123456789012:function:crpbot-risk-monitor

# SNS
SIGNALS_TOPIC_ARN=arn:aws:sns:us-east-1:123456789012:crpbot-signals
ALERTS_TOPIC_ARN=arn:aws:sns:us-east-1:123456789012:crpbot-alerts
```

### Python Integration Example

```python
# libs/aws/secrets.py
import boto3
import json
from functools import lru_cache

@lru_cache(maxsize=10)
def get_secret(secret_arn: str) -> dict:
    """Retrieve secret from AWS Secrets Manager."""
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_arn)
    return json.loads(response['SecretString'])

# Usage in trading system
from libs.aws.secrets import get_secret
coinbase_creds = get_secret(os.getenv('COINBASE_SECRET_ARN'))
api_key = coinbase_creds['api_key']
```

## ✅ Checklist for Amazon Q

When working on AWS tasks:

- [ ] Work on dedicated `aws/*` branch
- [ ] Document all ARNs in commit messages
- [ ] Include cost estimates for new resources
- [ ] Test in dev environment before production
- [ ] Update this document with actual ARNs after creation
- [ ] Create CloudFormation/CDK template for reproducibility
- [ ] Add IAM policies following least-privilege principle
- [ ] Enable CloudWatch logging for all services
- [ ] Tag all resources with `Project: crpbot` and `Environment: dev/prod`
- [ ] Push changes to GitHub for team visibility

---

**Last Updated**: 2025-11-08
**Maintained By**: Amazon Q (AWS Specialist)
**Contact**: See WORKFLOW_SETUP.md for coordination between AI tools
