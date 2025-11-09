# Amazon Q - Phase 2 Instructions

## 🎯 Current Status
- **Branch**: `aws/rds-setup` (ready for your work)
- **Phase 1**: ✅ Complete (S3, RDS, Secrets all deployed and tested)
- **Next**: Phase 2 Lambda Functions

## 📋 Required IAM Permissions ✅
User `ncldev` now has:
- ✅ `AmazonRDSFullAccess`
- ✅ `AmazonVPCFullAccess` 
- ✅ `SecretsManagerReadWrite`

## 🚀 Ready to Deploy: Task 2.1 Lambda Signal Processing

### Target Configuration
- **Branch**: `aws/rds-setup`
- **Region**: `us-east-1`
- **Account**: `980104576869`
- **Stack Name**: `crpbot-lambda-signal-dev`

### Lambda Function Requirements
**Function Name**: `crpbot-signal-processor-dev`
**Runtime**: Python 3.11
**Memory**: 512MB
**Timeout**: 30s
**Trigger**: EventBridge schedule (every 5 minutes)

**Function Responsibilities**:
1. Fetch latest market data from Coinbase
2. Run trained models (LSTM, Transformer, RL)
3. Generate trading signals
4. Store signals in RDS
5. Check FTMO rules and rate limits
6. Send high-confidence signals to SNS

### Environment Variables Needed
```bash
# From .env.aws
S3_MARKET_DATA_BUCKET=crpbot-market-data-dev
DB_HOST=crpbot-dev.cyjcoys82evx.us-east-1.rds.amazonaws.com
DB_NAME=postgres
DB_USERNAME=crpbot_admin
DB_PASSWORD=TempPassword123!
COINBASE_SECRET_ARN=arn:aws:secretsmanager:us-east-1:980104576869:secret:crpbot/coinbase-api/dev-dHLD4h
```

### Dependencies for Lambda Layer
```
boto3
psycopg[binary]
pandas
numpy
torch
sqlalchemy
requests
```

### Success Criteria
Please confirm:
- [ ] CloudFormation stack status: `CREATE_COMPLETE`
- [ ] Lambda function can be invoked successfully
- [ ] EventBridge schedule created (every 5 minutes)
- [ ] Lambda can connect to RDS
- [ ] Lambda can access Secrets Manager
- [ ] Lambda can write to S3
- [ ] SNS topic created for signals
- [ ] Cost estimate provided
- [ ] Update `AWS_TASKS_STATUS.md`

### CloudFormation Template Location
Create: `infra/aws/cloudformation/lambda-signal-processing.yaml`

### Expected Outputs
- Lambda Function ARN
- SNS Topic ARN
- EventBridge Rule ARN
- IAM Role ARN
- Monthly cost estimate

## 📚 Reference Materials
- **Existing Infrastructure**: See `docs/AWS_INFRASTRUCTURE_SUMMARY.md`
- **AWS Tasks Plan**: See `docs/AWS_TASKS.md` (Task 2.1)
- **Integration Code**: `libs/aws/s3_client.py`, `libs/aws/secrets.py`

## 🔄 Workflow
1. Create CloudFormation template
2. Deploy stack
3. Test Lambda function
4. Update documentation
5. Commit to `aws/rds-setup` branch
6. Push to GitHub

Ready when you are! 🚀