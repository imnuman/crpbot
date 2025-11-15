# 🚀 AWS Setup Checklist

**Created**: 2025-11-15 13:45 EST (Toronto)
**Last Updated**: 2025-11-15 13:20 EST (Toronto)
**Author**: QC Claude
**Status**: Ready for Execution
**Purpose**: Set up core AWS infrastructure for V5 deployment

**Agent**: Amazon Q (local machine)
**Timeline**: Complete 2025-11-15
**Budget**: Estimated ~$50-100/month

---

## 📋 Priority 1: Storage Infrastructure (S3)

### Task 1.1: Create S3 Bucket for Data/Models
**Estimated Time**: 10 minutes

**Amazon Q Commands:**
```bash
# Create S3 bucket
q "Create an S3 bucket named 'crpbot-data-v5' in us-east-1 region"

# Or with specific settings
q "Create S3 bucket 'crpbot-data-v5' with versioning enabled and server-side encryption"
```

**What to create:**
- Bucket name: `crpbot-data-v5`
- Region: `us-east-1` (same as your location)
- Versioning: Enabled
- Encryption: AES-256 (default)
- Block public access: Yes (all blocked)

**Expected Result:**
- ✅ Bucket created
- ✅ Bucket URL: `s3://crpbot-data-v5`

---

### Task 1.2: Set Up S3 Folder Structure
**Estimated Time**: 5 minutes

**Amazon Q Commands:**
```bash
q "Create folders in S3 bucket 'crpbot-data-v5' for: raw-data, features, models, backups"
```

**Folder Structure:**
```
s3://crpbot-data-v5/
├── raw-data/
│   ├── BTC-USD/
│   ├── ETH-USD/
│   └── SOL-USD/
├── features/
│   ├── BTC-USD/
│   ├── ETH-USD/
│   └── SOL-USD/
├── models/
│   ├── lstm/
│   ├── transformer/
│   └── promoted/
└── backups/
    └── daily/
```

---

### Task 1.3: Configure S3 Lifecycle Rules
**Estimated Time**: 10 minutes

**Amazon Q Commands:**
```bash
q "Set up lifecycle policy for S3 bucket 'crpbot-data-v5' to move objects to Glacier after 90 days"
```

**Lifecycle Rules:**
1. **Raw data**: Move to Glacier after 90 days
2. **Features**: Keep in Standard for 30 days, then Glacier
3. **Models**: Keep promoted models in Standard, archive old to Glacier after 60 days
4. **Backups**: Delete after 30 days

**Expected Result:**
- ✅ Cost savings: ~70% after 90 days

---

### Task 1.4: Test S3 Upload/Download
**Estimated Time**: 10 minutes

**Amazon Q Commands:**
```bash
# Upload test file
q "Upload file ./test.txt to s3://crpbot-data-v5/test/"

# Download test file
q "Download s3://crpbot-data-v5/test/test.txt to local /tmp/"

# List bucket contents
q "List all files in S3 bucket 'crpbot-data-v5'"
```

**Expected Result:**
- ✅ Upload works
- ✅ Download works
- ✅ Permissions verified

---

## 📋 Priority 2: Secrets Management

### Task 2.1: Set Up AWS Secrets Manager
**Estimated Time**: 15 minutes

**Amazon Q Commands:**
```bash
q "Create secret in AWS Secrets Manager named 'crpbot/api-keys' with following keys: COINBASE_API_KEY, COINBASE_PRIVATE_KEY, TARDIS_API_KEY"
```

**Secrets to Store:**
1. **crpbot/coinbase**
   - `COINBASE_API_KEY`: (from .env)
   - `COINBASE_PRIVATE_KEY`: (from .env)

2. **crpbot/tardis** (when subscribed)
   - `TARDIS_API_KEY`: (from Tardis.dev)

3. **crpbot/database**
   - `DB_PASSWORD`: (auto-generated for RDS)
   - `DB_URL`: (RDS connection string)

**Amazon Q Commands:**
```bash
# Create Coinbase secret
q "Create secret 'crpbot/coinbase' in Secrets Manager with JSON: {\"api_key\": \"your-key\", \"private_key\": \"your-private-key\"}"

# Retrieve secret (test)
q "Retrieve secret 'crpbot/coinbase' from Secrets Manager"
```

**Expected Result:**
- ✅ Secrets stored securely
- ✅ Can retrieve secrets
- ✅ No plaintext credentials in code

---

## 📋 Priority 3: Database (RDS PostgreSQL)

### Task 3.1: Deploy RDS PostgreSQL Instance
**Estimated Time**: 20 minutes (+ 10 min provisioning)

**Amazon Q Commands:**
```bash
q "Create RDS PostgreSQL database instance with following specs:
- Instance name: crpbot-db-v5
- Engine: PostgreSQL 15
- Instance class: db.t3.micro (free tier eligible)
- Storage: 20 GB SSD
- Multi-AZ: No (save cost)
- Public access: Yes (for local connection)
- Database name: crpbot
- Master username: crpbot_admin
- Auto-generate password and store in Secrets Manager"
```

**Expected Configuration:**
- Instance: `crpbot-db-v5`
- Engine: PostgreSQL 15.x
- Class: `db.t3.micro` (free tier - save money)
- Storage: 20 GB gp3
- Backup: 7 days retention
- Multi-AZ: No (Phase 1 - save $$$)
- Public: Yes (for dev access)

**Cost:** ~$15-20/month (free tier: $0 for 12 months)

**Expected Result:**
- ✅ RDS instance created
- ✅ Endpoint: `crpbot-db-v5.xxxxxx.us-east-1.rds.amazonaws.com`
- ✅ Password stored in Secrets Manager

---

### Task 3.2: Configure RDS Security Group
**Estimated Time**: 10 minutes

**Amazon Q Commands:**
```bash
q "Configure security group for RDS instance 'crpbot-db-v5' to allow PostgreSQL access from my current IP address"

# Get your IP first
q "What is my current public IP address?"
```

**Security Rules:**
1. **Inbound**: PostgreSQL (5432) from your IP
2. **Outbound**: All traffic

**Expected Result:**
- ✅ Can connect from local machine
- ✅ Secure (only your IP allowed)

---

### Task 3.3: Test RDS Connection
**Estimated Time**: 10 minutes

**Amazon Q Commands:**
```bash
# Get connection details
q "Get connection endpoint and credentials for RDS instance 'crpbot-db-v5'"

# Test connection
q "Test PostgreSQL connection to RDS instance 'crpbot-db-v5'"
```

**Manual Test (if needed):**
```bash
# Amazon Q will provide the connection string, then test with:
psql -h <endpoint> -p 5432 -U crpbot_admin -d crpbot
```

**Expected Result:**
- ✅ Connection successful
- ✅ Can run SQL queries
- ✅ Connection string saved to `.env`

---

### Task 3.4: Create Database Schema
**Estimated Time**: 10 minutes

**Amazon Q Commands:**
```bash
q "Execute SQL file 'scripts/infrastructure/create_db_schema.sql' on RDS instance 'crpbot-db-v5'"
```

**Or if file doesn't exist, create schema:**
```sql
-- Signals table
CREATE TABLE signals (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    confidence FLOAT NOT NULL,
    tier VARCHAR(10) NOT NULL,
    entry_price FLOAT,
    lstm_prediction FLOAT,
    transformer_prediction FLOAT,
    rl_prediction FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Risk book table
CREATE TABLE risk_book_snapshots (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    balance FLOAT NOT NULL,
    daily_pnl FLOAT NOT NULL,
    total_pnl FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_signals_timestamp ON signals(timestamp);
CREATE INDEX idx_signals_symbol ON signals(symbol);
CREATE INDEX idx_risk_timestamp ON risk_book_snapshots(timestamp);
```

**Expected Result:**
- ✅ Tables created
- ✅ Indexes created
- ✅ Ready for production data

---

## 📋 Priority 4: Monitoring (CloudWatch)

### Task 4.1: Set Up CloudWatch Alarms
**Estimated Time**: 15 minutes

**Amazon Q Commands:**
```bash
# S3 monitoring
q "Create CloudWatch alarm for S3 bucket 'crpbot-data-v5' to alert when storage exceeds 100 GB"

# RDS monitoring
q "Create CloudWatch alarms for RDS instance 'crpbot-db-v5' for:
- CPU utilization > 80%
- Free storage < 2 GB
- Database connections > 50"
```

**Alarms to Create:**
1. **S3 Storage Alert**: > 100 GB
2. **RDS CPU**: > 80% for 5 minutes
3. **RDS Storage**: < 2 GB free
4. **RDS Connections**: > 50 concurrent

**Notification:**
- Email: Your email address
- SNS topic: `crpbot-alerts`

**Expected Result:**
- ✅ 4 alarms active
- ✅ Email notifications configured
- ✅ Dashboard created

---

### Task 4.2: Create CloudWatch Dashboard
**Estimated Time**: 10 minutes

**Amazon Q Commands:**
```bash
q "Create CloudWatch dashboard named 'CRPBot-V5' with widgets for:
- S3 bucket size and request metrics
- RDS CPU, memory, and connections
- RDS read/write latency
- Cost tracking"
```

**Expected Result:**
- ✅ Dashboard URL saved
- ✅ All metrics visible
- ✅ Can monitor at a glance

---

## 📋 Priority 5: Cost Management

### Task 5.1: Set Up Cost Alerts
**Estimated Time**: 10 minutes

**Amazon Q Commands:**
```bash
q "Create billing alert for AWS account to notify when estimated charges exceed $50/month"

q "Create budget named 'CRPBot-V5-Monthly' with limit $100/month and alert at 80% threshold"
```

**Budget Settings:**
- Monthly budget: $100
- Alerts: 50%, 80%, 100%
- Email notification

**Expected Result:**
- ✅ Won't accidentally overspend
- ✅ Email alerts configured

---

### Task 5.2: Enable Cost Explorer
**Estimated Time**: 5 minutes

**Amazon Q Commands:**
```bash
q "Enable AWS Cost Explorer and show me current month-to-date costs by service"
```

**Expected Result:**
- ✅ Cost visibility
- ✅ Can track S3, RDS, Data Transfer costs

---

## 📋 Priority 6: Documentation & Testing

### Task 6.1: Save All Connection Details
**Estimated Time**: 10 minutes

**Create `.aws_resources` file:**
```bash
# S3
export S3_BUCKET_DATA="s3://crpbot-data-v5"
export S3_REGION="us-east-1"

# RDS
export RDS_ENDPOINT="crpbot-db-v5.xxxxxx.us-east-1.rds.amazonaws.com"
export RDS_PORT="5432"
export RDS_DATABASE="crpbot"
export RDS_USERNAME="crpbot_admin"
# Password stored in Secrets Manager: crpbot/database

# Secrets Manager
export SECRETS_REGION="us-east-1"
export SECRET_COINBASE="crpbot/coinbase"
export SECRET_TARDIS="crpbot/tardis"
export SECRET_DATABASE="crpbot/database"

# CloudWatch
export CLOUDWATCH_DASHBOARD="CRPBot-V5"
export SNS_TOPIC_ALERTS="crpbot-alerts"
```

**Amazon Q Command:**
```bash
q "Save RDS endpoint, S3 bucket name, and Secrets Manager ARNs to a file"
```

**Expected Result:**
- ✅ All connection info documented
- ✅ Can source file for scripts

---

### Task 6.2: Test End-to-End AWS Integration
**Estimated Time**: 15 minutes

**Test Script:**
```python
# test_aws_integration.py
import boto3
import psycopg2
from botocore.exceptions import ClientError

def test_s3():
    s3 = boto3.client('s3')
    s3.put_object(Bucket='crpbot-data-v5', Key='test/hello.txt', Body=b'Hello AWS!')
    print("✅ S3 upload works")

def test_secrets():
    secrets = boto3.client('secretsmanager', region_name='us-east-1')
    secret = secrets.get_secret_value(SecretId='crpbot/coinbase')
    print("✅ Secrets Manager works")

def test_rds():
    # Connection string from Secrets Manager
    conn = psycopg2.connect(
        host='<RDS_ENDPOINT>',
        port=5432,
        database='crpbot',
        user='crpbot_admin',
        password='<from-secrets-manager>'
    )
    cur = conn.cursor()
    cur.execute('SELECT version()')
    print(f"✅ RDS works: {cur.fetchone()}")

if __name__ == '__main__':
    test_s3()
    test_secrets()
    test_rds()
    print("\n🎉 All AWS services working!")
```

**Amazon Q Command:**
```bash
q "Run Python script test_aws_integration.py to verify S3, Secrets Manager, and RDS connections"
```

**Expected Result:**
- ✅ All 3 services working
- ✅ No errors
- ✅ Ready for production

---

## 📊 Completion Checklist

### S3 Storage
- [ ] Bucket created: `crpbot-data-v5`
- [ ] Folder structure created
- [ ] Lifecycle policies configured
- [ ] Upload/download tested

### Secrets Manager
- [ ] Coinbase credentials stored
- [ ] Tardis credentials ready (when subscribed)
- [ ] Database password stored
- [ ] Retrieval tested

### RDS PostgreSQL
- [ ] Instance created: `crpbot-db-v5`
- [ ] Security group configured
- [ ] Connection tested
- [ ] Schema created

### CloudWatch
- [ ] 4 alarms created
- [ ] Dashboard created
- [ ] Email notifications working

### Cost Management
- [ ] Billing alerts configured
- [ ] Budget created ($100/month)
- [ ] Cost Explorer enabled

### Documentation
- [ ] `.aws_resources` file created
- [ ] Connection strings saved
- [ ] End-to-end test passed

---

## 💰 Expected Monthly Costs

| Service | Usage | Cost |
|---------|-------|------|
| **S3** | 50 GB storage, 10K requests | ~$2 |
| **RDS** | db.t3.micro, 20 GB | ~$15-20 (free tier: $0) |
| **Data Transfer** | 10 GB/month | ~$1 |
| **Secrets Manager** | 3 secrets | ~$1.20 |
| **CloudWatch** | 10 alarms, 1 dashboard | ~$2 |
| **Total** | | **~$20-25/month** |

*(Free tier covers most costs for first 12 months)*

---

## 🚀 Quick Start Command Sequence

**Copy and paste these to Amazon Q (local machine):**

```bash
# 1. Create S3 bucket
q "Create S3 bucket 'crpbot-data-v5' in us-east-1 with versioning and encryption enabled"

# 2. Set up folder structure
q "Create folders in 'crpbot-data-v5': raw-data, features, models, backups"

# 3. Create secrets
q "Create secret 'crpbot/coinbase' in Secrets Manager"

# 4. Deploy RDS
q "Create PostgreSQL RDS instance 'crpbot-db-v5', db.t3.micro, 20GB storage, auto-generate password"

# 5. Configure security
q "Allow PostgreSQL access to 'crpbot-db-v5' from my IP"

# 6. Test connection
q "Test connection to RDS instance 'crpbot-db-v5'"

# 7. Set up monitoring
q "Create CloudWatch alarms for S3 storage and RDS metrics"

# 8. Cost alerts
q "Create billing alert for $50/month threshold"
```

---

## ⏱️ Estimated Total Time

- **S3 Setup**: 35 minutes
- **Secrets Manager**: 15 minutes
- **RDS Setup**: 50 minutes (includes provisioning wait)
- **CloudWatch**: 25 minutes
- **Cost Management**: 15 minutes
- **Documentation & Testing**: 25 minutes

**Total**: ~2.5 hours

---

## 📝 Notes

- Use Amazon Q for ALL AWS operations (it's faster and safer)
- Save all outputs to `.aws_resources` file
- Test each service before moving to next
- Take screenshots of dashboards for documentation
- Don't forget to update `.env` with new connection strings

---

**Ready to start?** Let's begin with Priority 1 (S3 Setup)! 🚀
