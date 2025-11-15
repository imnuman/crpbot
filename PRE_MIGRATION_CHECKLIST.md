# Pre-Migration Checklist

Complete this checklist before migrating CRPBot to your cloud server.

## 📋 Pre-Migration Tasks

### 1. Local Machine Preparation

- [ ] **Commit all code changes**
  ```bash
  cd /home/numan/crpbot
  git status
  git add .
  git commit -m "Pre-migration commit"
  git push origin main
  ```

- [ ] **Verify .env file is complete**
  ```bash
  cat .env | grep -E "COINBASE_API_KEY|DB_URL|AWS_ACCESS_KEY_ID"
  ```

- [ ] **Check data and models sizes**
  ```bash
  du -sh data/ models/
  # Expected: ~764MB data, ~3.2MB models
  ```

- [ ] **Test local setup works**
  ```bash
  source .venv/bin/activate
  make test
  python test_runtime_connection.py
  ```

- [ ] **Verify AWS credentials**
  ```bash
  aws sts get-caller-identity
  aws s3 ls s3://crpbot-market-data-dev/
  ```

### 2. Cloud Server Information

- [ ] **Server IP/Hostname**: ___________________________________

- [ ] **SSH Username**: ___________________________________

- [ ] **SSH Connection Method**:
  - [ ] Password
  - [ ] SSH Key (location: ___________________________________)

- [ ] **Test SSH Connection**:
  ```bash
  ssh user@server-ip "echo 'Connection OK'"
  ```

### 3. AWS Account Verification

- [ ] **AWS Access Key ID**: Available (don't write here!)

- [ ] **AWS Secret Access Key**: Available (don't write here!)

- [ ] **AWS Region**: us-east-1 (or your region: _____________)

- [ ] **S3 Bucket Name**: crpbot-market-data-dev

- [ ] **RDS Endpoint**: crpbot-rds-postgres-db.cyjcoys82evx.us-east-1.rds.amazonaws.com

- [ ] **Database Password**: Available in `.db_password` file

### 4. Decision: Data Transfer Method

**Option A: S3 Download** (Recommended)
- [ ] Data already uploaded to S3
- [ ] AWS credentials configured
- [ ] Estimated time: 10 minutes

**Option B: Direct Transfer**
- [ ] Local data directory ready (764MB)
- [ ] Stable internet connection
- [ ] Estimated time: 30+ minutes

**Option C: Manual Later**
- [ ] Will handle separately
- [ ] Acknowledged runtime won't work until data present

### 5. GitHub/GitLab Setup

- [ ] **Repository URL**: ___________________________________

- [ ] **Access Method**:
  - [ ] HTTPS (username/token)
  - [ ] SSH (key configured)

- [ ] **Latest code pushed to main branch**
  ```bash
  git log --oneline -1
  ```

### 6. Required Credentials/Files

Files that need to be transferred (will be handled by script):

- [ ] `.env` file exists and is complete
- [ ] `.db_password` file exists
- [ ] `.rds_connection_info` file exists (if present)
- [ ] AWS credentials available for cloud server

### 7. Cloud Server Requirements

Check your cloud server meets these requirements:

- [ ] **OS**: Ubuntu 20.04+ (preferably 22.04 LTS)
- [ ] **CPU**: 2+ cores
- [ ] **RAM**: 4GB+ (8GB recommended)
- [ ] **Disk**: 20GB+ free space
- [ ] **Network**: Public internet access
- [ ] **Firewall**: Port 22 (SSH) open

### 8. Backup Current State

Create local backups before migration:

- [ ] **Backup configuration**
  ```bash
  tar -czf ~/crpbot-backup-$(date +%Y%m%d).tar.gz \
    ~/crpbot/.env ~/crpbot/.db_password ~/crpbot/.rds_connection_info
  ```

- [ ] **Backup models locally**
  ```bash
  tar -czf ~/crpbot-models-backup.tar.gz ~/crpbot/models/
  ```

- [ ] **Note current git commit**
  ```bash
  cd ~/crpbot
  git log --oneline -1 > ~/crpbot-git-commit.txt
  ```

### 9. Test Database Connection

- [ ] **Test RDS connection from local machine**
  ```bash
  PGPASSWORD="$(cat .db_password)" psql \
    -h crpbot-rds-postgres-db.cyjcoys82evx.us-east-1.rds.amazonaws.com \
    -p 5432 \
    -U crpbot_admin \
    -d crpbot \
    -c "SELECT version();"
  ```

### 10. Review Migration Plan

- [ ] Read `MIGRATION_GUIDE.md` (estimated time: 30-60 minutes)
- [ ] Reviewed `CLOUD_SERVER_QUICKSTART.md`
- [ ] Understand automated vs manual migration options
- [ ] Scheduled migration time (recommended: off-hours)

## 🚀 Ready to Migrate?

Once all items are checked:

### Automated Migration (Recommended)
```bash
cd /home/numan/crpbot
./scripts/deploy_to_cloud.sh user@your-cloud-server-ip
```

### Manual Migration
Follow step-by-step instructions in `MIGRATION_GUIDE.md`

## 📞 Emergency Contacts

**If something goes wrong:**

1. **Backup still on local machine** ✅
2. **Can rollback with git**: `git reset --hard COMMIT_HASH`
3. **S3 data is safe**: Can re-download anytime
4. **RDS database unchanged**: Still accessible from local machine

## Estimated Timeline

| Phase | Time | Description |
|-------|------|-------------|
| Pre-flight checks | 15 min | Complete this checklist |
| Server setup | 10 min | Install packages on cloud server |
| Code deployment | 5 min | Clone repo, transfer config |
| Dependencies | 5 min | Install Python packages |
| Data transfer | 10-30 min | S3 download or direct transfer |
| Testing | 10 min | Verify everything works |
| **Total** | **55-80 min** | Complete migration |

## Post-Migration Verification

After migration, verify these work:

- [ ] SSH access to cloud server
- [ ] Code deployed to `~/crpbot`
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] `.env` file present with correct values
- [ ] Data directory populated
- [ ] Models directory populated
- [ ] AWS credentials configured
- [ ] Database connection works
- [ ] Tests pass (`make test`)
- [ ] Runtime starts in dry-run mode

## Notes Section

Use this space for notes, server IPs, or special configurations:

```
Server IP: _______________________________________

SSH Key Location: _______________________________________

Special Notes:
_________________________________________________
_________________________________________________
_________________________________________________
_________________________________________________
```

---

**Date Prepared**: _______________
**Migration Scheduled**: _______________
**Completed**: _______________

**✅ Ready to migrate when all boxes are checked!**
