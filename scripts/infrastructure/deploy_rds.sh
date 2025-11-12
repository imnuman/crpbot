#!/bin/bash
set -e

# Deploy RDS PostgreSQL (t4g.small) for CryptoBot
# Cost: ~$24/month (single-AZ), ~$48/month (multi-AZ)

STACK_NAME="crpbot-rds-postgres"
DB_NAME="crpbot"
DB_USERNAME="crpbot_admin"
REGION="us-east-1"
INSTANCE_CLASS="db.t4g.small"  # ARM-based Graviton (20% cheaper)
ALLOCATED_STORAGE="100"  # GB
MULTI_AZ="false"  # Start with single-AZ, upgrade later

# Generate random password if not exists
if [ ! -f .db_password ]; then
    echo "Generating database password..."
    DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
    echo "$DB_PASSWORD" > .db_password
    chmod 600 .db_password
    echo "Password saved to .db_password (keep this secure!)"
else
    DB_PASSWORD=$(cat .db_password)
    echo "Using existing password from .db_password"
fi

echo "=== Deploying RDS PostgreSQL ==="
echo "Stack: $STACK_NAME"
echo "Instance: $INSTANCE_CLASS"
echo "Database: $DB_NAME"
echo "Storage: ${ALLOCATED_STORAGE}GB"
echo "Multi-AZ: $MULTI_AZ"
echo "Region: $REGION"
echo ""

# Create CloudFormation template
cat > /tmp/rds-template.yaml <<'EOF'
AWSTemplateFormatVersion: '2010-09-09'
Description: 'RDS PostgreSQL database for CryptoBot'

Parameters:
  DBName:
    Type: String
    Description: Database name
  DBUsername:
    Type: String
    Description: Database admin username
  DBPassword:
    Type: String
    NoEcho: true
    Description: Database admin password
  DBInstanceClass:
    Type: String
    Description: Database instance class
  AllocatedStorage:
    Type: Number
    Description: Allocated storage in GB
  MultiAZ:
    Type: String
    AllowedValues: ['true', 'false']
    Description: Enable Multi-AZ deployment

Resources:
  DBSubnetGroup:
    Type: AWS::RDS::DBSubnetGroup
    Properties:
      DBSubnetGroupDescription: Subnet group for CryptoBot RDS
      SubnetIds:
        - !Ref PrivateSubnet1
        - !Ref PrivateSubnet2
      Tags:
        - Key: Project
          Value: CryptoBot
        - Key: Environment
          Value: Production

  DBSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for CryptoBot RDS
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 5432
          ToPort: 5432
          SourceSecurityGroupId: !Ref AppSecurityGroup
      Tags:
        - Key: Project
          Value: CryptoBot
        - Key: Environment
          Value: Production

  DBInstance:
    Type: AWS::RDS::DBInstance
    DeletionPolicy: Snapshot  # Take snapshot on stack deletion
    Properties:
      DBInstanceIdentifier: !Sub '${AWS::StackName}-db'
      DBName: !Ref DBName
      Engine: postgres
      EngineVersion: '16.10'  # Latest stable PostgreSQL 16
      DBInstanceClass: !Ref DBInstanceClass
      AllocatedStorage: !Ref AllocatedStorage
      StorageType: gp3  # SSD, better performance than gp2
      StorageEncrypted: true
      MasterUsername: !Ref DBUsername
      MasterUserPassword: !Ref DBPassword
      MultiAZ: !Ref MultiAZ
      DBSubnetGroupName: !Ref DBSubnetGroup
      VPCSecurityGroups:
        - !Ref DBSecurityGroup
      BackupRetentionPeriod: 7  # 7 days of automated backups
      PreferredBackupWindow: '03:00-04:00'  # 3-4 AM UTC
      PreferredMaintenanceWindow: 'sun:04:00-sun:05:00'  # Sunday 4-5 AM UTC
      EnableCloudwatchLogsExports:
        - postgresql
      DeletionProtection: true  # Prevent accidental deletion
      Tags:
        - Key: Project
          Value: CryptoBot
        - Key: Environment
          Value: Production
        - Key: CostCenter
          Value: Infrastructure

  # VPC and Networking (simplified - will use default VPC)
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: !Sub '${AWS::StackName}-vpc'
        - Key: Project
          Value: CryptoBot

  PrivateSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.1.0/24
      AvailabilityZone: !Select [0, !GetAZs '']
      Tags:
        - Key: Name
          Value: !Sub '${AWS::StackName}-private-subnet-1'

  PrivateSubnet2:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.2.0/24
      AvailabilityZone: !Select [1, !GetAZs '']
      Tags:
        - Key: Name
          Value: !Sub '${AWS::StackName}-private-subnet-2'

  AppSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for application servers
      VpcId: !Ref VPC
      Tags:
        - Key: Name
          Value: !Sub '${AWS::StackName}-app-sg'

Outputs:
  DBEndpoint:
    Description: RDS endpoint
    Value: !GetAtt DBInstance.Endpoint.Address
    Export:
      Name: !Sub '${AWS::StackName}-endpoint'

  DBPort:
    Description: RDS port
    Value: !GetAtt DBInstance.Endpoint.Port
    Export:
      Name: !Sub '${AWS::StackName}-port'

  DBName:
    Description: Database name
    Value: !Ref DBName

  ConnectionString:
    Description: PostgreSQL connection string
    Value: !Sub 'postgresql://${DBUsername}:***@${DBInstance.Endpoint.Address}:${DBInstance.Endpoint.Port}/${DBName}'
EOF

# Deploy CloudFormation stack
echo "Deploying CloudFormation stack..."
aws cloudformation create-stack \
    --stack-name "$STACK_NAME" \
    --template-body file:///tmp/rds-template.yaml \
    --parameters \
        ParameterKey=DBName,ParameterValue="$DB_NAME" \
        ParameterKey=DBUsername,ParameterValue="$DB_USERNAME" \
        ParameterKey=DBPassword,ParameterValue="$DB_PASSWORD" \
        ParameterKey=DBInstanceClass,ParameterValue="$INSTANCE_CLASS" \
        ParameterKey=AllocatedStorage,ParameterValue="$ALLOCATED_STORAGE" \
        ParameterKey=MultiAZ,ParameterValue="$MULTI_AZ" \
    --region "$REGION" \
    --capabilities CAPABILITY_IAM \
    --tags \
        Key=Project,Value=CryptoBot \
        Key=Environment,Value=Production \
        Key=ManagedBy,Value=CloudFormation

echo ""
echo "Stack creation initiated. This will take 10-15 minutes..."
echo "Monitoring stack creation..."

aws cloudformation wait stack-create-complete \
    --stack-name "$STACK_NAME" \
    --region "$REGION"

echo ""
echo "=== RDS Deployment Complete ==="

# Get outputs
DB_ENDPOINT=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`DBEndpoint`].OutputValue' \
    --output text)

DB_PORT=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`DBPort`].OutputValue' \
    --output text)

# Save connection info
cat > .rds_connection_info <<EOF
DB_HOST=$DB_ENDPOINT
DB_PORT=$DB_PORT
DB_NAME=$DB_NAME
DB_USER=$DB_USERNAME
DB_PASSWORD=$DB_PASSWORD
EOF

chmod 600 .rds_connection_info

echo "Connection details:"
echo "  Endpoint: $DB_ENDPOINT"
echo "  Port: $DB_PORT"
echo "  Database: $DB_NAME"
echo "  Username: $DB_USERNAME"
echo "  Password: (saved in .db_password)"
echo ""
echo "Connection info saved to .rds_connection_info"
echo ""
echo "Connection string:"
echo "  postgresql://$DB_USERNAME:***@$DB_ENDPOINT:$DB_PORT/$DB_NAME"
echo ""
echo "Next steps:"
echo "  1. Store password in AWS Secrets Manager"
echo "  2. Create database schema (run scripts/infrastructure/create_db_schema.sql)"
echo "  3. Update application config to use RDS"
