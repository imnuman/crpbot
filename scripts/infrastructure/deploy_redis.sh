#!/bin/bash
set -e

# Deploy ElastiCache Redis (t4g.micro) for CryptoBot
# Cost: ~$11/month (single node), ~$22/month (with replica)

STACK_NAME="crpbot-redis"
REGION="us-east-1"
NODE_TYPE="cache.t4g.micro"  # ARM-based Gravit on (cheapest)
ENGINE_VERSION="7.0"  # Latest Redis 7
NUM_CACHE_NODES="1"  # Start with 1, add replica later

echo "=== Deploying ElastiCache Redis ==="
echo "Stack: $STACK_NAME"
echo "Node type: $NODE_TYPE"
echo "Engine: Redis $ENGINE_VERSION"
echo "Nodes: $NUM_CACHE_NODES"
echo "Region: $REGION"
echo ""

# Create CloudFormation template
cat > /tmp/redis-template.yaml <<'EOF'
AWSTemplateFormatVersion: '2010-09-09'
Description: 'ElastiCache Redis for CryptoBot feature caching'

Parameters:
  NodeType:
    Type: String
    Description: Cache node instance type
  EngineVersion:
    Type: String
    Description: Redis engine version
  NumCacheNodes:
    Type: Number
    Description: Number of cache nodes

Resources:
  # Redis subnet group
  CacheSubnetGroup:
    Type: AWS::ElastiCache::SubnetGroup
    Properties:
      Description: Subnet group for CryptoBot Redis
      SubnetIds:
        - !Ref PrivateSubnet1
        - !Ref PrivateSubnet2
      CacheSubnetGroupName: !Sub '${AWS::StackName}-subnet-group'

  # Security group for Redis
  CacheSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for CryptoBot Redis
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 6379
          ToPort: 6379
          SourceSecurityGroupId: !Ref AppSecurityGroup
      Tags:
        - Key: Name
          Value: !Sub '${AWS::StackName}-sg'
        - Key: Project
          Value: CryptoBot

  # Redis cluster
  RedisCluster:
    Type: AWS::ElastiCache::CacheCluster
    Properties:
      CacheClusterIdentifier: !Sub '${AWS::StackName}-cluster'
      Engine: redis
      EngineVersion: !Ref EngineVersion
      CacheNodeType: !Ref NodeType
      NumCacheNodes: !Ref NumCacheNodes
      CacheSubnetGroupName: !Ref CacheSubnetGroup
      VpcSecurityGroupIds:
        - !Ref CacheSecurityGroup

      # Performance and reliability
      PreferredMaintenanceWindow: 'sun:04:00-sun:05:00'  # Sunday 4-5 AM UTC
      SnapshotWindow: '03:00-04:00'  # 3-4 AM UTC
      SnapshotRetentionLimit: 7  # 7 days of snapshots

      # Monitoring
      NotificationTopicArn: !Ref AlertTopic

      Tags:
        - Key: Project
          Value: CryptoBot
        - Key: Environment
          Value: Production
        - Key: CostCenter
          Value: Infrastructure

  # SNS topic for alerts
  AlertTopic:
    Type: AWS::SNS::Topic
    Properties:
      TopicName: !Sub '${AWS::StackName}-alerts'
      DisplayName: CryptoBot Redis Alerts
      Tags:
        - Key: Project
          Value: CryptoBot

  # VPC (simplified - reuse from RDS or create new)
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: !Sub '${AWS::StackName}-vpc'

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
  RedisEndpoint:
    Description: Redis endpoint address
    Value: !GetAtt RedisCluster.RedisEndpoint.Address
    Export:
      Name: !Sub '${AWS::StackName}-endpoint'

  RedisPort:
    Description: Redis port
    Value: !GetAtt RedisCluster.RedisEndpoint.Port
    Export:
      Name: !Sub '${AWS::StackName}-port'

  ConnectionString:
    Description: Redis connection string
    Value: !Sub 'redis://${RedisCluster.RedisEndpoint.Address}:${RedisCluster.RedisEndpoint.Port}'
EOF

# Deploy CloudFormation stack
echo "Deploying CloudFormation stack..."
aws cloudformation create-stack \
    --stack-name "$STACK_NAME" \
    --template-body file:///tmp/redis-template.yaml \
    --parameters \
        ParameterKey=NodeType,ParameterValue="$NODE_TYPE" \
        ParameterKey=EngineVersion,ParameterValue="$ENGINE_VERSION" \
        ParameterKey=NumCacheNodes,ParameterValue="$NUM_CACHE_NODES" \
    --region "$REGION" \
    --tags \
        Key=Project,Value=CryptoBot \
        Key=Environment,Value=Production \
        Key=ManagedBy,Value=CloudFormation

echo ""
echo "Stack creation initiated. This will take 5-10 minutes..."
echo "Monitoring stack creation..."

aws cloudformation wait stack-create-complete \
    --stack-name "$STACK_NAME" \
    --region "$REGION"

echo ""
echo "=== Redis Deployment Complete ==="

# Get outputs
REDIS_ENDPOINT=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`RedisEndpoint`].OutputValue' \
    --output text)

REDIS_PORT=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`RedisPort`].OutputValue' \
    --output text)

# Save connection info
cat > .redis_connection_info <<EOF
REDIS_HOST=$REDIS_ENDPOINT
REDIS_PORT=$REDIS_PORT
REDIS_URL=redis://$REDIS_ENDPOINT:$REDIS_PORT
EOF

chmod 600 .redis_connection_info

echo "Connection details:"
echo "  Endpoint: $REDIS_ENDPOINT"
echo "  Port: $REDIS_PORT"
echo "  URL: redis://$REDIS_ENDPOINT:$REDIS_PORT"
echo ""
echo "Connection info saved to .redis_connection_info"
echo ""
echo "Testing connection..."
if command -v redis-cli &> /dev/null; then
    redis-cli -h "$REDIS_ENDPOINT" -p "$REDIS_PORT" PING
else
    echo "redis-cli not found. Install with: sudo apt-get install redis-tools"
fi
echo ""
echo "Next steps:"
echo "  1. Store connection info in AWS Secrets Manager"
echo "  2. Update application config to use Redis for feature caching"
echo "  3. Implement cache warming logic"
echo "  4. Monitor cache hit rates in Grafana"
