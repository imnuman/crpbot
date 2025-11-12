#!/bin/bash
# Setup GPU training on AWS with multi-GPU support
# Instance: p3.8xlarge (4x V100 GPUs)
# Cost: $12.24/hour, ~3 minutes total = $0.61

set -e

# Configuration
INSTANCE_TYPE="p3.8xlarge"  # 4x V100 GPUs
AMI_ID="ami-0c55b159cbfafe1f0"  # Deep Learning AMI (Ubuntu 20.04)
KEY_NAME="crpbot-training"
REGION="us-east-1"
INSTANCE_NAME="crpbot-gpu-training"

# Get bucket name
if [ -f .s3_bucket_name ]; then
  BUCKET_NAME=$(cat .s3_bucket_name)
else
  echo "Error: Bucket name not found. Run setup_s3_storage.sh first."
  exit 1
fi

echo "=== GPU Training Setup ==="
echo "Instance: ${INSTANCE_TYPE}"
echo "Region: ${REGION}"
echo "Bucket: s3://${BUCKET_NAME}"
echo ""

# Check if key pair exists
echo "[1/7] Checking SSH key pair..."
if aws ec2 describe-key-pairs --key-names "${KEY_NAME}" --region "${REGION}" 2>/dev/null; then
  echo "✅ Key pair '${KEY_NAME}' exists"
else
  echo "Creating new key pair..."
  aws ec2 create-key-pair \
    --key-name "${KEY_NAME}" \
    --region "${REGION}" \
    --query 'KeyMaterial' \
    --output text > ~/.ssh/${KEY_NAME}.pem
  chmod 400 ~/.ssh/${KEY_NAME}.pem
  echo "✅ Key pair created: ~/.ssh/${KEY_NAME}.pem"
fi

# Create security group
echo "[2/7] Setting up security group..."
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=${INSTANCE_NAME}-sg" \
  --region "${REGION}" \
  --query 'SecurityGroups[0].GroupId' \
  --output text 2>/dev/null)

if [ "$SG_ID" == "None" ] || [ -z "$SG_ID" ]; then
  echo "Creating security group..."
  SG_ID=$(aws ec2 create-security-group \
    --group-name "${INSTANCE_NAME}-sg" \
    --description "Security group for CryptoBot GPU training" \
    --region "${REGION}" \
    --query 'GroupId' \
    --output text)

  # Allow SSH from your IP
  MY_IP=$(curl -s https://checkip.amazonaws.com)
  aws ec2 authorize-security-group-ingress \
    --group-id "${SG_ID}" \
    --protocol tcp \
    --port 22 \
    --cidr "${MY_IP}/32" \
    --region "${REGION}"

  echo "✅ Security group created: ${SG_ID}"
else
  echo "✅ Security group exists: ${SG_ID}"
fi

# Create IAM role for S3 access
echo "[3/7] Setting up IAM role for S3 access..."
ROLE_NAME="${INSTANCE_NAME}-role"

# Check if role exists
if aws iam get-role --role-name "${ROLE_NAME}" 2>/dev/null; then
  echo "✅ IAM role exists: ${ROLE_NAME}"
else
  echo "Creating IAM role..."

  # Create trust policy
  cat > /tmp/trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

  # Create role
  aws iam create-role \
    --role-name "${ROLE_NAME}" \
    --assume-role-policy-document file:///tmp/trust-policy.json

  # Attach S3 access policy
  aws iam attach-role-policy \
    --role-name "${ROLE_NAME}" \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

  # Create instance profile
  aws iam create-instance-profile \
    --instance-profile-name "${ROLE_NAME}-profile"

  # Add role to instance profile
  aws iam add-role-to-instance-profile \
    --instance-profile-name "${ROLE_NAME}-profile" \
    --role-name "${ROLE_NAME}"

  sleep 10  # Wait for IAM propagation
  echo "✅ IAM role created: ${ROLE_NAME}"
fi

# Create user data script for instance initialization
echo "[4/7] Preparing instance initialization script..."
cat > /tmp/user-data.sh << EOF
#!/bin/bash
set -e

# Update and install dependencies
apt-get update
apt-get install -y python3-pip git htop nvtop

# Clone repository
cd /home/ubuntu
git clone https://github.com/your-repo/crpbot.git || true
cd crpbot

# Install Python dependencies
pip3 install -r requirements.txt

# Configure AWS CLI
aws configure set region ${REGION}

# Download data from S3
echo "Downloading data from S3..."
aws s3 sync s3://${BUCKET_NAME}/features/ data/features/ --quiet

echo "✅ Instance ready for training!"
EOF

# Check spot pricing
echo "[5/7] Checking spot instance pricing..."
SPOT_PRICE=$(aws ec2 describe-spot-price-history \
  --instance-types "${INSTANCE_TYPE}" \
  --region "${REGION}" \
  --start-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --product-descriptions "Linux/UNIX" \
  --query 'SpotPriceHistory[0].SpotPrice' \
  --output text)

ON_DEMAND_PRICE="12.24"
echo "On-demand price: \$${ON_DEMAND_PRICE}/hour"
echo "Current spot price: \$${SPOT_PRICE}/hour"
echo "Potential savings: $(echo "scale=0; (${ON_DEMAND_PRICE} - ${SPOT_PRICE}) / ${ON_DEMAND_PRICE} * 100" | bc)%"
echo ""

# Ask user preference
echo "Would you like to use spot instances? (50-70% cheaper, may be interrupted)"
read -p "Use spot instances? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  USE_SPOT=true
  echo "Using spot instances (cheaper but may be interrupted)"
else
  USE_SPOT=false
  echo "Using on-demand instances (guaranteed availability)"
fi

echo ""
echo "[6/7] Ready to launch instance!"
echo ""
echo "Configuration:"
echo "  Instance type: ${INSTANCE_TYPE} (4x V100 GPUs)"
echo "  Instance mode: $([ "$USE_SPOT" = true ] && echo "Spot" || echo "On-demand")"
echo "  Estimated cost: \$0.61 for 3 minutes"
echo "  Region: ${REGION}"
echo "  Bucket: s3://${BUCKET_NAME}"
echo ""

read -p "Launch instance now? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Aborted. Run this script again when ready."
  exit 0
fi

echo "[7/7] Launching instance..."

# Launch instance
if [ "$USE_SPOT" = true ]; then
  # Launch spot instance
  INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "${AMI_ID}" \
    --instance-type "${INSTANCE_TYPE}" \
    --key-name "${KEY_NAME}" \
    --security-group-ids "${SG_ID}" \
    --iam-instance-profile Name="${ROLE_NAME}-profile" \
    --instance-market-options 'MarketType=spot,SpotOptions={MaxPrice='${ON_DEMAND_PRICE}',SpotInstanceType=one-time}' \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --user-data file:///tmp/user-data.sh \
    --region "${REGION}" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME}},{Key=Project,Value=CryptoBot}]" \
    --query 'Instances[0].InstanceId' \
    --output text)
else
  # Launch on-demand instance
  INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "${AMI_ID}" \
    --instance-type "${INSTANCE_TYPE}" \
    --key-name "${KEY_NAME}" \
    --security-group-ids "${SG_ID}" \
    --iam-instance-profile Name="${ROLE_NAME}-profile" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --user-data file:///tmp/user-data.sh \
    --region "${REGION}" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME}},{Key=Project,Value=CryptoBot}]" \
    --query 'Instances[0].InstanceId' \
    --output text)
fi

echo "✅ Instance launched: ${INSTANCE_ID}"
echo ""
echo "Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids "${INSTANCE_ID}" --region "${REGION}"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids "${INSTANCE_ID}" \
  --region "${REGION}" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo ""
echo "✅ Instance ready!"
echo ""
echo "Instance ID: ${INSTANCE_ID}"
echo "Public IP: ${PUBLIC_IP}"
echo "SSH command: ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo ""
echo "Next steps:"
echo "  1. SSH into instance (wait 2-3 minutes for initialization)"
echo "  2. Run training: cd crpbot && ./scripts/train_multi_gpu.sh"
echo "  3. Monitor: watch nvidia-smi"
echo "  4. After training, run: ./scripts/download_models.sh"
echo "  5. Terminate instance: aws ec2 terminate-instances --instance-ids ${INSTANCE_ID}"
echo ""

# Save instance info
cat > .gpu_instance_info << EOF
INSTANCE_ID=${INSTANCE_ID}
PUBLIC_IP=${PUBLIC_IP}
KEY_NAME=${KEY_NAME}
BUCKET_NAME=${BUCKET_NAME}
LAUNCHED_AT=$(date)
EOF

echo "Instance info saved to .gpu_instance_info"
