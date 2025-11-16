#!/bin/bash
# Launch AWS g5.xlarge GPU instance for model training
# Instance: g5.xlarge (NVIDIA A10G, 24GB VRAM)
# Cost: ~$0.30/hour spot, ~$1.01/hour on-demand

set -e

echo "🚀 Launching AWS g5.xlarge GPU Training Instance"
echo "================================================"

# Configuration
INSTANCE_TYPE="g5.xlarge"
AMI_ID="ami-0c94855ba95c574c8"  # Deep Learning AMI (PyTorch)
KEY_NAME="crpbot-gpu"
SECURITY_GROUP="crpbot-training-sg"
SUBNET_ID=""  # Will auto-select
REGION="us-east-1"

# Check if key pair exists
if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" &>/dev/null; then
    echo "Creating SSH key pair..."
    aws ec2 create-key-pair \
        --key-name "$KEY_NAME" \
        --region "$REGION" \
        --query 'KeyMaterial' \
        --output text > ~/.ssh/${KEY_NAME}.pem
    chmod 400 ~/.ssh/${KEY_NAME}.pem
    echo "✅ Key saved to ~/.ssh/${KEY_NAME}.pem"
fi

# Check if security group exists
if ! aws ec2 describe-security-groups --group-names "$SECURITY_GROUP" --region "$REGION" &>/dev/null; then
    echo "Creating security group..."
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SECURITY_GROUP" \
        --description "CRPBot GPU training security group" \
        --region "$REGION" \
        --output text)

    # Allow SSH from anywhere (restrict in production)
    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 \
        --region "$REGION"

    echo "✅ Security group created: $SG_ID"
else
    SG_ID=$(aws ec2 describe-security-groups \
        --group-names "$SECURITY_GROUP" \
        --region "$REGION" \
        --query 'SecurityGroups[0].GroupId' \
        --output text)
    echo "✅ Using existing security group: $SG_ID"
fi

# Launch spot instance (70% cheaper)
echo ""
echo "Launching g5.xlarge spot instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=crpbot-gpu-training},{Key=Project,Value=crpbot}]" \
    --region "$REGION" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "✅ Instance launched: $INSTANCE_ID"
echo ""

# Wait for instance to be running
echo "Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
echo "✅ Instance is running"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo "================================================"
echo "✅ GPU Instance Ready!"
echo "================================================"
echo ""
echo "Instance ID: $INSTANCE_ID"
echo "Public IP:   $PUBLIC_IP"
echo "GPU:         NVIDIA A10G (24GB VRAM)"
echo "Cost:        ~$0.30/hour (spot)"
echo ""
echo "SSH Command:"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo ""
echo "Wait 2-3 minutes for initialization, then SSH and run:"
echo "  git clone https://github.com/imnuman/crpbot.git"
echo "  cd crpbot"
echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
echo "  source ~/.bashrc"
echo "  uv pip install -e ."
echo ""
echo "⚠️  REMEMBER TO TERMINATE INSTANCE WHEN DONE:"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
echo ""

# Save instance info
cat > .gpu_instance_info << EOF
INSTANCE_ID=$INSTANCE_ID
PUBLIC_IP=$PUBLIC_IP
KEY_FILE=~/.ssh/${KEY_NAME}.pem
REGION=$REGION
LAUNCHED=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
EOF

echo "Instance info saved to .gpu_instance_info"
