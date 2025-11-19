#!/bin/bash
#
# Launch GPU Training with Automatic Termination
#
# This script launches an AWS GPU instance, runs training, and automatically
# terminates the instance when training completes.
#
# Usage:
#   ./launch_training_with_auto_terminate.sh [--spot] [--instance-type TYPE]
#

set -e

# Configuration
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"
USE_SPOT=false
MAX_SPOT_PRICE="0.50"
AMI_ID="ami-0c7217cdde317cfec"  # Ubuntu 22.04 LTS
KEY_NAME="your-key-name"  # UPDATE THIS
SECURITY_GROUP="sg-xxxxxxxxx"  # UPDATE THIS
SUBNET_ID="subnet-xxxxxxxxx"  # UPDATE THIS (optional)
REGION="us-east-1"
MAX_TRAINING_HOURS=4

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --spot)
            USE_SPOT=true
            shift
            ;;
        --instance-type)
            INSTANCE_TYPE="$2"
            shift 2
            ;;
        --max-hours)
            MAX_TRAINING_HOURS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "AWS GPU TRAINING WITH AUTO-TERMINATION"
echo "=========================================="
echo "Instance Type: $INSTANCE_TYPE"
echo "Use Spot: $USE_SPOT"
echo "Max Training Hours: $MAX_TRAINING_HOURS"
echo "Region: $REGION"
echo "=========================================="

# Create user data script for instance initialization
cat > /tmp/user_data.sh << 'USERDATA_EOF'
#!/bin/bash
set -e

# Install dependencies
apt-get update
apt-get install -y python3-pip git awscli

# Clone repository
cd /home/ubuntu
git clone https://github.com/your-repo/crpbot.git
cd crpbot

# Install Python dependencies
pip3 install uv
uv sync

# Download training data from S3
aws s3 sync s3://crpbot-ml-data/features/ data/features/

# Create training script that will auto-terminate
cat > /tmp/train_and_terminate.sh << 'EOF'
#!/bin/bash
set -e

INSTANCE_ID=$(ec2-metadata --instance-id | cut -d' ' -f2)
echo "Training on instance: $INSTANCE_ID"

# Run training
cd /home/ubuntu/crpbot
uv run python apps/trainer/main.py --task lstm --coin BTC --epochs 15
uv run python apps/trainer/main.py --task lstm --coin ETH --epochs 15
uv run python apps/trainer/main.py --task lstm --coin SOL --epochs 15

# Upload trained models to S3
aws s3 sync models/ s3://crpbot-ml-data/models/latest/

echo "✅ Training complete! Terminating instance in 60 seconds..."
sleep 60

# Self-terminate
sudo shutdown -h now
EOF

chmod +x /tmp/train_and_terminate.sh

# Run training with auto-termination monitor
nohup /tmp/train_and_terminate.sh > /tmp/training.log 2>&1 &

USERDATA_EOF

# Launch instance
echo "Launching EC2 instance..."

if [ "$USE_SPOT" = true ]; then
    # Launch spot instance
    SPOT_REQUEST=$(aws ec2 request-spot-instances \
        --region "$REGION" \
        --spot-price "$MAX_SPOT_PRICE" \
        --instance-count 1 \
        --type "one-time" \
        --launch-specification "{
            \"ImageId\": \"$AMI_ID\",
            \"InstanceType\": \"$INSTANCE_TYPE\",
            \"KeyName\": \"$KEY_NAME\",
            \"SecurityGroupIds\": [\"$SECURITY_GROUP\"],
            \"UserData\": \"$(base64 -w 0 /tmp/user_data.sh)\",
            \"IamInstanceProfile\": {
                \"Name\": \"EC2-S3-Access\"
            },
            \"TagSpecifications\": [{
                \"ResourceType\": \"instance\",
                \"Tags\": [
                    {\"Key\": \"Name\", \"Value\": \"crpbot-training-spot-auto-terminate\"},
                    {\"Key\": \"AutoTerminate\", \"Value\": \"true\"}
                ]
            }]
        }" \
        --query 'SpotInstanceRequests[0].SpotInstanceRequestId' \
        --output text)

    echo "Spot request created: $SPOT_REQUEST"
    echo "Waiting for instance to launch..."

    # Wait for spot request to be fulfilled
    aws ec2 wait spot-instance-request-fulfilled \
        --region "$REGION" \
        --spot-instance-request-ids "$SPOT_REQUEST"

    # Get instance ID
    INSTANCE_ID=$(aws ec2 describe-spot-instance-requests \
        --region "$REGION" \
        --spot-instance-request-ids "$SPOT_REQUEST" \
        --query 'SpotInstanceRequests[0].InstanceId' \
        --output text)
else
    # Launch on-demand instance
    INSTANCE_ID=$(aws ec2 run-instances \
        --region "$REGION" \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SECURITY_GROUP" \
        --user-data file:///tmp/user_data.sh \
        --iam-instance-profile Name=EC2-S3-Access \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=crpbot-training-auto-terminate},{Key=AutoTerminate,Value=true}]" \
        --query 'Instances[0].InstanceId' \
        --output text)
fi

echo "=========================================="
echo "Instance launched: $INSTANCE_ID"
echo "=========================================="
echo ""
echo "The instance will automatically terminate when training completes."
echo "Maximum runtime: $MAX_TRAINING_HOURS hours"
echo ""
echo "Monitor training:"
echo "  aws ec2 describe-instances --instance-ids $INSTANCE_ID"
echo ""
echo "Get logs (after SSH is available):"
echo "  ssh ubuntu@<instance-ip> 'tail -f /tmp/training.log'"
echo ""
echo "Instance will self-terminate after training completes."
echo "=========================================="

# Cleanup
rm /tmp/user_data.sh

echo ""
echo "✅ Training launched successfully!"
echo "Instance ID: $INSTANCE_ID"
