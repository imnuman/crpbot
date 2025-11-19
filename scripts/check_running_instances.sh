#!/bin/bash
#
# Check Running AWS Instances
#
# Monitors all running EC2 instances and alerts if any have been running too long
#

set -e

REGION="${AWS_REGION:-us-east-1}"
WARNING_HOURS=2
CRITICAL_HOURS=6

echo "=========================================="
echo "AWS EC2 INSTANCES MONITOR"
echo "=========================================="
echo "Region: $REGION"
echo "Warning threshold: ${WARNING_HOURS}h"
echo "Critical threshold: ${CRITICAL_HOURS}h"
echo "=========================================="
echo ""

# Get all running instances
RUNNING_INSTANCES=$(aws ec2 describe-instances \
    --region "$REGION" \
    --filters "Name=instance-state-name,Values=running" \
    --query 'Reservations[*].Instances[*].[InstanceId,InstanceType,LaunchTime,Tags[?Key==`Name`].Value|[0],InstanceLifecycle]' \
    --output text)

if [ -z "$RUNNING_INSTANCES" ]; then
    echo "✅ No running instances found"
    exit 0
fi

CURRENT_TIME=$(date +%s)
ALERT_COUNT=0

echo "RUNNING INSTANCES:"
echo ""

while IFS=$'\t' read -r INSTANCE_ID INSTANCE_TYPE LAUNCH_TIME NAME LIFECYCLE; do
    # Parse launch time
    LAUNCH_EPOCH=$(date -d "$LAUNCH_TIME" +%s 2>/dev/null || echo "0")
    RUNNING_SECONDS=$((CURRENT_TIME - LAUNCH_EPOCH))
    RUNNING_HOURS=$((RUNNING_SECONDS / 3600))
    RUNNING_MINS=$(((RUNNING_SECONDS % 3600) / 60))

    # Determine pricing
    if [ "$LIFECYCLE" = "spot" ]; then
        PRICING="SPOT"
    else
        PRICING="ON-DEMAND"
    fi

    # Estimate cost (rough approximation)
    case $INSTANCE_TYPE in
        g5.xlarge)
            HOURLY_RATE=1.006
            ;;
        g4dn.xlarge)
            HOURLY_RATE=0.526
            ;;
        p3.2xlarge)
            HOURLY_RATE=3.06
            ;;
        *)
            HOURLY_RATE=0.50
            ;;
    esac

    if [ "$PRICING" = "SPOT" ]; then
        HOURLY_RATE=$(echo "$HOURLY_RATE * 0.3" | bc)
    fi

    ESTIMATED_COST=$(echo "$HOURLY_RATE * $RUNNING_HOURS" | bc)

    # Status indicator
    if [ $RUNNING_HOURS -ge $CRITICAL_HOURS ]; then
        STATUS="🔴 CRITICAL"
        ALERT_COUNT=$((ALERT_COUNT + 1))
    elif [ $RUNNING_HOURS -ge $WARNING_HOURS ]; then
        STATUS="⚠️  WARNING"
        ALERT_COUNT=$((ALERT_COUNT + 1))
    else
        STATUS="✅ OK"
    fi

    echo "$STATUS | $INSTANCE_ID | $INSTANCE_TYPE ($PRICING)"
    echo "       Name: ${NAME:-<no name>}"
    echo "       Running: ${RUNNING_HOURS}h ${RUNNING_MINS}m"
    echo "       Est. Cost: \$${ESTIMATED_COST}"
    echo ""

done <<< "$RUNNING_INSTANCES"

echo "=========================================="
if [ $ALERT_COUNT -gt 0 ]; then
    echo "⚠️  $ALERT_COUNT instance(s) need attention!"
    echo ""
    echo "To terminate an instance:"
    echo "  aws ec2 terminate-instances --instance-ids <instance-id>"
    echo ""
    echo "To terminate all long-running instances:"
    echo "  aws ec2 describe-instances --filters \"Name=instance-state-name,Values=running\" --query 'Reservations[*].Instances[?LaunchTime<=\`$(date -u -d '6 hours ago' --iso-8601=seconds)\`].InstanceId' --output text | xargs aws ec2 terminate-instances --instance-ids"
else
    echo "✅ All instances within acceptable runtime"
fi
echo "=========================================="

exit $ALERT_COUNT
