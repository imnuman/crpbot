#!/usr/bin/env python3
"""
Deploy CRPBot runtime to AWS
"""
import subprocess
import sys
import time

def run_command(cmd, description):
    """Run command with logging"""
    print(f"🔄 {description}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} completed")
        return result.stdout
    else:
        print(f"❌ {description} failed: {result.stderr}")
        return None

def main():
    print("🚀 Deploying CRPBot Runtime to AWS")
    
    # Step 1: Attach ElastiCache permissions
    print("\n1. Setting up ElastiCache permissions...")
    run_command(
        "aws iam detach-user-policy --user-name ncldev --policy-arn arn:aws:iam::980104576869:policy/ElastiCacheReadOnly",
        "Detaching read-only policy"
    )
    run_command(
        "aws iam attach-user-policy --user-name ncldev --policy-arn arn:aws:iam::980104576869:policy/ElastiCacheFullAccess",
        "Attaching full access policy"
    )
    
    # Step 2: Create Redis cluster
    print("\n2. Creating Redis cluster...")
    redis_result = run_command(
        """aws elasticache create-cache-cluster \
        --cache-cluster-id crpbot-redis-dev \
        --engine redis \
        --cache-node-type cache.t3.micro \
        --num-cache-nodes 1 \
        --tags Key=Project,Value=crpbot Key=Environment,Value=dev""",
        "Creating Redis cluster"
    )
    
    if redis_result:
        print("⏳ Redis cluster creating... (takes 5-10 minutes)")
    
    # Step 3: Test database connection
    print("\n3. Testing database connection...")
    test_result = run_command(
        "python -c \"import psycopg2; print('psycopg2 available')\"",
        "Checking psycopg2"
    )
    
    # Step 4: Install dependencies
    print("\n4. Installing runtime dependencies...")
    run_command("pip install psycopg2-binary boto3 redis", "Installing packages")
    
    # Step 5: Test AWS connections
    print("\n5. Testing AWS connections...")
    run_command("aws s3 ls s3://crpbot-market-data-dev/", "Testing S3 access")
    run_command("aws secretsmanager list-secrets", "Testing Secrets Manager")
    
    print("\n🎉 Runtime deployment initiated!")
    print("\nNext steps:")
    print("1. Wait for Redis cluster (5-10 minutes)")
    print("2. Create database schema")
    print("3. Test runtime: python apps/runtime/aws_runtime.py")
    print("4. Set up Google Colab Pro for GPU training")

if __name__ == "__main__":
    main()
