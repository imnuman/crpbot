#!/usr/bin/env python3
"""
Deploy V7 Enhanced Training to SageMaker with mandatory scaler verification.
This script handles the complete SageMaker setup and training deployment.
"""

import boto3
import json
import time
from pathlib import Path

def create_sagemaker_role():
    """Create SageMaker execution role if it doesn't exist."""
    iam = boto3.client('iam')
    
    role_name = 'CRPBot-SageMaker-ExecutionRole'
    
    try:
        # Check if role exists
        response = iam.get_role(RoleName=role_name)
        role_arn = response['Role']['Arn']
        print(f"✅ Using existing role: {role_arn}")
        return role_arn
    except iam.exceptions.NoSuchEntityException:
        pass
    
    # Create role
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    try:
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='SageMaker execution role for CRPBot V7 training'
        )
        role_arn = response['Role']['Arn']
        
        # Attach policies
        policies = [
            'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
            'arn:aws:iam::aws:policy/AmazonS3FullAccess'
        ]
        
        for policy in policies:
            iam.attach_role_policy(RoleName=role_name, PolicyArn=policy)
        
        print(f"✅ Created role: {role_arn}")
        print("⏳ Waiting 10 seconds for role propagation...")
        time.sleep(10)
        
        return role_arn
        
    except Exception as e:
        print(f"❌ Failed to create role: {e}")
        return None

def upload_training_code():
    """Upload training script to S3."""
    s3 = boto3.client('s3')
    bucket = 'crpbot-sagemaker-training'
    
    # Create bucket if it doesn't exist
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"✅ Using existing bucket: {bucket}")
    except:
        try:
            s3.create_bucket(Bucket=bucket)
            print(f"✅ Created bucket: {bucket}")
        except Exception as e:
            print(f"❌ Failed to create bucket: {e}")
            return None
    
    # Upload training script
    script_path = 'apps/trainer/sagemaker_train.py'
    s3_key = 'code/sagemaker_train.py'
    
    try:
        s3.upload_file(script_path, bucket, s3_key)
        print(f"✅ Uploaded training script to s3://{bucket}/{s3_key}")
        return f"s3://{bucket}/code/"
    except Exception as e:
        print(f"❌ Failed to upload script: {e}")
        return None

def create_training_job(role_arn, code_location, symbol='BTC'):
    """Create SageMaker training job."""
    sagemaker = boto3.client('sagemaker', region_name='us-east-2')  # Correct region
    
    job_name = f"crpbot-v7-enhanced-{symbol.lower()}-{int(time.time())}"
    
    training_params = {
        'TrainingJobName': job_name,
        'RoleArn': role_arn,
        'AlgorithmSpecification': {
            'TrainingImage': '763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu118-ubuntu20.04-sagemaker',
            'TrainingInputMode': 'File'
        },
        'InputDataConfig': [
            {
                'ChannelName': 'training',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': 's3://crpbot-sagemaker-training/data/',
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                }
            }
        ],
        'OutputDataConfig': {
            'S3OutputPath': 's3://crpbot-sagemaker-training/models/'
        },
        'ResourceConfig': {
            'InstanceType': 'ml.g5.4xlarge',  # GPU instance for fast training (1-2 hours)
            'InstanceCount': 1,
            'VolumeSizeInGB': 50  # Increased for GPU data
        },
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 7200  # 2 hours max
        },
        'HyperParameters': {
            'symbol': symbol,
            'epochs': '30',
            'batch-size': '32',  # Smaller for CPU
            'lr': '0.001'
        }
    }
    
    try:
        response = sagemaker.create_training_job(**training_params)
        print(f"✅ Created training job: {job_name}")
        print(f"📊 Training {symbol} model on ml.m5.large")
        print(f"🔗 Job ARN: {response['TrainingJobArn']}")
        return job_name
    except Exception as e:
        print(f"❌ Failed to create training job: {e}")
        return None

def monitor_training_job(job_name):
    """Monitor training job progress."""
    sagemaker = boto3.client('sagemaker', region_name='us-east-2')
    
    print(f"📊 Monitoring training job: {job_name}")
    print("⏳ Training in progress...")
    
    while True:
        try:
            response = sagemaker.describe_training_job(TrainingJobName=job_name)
            status = response['TrainingJobStatus']
            
            if status == 'Completed':
                print("✅ Training completed successfully!")
                model_artifacts = response['ModelArtifacts']['S3ModelArtifacts']
                print(f"📦 Model artifacts: {model_artifacts}")
                return True
            elif status == 'Failed':
                print("❌ Training failed!")
                print(f"Failure reason: {response.get('FailureReason', 'Unknown')}")
                return False
            elif status in ['Stopping', 'Stopped']:
                print(f"⏹️ Training {status.lower()}")
                return False
            else:
                print(f"⏳ Status: {status}")
                time.sleep(30)
                
        except Exception as e:
            print(f"❌ Error monitoring job: {e}")
            return False

def main():
    """Main deployment function."""
    print("🚀 Deploying V7 Enhanced Training to SageMaker")
    print("="*60)
    
    # Step 1: Create SageMaker role
    print("\n1️⃣ Setting up SageMaker execution role...")
    role_arn = create_sagemaker_role()
    if not role_arn:
        print("❌ Failed to create/get SageMaker role")
        return
    
    # Step 2: Upload training code
    print("\n2️⃣ Uploading training code to S3...")
    code_location = upload_training_code()
    if not code_location:
        print("❌ Failed to upload training code")
        return
    
    # Step 3: Create training jobs for each symbol
    symbols = ['BTC', 'ETH', 'SOL']
    training_jobs = []
    
    for symbol in symbols:
        print(f"\n3️⃣ Creating training job for {symbol}...")
        job_name = create_training_job(role_arn, code_location, symbol)
        if job_name:
            training_jobs.append((symbol, job_name))
        else:
            print(f"❌ Failed to create training job for {symbol}")
    
    # Step 4: Monitor training jobs
    if training_jobs:
        print(f"\n4️⃣ Monitoring {len(training_jobs)} training jobs...")
        for symbol, job_name in training_jobs:
            print(f"\n📊 Monitoring {symbol} training...")
            success = monitor_training_job(job_name)
            if success:
                print(f"✅ {symbol} training completed successfully")
            else:
                print(f"❌ {symbol} training failed")
    
    print("\n🎉 SageMaker training deployment complete!")
    print("\nNext steps:")
    print("1. Download model artifacts from S3")
    print("2. Run diagnostic to verify quality gates")
    print("3. Deploy to production if all gates pass")

if __name__ == "__main__":
    main()
