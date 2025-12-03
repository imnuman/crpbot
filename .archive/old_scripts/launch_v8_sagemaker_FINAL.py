#!/usr/bin/env python3
"""
Launch V8 Training on SageMaker
Simple script to start V8 training with all fixes
"""

import boto3
import os
import json
from datetime import datetime

def check_requirements():
    """Check if we have everything needed"""

    print("🔍 Checking SageMaker Requirements...")

    # Check data files in S3
    bucket = 'crpbot-sagemaker-training'
    required_files = [
        'data/BTC_features.parquet',
        'data/ETH_features.parquet',
        'data/SOL_features.parquet'
    ]

    try:
        s3 = boto3.client('s3')
        missing_files = []
        for s3_key in required_files:
            try:
                response = s3.head_object(Bucket=bucket, Key=s3_key)
                size_mb = response['ContentLength'] / (1024*1024)
                print(f"  ✅ s3://{bucket}/{s3_key} ({size_mb:.1f} MB)")
            except:
                print(f"  ❌ s3://{bucket}/{s3_key} - MISSING")
                missing_files.append(s3_key)

        if missing_files:
            print(f"  ❌ Missing S3 files: {missing_files}")
            return False
    except Exception as e:
        print(f"  ❌ S3 access error: {e}")
        return False
    
    # Check AWS credentials
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"  ✅ AWS Account: {identity['Account']}")
        print(f"  ✅ AWS User: {identity['Arn'].split('/')[-1]}")
    except Exception as e:
        print(f"  ❌ AWS credentials issue: {e}")
        return False
    
    # Check S3 bucket
    bucket = 'crpbot-sagemaker-training'
    try:
        s3 = boto3.client('s3')
        s3.head_bucket(Bucket=bucket)
        print(f"  ✅ S3 Bucket: {bucket}")
    except Exception as e:
        print(f"  ❌ S3 bucket issue: {e}")
        return False
    
    return True

def upload_training_code():
    """Upload training code to S3"""

    print("\n📤 Uploading Training Code...")

    s3 = boto3.client('s3')
    bucket = 'crpbot-sagemaker-training'
    prefix = 'v8-final'

    files_to_upload = [
        'v8_sagemaker_train.py',
        'requirements_sagemaker.txt'
    ]

    uploaded = []

    for file in files_to_upload:
        if os.path.exists(file):
            s3_key = f'{prefix}/code/{file}'
            try:
                s3.upload_file(file, bucket, s3_key)
                print(f"  ✅ {file} → s3://{bucket}/{s3_key}")
                uploaded.append(file)
            except Exception as e:
                print(f"  ❌ {file}: {e}")
        else:
            print(f"  ⚠️  {file} not found")

    return len(uploaded) == 2  # Need training script + requirements

def create_sagemaker_training_job():
    """Create SageMaker training job using boto3"""
    
    print("\n🚀 Creating SageMaker Training Job...")
    
    sagemaker = boto3.client('sagemaker')
    
    # Job configuration
    job_name = f"v8-enhanced-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    training_job_config = {
        'TrainingJobName': job_name,
        'RoleArn': 'arn:aws:iam::980104576869:role/service-role/AmazonBraketServiceSageMakerNotebookRole',
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
                },
                'ContentType': 'application/x-parquet',
                'CompressionType': 'None'
            }
        ],
        'OutputDataConfig': {
            'S3OutputPath': 's3://crpbot-sagemaker-training/v8-final/output/'
        },
        'ResourceConfig': {
            'InstanceType': 'ml.g5.4xlarge',
            'InstanceCount': 1,
            'VolumeSizeInGB': 100
        },
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 7200  # 2 hours max
        },
        'HyperParameters': {
            'epochs': '100',
            'batch-size': '256',
            'learning-rate': '0.001',
            'all': 'True'
        },
        'Tags': [
            {'Key': 'Project', 'Value': 'CRPBot'},
            {'Key': 'Version', 'Value': 'V8'},
            {'Key': 'Purpose', 'Value': 'Fix-V6-Issues'}
        ]
    }
    
    try:
        response = sagemaker.create_training_job(**training_job_config)

        print(f"✅ Training Job Created: {job_name}")
        print(f"✅ Instance: ml.g5.4xlarge (16 vCPUs, 64GB RAM, 1x A10G GPU)")
        print(f"✅ Max Runtime: 2 hours")
        print(f"✅ Expected Cost: ~$4.06 ($2.03/hr × 2 hrs)")
        print(f"✅ Dataset: 835 MB parquet files (2 years of data)")

        # Monitor URL
        region = boto3.Session().region_name or 'us-east-2'
        monitor_url = f"https://console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{job_name}"
        print(f"✅ Monitor: {monitor_url}")

        return job_name
        
    except Exception as e:
        print(f"❌ Failed to create training job: {e}")
        return None

def monitor_training_job(job_name):
    """Monitor training job progress"""
    
    if not job_name:
        return
    
    print(f"\n📊 Monitoring Training Job: {job_name}")
    
    sagemaker = boto3.client('sagemaker')
    
    try:
        # Get job status
        response = sagemaker.describe_training_job(TrainingJobName=job_name)
        
        status = response['TrainingJobStatus']
        print(f"Status: {status}")
        
        if 'TrainingStartTime' in response:
            start_time = response['TrainingStartTime']
            print(f"Started: {start_time}")
        
        if status == 'InProgress':
            print("Training is running...")
            print("Use AWS Console to monitor logs and progress")
        elif status == 'Completed':
            print("✅ Training completed successfully!")
            
            # Model artifacts
            if 'ModelArtifacts' in response:
                model_uri = response['ModelArtifacts']['S3ModelArtifacts']
                print(f"Model artifacts: {model_uri}")
        elif status == 'Failed':
            print("❌ Training failed")
            if 'FailureReason' in response:
                print(f"Reason: {response['FailureReason']}")
        
        return response
        
    except Exception as e:
        print(f"❌ Error monitoring job: {e}")
        return None

def main():
    """Main execution"""
    
    print("🚀 SageMaker V8 Enhanced Training Launcher")
    print("Fixing all V6 model issues with GPU training")
    print("="*60)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements not met. Please fix issues above.")
        return
    
    # Upload training code
    if not upload_training_code():
        print("\n❌ Failed to upload training code.")
        return
    
    # Create training job
    job_name = create_sagemaker_training_job()
    
    if job_name:
        print(f"\n🎉 V8 Training Launched Successfully!")
        print(f"Job Name: {job_name}")
        print(f"Expected Duration: 1-2 hours")
        print(f"Expected Cost: ~$4.06")
        print(f"Expected Results:")
        print(f"  - 3 trained V8 models (BTC, ETH, SOL)")
        print(f"  - 3 feature processors")
        print(f"  - <10% overconfident predictions")
        print(f"  - Balanced class distributions (15-60% per class)")
        print(f"  - Realistic confidence scores (60-85%)")
        print(f"  - Logit range: ±15 (not ±40,000)")

        # Initial status check
        monitor_training_job(job_name)

        print(f"\nTo check status later:")
        print(f"aws sagemaker describe-training-job --training-job-name {job_name}")

    else:
        print("\n❌ Failed to launch training")

if __name__ == "__main__":
    main()
