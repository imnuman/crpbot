#!/usr/bin/env python3
"""
Launch SageMaker Training on AWS
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
import os

def launch_training():
    """Launch SageMaker training job on AWS"""
    
    # Get AWS session and region
    session = boto3.Session()
    region = session.region_name or 'us-east-1'
    
    # Initialize SageMaker
    sagemaker_session = sagemaker.Session()
    bucket = sagemaker_session.default_bucket()
    
    # IAM role for SageMaker
    role = f"arn:aws:iam::{session.client('sts').get_caller_identity()['Account']}:role/SageMakerExecutionRole"
    
    print(f"🌍 Region: {region}")
    print(f"🪣 S3 Bucket: {bucket}")
    print(f"🔑 Role: {role}")
    
    # Create PyTorch estimator
    estimator = PyTorch(
        entry_point='v8_sagemaker_train_aws.py',
        source_dir='.',
        role=role,
        instance_type='ml.g4dn.xlarge',  # GPU instance
        instance_count=1,
        framework_version='2.1.0',
        py_version='py310',
        hyperparameters={
            'epochs': 20,
            'batch_size': 32,
            'learning_rate': 0.001,
            'coin': 'BTC'
        },
        output_path=f's3://{bucket}/crpbot/models',
        max_run=1800,  # 30 minutes
        volume_size=20,
        use_spot_instances=True,  # Save costs
        max_wait=3600,
    )
    
    print("🚀 Starting SageMaker training job...")
    estimator.fit()
    
    print("✅ Training completed!")
    return estimator

if __name__ == "__main__":
    estimator = launch_training()
