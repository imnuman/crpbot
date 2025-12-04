#!/usr/bin/env python3
"""
Simple SageMaker Training Launch - Guaranteed to Work
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

def main():
    print("🚀 Launching simple SageMaker training...")
    
    sagemaker_session = sagemaker.Session()
    bucket = sagemaker_session.default_bucket()
    
    sts = boto3.client('sts')
    account_id = sts.get_caller_identity()['Account']
    role = f"arn:aws:iam::{account_id}:role/SageMakerExecutionRole"
    
    print(f"📍 Using bucket: {bucket}")
    print(f"🔑 Using role: {role}")
    
    # Simple single GPU training
    estimator = PyTorch(
        entry_point='v8_complete_4x_t4_training.py',
        source_dir='.',
        role=role,
        instance_type='ml.g4dn.xlarge',  # Single T4 GPU
        instance_count=1,
        framework_version='2.1.0',
        py_version='py310',
        hyperparameters={
            'epochs': 20,
            'batch_size': 64,
            'learning_rate': 0.001,
            'coin': 'BTC'
        },
        output_path=f's3://{bucket}/crpbot/models/simple',
        max_run=3600,  # 1 hour
        volume_size=30,
        use_spot_instances=True,
        max_wait=7200
    )
    
    print("🔥 Starting training job...")
    estimator.fit()
    
    print("✅ Training job submitted!")
    print(f"📊 Monitor at: https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs")
    
    return estimator

if __name__ == "__main__":
    estimator = main()
