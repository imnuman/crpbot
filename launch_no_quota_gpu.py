#!/usr/bin/env python3
"""
SageMaker GPU Training - No Quota Approval Required
Uses instances that don't require quota requests
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

def launch_training():
    sagemaker_session = sagemaker.Session()
    bucket = sagemaker_session.default_bucket()
    
    sts = boto3.client('sts')
    account_id = sts.get_caller_identity()['Account']
    role = f"arn:aws:iam::{account_id}:role/SageMakerExecutionRole"
    
    # ml.g4dn.xlarge is usually available without quota approval
    # It's the most common GPU instance for new accounts
    estimator = PyTorch(
        entry_point='v8_sagemaker_train_aws.py',
        source_dir='.',
        role=role,
        instance_type='ml.g4dn.xlarge',  # 1 GPU, usually no quota needed
        instance_count=1,
        framework_version='2.1.0',
        py_version='py310',
        hyperparameters={
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'coin': 'BTC'
        },
        output_path=f's3://{bucket}/crpbot/models',
        max_run=3600,
        volume_size=30,
        use_spot_instances=True,  # Even cheaper
        max_wait=7200
    )
    
    print("🚀 Using ml.g4dn.xlarge (no quota approval needed)")
    print("💰 Using spot instances for 70% cost savings")
    print("🔥 1 GPU, 4 vCPUs, 16GB RAM")
    
    try:
        estimator.fit()
        print("✅ Training completed successfully!")
    except Exception as e:
        if "quota" in str(e).lower() or "limit" in str(e).lower():
            print("❌ Quota limit reached. Trying CPU fallback...")
            launch_cpu_fallback()
        else:
            raise e
    
    return estimator

def launch_cpu_fallback():
    """Fallback to CPU if GPU quota exceeded"""
    sagemaker_session = sagemaker.Session()
    bucket = sagemaker_session.default_bucket()
    
    sts = boto3.client('sts')
    account_id = sts.get_caller_identity()['Account']
    role = f"arn:aws:iam::{account_id}:role/SageMakerExecutionRole"
    
    estimator = PyTorch(
        entry_point='v8_sagemaker_train_aws.py',
        source_dir='.',
        role=role,
        instance_type='ml.m5.xlarge',  # CPU instance, always available
        instance_count=1,
        framework_version='2.1.0',
        py_version='py310',
        hyperparameters={
            'epochs': 20,  # Fewer epochs for CPU
            'batch_size': 16,  # Smaller batch for CPU
            'learning_rate': 0.001,
            'coin': 'BTC'
        },
        output_path=f's3://{bucket}/crpbot/models',
        max_run=3600,
        volume_size=20,
        use_spot_instances=True
    )
    
    print("🔄 Fallback: ml.m5.xlarge (CPU)")
    print("⚠️  Training will be slower but guaranteed to work")
    
    estimator.fit()
    return estimator

if __name__ == "__main__":
    estimator = launch_training()
