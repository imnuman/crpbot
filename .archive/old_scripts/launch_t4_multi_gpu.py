#!/usr/bin/env python3
"""
T4 GPU Multi-Instance Training
Uses multiple ml.g4dn.xlarge instances (T4 GPUs) for faster training
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

def launch_t4_training():
    sagemaker_session = sagemaker.Session()
    bucket = sagemaker_session.default_bucket()
    
    sts = boto3.client('sts')
    account_id = sts.get_caller_identity()['Account']
    role = f"arn:aws:iam::{account_id}:role/SageMakerExecutionRole"
    
    # Multiple T4 GPUs (ml.g4dn.xlarge instances)
    estimator = PyTorch(
        entry_point='v8_sagemaker_train_multi_gpu.py',
        source_dir='.',
        role=role,
        instance_type='ml.g4dn.xlarge',  # T4 GPU - no quota needed
        instance_count=4,                # 4 instances = 4 T4 GPUs
        framework_version='2.1.0',
        py_version='py310',
        hyperparameters={
            'epochs': 100,
            'batch_size': 32,  # Per GPU
            'learning_rate': 0.001,
            'coin': 'BTC'
        },
        output_path=f's3://{bucket}/crpbot/models',
        max_run=7200,  # 2 hours
        volume_size=30,
        distribution={'torch_distributed': {'enabled': True}},
        use_spot_instances=True,  # 70% cheaper
        max_wait=10800
    )
    
    print("🚀 T4 Multi-GPU Training Setup:")
    print("📍 4x ml.g4dn.xlarge (4 T4 GPUs)")
    print("🔥 Distributed training across instances")
    print("💰 Using spot instances (~$0.64/hour total)")
    print("⚡ 4x faster than single GPU")
    
    try:
        estimator.fit()
        print("✅ T4 multi-GPU training completed!")
        return estimator
    except Exception as e:
        if "quota" in str(e).lower() or "limit" in str(e).lower():
            print("❌ T4 quota exceeded. Trying single T4...")
            return launch_single_t4()
        else:
            raise e

def launch_single_t4():
    """Fallback to single T4 GPU"""
    sagemaker_session = sagemaker.Session()
    bucket = sagemaker_session.default_bucket()
    
    sts = boto3.client('sts')
    account_id = sts.get_caller_identity()['Account']
    role = f"arn:aws:iam::{account_id}:role/SageMakerExecutionRole"
    
    estimator = PyTorch(
        entry_point='v8_sagemaker_train_aws.py',
        source_dir='.',
        role=role,
        instance_type='ml.g4dn.xlarge',  # Single T4 GPU
        instance_count=1,
        framework_version='2.1.0',
        py_version='py310',
        hyperparameters={
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'coin': 'BTC'
        },
        output_path=f's3://{bucket}/crpbot/models',
        max_run=7200,
        volume_size=30,
        use_spot_instances=True
    )
    
    print("🔄 Fallback: Single T4 GPU")
    print("📍 1x ml.g4dn.xlarge")
    print("💰 ~$0.16/hour (spot)")
    
    estimator.fit()
    return estimator

if __name__ == "__main__":
    estimator = launch_t4_training()
