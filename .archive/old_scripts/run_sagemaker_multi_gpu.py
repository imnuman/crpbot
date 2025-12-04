#!/usr/bin/env python3
"""
Multi-GPU SageMaker Training
Uses multiple GPUs to speed up training
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

def main():
    print("🚀 Launching Multi-GPU SageMaker training...")
    
    sagemaker_session = sagemaker.Session()
    bucket = sagemaker_session.default_bucket()
    
    sts = boto3.client('sts')
    account_id = sts.get_caller_identity()['Account']
    role = f"arn:aws:iam::{account_id}:role/SageMakerExecutionRole"
    
    # Multi-GPU configuration options:
    
    # Option 1: Single instance with multiple GPUs
    estimator_multi_gpu = PyTorch(
        entry_point='v8_sagemaker_train_multi_gpu.py',
        source_dir='.',
        role=role,
        instance_type='ml.g4dn.12xlarge',  # 4 GPUs
        instance_count=1,
        framework_version='2.1.0',
        py_version='py310',
        hyperparameters={
            'epochs': 50,
            'batch_size': 128,  # Larger batch for multi-GPU
            'coin': 'BTC'
        },
        output_path=f's3://{bucket}/crpbot/models',
        max_run=3600,
        volume_size=50,
        distribution={'torch_distributed': {'enabled': True}}  # Enable distributed training
    )
    
    # Option 2: Multiple instances (distributed)
    estimator_distributed = PyTorch(
        entry_point='v8_sagemaker_train_multi_gpu.py',
        source_dir='.',
        role=role,
        instance_type='ml.g4dn.xlarge',  # 1 GPU each
        instance_count=4,  # 4 instances = 4 GPUs total
        framework_version='2.1.0',
        py_version='py310',
        hyperparameters={
            'epochs': 50,
            'batch_size': 32,
            'coin': 'BTC'
        },
        output_path=f's3://{bucket}/crpbot/models',
        max_run=3600,
        volume_size=30,
        distribution={'torch_distributed': {'enabled': True}}
    )
    
    print("Choose GPU configuration:")
    print("1. Single instance, 4 GPUs (ml.g4dn.12xlarge)")
    print("2. 4 instances, 1 GPU each (4x ml.g4dn.xlarge)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        print("🔥 Using ml.g4dn.12xlarge (4 GPUs)")
        estimator = estimator_multi_gpu
    else:
        print("🔥 Using 4x ml.g4dn.xlarge (distributed)")
        estimator = estimator_distributed
    
    estimator.fit()
    print("✅ Multi-GPU training completed!")

if __name__ == "__main__":
    main()
