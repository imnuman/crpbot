#!/usr/bin/env python3
"""
Quick Multi-GPU Training Launcher
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

# GPU instance options
GPU_CONFIGS = {
    '1': {
        'instance_type': 'ml.g4dn.12xlarge',  # 4 GPUs, 48 vCPUs, 192GB RAM
        'instance_count': 1,
        'description': '4 GPUs on single instance'
    },
    '2': {
        'instance_type': 'ml.g4dn.xlarge',    # 1 GPU each
        'instance_count': 4,                   # 4 instances = 4 GPUs total
        'description': '4 instances with 1 GPU each'
    },
    '3': {
        'instance_type': 'ml.p3.8xlarge',     # 4 V100 GPUs (faster)
        'instance_count': 1,
        'description': '4 V100 GPUs (premium)'
    }
}

def launch_training(config_choice='1'):
    sagemaker_session = sagemaker.Session()
    bucket = sagemaker_session.default_bucket()
    
    sts = boto3.client('sts')
    account_id = sts.get_caller_identity()['Account']
    role = f"arn:aws:iam::{account_id}:role/SageMakerExecutionRole"
    
    config = GPU_CONFIGS[config_choice]
    
    estimator = PyTorch(
        entry_point='v8_sagemaker_train_multi_gpu.py',
        source_dir='.',
        role=role,
        instance_type=config['instance_type'],
        instance_count=config['instance_count'],
        framework_version='2.1.0',
        py_version='py310',
        hyperparameters={
            'epochs': 100,
            'batch_size': 64,  # Larger batch for multi-GPU
            'learning_rate': 0.001,
            'coin': 'BTC'
        },
        output_path=f's3://{bucket}/crpbot/models',
        max_run=7200,  # 2 hours
        volume_size=50,
        distribution={'torch_distributed': {'enabled': True}},
        use_spot_instances=True,  # Save 70% cost
        max_wait=10800
    )
    
    print(f"🚀 Launching: {config['description']}")
    print(f"📍 Instance: {config['instance_type']} x{config['instance_count']}")
    print(f"💰 Using spot instances (70% cheaper)")
    
    estimator.fit()
    return estimator

if __name__ == "__main__":
    print("Multi-GPU Training Options:")
    for key, config in GPU_CONFIGS.items():
        print(f"{key}. {config['description']} ({config['instance_type']})")
    
    choice = input("Choose (1-3): ").strip() or '1'
    estimator = launch_training(choice)
