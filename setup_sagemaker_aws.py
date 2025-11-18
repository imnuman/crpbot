#!/usr/bin/env python3
"""
Setup SageMaker Training Job on AWS
This script launches a SageMaker training job with all dependencies
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role
import os

def setup_sagemaker_training():
    """Launch SageMaker training job with dependencies"""
    
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    
    # Get execution role (or create one)
    try:
        role = get_execution_role()
    except:
        # If not in SageMaker environment, use IAM role
        role = "arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole"
        print(f"Using IAM role: {role}")
    
    # Define training job
    estimator = PyTorch(
        entry_point='v8_sagemaker_train.py',
        source_dir='.',
        role=role,
        instance_type='ml.g4dn.xlarge',  # GPU instance
        instance_count=1,
        framework_version='2.1.0',
        py_version='py310',
        requirements_file='requirements_sagemaker.txt',  # This installs our deps
        hyperparameters={
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'coin': 'BTC'
        },
        output_path=f's3://{sagemaker_session.default_bucket()}/models',
        code_location=f's3://{sagemaker_session.default_bucket()}/code',
        max_run=3600,  # 1 hour max
        volume_size=30,  # 30GB storage
        environment={
            'PYTHONPATH': '/opt/ml/code',
        }
    )
    
    print("🚀 Launching SageMaker training job...")
    print(f"Instance: ml.g4dn.xlarge (GPU)")
    print(f"Dependencies: requirements_sagemaker.txt")
    print(f"Output: s3://{sagemaker_session.default_bucket()}/models")
    
    # Start training
    estimator.fit()
    
    return estimator

if __name__ == "__main__":
    estimator = setup_sagemaker_training()
    print("✅ SageMaker training job launched!")
