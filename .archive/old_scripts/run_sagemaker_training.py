#!/usr/bin/env python3
"""
Simple command to launch SageMaker training on AWS
Usage: python run_sagemaker_training.py
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

def main():
    print("🚀 Launching SageMaker training on AWS...")
    
    # Get SageMaker session
    sagemaker_session = sagemaker.Session()
    bucket = sagemaker_session.default_bucket()
    
    # Get account ID for role
    sts = boto3.client('sts')
    account_id = sts.get_caller_identity()['Account']
    role = f"arn:aws:iam::{account_id}:role/SageMakerExecutionRole"
    
    # Create estimator
    estimator = PyTorch(
        entry_point='v8_sagemaker_train_aws.py',
        source_dir='.',
        role=role,
        instance_type='ml.g4dn.xlarge',
        instance_count=1,
        framework_version='2.1.0',
        py_version='py310',
        hyperparameters={'epochs': 10, 'coin': 'BTC'},
        output_path=f's3://{bucket}/crpbot/models',
        max_run=1800,
        volume_size=20,
    )
    
    print(f"📍 Training on: ml.g4dn.xlarge (GPU)")
    print(f"📦 Dependencies: requirements_sagemaker.txt")
    print(f"💾 Output: s3://{bucket}/crpbot/models")
    
    # Launch training
    estimator.fit()
    print("✅ SageMaker training completed!")

if __name__ == "__main__":
    main()
