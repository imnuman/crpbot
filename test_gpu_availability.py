#!/usr/bin/env python3
"""
Test GPU Instance Availability
Tests which GPU instances work without quota approval
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

# GPU instances in order of likelihood to work without quota
GPU_INSTANCES = [
    'ml.g4dn.xlarge',    # Most likely to work (1 GPU)
    'ml.g4dn.2xlarge',   # Sometimes works (1 GPU, more RAM)
    'ml.g5.xlarge',      # Newer generation, might work
    'ml.m5.xlarge'       # CPU fallback
]

def test_instance(instance_type):
    """Test if instance type works"""
    try:
        sagemaker_session = sagemaker.Session()
        bucket = sagemaker_session.default_bucket()
        
        sts = boto3.client('sts')
        account_id = sts.get_caller_identity()['Account']
        role = f"arn:aws:iam::{account_id}:role/SageMakerExecutionRole"
        
        estimator = PyTorch(
            entry_point='v8_sagemaker_train_aws.py',
            source_dir='.',
            role=role,
            instance_type=instance_type,
            instance_count=1,
            framework_version='2.1.0',
            py_version='py310',
            hyperparameters={'epochs': 1, 'coin': 'BTC'},  # Quick test
            output_path=f's3://{bucket}/crpbot/test',
            max_run=300,  # 5 minutes max
            volume_size=20
        )
        
        print(f"✅ {instance_type} - Available")
        return True, estimator
        
    except Exception as e:
        error_msg = str(e).lower()
        if "quota" in error_msg or "limit" in error_msg:
            print(f"❌ {instance_type} - Quota limit")
        elif "capacity" in error_msg:
            print(f"⚠️  {instance_type} - No capacity (try later)")
        else:
            print(f"❌ {instance_type} - Error: {e}")
        return False, None

def find_available_gpu():
    """Find first available GPU instance"""
    print("🔍 Testing GPU instance availability...")
    
    for instance_type in GPU_INSTANCES:
        available, estimator = test_instance(instance_type)
        if available:
            print(f"🎯 Using: {instance_type}")
            return instance_type, estimator
    
    print("❌ No GPU instances available")
    return None, None

if __name__ == "__main__":
    instance_type, estimator = find_available_gpu()
    
    if instance_type:
        print(f"\n🚀 Launch training with: {instance_type}")
        print("Run: python launch_no_quota_gpu.py")
    else:
        print("\n💡 Request quota increase for GPU instances")
        print("Or use CPU training with ml.m5.xlarge")
