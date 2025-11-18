#!/usr/bin/env python3
"""
Test T4 GPU Availability and Launch Optimal Training
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

def test_t4_capacity():
    """Test how many T4 instances we can use"""
    sagemaker_session = sagemaker.Session()
    bucket = sagemaker_session.default_bucket()
    
    sts = boto3.client('sts')
    account_id = sts.get_caller_identity()['Account']
    role = f"arn:aws:iam::{account_id}:role/SageMakerExecutionRole"
    
    # Test configurations in order of preference
    configs = [
        {'count': 4, 'desc': '4 T4 GPUs (fastest)'},
        {'count': 2, 'desc': '2 T4 GPUs (2x speed)'},
        {'count': 1, 'desc': '1 T4 GPU (baseline)'}
    ]
    
    for config in configs:
        try:
            print(f"🧪 Testing {config['desc']}...")
            
            estimator = PyTorch(
                entry_point='v8_sagemaker_train_multi_gpu.py',
                source_dir='.',
                role=role,
                instance_type='ml.g4dn.xlarge',
                instance_count=config['count'],
                framework_version='2.1.0',
                py_version='py310',
                hyperparameters={'epochs': 1, 'coin': 'BTC'},  # Quick test
                output_path=f's3://{bucket}/crpbot/test',
                max_run=300,
                volume_size=20,
                distribution={'torch_distributed': {'enabled': True}} if config['count'] > 1 else None
            )
            
            print(f"✅ {config['desc']} - Available!")
            return config['count'], estimator
            
        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "limit" in error_msg:
                print(f"❌ {config['desc']} - Quota exceeded")
                continue
            elif "capacity" in error_msg:
                print(f"⚠️  {config['desc']} - No capacity")
                continue
            else:
                print(f"❌ {config['desc']} - Error: {e}")
                continue
    
    return 0, None

def launch_optimal_t4_training(instance_count):
    """Launch training with optimal T4 configuration"""
    sagemaker_session = sagemaker.Session()
    bucket = sagemaker_session.default_bucket()
    
    sts = boto3.client('sts')
    account_id = sts.get_caller_identity()['Account']
    role = f"arn:aws:iam::{account_id}:role/SageMakerExecutionRole"
    
    # Adjust batch size based on GPU count
    batch_size = 32 * instance_count  # Scale batch size with GPUs
    
    estimator = PyTorch(
        entry_point='v8_sagemaker_train_multi_gpu.py',
        source_dir='.',
        role=role,
        instance_type='ml.g4dn.xlarge',  # T4 GPU
        instance_count=instance_count,
        framework_version='2.1.0',
        py_version='py310',
        hyperparameters={
            'epochs': 100,
            'batch_size': batch_size,
            'learning_rate': 0.001,
            'coin': 'BTC'
        },
        output_path=f's3://{bucket}/crpbot/models',
        max_run=7200,
        volume_size=30,
        distribution={'torch_distributed': {'enabled': True}} if instance_count > 1 else None,
        use_spot_instances=True,
        max_wait=10800
    )
    
    cost_per_hour = 0.526 * instance_count * 0.3  # Spot pricing ~30% of on-demand
    
    print(f"🚀 Launching {instance_count}x T4 GPU Training:")
    print(f"📍 {instance_count}x ml.g4dn.xlarge")
    print(f"🔥 Batch size: {batch_size} (scaled)")
    print(f"💰 Cost: ~${cost_per_hour:.2f}/hour (spot)")
    print(f"⚡ Speed: {instance_count}x faster than single GPU")
    
    estimator.fit()
    return estimator

def main():
    print("🔍 Testing T4 GPU availability...")
    
    instance_count, _ = test_t4_capacity()
    
    if instance_count > 0:
        print(f"\n🎯 Optimal configuration: {instance_count}x T4 GPUs")
        estimator = launch_optimal_t4_training(instance_count)
        print("✅ T4 training completed!")
    else:
        print("\n❌ No T4 GPUs available")
        print("💡 Try again later or request quota increase")

if __name__ == "__main__":
    main()
