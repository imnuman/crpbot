#!/usr/bin/env python3
"""
Launch 4x T4 GPU Distributed Training - Final Version
All dependencies included
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

def launch_4x_t4_training():
    """Launch 4x T4 GPU distributed training with all dependencies"""
    
    sagemaker_session = sagemaker.Session()
    bucket = sagemaker_session.default_bucket()
    
    sts = boto3.client('sts')
    account_id = sts.get_caller_identity()['Account']
    role = f"arn:aws:iam::{account_id}:role/SageMakerExecutionRole"
    
    # 4x T4 GPU distributed training
    estimator = PyTorch(
        entry_point='v8_t4_distributed_train.py',
        source_dir='.',
        role=role,
        instance_type='ml.g4dn.xlarge',  # T4 GPU
        instance_count=4,                # 4 instances = 4 T4 GPUs
        framework_version='2.1.0',
        py_version='py310',
        hyperparameters={
            'epochs': 100,
            'batch_size': 512,  # Large batch for 4 GPUs
            'learning_rate': 0.002,  # Higher LR for larger batch
            'coin': 'BTC'
        },
        output_path=f's3://{bucket}/crpbot/models/4x-t4',
        max_run=10800,  # 3 hours
        volume_size=50,
        distribution={
            'torch_distributed': {
                'enabled': True
            }
        },
        use_spot_instances=True,
        max_wait=14400,  # 4 hours wait
        environment={
            'NCCL_DEBUG': 'INFO',  # Debug distributed training
            'PYTHONPATH': '/opt/ml/code'
        }
    )
    
    print("🚀 4x T4 GPU Distributed Training Launch:")
    print("=" * 50)
    print("📍 Instance: 4x ml.g4dn.xlarge (T4 GPUs)")
    print("🔥 Total GPUs: 4 NVIDIA T4 Tensor Core")
    print("💾 Memory: 64GB total (16GB per GPU)")
    print("⚡ Speed: ~4x faster than single GPU")
    print("💰 Cost: ~$0.64/hour (spot instances)")
    print("📦 Dependencies: All ML/distributed packages")
    print("🎯 Model: Advanced LSTM + Attention")
    print("=" * 50)
    
    try:
        print("🚀 Starting distributed training...")
        estimator.fit()
        print("✅ 4x T4 distributed training completed!")
        
        # Print model location
        print(f"📁 Model saved to: s3://{bucket}/crpbot/models/4x-t4")
        
        return estimator
        
    except Exception as e:
        error_msg = str(e).lower()
        if "quota" in error_msg or "limit" in error_msg:
            print("❌ T4 quota exceeded")
            print("💡 Try single T4 or request quota increase")
        elif "capacity" in error_msg:
            print("⚠️  No T4 capacity available")
            print("💡 Try again in a few minutes")
        else:
            print(f"❌ Training failed: {e}")
        raise e

if __name__ == "__main__":
    estimator = launch_4x_t4_training()
