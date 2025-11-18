#!/usr/bin/env python3
"""
SageMaker V8 Training - Complete V6 Fix
Launch V8 training on SageMaker with GPU
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role
import os
from datetime import datetime

def setup_sagemaker_training():
    """Setup and launch V8 training on SageMaker"""
    
    print("🚀 SageMaker V8 Training Setup")
    print("="*40)
    
    # SageMaker session
    sagemaker_session = sagemaker.Session()
    region = sagemaker_session.boto_region_name
    
    print(f"Region: {region}")
    
    # S3 bucket for training
    bucket = 'crpbot-sagemaker-training'
    prefix = 'v8-training'
    
    print(f"S3 Bucket: {bucket}")
    print(f"Prefix: {prefix}")
    
    # IAM role for SageMaker
    try:
        role = get_execution_role()
        print(f"Using execution role: {role}")
    except:
        # Use existing SageMaker role
        role = 'arn:aws:iam::980104576869:role/service-role/AmazonBraketServiceSageMakerNotebookRole'
        print(f"Using service role: {role}")
    
    # Training configuration
    training_config = {
        'entry_point': 'v8_enhanced_training.py',
        'source_dir': '.',
        'role': role,
        'instance_type': 'ml.g5.xlarge',  # 1x NVIDIA A10G (24GB)
        'instance_count': 1,
        'framework_version': '2.1.0',
        'py_version': 'py310',
        'volume_size': 100,  # GB
        'max_run': 6 * 3600,  # 6 hours max
        'use_spot_instances': True,  # Save up to 90%
        'max_wait': 8 * 3600,  # Wait up to 8 hours for spot
        'checkpoint_s3_uri': f's3://{bucket}/{prefix}/checkpoints',
        'output_path': f's3://{bucket}/{prefix}/output',
        'base_job_name': 'v8-enhanced-training',
        'hyperparameters': {
            'epochs': 100,
            'batch-size': 256,
            'learning-rate': 0.001,
            'all': True  # Train all symbols
        },
        'environment': {
            'PYTHONPATH': '/opt/ml/code',
            'CUDA_VISIBLE_DEVICES': '0'
        },
        'tags': [
            {'Key': 'Project', 'Value': 'CRPBot'},
            {'Key': 'Version', 'Value': 'V8'},
            {'Key': 'Purpose', 'Value': 'Fix-V6-Issues'}
        ]
    }
    
    print("\n📋 Training Configuration:")
    print(f"  Instance: {training_config['instance_type']}")
    print(f"  Framework: PyTorch {training_config['framework_version']}")
    print(f"  Max Runtime: {training_config['max_run']/3600:.1f} hours")
    print(f"  Spot Instances: {training_config['use_spot_instances']}")
    print(f"  Volume Size: {training_config['volume_size']} GB")
    
    # Create PyTorch estimator
    estimator = PyTorch(**training_config)
    
    return estimator, bucket, prefix

def upload_training_data(bucket, prefix):
    """Upload training data to S3"""
    
    print("\n📤 Uploading Training Data...")
    
    s3 = boto3.client('s3')
    
    # Data files to upload
    data_files = [
        'btc_data.csv',
        'eth_data.csv', 
        'sol_data.csv',
        'v8_enhanced_training.py',
        'diagnose_v8_models.py'
    ]
    
    uploaded_files = []
    
    for file in data_files:
        if os.path.exists(file):
            s3_key = f'{prefix}/data/{file}'
            try:
                s3.upload_file(file, bucket, s3_key)
                print(f"  ✅ {file} -> s3://{bucket}/{s3_key}")
                uploaded_files.append(file)
            except Exception as e:
                print(f"  ❌ Failed to upload {file}: {e}")
        else:
            print(f"  ⚠️  {file} not found locally")
    
    return uploaded_files

def launch_training():
    """Launch the complete V8 training"""
    
    # Setup
    estimator, bucket, prefix = setup_sagemaker_training()
    
    # Upload data
    uploaded_files = upload_training_data(bucket, prefix)
    
    if len(uploaded_files) < 3:
        print("\n❌ Missing training data files. Please ensure you have:")
        print("  - btc_data.csv")
        print("  - eth_data.csv") 
        print("  - sol_data.csv")
        return None
    
    # Training data S3 path
    training_data = f's3://{bucket}/{prefix}/data'
    
    print(f"\n🎯 Launching V8 Training...")
    print(f"Training Data: {training_data}")
    
    # Start training
    try:
        estimator.fit({'training': training_data}, wait=False)
        
        job_name = estimator.latest_training_job.name
        print(f"\n✅ Training Job Started: {job_name}")
        print(f"Monitor at: https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs/{job_name}")
        
        return estimator
        
    except Exception as e:
        print(f"\n❌ Failed to start training: {e}")
        return None

def monitor_training(estimator):
    """Monitor training progress"""
    
    if not estimator:
        return
    
    job_name = estimator.latest_training_job.name
    
    print(f"\n📊 Monitoring Training Job: {job_name}")
    print("="*50)
    
    # Wait for completion
    try:
        estimator.logs()  # Stream logs
        
        print(f"\n🎉 Training Complete!")
        
        # Download results
        model_artifacts = estimator.model_data
        print(f"Model artifacts: {model_artifacts}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return False

def main():
    """Main execution"""
    
    print("🚀 SageMaker V8 Enhanced Training")
    print("Fixing all V6 model issues with GPU training")
    print("="*60)
    
    # Launch training
    estimator = launch_training()
    
    if estimator:
        print(f"\n✅ Training launched successfully!")
        print(f"Job Name: {estimator.latest_training_job.name}")
        print(f"Instance: ml.g5.xlarge (1x NVIDIA A10G)")
        print(f"Expected Duration: 3-4 hours")
        print(f"Expected Cost: $3-4 (with spot instances)")
        
        # Option to monitor
        monitor = input("\nMonitor training progress? (y/n): ").lower().strip()
        if monitor == 'y':
            monitor_training(estimator)
        else:
            print(f"\nTo monitor later:")
            print(f"aws sagemaker describe-training-job --training-job-name {estimator.latest_training_job.name}")
    
    else:
        print("\n❌ Failed to launch training")

if __name__ == "__main__":
    main()
