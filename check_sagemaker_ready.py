#!/usr/bin/env python3
"""
SageMaker Readiness Check
Verify we have everything needed for V8 training
"""

import os
import boto3

def main():
    print("🔍 SageMaker V8 Training Readiness Check")
    print("="*50)
    
    all_good = True
    
    # 1. Check training data files in S3
    print("\n📊 Training Data Files (S3):")
    bucket = 'crpbot-sagemaker-training'
    required_files = [
        'data/BTC_features.parquet',
        'data/ETH_features.parquet',
        'data/SOL_features.parquet'
    ]

    try:
        s3 = boto3.client('s3')
        for s3_key in required_files:
            try:
                response = s3.head_object(Bucket=bucket, Key=s3_key)
                size_mb = response['ContentLength'] / (1024*1024)
                print(f"  ✅ s3://{bucket}/{s3_key} ({size_mb:.1f} MB)")
            except:
                print(f"  ❌ s3://{bucket}/{s3_key} - MISSING")
                all_good = False
    except Exception as e:
        print(f"  ❌ S3 access error: {e}")
        all_good = False
    
    # 2. Check training scripts
    print("\n🐍 Training Scripts:")
    script_files = ['v8_sagemaker_train.py', 'launch_v8_sagemaker.py']
    
    for file in script_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING")
            all_good = False
    
    # 3. Check AWS credentials
    print("\n🔐 AWS Access:")
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"  ✅ Account: {identity['Account']}")
        print(f"  ✅ User: {identity['Arn'].split('/')[-1]}")
    except Exception as e:
        print(f"  ❌ AWS credentials: {e}")
        all_good = False
    
    # 4. Check S3 bucket access
    print("\n🪣 S3 Bucket:")
    bucket = 'crpbot-sagemaker-training'
    try:
        s3 = boto3.client('s3')
        s3.head_bucket(Bucket=bucket)
        print(f"  ✅ {bucket} - accessible")
    except Exception as e:
        print(f"  ❌ {bucket}: {e}")
        all_good = False
    
    # 5. Check SageMaker access
    print("\n🤖 SageMaker Access:")
    try:
        sagemaker = boto3.client('sagemaker')
        # Try to list training jobs (will work if we have access)
        sagemaker.list_training_jobs(MaxResults=1)
        print(f"  ✅ SageMaker API accessible")
    except Exception as e:
        print(f"  ❌ SageMaker access: {e}")
        all_good = False
    
    # 6. Check IAM role
    print("\n👤 IAM Role:")
    role_arn = 'arn:aws:iam::980104576869:role/service-role/AmazonBraketServiceSageMakerNotebookRole'
    try:
        iam = boto3.client('iam')
        role_name = role_arn.split('/')[-1]
        iam.get_role(RoleName=role_name)
        print(f"  ✅ {role_name} - exists")
    except Exception as e:
        print(f"  ❌ Role check: {e}")
        all_good = False
    
    # Summary
    print("\n" + "="*50)
    if all_good:
        print("🎉 ALL CHECKS PASSED!")
        print("Ready to launch V8 SageMaker training")
        print("\nNext steps:")
        print("  python3 launch_v8_sagemaker.py")
    else:
        print("❌ SOME CHECKS FAILED")
        print("Please fix the issues above before launching training")
    
    print("="*50)

if __name__ == "__main__":
    main()
