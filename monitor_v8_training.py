#!/usr/bin/env python3
"""
Monitor V8 SageMaker Training Job
"""

import json
import time
from datetime import datetime

def monitor_training():
    """Monitor the V8 training job"""
    
    job_name = "v8-enhanced-p3-20251116-165424"
    
    print("🔍 V8 SageMaker Training Monitor")
    print("="*50)
    print(f"Job Name: {job_name}")
    print(f"Instance: ml.p3.2xlarge (1x Tesla V100)")
    print(f"Expected Duration: 2-3 hours")
    print(f"Expected Cost: ~$6-9")
    print("="*50)
    
    # Monitor URL
    region = "us-east-1"
    monitor_url = f"https://console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{job_name}"
    print(f"AWS Console: {monitor_url}")
    
    print(f"\nTo check status manually:")
    print(f"aws sagemaker describe-training-job --training-job-name {job_name}")
    
    print(f"\nTraining started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Monitor will show status updates...")

if __name__ == "__main__":
    monitor_training()
