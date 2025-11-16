#!/usr/bin/env python3
"""
Colab File Verification Script
Run this in your Colab notebook to verify files are accessible
"""

import os
import pandas as pd
from pathlib import Path

def verify_colab_setup():
    """Verify all required files are present and accessible."""
    print("🔍 Verifying Colab Setup")
    print("=" * 50)
    
    # Check current directory
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"📁 Directory contents:")
    for item in sorted(os.listdir('.')):
        if os.path.isdir(item):
            print(f"   📂 {item}/")
        else:
            size = os.path.getsize(item) / (1024*1024)  # MB
            print(f"   📄 {item} ({size:.1f} MB)")
    
    print("\n" + "=" * 50)
    
    # Required files
    required_files = [
        "data/features/features_BTC-USD_1m_2025-11-10.parquet",
        "data/features/features_ETH-USD_1m_2025-11-10.parquet", 
        "data/features/features_SOL-USD_1m_2025-11-10.parquet",
        "train_gpu_proper.py"
    ]
    
    print("📋 Checking Required Files:")
    all_present = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"   ✅ {file_path} ({size:.1f} MB)")
            
            # Quick data validation for parquet files
            if file_path.endswith('.parquet'):
                try:
                    df = pd.read_parquet(file_path)
                    print(f"      📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
                    print(f"      📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                except Exception as e:
                    print(f"      ⚠️  Error reading: {e}")
        else:
            print(f"   ❌ MISSING: {file_path}")
            all_present = False
    
    print("\n" + "=" * 50)
    
    # Check Python script
    if os.path.exists("train_gpu_proper.py"):
        print("🐍 Checking Training Script:")
        with open("train_gpu_proper.py", 'r') as f:
            content = f.read()
            
        # Check for key components
        checks = {
            "Feature loading": "features_BTC-USD_1m_2025-11-10.parquet" in content,
            "GPU detection": "torch.cuda.is_available()" in content,
            "Model saving": "torch.save" in content,
            "Multiple coins": all(coin in content for coin in ['BTC', 'ETH', 'SOL'])
        }
        
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {check}")
    
    print("\n" + "=" * 50)
    
    # GPU Check
    try:
        import torch
        print("🚀 GPU Status:")
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU device: {torch.cuda.get_device_name(0)}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("   ⚠️  No GPU detected - training will be slow!")
    except ImportError:
        print("   ❌ PyTorch not installed")
    
    print("\n" + "=" * 50)
    
    # Final status
    if all_present:
        print("🎉 ALL FILES PRESENT - Ready to train!")
        print("\nNext steps:")
        print("1. Ensure GPU runtime is enabled (Runtime → Change runtime type → GPU)")
        print("2. Run: !python train_gpu_proper.py")
        print("3. Wait 40-50 minutes for training to complete")
        return True
    else:
        print("❌ MISSING FILES - Upload required files first")
        print("\nRequired uploads:")
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"   - {file_path}")
        return False

if __name__ == "__main__":
    verify_colab_setup()
