"""
CRPBot LSTM Retraining on Colab Pro GPU - Improved Models
Complete script - copy entire file into Colab cell
Estimated time: 30 minutes for 3 models (BTC, ETH, SOL)

Architecture Improvements:
- Hidden size: 64 → 128
- Num layers: 2 → 3
- Dropout: 0.2 → 0.35
- Epochs: 15 → 50
- Added: LR scheduler, weighted loss, increased early stopping patience
"""

# ============================================================================
# STEP 1: Install Dependencies
# ============================================================================
print("📦 Installing dependencies...")
!pip install -q torch pandas pyarrow scikit-learn boto3 pyyaml tqdm awscli

print("✅ Dependencies installed")

# ============================================================================
# STEP 2: Setup AWS Credentials
# ============================================================================
import os
from google.colab import userdata

# Get credentials from Colab Secrets
# Setup: Click 🔑 in left sidebar → Add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
os.environ['AWS_ACCESS_KEY_ID'] = userdata.get('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = userdata.get('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

print("✅ AWS credentials configured")

# ============================================================================
# STEP 3: Clone Repository
# ============================================================================
print("\n📂 Cloning CRPBot repository...")

# Remove old directory if exists
!rm -rf crpbot

!git clone https://github.com/imnuman/crpbot.git
%cd crpbot

# Install project
!pip install -q -e .

print("✅ Repository cloned and installed")

# ============================================================================
# STEP 4: Setup Data from Google Drive
# ============================================================================
print("\n📊 Setting up training data...")

from google.colab import drive
from pathlib import Path
import shutil

# Mount Google Drive
drive.mount('/content/drive')

# Create data directories
Path('data/features').mkdir(parents=True, exist_ok=True)

# Copy from Google Drive
drive_data_path = '/content/drive/MyDrive/crpbot/data/features'

if Path(drive_data_path).exists():
    print(f"✅ Found data in Google Drive: {drive_data_path}")

    for symbol in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
        src = f'{drive_data_path}/features_{symbol}_1m_latest.parquet'
        dst = f'data/features/features_{symbol}_1m_latest.parquet'

        if Path(src).exists():
            shutil.copy(src, dst)
            size_mb = Path(dst).stat().st_size / (1024 * 1024)
            print(f"  ✅ {symbol}: {size_mb:.1f} MB copied from Drive")
        else:
            print(f"  ⚠️ {symbol}: Not found in Drive")

    print("✅ Data loaded from Google Drive")
else:
    print(f"❌ Google Drive path not found: {drive_data_path}")
    print("Please ensure data is uploaded to Google Drive first")

# Verify all data present
print("\n🔍 Verifying data files...")
for symbol in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
    file_path = f'data/features/features_{symbol}_1m_latest.parquet'
    if Path(file_path).exists():
        size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        print(f"  ✅ {symbol}: {size_mb:.1f} MB ready")
    else:
        print(f"  ❌ {symbol}: MISSING!")
        raise FileNotFoundError(f"Missing data file: {file_path}")

print("✅ All data files verified and ready")

# ============================================================================
# STEP 5: Check GPU Availability
# ============================================================================
print("\n🔍 Checking GPU availability...")
import torch

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU available: {gpu_name}")
    print(f"   Memory: {gpu_memory:.2f} GB")
    device = "cuda"
else:
    print("❌ WARNING: No GPU detected! Training will be slow.")
    device = "cpu"

# ============================================================================
# STEP 6: Train All 3 LSTM Models on GPU (IMPROVED ARCHITECTURE)
# ============================================================================
print("\n" + "="*70)
print("🚀 Starting IMPROVED GPU Training for 3 LSTM Models")
print("Architecture: 128 hidden units, 3 layers, 0.35 dropout, 50 epochs")
print("="*70)

import time
import subprocess

start_time = time.time()
results = {}

for symbol in ['BTC', 'ETH', 'SOL']:
    print(f"\n{'='*70}")
    print(f"Training {symbol}-USD LSTM Model (IMPROVED) on GPU")
    print(f"{'='*70}\n")

    model_start = time.time()

    # Train model using subprocess to capture output
    # Note: NO --device argument per Amazon Q guidance
    result = subprocess.run(
        [
            'python', 'apps/trainer/main.py',
            '--task', 'lstm',
            '--coin', symbol,
            '--epochs', '50'  # ← 50 epochs instead of 15
        ],
        capture_output=True,
        text=True
    )

    # Print training output
    print(result.stdout)
    if result.stderr:
        print("Errors/Warnings:")
        print(result.stderr)

    model_time = time.time() - model_start
    results[symbol] = {
        'time': model_time,
        'success': result.returncode == 0
    }

    if result.returncode == 0:
        print(f"\n✅ {symbol}-USD training complete in {model_time/60:.1f} minutes!")
    else:
        print(f"\n❌ {symbol}-USD training FAILED!")

total_time = time.time() - start_time

print(f"\n{'='*70}")
print(f"Training Summary (IMPROVED MODELS)")
print(f"{'='*70}")
for symbol, info in results.items():
    status = "✅ SUCCESS" if info['success'] else "❌ FAILED"
    print(f"{symbol}-USD: {status} ({info['time']/60:.1f} min)")
print(f"Total time: {total_time/60:.1f} minutes")
print(f"{'='*70}\n")

# ============================================================================
# STEP 7: List Trained Models
# ============================================================================
print("\n📋 Trained models:")
!ls -lh models/*.pt

# ============================================================================
# STEP 8: Upload Models to S3
# ============================================================================
print("\n📤 Uploading models to S3...")

result = subprocess.run(
    ['aws', 's3', 'sync', 'models/', 's3://crpbot-market-data-dev/models/',
     '--exclude', '*', '--include', '*.pt'],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print("✅ Models uploaded to S3 successfully!")
    print(result.stdout)
else:
    print("❌ Upload failed:")
    print(result.stderr)

# ============================================================================
# STEP 9: Verify Upload
# ============================================================================
print("\n🔍 Verifying S3 upload...")
result = subprocess.run(
    ['aws', 's3', 'ls', 's3://crpbot-market-data-dev/models/',
     '--recursive', '--human-readable'],
    capture_output=True,
    text=True
)
print(result.stdout)

# ============================================================================
# STEP 10: Save to Google Drive (Backup)
# ============================================================================
print("\n💾 Backing up models to Google Drive...")

drive_models_path = '/content/drive/MyDrive/crpbot/models'
Path(drive_models_path).mkdir(parents=True, exist_ok=True)

for model_file in Path('models').glob('*.pt'):
    dst = Path(drive_models_path) / model_file.name
    shutil.copy(model_file, dst)
    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"  ✅ {model_file.name}: {size_mb:.1f} MB backed up to Drive")

print("✅ Models backed up to Google Drive")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("🎉 IMPROVED MODEL TRAINING COMPLETE!")
print("="*70)
print(f"Total training time: {total_time/60:.1f} minutes")
print(f"Models trained: {len([r for r in results.values() if r['success']])}/3")
print(f"Architecture: 128 hidden units, 3 layers, 0.35 dropout, 50 epochs")
print(f"\nModels uploaded to:")
print(f"  - S3: s3://crpbot-market-data-dev/models/")
print(f"  - Google Drive: {drive_models_path}")
print("\n📋 Next steps:")
print("  1. Notify Cloud Claude that training is complete")
print("  2. Cloud Claude downloads models from S3")
print("  3. Evaluate models against 68% promotion gate")
print("  4. Check if models now pass promotion criteria")
print("="*70)
