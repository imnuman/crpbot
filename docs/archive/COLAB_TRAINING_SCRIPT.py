"""
CRPBot LSTM Training on Colab Pro GPU
Complete script - copy entire file into Colab cell
"""

# ============================================================================
# STEP 1: Install Dependencies
# ============================================================================
print("📦 Installing dependencies...")
!pip install -q torch pandas pyarrow scikit-learn boto3 pyyaml tqdm

print("✅ Dependencies installed")

# ============================================================================
# STEP 2: Setup AWS Credentials (for S3 upload)
# ============================================================================
import os

# 🔑 REPLACE WITH YOUR AWS CREDENTIALS
os.environ['AWS_ACCESS_KEY_ID'] = 'YOUR_AWS_ACCESS_KEY_ID'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'YOUR_AWS_SECRET_ACCESS_KEY'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

print("✅ AWS credentials configured")

# ============================================================================
# STEP 3: Clone Repository
# ============================================================================
print("\n📂 Cloning CRPBot repository...")
!git clone https://github.com/imnuman/crpbot.git
%cd crpbot

# Install project
!pip install -q -e .

print("✅ Repository cloned and installed")

# ============================================================================
# STEP 4: Setup Data (Option A: From Google Drive OR Option B: From S3)
# ============================================================================
print("\n📊 Setting up training data...")

from pathlib import Path
import shutil

# Create data directories
Path('data/features').mkdir(parents=True, exist_ok=True)

# OPTION A: Copy from Google Drive (if data already there)
drive_data_path = '/content/drive/MyDrive/crpbot/data/features'

if Path(drive_data_path).exists():
    print(f"✅ Found data in Google Drive: {drive_data_path}")

    # Copy feature files
    for symbol in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
        src = f'{drive_data_path}/features_{symbol}_1m_latest.parquet'
        dst = f'data/features/features_{symbol}_1m_latest.parquet'

        if Path(src).exists():
            shutil.copy(src, dst)
            size_mb = Path(dst).stat().st_size / (1024 * 1024)
            print(f"  ✅ {symbol}: {size_mb:.1f} MB copied from Drive")
        else:
            print(f"  ⚠️ {symbol}: Not found in Drive, will download from S3")

    print("✅ Data loaded from Google Drive")

# OPTION B: Download from S3 (fallback if not in Drive)
else:
    print("⚠️ Data not found in Google Drive, downloading from S3...")

    import boto3
    s3 = boto3.client('s3')
    bucket = 'crpbot-market-data-dev'

    for symbol in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
        file_key = f'data/features/features_{symbol}_1m_latest.parquet'
        local_path = f'data/features/features_{symbol}_1m_latest.parquet'

        print(f"  Downloading {symbol}...")
        s3.download_file(bucket, file_key, local_path)

        size_mb = Path(local_path).stat().st_size / (1024 * 1024)
        print(f"  ✅ {symbol}: {size_mb:.1f} MB")

    print("✅ Data downloaded from S3")

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
# STEP 6: Train All 3 LSTM Models on GPU
# ============================================================================
print("\n" + "="*70)
print("🚀 Starting GPU Training for 3 LSTM Models")
print("="*70)

import time
import subprocess

start_time = time.time()
results = {}

for symbol in ['BTC', 'ETH', 'SOL']:
    print(f"\n{'='*70}")
    print(f"Training {symbol}-USD LSTM Model on GPU")
    print(f"{'='*70}\n")

    model_start = time.time()

    # Train model using subprocess to capture output
    result = subprocess.run(
        [
            'python', 'apps/trainer/main.py',
            '--task', 'lstm',
            '--coin', symbol,
            '--epochs', '15',
            '--device', device
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
print(f"Training Summary")
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
print("🎉 COLAB PRO GPU TRAINING COMPLETE!")
print("="*70)
print(f"Total training time: {total_time/60:.1f} minutes")
print(f"Models trained: {len([r for r in results.values() if r['success']])}/3")
print(f"GPU used: {gpu_name if device == 'cuda' else 'CPU (no GPU)'}")
print(f"\nModels uploaded to:")
print(f"  - S3: s3://crpbot-market-data-dev/models/")
print(f"  - Google Drive: {drive_models_path}")
print("\n📋 Next steps:")
print("  1. Notify Cloud Claude that training is complete")
print("  2. Cloud Claude downloads models from S3")
print("  3. Evaluate models against 68% promotion gate")
print("  4. Train Transformer model (next)")
print("="*70)
