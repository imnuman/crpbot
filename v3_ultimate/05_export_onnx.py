#!/usr/bin/env python3
"""
V3 Ultimate - Step 5: Export ONNX
Convert trained models to ONNX format and upload to production S3.

Expected output: ONNX models ready for production deployment
Runtime: ~1 hour

Requirements:
- pip install onnx onnxruntime skl2onnx boto3
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

# ONNX conversion
try:
    import onnx
    import onnxruntime as rt
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    HAS_ONNX = True
except ImportError:
    print("⚠️  ONNX libraries not available")
    HAS_ONNX = False

# AWS S3
try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    print("⚠️  boto3 not available, will skip S3 upload")
    HAS_BOTO3 = False

# Configuration
MODELS_DIR = Path('/content/drive/MyDrive/crpbot/models')
OUTPUT_DIR = Path('/content/drive/MyDrive/crpbot/models/onnx')
BACKTEST_DIR = Path('/content/drive/MyDrive/crpbot/backtest')

# S3 configuration
S3_BUCKET = 'crpbot-models-production'
S3_PREFIX = 'v3_ultimate'

def convert_model_to_onnx(model, model_name, num_features, output_dir):
    """Convert sklearn/xgboost model to ONNX."""
    print(f"\n🔄 Converting {model_name} to ONNX...")

    if not HAS_ONNX:
        print(f"   ❌ ONNX not available, skipping")
        return None

    try:
        # Define input type
        initial_type = [('float_input', FloatTensorType([None, num_features]))]

        # Convert to ONNX
        onnx_model = convert_sklearn(
            model,
            initial_types=initial_type,
            target_opset=12
        )

        # Save ONNX model
        output_path = output_dir / f"{model_name}.onnx"
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"   ✅ Saved: {output_path.name}")

        # Validate ONNX model
        print(f"   Validating ONNX model...")
        sess = rt.InferenceSession(str(output_path))

        # Test inference
        test_input = np.random.randn(1, num_features).astype(np.float32)
        input_name = sess.get_inputs()[0].name
        output = sess.run(None, {input_name: test_input})

        print(f"   ✅ ONNX model validated (output shape: {output[0].shape})")

        return output_path

    except Exception as e:
        print(f"   ❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def package_deployment_bundle(models_dir, backtest_dir, output_dir):
    """Package all models and metadata for deployment."""
    print(f"\n📦 Packaging deployment bundle...")

    bundle = {
        'version': 'v3_ultimate',
        'timestamp': datetime.now().isoformat(),
        'models': [],
        'files': []
    }

    # Copy model files
    for model_name in ['xgboost', 'lightgbm', 'catboost', 'tabnet', 'automl', 'meta_learner']:
        pkl_file = models_dir / f"{model_name}.pkl"
        onnx_file = output_dir / f"{model_name}.onnx"

        if pkl_file.exists():
            bundle['models'].append({
                'name': model_name,
                'pkl_path': str(pkl_file),
                'onnx_path': str(onnx_file) if onnx_file.exists() else None
            })
            bundle['files'].append(str(pkl_file))
            if onnx_file.exists():
                bundle['files'].append(str(onnx_file))

    # Include metadata
    metadata_file = models_dir / 'metadata.json'
    if metadata_file.exists():
        bundle['files'].append(str(metadata_file))

    # Include backtest results
    backtest_summary = backtest_dir / 'backtest_summary.json'
    backtest_results = backtest_dir / 'backtest_results.csv'

    if backtest_summary.exists():
        bundle['files'].append(str(backtest_summary))

    if backtest_results.exists():
        bundle['files'].append(str(backtest_results))

    # Save bundle manifest
    bundle_path = output_dir / 'deployment_bundle.json'
    with open(bundle_path, 'w') as f:
        json.dump(bundle, f, indent=2)

    print(f"   ✅ Bundle manifest: {bundle_path}")
    print(f"   Files in bundle: {len(bundle['files'])}")

    return bundle

def upload_to_s3(bundle, s3_bucket, s3_prefix):
    """Upload models and metadata to S3."""
    print(f"\n☁️  Uploading to S3...")
    print(f"   Bucket: {s3_bucket}")
    print(f"   Prefix: {s3_prefix}")

    if not HAS_BOTO3:
        print(f"   ❌ boto3 not available, skipping upload")
        return False

    try:
        s3_client = boto3.client('s3')

        uploaded_files = []

        for file_path in bundle['files']:
            file_path = Path(file_path)

            if not file_path.exists():
                print(f"   ⚠️  File not found: {file_path}, skipping")
                continue

            s3_key = f"{s3_prefix}/{file_path.name}"

            print(f"   Uploading {file_path.name}...")

            s3_client.upload_file(
                str(file_path),
                s3_bucket,
                s3_key
            )

            uploaded_files.append(s3_key)
            print(f"      ✅ s3://{s3_bucket}/{s3_key}")

        print(f"\n   ✅ Uploaded {len(uploaded_files)} files to S3")

        # Upload bundle manifest
        bundle_path = Path(bundle['files'][0]).parent / 'deployment_bundle.json'
        if bundle_path.exists():
            s3_key = f"{s3_prefix}/deployment_bundle.json"
            s3_client.upload_file(
                str(bundle_path),
                s3_bucket,
                s3_key
            )
            print(f"   ✅ Uploaded bundle manifest: s3://{s3_bucket}/{s3_key}")

        return True

    except Exception as e:
        print(f"   ❌ S3 upload failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_deployment(models_dir, backtest_dir):
    """Final validation before deployment."""
    print(f"\n✅ Final Deployment Validation")
    print("=" * 70)

    checks = []

    # Check models exist
    required_models = ['xgboost', 'lightgbm', 'catboost', 'tabnet', 'automl', 'meta_learner']

    print(f"\n1️⃣  Checking required models...")
    for model_name in required_models:
        model_path = models_dir / f"{model_name}.pkl"
        if model_path.exists():
            print(f"   ✅ {model_name}.pkl")
            checks.append(True)
        else:
            print(f"   ❌ {model_name}.pkl MISSING")
            checks.append(False)

    # Check metadata
    print(f"\n2️⃣  Checking metadata...")
    metadata_path = models_dir / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        print(f"   ✅ metadata.json")
        print(f"      Features: {metadata.get('num_features', 'N/A')}")
        print(f"      Test Accuracy: {metadata.get('metrics', {}).get('accuracy', 'N/A'):.3f}")

        if metadata.get('gates_passed', False):
            print(f"   ✅ Training gates passed")
            checks.append(True)
        else:
            print(f"   ⚠️  Training gates not passed")
            checks.append(False)
    else:
        print(f"   ❌ metadata.json MISSING")
        checks.append(False)

    # Check backtest results
    print(f"\n3️⃣  Checking backtest results...")
    backtest_summary_path = backtest_dir / 'backtest_summary.json'
    backtest_results_path = backtest_dir / 'backtest_results.csv'

    if backtest_summary_path.exists():
        with open(backtest_summary_path, 'r') as f:
            backtest = json.load(f)

        print(f"   ✅ backtest_summary.json")
        print(f"      Win Rate: {backtest.get('metrics', {}).get('win_rate', 'N/A'):.1%}")
        print(f"      Sharpe: {backtest.get('metrics', {}).get('sharpe_ratio', 'N/A'):.2f}")
        print(f"      Total Trades: {backtest.get('metrics', {}).get('total_trades', 'N/A'):,}")

        if backtest.get('gates_passed', False):
            print(f"   ✅ Backtest gates passed")
            checks.append(True)
        else:
            print(f"   ⚠️  Backtest gates not passed")
            checks.append(False)
    else:
        print(f"   ❌ backtest_summary.json MISSING")
        checks.append(False)

    if backtest_results_path.exists():
        print(f"   ✅ backtest_results.csv")
        checks.append(True)
    else:
        print(f"   ❌ backtest_results.csv MISSING")
        checks.append(False)

    # Final verdict
    all_passed = all(checks)

    print(f"\n" + "=" * 70)
    if all_passed:
        print(f"✅ ALL VALIDATION CHECKS PASSED")
        print(f"🚀 Models are ready for production deployment!")
    else:
        print(f"⚠️  SOME VALIDATION CHECKS FAILED")
        print(f"   Please review the issues above before deploying.")

    return all_passed

def main():
    """Main ONNX export workflow."""
    print("=" * 70)
    print("🚀 V3 ULTIMATE - STEP 5: EXPORT ONNX")
    print("=" * 70)

    start_time = datetime.now()

    # Load metadata to get feature count
    metadata_path = MODELS_DIR / 'metadata.json'
    if not metadata_path.exists():
        print("❌ Metadata not found! Cannot determine feature count.")
        return False

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    num_features = metadata['num_features']
    print(f"\n📊 Model Configuration:")
    print(f"   Features: {num_features}")

    # Convert models to ONNX
    print(f"\n{'='*70}")
    print("🔄 CONVERTING MODELS TO ONNX")
    print(f"{'='*70}")

    model_names = ['xgboost', 'lightgbm', 'catboost', 'automl', 'meta_learner']
    # Note: TabNet requires special handling, skip for now

    onnx_models = {}

    for model_name in model_names:
        model_path = MODELS_DIR / f"{model_name}.pkl"

        if not model_path.exists():
            print(f"\n⚠️  {model_name}.pkl not found, skipping")
            continue

        model = joblib.load(model_path)

        onnx_path = convert_model_to_onnx(
            model=model,
            model_name=model_name,
            num_features=num_features,
            output_dir=OUTPUT_DIR
        )

        if onnx_path:
            onnx_models[model_name] = onnx_path

    print(f"\n✅ Converted {len(onnx_models)}/{len(model_names)} models to ONNX")

    # Package deployment bundle
    bundle = package_deployment_bundle(MODELS_DIR, BACKTEST_DIR, OUTPUT_DIR)

    # Upload to S3
    upload_success = upload_to_s3(bundle, S3_BUCKET, S3_PREFIX)

    # Final validation
    deployment_ready = validate_deployment(MODELS_DIR, BACKTEST_DIR)

    duration = (datetime.now() - start_time).total_seconds()

    # Summary
    print(f"\n" + "=" * 70)
    print("📋 EXPORT SUMMARY")
    print("=" * 70)
    print(f"\n⏱️  Duration: {duration/60:.1f} minutes")
    print(f"🔄 ONNX Models: {len(onnx_models)}")
    print(f"☁️  S3 Upload: {'✅ Success' if upload_success else '❌ Failed'}")
    print(f"✅ Deployment Ready: {'✅ Yes' if deployment_ready else '⚠️  No'}")

    if deployment_ready:
        print(f"\n🎉 V3 ULTIMATE COMPLETE!")
        print(f"\n📦 Deployment Bundle Location:")
        print(f"   Local: {OUTPUT_DIR}")
        if upload_success:
            print(f"   S3: s3://{S3_BUCKET}/{S3_PREFIX}/")

        print(f"\n🚀 Next Steps:")
        print(f"   1. Download ONNX models from S3")
        print(f"   2. Deploy to production infrastructure")
        print(f"   3. Configure real-time data feeds")
        print(f"   4. Start paper trading to validate")
        print(f"   5. Monitor performance and recalibrate as needed")

    print(f"\n✅ Step 5 Complete!")

    return deployment_ready

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
