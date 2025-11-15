"""
V5 Model Accuracy Investigation Script

Performs local investigation of V5 model performance to help determine
if the 70-74% (first run) or 63-66% (second run) accuracies are correct.
"""
import sys
from pathlib import Path

import pandas as pd
import torch
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_model_info(model_path: Path) -> dict:
    """Load and display model checkpoint information."""
    checkpoint = torch.load(model_path, map_location='cpu')

    info = {
        'path': model_path.name,
        'size_mb': model_path.stat().st_size / (1024 * 1024),
        'keys': list(checkpoint.keys()),
        'has_weights': 'model_state_dict' in checkpoint
    }

    if 'accuracy' in checkpoint:
        info['accuracy'] = checkpoint['accuracy']
    if 'epoch' in checkpoint:
        info['epoch'] = checkpoint['epoch']
    if 'model_state_dict' in checkpoint:
        total_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
        info['parameters'] = total_params
    if 'input_size' in checkpoint:
        info['input_size'] = checkpoint['input_size']
    if 'model_config' in checkpoint:
        info['model_config'] = checkpoint['model_config']

    return info


def compare_model_checkpoints():
    """Compare all available V5 model checkpoints."""

    logger.info("="*70)
    logger.info("V5 MODEL CHECKPOINT COMPARISON")
    logger.info("="*70)

    # Find all V5 models
    models_dir = Path('models')

    v5_models = []
    v5_models.extend(list(models_dir.glob('v5/*.pt')))
    v5_models.extend(list(models_dir.glob('v5_fixed/*.pt')))

    if not v5_models:
        logger.error("No V5 models found!")
        return

    logger.info(f"\nFound {len(v5_models)} V5 model files:\n")

    for model_path in sorted(v5_models):
        logger.info(f"📦 {model_path}")
        info = load_model_info(model_path)

        logger.info(f"   Size: {info['size_mb']:.2f} MB")
        logger.info(f"   Has weights: {'✅ YES' if info['has_weights'] else '❌ NO'}")

        if 'accuracy' in info:
            logger.info(f"   Accuracy: {info['accuracy']:.1%}")
        if 'epoch' in info:
            logger.info(f"   Epoch: {info['epoch']}")
        if 'parameters' in info:
            logger.info(f"   Parameters: {info['parameters']:,}")
        if 'input_size' in info:
            logger.info(f"   Input size: {info['input_size']}")

        logger.info("")

    logger.info("="*70)


def check_training_data():
    """Check training data checksums and metadata."""

    logger.info("\n" + "="*70)
    logger.info("TRAINING DATA VERIFICATION")
    logger.info("="*70)

    training_dir = Path('data/training')

    if not training_dir.exists():
        logger.warning("Training data directory not found!")
        return

    for symbol_dir in sorted(training_dir.glob('*')):
        if not symbol_dir.is_dir():
            continue

        logger.info(f"\n📊 {symbol_dir.name}:")

        # Check metadata
        metadata_file = symbol_dir / 'metadata.json'
        if metadata_file.exists():
            import json
            metadata = json.loads(metadata_file.read_text())
            logger.info(f"   Features: {metadata.get('num_features', 'unknown')}")
            logger.info(f"   Train rows: {metadata.get('train_rows', 'unknown'):,}")
            logger.info(f"   Val rows: {metadata.get('val_rows', 'unknown'):,}")
            logger.info(f"   Test rows: {metadata.get('test_rows', 'unknown'):,}")

        # Check data files
        for split in ['train', 'val', 'test']:
            data_file = symbol_dir / f'{split}.parquet'
            if data_file.exists():
                df = pd.read_parquet(data_file)
                logger.info(f"   {split}.parquet: {len(df):,} rows, {len(df.columns)} cols")

                # Check for issues
                if df.isnull().any().any():
                    null_cols = df.columns[df.isnull().any()].tolist()
                    logger.warning(f"      ⚠️  Null values in: {null_cols[:5]}")

                if (df.select_dtypes(include=['float64', 'float32']).isin([float('inf'), float('-inf')])).any().any():
                    logger.warning(f"      ⚠️  Infinite values detected")

    logger.info("\n" + "="*70)


def search_for_training_logs():
    """Search for any training logs or artifacts."""

    logger.info("\n" + "="*70)
    logger.info("TRAINING LOGS SEARCH")
    logger.info("="*70)

    # Common log locations
    log_locations = [
        'logs/',
        'models/v5/',
        'models/v5_fixed/',
        'training_logs/',
        './',
    ]

    log_files_found = []

    for location in log_locations:
        location_path = Path(location)
        if not location_path.exists():
            continue

        # Look for log files
        for pattern in ['*.log', '*.txt', 'training_*.json', 'history_*.json']:
            log_files_found.extend(list(location_path.glob(pattern)))

    if log_files_found:
        logger.info(f"\n✅ Found {len(log_files_found)} log files:")
        for log_file in log_files_found:
            logger.info(f"   📄 {log_file}")
    else:
        logger.warning("\n❌ No training logs found locally")
        logger.info("   Logs may be on GPU instance or in CloudWatch")

    logger.info("\n" + "="*70)


def check_git_history():
    """Check git history for training configuration changes."""

    logger.info("\n" + "="*70)
    logger.info("GIT HISTORY CHECK")
    logger.info("="*70)

    import subprocess

    # Check recent commits related to training
    try:
        result = subprocess.run(
            ['git', 'log', '--oneline', '--all', '--grep=train', '-10'],
            capture_output=True,
            text=True,
            check=True
        )

        if result.stdout:
            logger.info("\nRecent training-related commits:")
            logger.info(result.stdout)
        else:
            logger.info("\nNo training-related commits found in recent history")

    except subprocess.CalledProcessError as e:
        logger.warning(f"Could not check git history: {e}")

    logger.info("="*70)


def generate_investigation_report():
    """Generate a summary report of the investigation."""

    logger.info("\n" + "="*70)
    logger.info("INVESTIGATION SUMMARY")
    logger.info("="*70)

    logger.info("""
📋 Key Questions to Answer:

1. ACCURACY DISCREPANCY
   - First run: BTC 74.0%, ETH 70.6%, SOL 72.1%
   - Second run: BTC 63.6%, ETH 65.1%, SOL 65.6%
   - Difference: -8 to -10 percentage points

   ❓ Why did accuracy drop?

2. POSSIBLE CAUSES

   A. First run had data leakage
      - Check: Did first run accidentally use test data in training?
      - Check: Were features calculated with look-ahead bias?
      - Evidence needed: First run training script

   B. Second run had training issues
      - Check: Different hyperparameters?
      - Check: Early stopping triggered too early?
      - Check: Optimizer state corrupted?
      - Evidence needed: Compare training configs

   C. First run accuracy was measurement error
      - Check: Was accuracy calculated on correct dataset?
      - Check: Was correct metric logged?
      - Evidence needed: First run logs showing calculation

   D. Models are architecturally different
      - Check: Same model architecture?
      - Check: Same feature engineering?
      - Evidence needed: Model configs from both runs

3. NEXT STEPS BASED ON FINDINGS

   IF first run accuracies were REAL:
   → Retrain to reproduce 70-74% with complete saves

   IF first run accuracies were ERROR:
   → Accept 63-66% models and deploy

   IF unclear:
   → Do third training run with extra validation

📞 AMAZON Q INVESTIGATION NEEDED:
   - Review first training run logs
   - Compare training configurations
   - Verify data between runs
   - Check for any AWS/instance issues
   - Recommend path forward

⏱️  Local investigation complete. Awaiting Amazon Q findings...
""")

    logger.info("="*70)


def main():
    """Run full investigation."""

    logger.info("\n\n")
    logger.info("╔" + "="*68 + "╗")
    logger.info("║" + " "*15 + "V5 ACCURACY INVESTIGATION" + " "*28 + "║")
    logger.info("╚" + "="*68 + "╝")
    logger.info("\n")

    # Run all checks
    compare_model_checkpoints()
    check_training_data()
    search_for_training_logs()
    check_git_history()
    generate_investigation_report()

    logger.info("\n✅ Local investigation complete!")
    logger.info("📨 Review findings above and await Amazon Q investigation")
    logger.info("\n")


if __name__ == '__main__':
    main()
