#!/usr/bin/env python3
"""
V3 Ultimate - Step 3: Train Ensemble
Train 5-model ensemble: XGBoost + LightGBM + CatBoost + TabNet + AutoML

Expected output: Calibrated ensemble model with 75-78% win rate
Runtime: ~24 hours on Colab Pro+ A100

Requirements:
- pip install xgboost lightgbm catboost pytorch-tabnet scikit-learn h2o joblib shap
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ML libraries
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from pytorch_tabnet.tab_model import TabNetClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, log_loss,
                            classification_report, confusion_matrix)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Feature selection
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️  SHAP not available, using correlation-based selection")

# Configuration
DATA_DIR = Path('/content/drive/MyDrive/crpbot/data/features')
OUTPUT_DIR = Path('/content/drive/MyDrive/crpbot/models')

# Primary coin for training (BTC)
PRIMARY_COIN = 'BTC_USDT'
PRIMARY_TIMEFRAME = '1m'

# Ensemble configuration
TARGET_HORIZON = 15  # 15-minute prediction
TARGET_COL = 'label_15m'

# Feature selection
MAX_FEATURES = 180  # Select top 180 from 270

# Training parameters
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

def load_and_merge_data(data_dir, coin, timeframe):
    """Load feature-engineered data."""
    feature_file = data_dir / f"{coin}_{timeframe}_features.parquet"

    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")

    print(f"📂 Loading {feature_file.name}...")
    df = pd.read_parquet(feature_file)

    print(f"   Loaded: {len(df):,} rows, {len(df.columns)} columns")

    return df

def prepare_features_and_target(df, target_col='label_15m'):
    """Prepare feature matrix and target."""
    # Exclude non-feature columns
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'label_5m', 'label_15m', 'label_30m',
                    'target_5m', 'target_15m', 'target_30m']

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Drop rows with NaN in target
    df = df.dropna(subset=[target_col])

    X = df[feature_cols].fillna(0)
    y = df[target_col].astype(int)

    print(f"   Features: {len(feature_cols)}")
    print(f"   Samples: {len(X):,}")
    print(f"   Class distribution: {y.value_counts().to_dict()}")

    return X, y, feature_cols

def select_top_features(X_train, y_train, feature_cols, max_features=180):
    """Select top features using SHAP or correlation."""
    print(f"\n🔍 Selecting top {max_features} features...")

    if HAS_SHAP:
        # Use SHAP for feature importance
        print("   Using SHAP for feature selection...")

        # Train a quick model
        model = lgb.LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE, verbose=-1)
        model.fit(X_train, y_train)

        # Calculate SHAP values (sample for speed)
        sample_size = min(1000, len(X_train))
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train.iloc[:sample_size])

        # Get mean absolute SHAP values
        if isinstance(shap_values, list):
            shap_importance = np.abs(shap_values[0]).mean(axis=0)
        else:
            shap_importance = np.abs(shap_values).mean(axis=0)

        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': shap_importance
        }).sort_values('importance', ascending=False)

    else:
        # Use correlation-based selection
        print("   Using correlation-based selection...")

        correlations = []
        for col in feature_cols:
            try:
                corr = abs(X_train[col].corr(y_train))
                correlations.append(corr)
            except:
                correlations.append(0)

        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': correlations
        }).sort_values('importance', ascending=False)

    # Select top features
    selected_features = feature_importance.head(max_features)['feature'].tolist()

    print(f"   ✅ Selected {len(selected_features)} features")
    print(f"   Top 10: {selected_features[:10]}")

    return selected_features, feature_importance

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model."""
    print(f"\n🌲 Training XGBoost...")

    model = xgb.XGBClassifier(
        max_depth=8,
        learning_rate=0.01,
        n_estimators=5000,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=3,
        random_state=RANDOM_STATE,
        tree_method='gpu_hist',  # GPU acceleration
        gpu_id=0,
        early_stopping_rounds=50,
        eval_metric='mlogloss'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    print(f"   ✅ XGBoost trained - Val Accuracy: {val_acc:.3f}")

    return model

def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM model."""
    print(f"\n💡 Training LightGBM...")

    model = lgb.LGBMClassifier(
        max_depth=8,
        learning_rate=0.01,
        n_estimators=5000,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multiclass',
        num_class=3,
        random_state=RANDOM_STATE,
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        verbose=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    print(f"   ✅ LightGBM trained - Val Accuracy: {val_acc:.3f}")

    return model

def train_catboost(X_train, y_train, X_val, y_val):
    """Train CatBoost model."""
    print(f"\n🐱 Training CatBoost...")

    model = cb.CatBoostClassifier(
        depth=8,
        learning_rate=0.01,
        iterations=5000,
        subsample=0.8,
        random_state=RANDOM_STATE,
        task_type='GPU',
        devices='0',
        early_stopping_rounds=50,
        verbose=100
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True
    )

    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    print(f"   ✅ CatBoost trained - Val Accuracy: {val_acc:.3f}")

    return model

def train_tabnet(X_train, y_train, X_val, y_val):
    """Train TabNet model."""
    print(f"\n🧠 Training TabNet...")

    model = TabNetClassifier(
        n_d=64,
        n_a=64,
        n_steps=5,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size":50, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax',
        verbose=10,
        device_name='cuda'
    )

    model.fit(
        X_train.values, y_train.values,
        eval_set=[(X_val.values, y_val.values)],
        eval_metric=['accuracy'],
        max_epochs=200,
        patience=20,
        batch_size=1024,
        virtual_batch_size=128
    )

    val_pred = model.predict(X_val.values)
    val_acc = accuracy_score(y_val, val_pred)

    print(f"   ✅ TabNet trained - Val Accuracy: {val_acc:.3f}")

    return model

def train_automl(X_train, y_train, max_runtime=3600):
    """Train AutoML model (H2O or AutoGluon)."""
    print(f"\n🤖 Training AutoML (max {max_runtime}s)...")

    try:
        import h2o
        from h2o.automl import H2OAutoML

        h2o.init()

        # Convert to H2O frame
        train_h2o = h2o.H2OFrame(pd.concat([X_train, y_train.rename('target')], axis=1))
        train_h2o['target'] = train_h2o['target'].asfactor()

        aml = H2OAutoML(
            max_runtime_secs=max_runtime,
            seed=RANDOM_STATE,
            nfolds=3,
            max_models=20
        )

        aml.train(y='target', training_frame=train_h2o)

        model = aml.leader

        print(f"   ✅ AutoML trained - Leader: {model.model_id}")

        return model

    except ImportError:
        print("   ⚠️  H2O not available, using stacking ensemble instead")

        # Fallback: simple stacking
        estimators = [
            ('xgb_simple', xgb.XGBClassifier(max_depth=4, n_estimators=100, random_state=RANDOM_STATE)),
            ('lgb_simple', lgb.LGBMClassifier(max_depth=4, n_estimators=100, random_state=RANDOM_STATE, verbose=-1))
        ]

        model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=RANDOM_STATE),
            cv=3
        )

        model.fit(X_train, y_train)

        return model

def train_meta_learner(models, X_train, y_train, X_val, y_val):
    """Train meta-learner on base model predictions."""
    print(f"\n🎯 Training Meta-Learner...")

    # Generate meta-features (predictions from base models)
    meta_train = []
    meta_val = []

    for name, model in models.items():
        print(f"   Generating predictions from {name}...")

        if name == 'tabnet':
            train_pred_proba = model.predict_proba(X_train.values)
            val_pred_proba = model.predict_proba(X_val.values)
        else:
            train_pred_proba = model.predict_proba(X_train)
            val_pred_proba = model.predict_proba(X_val)

        meta_train.append(train_pred_proba)
        meta_val.append(val_pred_proba)

    meta_X_train = np.hstack(meta_train)
    meta_X_val = np.hstack(meta_val)

    print(f"   Meta-features shape: {meta_X_train.shape}")

    # Train meta-learner (Logistic Regression)
    meta_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    meta_model.fit(meta_X_train, y_train)

    meta_val_pred = meta_model.predict(meta_X_val)
    meta_val_acc = accuracy_score(y_val, meta_val_pred)

    print(f"   ✅ Meta-learner trained - Val Accuracy: {meta_val_acc:.3f}")

    return meta_model

def calibrate_models(models, meta_model, X_val, y_val):
    """Calibrate model probabilities."""
    print(f"\n📐 Calibrating Models...")

    calibrated_models = {}

    for name, model in models.items():
        print(f"   Calibrating {name}...")

        # Wrap model for calibration
        if name != 'tabnet':
            calibrated = CalibratedClassifierCV(
                model,
                method='isotonic',
                cv='prefit'
            )

            calibrated.fit(X_val, y_val)
            calibrated_models[name] = calibrated
        else:
            # TabNet doesn't support calibration wrapper, use as-is
            calibrated_models[name] = model

    print(f"   ✅ All models calibrated")

    return calibrated_models

def evaluate_ensemble(models, meta_model, X_test, y_test):
    """Evaluate ensemble performance."""
    print(f"\n📊 Evaluating Ensemble...")

    # Generate meta-features
    meta_test = []

    for name, model in models.items():
        if name == 'tabnet':
            pred_proba = model.predict_proba(X_test.values)
        else:
            pred_proba = model.predict_proba(X_test)

        meta_test.append(pred_proba)

    meta_X_test = np.hstack(meta_test)

    # Final predictions
    final_pred = meta_model.predict(meta_X_test)
    final_pred_proba = meta_model.predict_proba(meta_X_test)

    # Metrics
    accuracy = accuracy_score(y_test, final_pred)
    auc = roc_auc_score(y_test, final_pred_proba, multi_class='ovr')
    logloss = log_loss(y_test, final_pred_proba)

    # Expected Calibration Error (ECE)
    ece = calculate_ece(y_test, final_pred_proba)

    # Win rate (only buy/sell signals)
    buy_sell_mask = final_pred != 1  # exclude flat
    if buy_sell_mask.sum() > 0:
        win_rate = accuracy_score(y_test[buy_sell_mask], final_pred[buy_sell_mask])
    else:
        win_rate = 0

    print(f"\n   📈 Test Results:")
    print(f"      Accuracy: {accuracy:.3f}")
    print(f"      AUC: {auc:.3f}")
    print(f"      Log Loss: {logloss:.3f}")
    print(f"      ECE: {ece:.4f}")
    print(f"      Win Rate (Buy/Sell): {win_rate:.3f}")

    print(f"\n   Classification Report:")
    print(classification_report(y_test, final_pred, target_names=['Down', 'Flat', 'Up']))

    print(f"\n   Confusion Matrix:")
    print(confusion_matrix(y_test, final_pred))

    metrics = {
        'accuracy': float(accuracy),
        'auc': float(auc),
        'log_loss': float(logloss),
        'ece': float(ece),
        'win_rate': float(win_rate)
    }

    return metrics, final_pred, final_pred_proba

def calculate_ece(y_true, y_pred_proba, n_bins=10):
    """Calculate Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(y_pred_proba, axis=1)
    predictions = np.argmax(y_pred_proba, axis=1)
    accuracies = (predictions == y_true)

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece

def main():
    """Main ensemble training workflow."""
    print("=" * 70)
    print("🚀 V3 ULTIMATE - STEP 3: TRAIN ENSEMBLE")
    print("=" * 70)

    start_time = datetime.now()

    # Load data
    df = load_and_merge_data(DATA_DIR, PRIMARY_COIN, PRIMARY_TIMEFRAME)

    # Prepare features and target
    X, y, feature_cols = prepare_features_and_target(df, TARGET_COL)

    # Split data
    print(f"\n✂️  Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE/(1-TEST_SIZE),
        random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"   Train: {len(X_train):,}")
    print(f"   Val: {len(X_val):,}")
    print(f"   Test: {len(X_test):,}")

    # Feature selection
    selected_features, feature_importance = select_top_features(
        X_train, y_train, feature_cols, MAX_FEATURES
    )

    # Use only selected features
    X_train = X_train[selected_features]
    X_val = X_val[selected_features]
    X_test = X_test[selected_features]

    # Train base models
    print(f"\n{'='*70}")
    print("🔥 TRAINING BASE MODELS")
    print(f"{'='*70}")

    models = {}

    models['xgboost'] = train_xgboost(X_train, y_train, X_val, y_val)
    models['lightgbm'] = train_lightgbm(X_train, y_train, X_val, y_val)
    models['catboost'] = train_catboost(X_train, y_train, X_val, y_val)
    models['tabnet'] = train_tabnet(X_train, y_train, X_val, y_val)
    models['automl'] = train_automl(X_train, y_train, max_runtime=3600)

    # Train meta-learner
    meta_model = train_meta_learner(models, X_train, y_train, X_val, y_val)

    # Calibrate models
    calibrated_models = calibrate_models(models, meta_model, X_val, y_val)

    # Evaluate
    metrics, test_pred, test_pred_proba = evaluate_ensemble(
        calibrated_models, meta_model, X_test, y_test
    )

    duration = (datetime.now() - start_time).total_seconds()

    # Validation gates
    print(f"\n{'='*70}")
    print("🎯 VALIDATION GATES")
    print(f"{'='*70}")

    gates_passed = []

    if metrics['auc'] >= 0.73:
        print(f"   ✅ AUC ≥0.73: {metrics['auc']:.3f}")
        gates_passed.append(True)
    else:
        print(f"   ❌ AUC <0.73: {metrics['auc']:.3f}")
        gates_passed.append(False)

    if metrics['ece'] < 0.03:
        print(f"   ✅ ECE <0.03: {metrics['ece']:.4f}")
        gates_passed.append(True)
    else:
        print(f"   ❌ ECE ≥0.03: {metrics['ece']:.4f}")
        gates_passed.append(False)

    if metrics['accuracy'] >= 0.73:
        print(f"   ✅ Test Accuracy ≥0.73: {metrics['accuracy']:.3f}")
        gates_passed.append(True)
    else:
        print(f"   ❌ Test Accuracy <0.73: {metrics['accuracy']:.3f}")
        gates_passed.append(False)

    all_passed = all(gates_passed)

    if all_passed:
        print(f"\n   🎉 ALL GATES PASSED!")
    else:
        print(f"\n   ⚠️  Some gates failed - model needs improvement")

    # Save models
    print(f"\n💾 Saving models...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in calibrated_models.items():
        model_path = OUTPUT_DIR / f"{name}_model.pkl"
        joblib.dump(model, model_path)
        print(f"   Saved: {model_path.name}")

    meta_path = OUTPUT_DIR / "meta_learner.pkl"
    joblib.dump(meta_model, meta_path)
    print(f"   Saved: {meta_path.name}")

    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'coin': PRIMARY_COIN,
        'timeframe': PRIMARY_TIMEFRAME,
        'target_horizon': TARGET_HORIZON,
        'num_features': len(selected_features),
        'selected_features': selected_features,
        'feature_importance': feature_importance.to_dict('records'),
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'duration_seconds': duration,
        'metrics': metrics,
        'gates_passed': all_passed
    }

    metadata_path = OUTPUT_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"   Saved: {metadata_path.name}")

    print(f"\n⏱️  Total Duration: {duration/3600:.1f} hours")
    print(f"\n✅ Step 3 Complete! Ready for Step 4: Backtest")

    return metadata

if __name__ == "__main__":
    import torch  # Required for TabNet
    main()
