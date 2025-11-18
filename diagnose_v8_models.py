#!/usr/bin/env python3
"""
V8 Model Diagnostic - Validate All Fixes
Checks: Feature normalization, confidence calibration, class balance, logit ranges
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import pickle
import argparse
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

def load_v8_model(symbol):
    """Load V8 model and processor"""
    model_path = f'models/v8_enhanced/lstm_{symbol}_v8_enhanced.pt'
    processor_path = f'models/v8_enhanced/processor_{symbol}_v8.pkl'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if not os.path.exists(processor_path):
        raise FileNotFoundError(f"Processor not found: {processor_path}")
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Load processor
    with open(processor_path, 'rb') as f:
        processor = pickle.load(f)
    
    return checkpoint, processor

def create_test_data(symbol, processor, n_samples=1000):
    """Create test data for diagnostic"""
    filename_map = {
        'BTC-USD': 'btc_data.csv',
        'ETH-USD': 'eth_data.csv', 
        'SOL-USD': 'sol_data.csv'
    }
    
    df = pd.read_csv(filename_map[symbol])
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    
    # Use last n_samples for testing
    df_test = df.tail(n_samples + 500)  # Extra for feature calculation
    
    # Transform using processor
    X_scaled, features_df = processor.transform(df_test)
    
    # Take last n_samples
    X_test = X_scaled[-n_samples:]
    
    return X_test, df_test.tail(n_samples)

def reconstruct_v8_model(input_size, checkpoint):
    """Reconstruct V8 model from checkpoint"""
    from v8_enhanced_training import V8TradingNet
    
    model = V8TradingNet(
        input_size=input_size,
        hidden_size=512,
        num_classes=3,
        dropout=0.3
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def diagnose_model(symbol, verbose=True):
    """Comprehensive diagnostic of V8 model"""
    
    if verbose:
        print(f"\n🔍 Diagnosing V8 {symbol} Model")
        print("="*50)
    
    try:
        # Load model and processor
        checkpoint, processor = load_v8_model(symbol)
        
        # Create test data
        X_test, df_test = create_test_data(symbol, processor, n_samples=1000)
        
        # Reconstruct model
        model = reconstruct_v8_model(X_test.shape[1], checkpoint)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_test)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            max_probs = torch.max(probabilities, dim=1)[0]
        
        # Convert to numpy
        outputs_np = outputs.numpy()
        probabilities_np = probabilities.numpy()
        predictions_np = predictions.numpy()
        max_probs_np = max_probs.numpy()
        
        # Diagnostic metrics
        diagnostic = {
            'symbol': symbol,
            'model_version': 'v8_enhanced',
            'diagnostic_date': datetime.now().isoformat(),
            'test_samples': len(X_test),
            'input_features': X_test.shape[1],
            
            # Feature normalization check
            'feature_stats': {
                'mean': float(X_test.mean()),
                'std': float(X_test.std()),
                'min': float(X_test.min()),
                'max': float(X_test.max()),
                'normalized': abs(X_test.mean()) < 0.1 and abs(X_test.std() - 1.0) < 0.1
            },
            
            # Logit analysis
            'logit_stats': {
                'mean': float(outputs_np.mean()),
                'std': float(outputs_np.std()),
                'min': float(outputs_np.min()),
                'max': float(outputs_np.max()),
                'range': float(outputs_np.max() - outputs_np.min()),
                'healthy_range': abs(outputs_np.min()) < 15 and abs(outputs_np.max()) < 15
            },
            
            # Confidence analysis
            'confidence_stats': {
                'mean': float(max_probs_np.mean()),
                'std': float(max_probs_np.std()),
                'min': float(max_probs_np.min()),
                'max': float(max_probs_np.max()),
                'overconfident_99': float(np.mean(max_probs_np > 0.99)),
                'overconfident_95': float(np.mean(max_probs_np > 0.95)),
                'overconfident_90': float(np.mean(max_probs_np > 0.90)),
                'underconfident_50': float(np.mean(max_probs_np < 0.50)),
                'healthy_confidence': np.mean(max_probs_np > 0.99) < 0.1
            },
            
            # Class distribution
            'class_distribution': {
                'sell_pct': float(np.mean(predictions_np == 0)),
                'hold_pct': float(np.mean(predictions_np == 1)),
                'buy_pct': float(np.mean(predictions_np == 2)),
                'balanced': all(0.15 < np.mean(predictions_np == i) < 0.6 for i in range(3))
            },
            
            # Model metadata
            'model_metadata': {
                'training_accuracy': checkpoint.get('accuracy', 'N/A'),
                'temperature': checkpoint.get('temperature', 'N/A'),
                'training_epoch': checkpoint.get('epoch', 'N/A'),
                'training_date': checkpoint.get('training_date', 'N/A')
            }
        }
        
        # Quality gates
        quality_gates = {
            'feature_normalization': diagnostic['feature_stats']['normalized'],
            'logit_range_healthy': diagnostic['logit_stats']['healthy_range'],
            'confidence_calibrated': diagnostic['confidence_stats']['healthy_confidence'],
            'class_balanced': diagnostic['class_distribution']['balanced'],
            'no_nan_inf': not (np.isnan(outputs_np).any() or np.isinf(outputs_np).any())
        }
        
        diagnostic['quality_gates'] = quality_gates
        diagnostic['all_gates_passed'] = all(quality_gates.values())
        
        if verbose:
            print_diagnostic_results(diagnostic)
        
        return diagnostic
        
    except Exception as e:
        error_diagnostic = {
            'symbol': symbol,
            'error': str(e),
            'diagnostic_date': datetime.now().isoformat(),
            'all_gates_passed': False
        }
        
        if verbose:
            print(f"❌ Error diagnosing {symbol}: {e}")
        
        return error_diagnostic

def print_diagnostic_results(diagnostic):
    """Print formatted diagnostic results"""
    symbol = diagnostic['symbol']
    
    print(f"Model: {symbol} ({diagnostic['model_version']})")
    print(f"Test Samples: {diagnostic['test_samples']:,}")
    print(f"Input Features: {diagnostic['input_features']}")
    
    # Feature normalization
    fs = diagnostic['feature_stats']
    status = "✅ PASS" if fs['normalized'] else "❌ FAIL"
    print(f"\n📊 Feature Normalization: {status}")
    print(f"   Mean: {fs['mean']:.6f} (target: ~0.0)")
    print(f"   Std:  {fs['std']:.6f} (target: ~1.0)")
    print(f"   Range: [{fs['min']:.3f}, {fs['max']:.3f}]")
    
    # Logit analysis
    ls = diagnostic['logit_stats']
    status = "✅ PASS" if ls['healthy_range'] else "❌ FAIL"
    print(f"\n🎯 Logit Range: {status}")
    print(f"   Range: [{ls['min']:.1f}, {ls['max']:.1f}] (target: ±15)")
    print(f"   Mean: {ls['mean']:.3f}, Std: {ls['std']:.3f}")
    
    # Confidence analysis
    cs = diagnostic['confidence_stats']
    status = "✅ PASS" if cs['healthy_confidence'] else "❌ FAIL"
    print(f"\n🎲 Confidence Calibration: {status}")
    print(f"   Mean Confidence: {cs['mean']:.1%}")
    print(f"   >99% Confident: {cs['overconfident_99']:.1%} (target: <10%)")
    print(f"   >95% Confident: {cs['overconfident_95']:.1%}")
    print(f"   >90% Confident: {cs['overconfident_90']:.1%}")
    print(f"   <50% Confident: {cs['underconfident_50']:.1%}")
    
    # Class distribution
    cd = diagnostic['class_distribution']
    status = "✅ PASS" if cd['balanced'] else "❌ FAIL"
    print(f"\n⚖️  Class Balance: {status}")
    print(f"   SELL: {cd['sell_pct']:.1%}")
    print(f"   HOLD: {cd['hold_pct']:.1%}")
    print(f"   BUY:  {cd['buy_pct']:.1%}")
    
    # Quality gates summary
    qg = diagnostic['quality_gates']
    overall = "✅ ALL PASS" if diagnostic['all_gates_passed'] else "❌ SOME FAIL"
    print(f"\n🚪 Quality Gates: {overall}")
    for gate, passed in qg.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {gate.replace('_', ' ').title()}")
    
    # Model metadata
    mm = diagnostic['model_metadata']
    print(f"\n📋 Model Info:")
    print(f"   Training Accuracy: {mm['training_accuracy']}")
    print(f"   Temperature: {mm['temperature']}")
    print(f"   Training Epoch: {mm['training_epoch']}")

def main():
    parser = argparse.ArgumentParser(description='V8 Model Diagnostic')
    parser.add_argument('--symbol', type=str, default='BTC-USD',
                       choices=['BTC-USD', 'ETH-USD', 'SOL-USD'],
                       help='Symbol to diagnose')
    parser.add_argument('--all-models', action='store_true',
                       help='Diagnose all models')
    parser.add_argument('--output', type=str, default='v8_diagnostic_report.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    if args.all_models:
        symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    else:
        symbols = [args.symbol]
    
    print("🔍 V8 Model Diagnostic Suite")
    print(f"Symbols: {symbols}")
    print("="*60)
    
    results = {}
    all_passed = True
    
    for symbol in symbols:
        diagnostic = diagnose_model(symbol, verbose=True)
        results[symbol] = diagnostic
        
        if not diagnostic.get('all_gates_passed', False):
            all_passed = False
    
    # Summary report
    summary = {
        'diagnostic_suite': 'v8_enhanced',
        'diagnostic_date': datetime.now().isoformat(),
        'symbols_tested': len(symbols),
        'all_models_passed': all_passed,
        'results': results
    }
    
    # Calculate aggregate stats
    if len(results) > 1:
        valid_results = [r for r in results.values() if 'confidence_stats' in r]
        if valid_results:
            summary['aggregate_stats'] = {
                'avg_overconfident_99': np.mean([r['confidence_stats']['overconfident_99'] for r in valid_results]),
                'avg_confidence': np.mean([r['confidence_stats']['mean'] for r in valid_results]),
                'avg_logit_range': np.mean([r['logit_stats']['range'] for r in valid_results]),
                'models_passed': sum(1 for r in valid_results if r.get('all_gates_passed', False))
            }
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    for symbol, result in results.items():
        if 'error' in result:
            print(f"{symbol}: ❌ ERROR - {result['error']}")
        else:
            status = "✅ PASS" if result['all_gates_passed'] else "❌ FAIL"
            overconf = result['confidence_stats']['overconfident_99']
            logit_range = result['logit_stats']['range']
            print(f"{symbol}: {status} - {overconf:.1%} overconfident, logit range {logit_range:.1f}")
    
    if all_passed:
        print(f"\n🎉 SUCCESS: All {len(symbols)} models passed quality gates!")
        print("V8 models are ready for production deployment.")
    else:
        print(f"\n⚠️  WARNING: Some models failed quality gates.")
        print("Review diagnostic details and consider retraining.")
    
    print(f"\n📄 Full report saved to: {args.output}")

if __name__ == "__main__":
    main()
