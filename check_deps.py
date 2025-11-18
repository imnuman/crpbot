#!/usr/bin/env python3
import sys
import os

print("🔍 SageMaker Container Dependency Check")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.executable}")

# Check available packages
packages_to_check = [
    'pandas', 'numpy', 'sklearn', 'torch', 'joblib', 
    'matplotlib', 'seaborn', 'scipy', 'json', 'os'
]

available = []
missing = []

for package in packages_to_check:
    try:
        __import__(package)
        available.append(package)
        print(f"✅ {package}")
    except ImportError:
        missing.append(package)
        print(f"❌ {package}")

print(f"\nSummary:")
print(f"Available: {len(available)} packages")
print(f"Missing: {len(missing)} packages")

# Save results
model_path = os.environ.get('SM_MODEL_DIR', './output')
os.makedirs(model_path, exist_ok=True)

results = {
    'python_version': sys.version,
    'available_packages': available,
    'missing_packages': missing,
    'check_complete': True
}

import json
with open(os.path.join(model_path, 'dependency_check.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Results saved to {model_path}/dependency_check.json")
exit(0)
