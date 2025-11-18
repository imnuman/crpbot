#!/usr/bin/env python3
import subprocess
import sys
import importlib

# Core dependencies from pyproject.toml
REQUIRED_PACKAGES = [
    "numpy>=1.24.0",
    "pandas>=2.0.0", 
    "torch>=2.0.0",
    "scikit-learn>=1.3.0",
    "matplotlib>=3.7.0",
    "tqdm>=4.65.0",
    "pyyaml>=6.0",
    "requests>=2.31.0"
]

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Installed: {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def check_import(module_name):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ Import OK: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {module_name} - {e}")
        return False

def main():
    print("🔧 Installing SageMaker Training Dependencies...")
    
    # Install packages
    failed_installs = []
    for package in REQUIRED_PACKAGES:
        if not install_package(package):
            failed_installs.append(package)
    
    print("\n📋 Verifying imports...")
    
    # Test imports
    test_modules = [
        "numpy", "pandas", "torch", "sklearn", 
        "matplotlib", "tqdm", "yaml", "requests"
    ]
    
    failed_imports = []
    for module in test_modules:
        if not check_import(module):
            failed_imports.append(module)
    
    # Summary
    print(f"\n📊 Installation Summary:")
    print(f"   Packages installed: {len(REQUIRED_PACKAGES) - len(failed_installs)}/{len(REQUIRED_PACKAGES)}")
    print(f"   Imports working: {len(test_modules) - len(failed_imports)}/{len(test_modules)}")
    
    if failed_installs:
        print(f"   ❌ Failed installs: {failed_installs}")
    if failed_imports:
        print(f"   ❌ Failed imports: {failed_imports}")
    
    if not failed_installs and not failed_imports:
        print("✅ All dependencies ready for training!")
        return 0
    else:
        print("❌ Some dependencies failed - check logs above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
