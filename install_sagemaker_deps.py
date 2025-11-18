#!/usr/bin/env python3
"""Install SageMaker dependencies"""

import subprocess
import sys

def install_requirements():
    """Install all SageMaker requirements"""
    try:
        print("Installing SageMaker dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", "requirements_sagemaker.txt", 
            "--upgrade"
        ])
        print("✅ SageMaker dependencies installed successfully")
        
        # Verify key imports
        print("Verifying installations...")
        import sagemaker
        import torch
        import pandas as pd
        import numpy as np
        import sklearn
        print(f"✅ SageMaker: {sagemaker.__version__}")
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ Pandas: {pd.__version__}")
        print(f"✅ NumPy: {np.__version__}")
        print(f"✅ Scikit-learn: {sklearn.__version__}")
        
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_requirements()
