#!/usr/bin/env python3
"""
Run this in Google Colab to find and download trained models.
"""

import os
import glob
from pathlib import Path

print("🔍 Searching for trained model files...")
print("=" * 60)

# Search for .pt model files
model_files = []
for pattern in ["**/*.pt", "**/*_lstm_model.pt", "**/models/**/*.pt"]:
    found = glob.glob(pattern, recursive=True)
    model_files.extend(found)

# Remove duplicates
model_files = list(set(model_files))

if not model_files:
    print("❌ No .pt model files found!")
    print("\nSearching in common locations:")

    # Check common directories
    common_dirs = [
        "models/",
        "models/gpu_trained/",
        "models/gpu_trained_proper/",
        "/content/models/",
        str(Path.home() / "models/")
    ]

    for dir_path in common_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path} exists")
            contents = os.listdir(dir_path)
            if contents:
                print(f"  Contents: {contents}")
            else:
                print(f"  (empty)")
        else:
            print(f"✗ {dir_path} does not exist")
else:
    print(f"✅ Found {len(model_files)} model file(s):\n")

    total_size = 0
    for model_file in sorted(model_files):
        size = os.path.getsize(model_file) / 1024  # KB
        total_size += size
        print(f"  {model_file} ({size:.1f} KB)")

    print(f"\n📊 Total size: {total_size/1024:.2f} MB")

    # Create zip command
    print("\n📦 To download these models, run:")
    print("-" * 60)

    # Group by directory
    dirs = set(str(Path(f).parent) for f in model_files)

    if len(dirs) == 1:
        # All in one directory
        dir_path = list(dirs)[0]
        print(f"!zip -r gpu_models.zip {dir_path}/")
        print("\nfrom google.colab import files")
        print("files.download('gpu_models.zip')")
    else:
        # Multiple directories - zip each model individually
        print("# Models in multiple directories, zip all together:")
        print(f"!zip gpu_models.zip {' '.join(model_files)}")
        print("\nfrom google.colab import files")
        print("files.download('gpu_models.zip')")

    print("-" * 60)

# Also check for metadata/summary files
print("\n🔍 Checking for training metadata...")
metadata_patterns = ["**/training_summary.json", "**/*_metadata.json", "**/manifest.json"]
metadata_files = []

for pattern in metadata_patterns:
    found = glob.glob(pattern, recursive=True)
    metadata_files.extend(found)

if metadata_files:
    print(f"✅ Found {len(metadata_files)} metadata file(s):")
    for f in metadata_files:
        print(f"  {f}")
else:
    print("⚠️  No metadata files found")

print("\n" + "=" * 60)
print("💡 Tip: If models aren't found, check your current directory:")
print("   !pwd")
print("   !ls -la")
