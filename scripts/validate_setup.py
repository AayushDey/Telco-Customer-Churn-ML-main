#!/usr/bin/env python3
"""
Project Requirements & Setup Validator
========================================
This script validates that all project requirements are met before running.
Run this to diagnose setup issues.
"""

import os
import sys
import subprocess

def check_python_version():
    """Verify Python 3.11+ is installed."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.11+ required.")
        return False

def check_packages():
    """Verify all required packages are installed."""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'xgboost',
        'mlflow', 'fastapi', 'uvicorn', 'gradio',
        'great_expectations', 'pydantic'
    ]
    
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"✅ {pkg:20s} installed")
        except ImportError:
            print(f"❌ {pkg:20s} NOT installed")
            missing.append(pkg)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print(f"   Fix: pip install -r requirements.txt")
        return False
    return True

def check_directories():
    """Verify required directories exist."""
    required_dirs = [
        'data/raw',
        'data/processed',
        'artifacts',
        'src/app',
        'src/data',
        'src/features',
        'src/models',
        'src/serving',
        'src/utils',
        'scripts'
    ]
    
    missing = []
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"✅ {dir_path:30s} exists")
        else:
            print(f"❌ {dir_path:30s} MISSING")
            missing.append(dir_path)
    
    if missing:
        print(f"\n❌ Missing directories: {', '.join(missing)}")
        print(f"   Fix: Run 'mkdir -p' for each missing directory")
        return False
    return True

def check_files():
    """Verify critical files exist."""
    required_files = [
        'requirements.txt',
        'src/app/main.py',
        'src/data/load_data.py',
        'src/data/preprocess.py',
        'src/features/build_features.py',
        'src/serving/inference.py',
        'src/utils/validate_data.py',
        'scripts/run_pipeline.py'
    ]
    
    missing = []
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"✅ {file_path:40s} exists")
        else:
            print(f"❌ {file_path:40s} MISSING")
            missing.append(file_path)
    
    if missing:
        print(f"\n❌ Missing files: {', '.join(missing)}")
        return False
    return True

def check_dataset():
    """Check if Telco dataset is available."""
    dataset_path = 'data/raw/Telco-Customer-Churn.csv'
    
    if os.path.isfile(dataset_path):
        print(f"✅ Dataset found: {dataset_path}")
        # Check file size
        size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
        print(f"   Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"❌ Dataset NOT found: {dataset_path}")
        print(f"   You must download: 'Telco Customer Churn' dataset from Kaggle")
        print(f"   or another compatible dataset with required columns")
        return False

def check_init_files():
    """Verify __init__.py files exist in all packages."""
    packages = [
        'src',
        'src/app',
        'src/data',
        'src/features',
        'src/models',
        'src/serving',
        'src/utils'
    ]
    
    missing = []
    for pkg in packages:
        init_file = os.path.join(pkg, '__init__.py')
        if os.path.isfile(init_file):
            print(f"✅ {init_file:30s} exists")
        else:
            print(f"❌ {init_file:30s} MISSING")
            missing.append(init_file)
    
    if missing:
        print(f"\n❌ Missing __init__.py files: {', '.join(missing)}")
        return False
    return True

def main():
    """Run all validation checks."""
    print("=" * 70)
    print("Telco Churn ML Project - Setup Validator")
    print("=" * 70)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_packages),
        ("Required Directories", check_directories),
        ("Required Files", check_files),
        ("Package Init Files", check_init_files),
        ("Dataset", check_dataset),
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n🔍 Checking: {check_name}")
        print("-" * 70)
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ Error during check: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8s} {check_name}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✅ All checks passed! You're ready to run the project.")
        print("\nNext steps:")
        print("1. If dataset not found, download from Kaggle:")
        print("   https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        print("2. Run training: python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn")
        print("3. View results: mlflow ui")
        print("4. Start serving: python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000")
    else:
        print(f"\n❌ {total - passed} check(s) failed. Fix issues above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
