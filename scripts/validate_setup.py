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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'xgboost': 'xgboost',
        'mlflow': 'mlflow',
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'great_expectations': 'great_expectations',
        'pydantic': 'pydantic',
    }
    
    missing = []
    for pkg, import_name in required_packages.items():
        try:
            __import__(import_name)
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
        'src/serving',
        'scripts',
        'scripts/pipeline'
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
        'src/serving/inference.py',
        'scripts/run_pipeline.py',
        'scripts/pipeline/load_data.py',
        'scripts/pipeline/preprocess.py',
        'scripts/pipeline/build_features.py',
        'scripts/pipeline/validate_data.py'
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
    dataset_paths = [
        'data/raw/Telco-Customer-Churn.csv',
        'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv',
    ]
    dataset_path = next((path for path in dataset_paths if os.path.isfile(path)), None)
    
    if dataset_path:
        print(f"✅ Dataset found: {dataset_path}")
        # Check file size
        size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
        print(f"   Size: {size_mb:.2f} MB")
        return True
    else:
        print("❌ Dataset NOT found in data/raw")
        print(f"   You must download: 'Telco Customer Churn' dataset from Kaggle")
        print(f"   or another compatible dataset with required columns")
        return False

def check_init_files():
    """Verify __init__.py files exist in all packages."""
    packages = [
        'src',
        'src/app',
        'src/serving',
        'scripts/pipeline'
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
