import pandas as pd
from typing import Tuple, List


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Basic data validation for Telco Customer Churn dataset.
    
    This function validates data integrity and basic sanity checks
    before model training.
    """
    print("🔍 Starting data validation...")
    
    failed_checks = []
    
    # === REQUIRED COLUMNS CHECK ===
    print("   📋 Validating required columns...")
    required_cols = [
        "customerID", "gender", "tenure", "MonthlyCharges", "TotalCharges",
        "Partner", "Dependents", "PhoneService", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "Churn"
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        failed_checks.append(f"Missing columns: {missing_cols}")
        print(f"   ❌ Missing columns: {missing_cols}")
    else:
        print(f"   ✅ All {len(required_cols)} required columns present")
    
    # === NULL VALUES CHECK ===
    print("   📊 Validating for critical null values...")
    critical_numeric = ["tenure", "MonthlyCharges"]
    null_count = df[critical_numeric].isnull().sum().sum()
    if null_count > 0:
        failed_checks.append(f"Null values found in critical columns: {null_count}")
        print(f"   ❌ Found {null_count} null values in critical columns")
    else:
        print(f"   ✅ No nulls in critical numeric columns")
    
    # === VALUE RANGES CHECK ===
    print("   📈 Validating numeric ranges...")
    
    # Tenure check (0-120 months reasonable)
    if (df["tenure"] < 0).any() or (df["tenure"] > 150).any():
        failed_checks.append("Unrealistic tenure values found")
        print(f"   ⚠️  Unusual tenure values (min: {df['tenure'].min()}, max: {df['tenure'].max()})")
    else:
        print(f"   ✅ Tenure values reasonable (min: {df['tenure'].min()}, max: {df['tenure'].max()})")
    
    # Charges check (should be positive)
    if (df["MonthlyCharges"] < 0).any():
        failed_checks.append("Negative monthly charges found")
        print(f"   ❌ Negative charges found")
    else:
        print(f"   ✅ All charges are non-negative")
    
    # === DATA TYPES CHECK ===
    print("   🔄 Validating data types...")
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_cols:
        try:
            pd.to_numeric(df[col], errors="coerce")
        except:
            failed_checks.append(f"Cannot convert {col} to numeric")
            print(f"   ❌ {col} is not numeric")
    
    if not any(col.startswith("Cannot convert") for col in failed_checks):
        print(f"   ✅ Numeric columns valid")
    
    # === RESULTS ===
    print("   🔗 Running final checks...")
    
    if failed_checks:
        print(f"\n❌ Data validation FAILED with {len(failed_checks)} issue(s)")
        for check in failed_checks:
            print(f"   - {check}")
        return False, failed_checks
    else:
        print(f"\n✅ Data validation PASSED - all checks successful!")
        return True, []

