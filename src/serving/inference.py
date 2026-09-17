"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================

This module provides the core inference functionality for the Telco Churn prediction model.
It ensures that serving-time feature transformations exactly match training-time transformations,
which is CRITICAL for model accuracy in production.

Key Responsibilities:
1. Load MLflow-logged model and feature metadata from training
2. Apply identical feature transformations as used during training
3. Ensure correct feature ordering for model input
4. Convert model predictions to user-friendly output
5. Return real churn probability + SHAP feature importance values

CRITICAL PATTERN: Training/Serving Consistency
- Uses fixed BINARY_MAP for deterministic binary encoding
- Applies same one-hot encoding with drop_first=True
- Maintains exact feature column order from training
- Handles missing/new categorical values gracefully

Production Deployment:
- MODEL_DIR points to containerized model artifacts
- Feature schema loaded from training-time artifacts
- Optimized for single-row inference (real-time serving)
"""

import os
import sys
import pandas as pd
import joblib

# MLflow is optional for serving (direct joblib loading is much faster and cleaner)
try:
    import mlflow
    import mlflow.sklearn
except Exception:
    mlflow = None

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# === MODEL LOADING CONFIGURATION ===
# IMPORTANT: This path is set during Docker container build
# In development: uses local bundled model artifacts
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOCAL_BUNDLED_MODEL = os.path.join(os.path.dirname(__file__), "model", "model")
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model" if os.path.exists("/app/model") else LOCAL_BUNDLED_MODEL)
LOCAL_MLRUNS_DIR = os.path.join(PROJECT_ROOT, "mlruns")

model = None
sklearn_model = None

# 1. Direct Joblib Load (Fastest, zero MLflow/pkg_resources dependency)
pkl_candidates = [
    os.path.join(MODEL_DIR, "model.pkl"),
    os.path.join(LOCAL_BUNDLED_MODEL, "model.pkl"),
]

for pkl_path in pkl_candidates:
    if os.path.exists(pkl_path):
        try:
            loaded_model = joblib.load(pkl_path)
            model = loaded_model
            sklearn_model = loaded_model
            print(f"✅ Model loaded successfully from {pkl_path}")
            break
        except Exception as e:
            print(f"⚠️ Direct model load from {pkl_path} failed: {e}")

# 2. MLflow pyfunc Fallback (if direct load was not possible and mlflow is installed)
if model is None and mlflow is not None:
    try:
        model = mlflow.pyfunc.load_model(MODEL_DIR)
        print(f"✅ MLflow pyfunc model loaded from {MODEL_DIR}")
    except Exception as e:
        print(f"⚠️ MLflow pyfunc model load failed: {e}")
        try:
            if os.path.exists(LOCAL_BUNDLED_MODEL):
                model = mlflow.pyfunc.load_model(LOCAL_BUNDLED_MODEL)
                MODEL_DIR = LOCAL_BUNDLED_MODEL
        except Exception:
            pass

    try:
        sklearn_model = mlflow.sklearn.load_model(MODEL_DIR)
        print("✅ MLflow sklearn model loaded for probability + SHAP")
    except Exception as e:
        print(f"⚠️ MLflow sklearn model load failed: {e}")

if model is None:
    raise RuntimeError("Failed to load customer churn prediction model from any source.")


# === SHAP EXPLAINER SETUP ===
SHAP_AVAILABLE = False
explainer = None
try:
    import shap
    if sklearn_model is not None:
        explainer = shap.TreeExplainer(sklearn_model)
        SHAP_AVAILABLE = True
        print("✅ SHAP TreeExplainer ready")
except Exception as e:
    print(f"⚠️ SHAP not available: {e}")

# === FEATURE SCHEMA LOADING ===
# CRITICAL: Load the exact feature column order used during training
# This ensures the model receives features in the expected order
try:
    # Try loading from model directory first
    feature_file = os.path.join(MODEL_DIR, "feature_columns.txt")
    if not os.path.exists(feature_file):
        # If not in model dir, try parent artifacts directory
        artifacts_dir = os.path.dirname(MODEL_DIR)
        feature_file = os.path.join(artifacts_dir, "feature_columns.txt")
    
    with open(feature_file) as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
    print(f"✅ Loaded {len(FEATURE_COLS)} feature columns from training")
except Exception as e:
    raise Exception(f"Failed to load feature columns: {e}")

# === FEATURE TRANSFORMATION CONSTANTS ===
# CRITICAL: These mappings must exactly match those used in training
# Any changes here will cause train/serve skew and degrade model performance

# Deterministic binary feature mappings (consistent with training)
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},           # Demographics
    "Partner": {"No": 0, "Yes": 1},               # Has partner
    "Dependents": {"No": 0, "Yes": 1},            # Has dependents  
    "PhoneService": {"No": 0, "Yes": 1},          # Phone service
    "PaperlessBilling": {"No": 0, "Yes": 1},      # Billing preference
}

# Numeric columns that need type coercion
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Friendly display names for SHAP features (one-hot encoded cols → readable)
FEATURE_DISPLAY_NAMES = {
    "tenure": "Tenure (months)",
    "MonthlyCharges": "Monthly Charges",
    "TotalCharges": "Total Charges",
    "gender": "Gender",
    "Partner": "Has Partner",
    "Dependents": "Has Dependents",
    "PhoneService": "Phone Service",
    "PaperlessBilling": "Paperless Billing",
    "MultipleLines_No phone service": "Multiple Lines (No phone)",
    "MultipleLines_Yes": "Multiple Lines",
    "InternetService_Fiber optic": "Internet: Fiber Optic",
    "InternetService_No": "Internet: None",
    "OnlineSecurity_No internet service": "Online Security (No internet)",
    "OnlineSecurity_Yes": "Online Security",
    "OnlineBackup_No internet service": "Online Backup (No internet)",
    "OnlineBackup_Yes": "Online Backup",
    "DeviceProtection_No internet service": "Device Protection (No internet)",
    "DeviceProtection_Yes": "Device Protection",
    "TechSupport_No internet service": "Tech Support (No internet)",
    "TechSupport_Yes": "Tech Support",
    "StreamingTV_No internet service": "Streaming TV (No internet)",
    "StreamingTV_Yes": "Streaming TV",
    "StreamingMovies_No internet service": "Streaming Movies (No internet)",
    "StreamingMovies_Yes": "Streaming Movies",
    "Contract_One year": "Contract: One Year",
    "Contract_Two year": "Contract: Two Year",
    "PaymentMethod_Credit card (automatic)": "Payment: Credit Card",
    "PaymentMethod_Electronic check": "Payment: Electronic Check",
    "PaymentMethod_Mailed check": "Payment: Mailed Check",
}


def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply identical feature transformations as used during model training.
    
    This function is CRITICAL for production ML - it ensures that features are
    transformed exactly as they were during training to prevent train/serve skew.
    
    Transformation Pipeline:
    1. Clean column names and handle data types
    2. Apply deterministic binary encoding (using BINARY_MAP)
    3. One-hot encode remaining categorical features  
    4. Convert boolean columns to integers
    5. Align features with training schema and order
    
    Args:
        df: Single-row DataFrame with raw customer data
        
    Returns:
        DataFrame with features transformed and ordered for model input
        
    IMPORTANT: Any changes to this function must be reflected in training
    feature engineering to maintain consistency.
    """
    df = df.copy()
    
    # Clean column names (remove any whitespace)
    df.columns = df.columns.str.strip()
    
    # === STEP 1: Numeric Type Coercion ===
    # Ensure numeric columns are properly typed (handle string inputs)
    for c in NUMERIC_COLS:
        if c in df.columns:
            # Convert to numeric, replacing invalid values with NaN
            df[c] = pd.to_numeric(df[c], errors="coerce")
            # Fill NaN with 0 (same as training preprocessing)
            df[c] = df[c].fillna(0)
    
    # === STEP 2: Binary Feature Encoding ===
    # Apply deterministic mappings for binary features
    # CRITICAL: Must use exact same mappings as training
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)                    # Convert to string
                .str.strip()                    # Remove whitespace
                .map(mapping)                   # Apply binary mapping
                .astype("Int64")                # Handle NaN values
                .fillna(0)                      # Fill unknown values with 0
                .astype(int)                    # Final integer conversion
            )
    
    # === STEP 3: One-Hot Encoding for Remaining Categorical Features ===
    # Find remaining object/categorical columns (not in BINARY_MAP)
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        # Apply one-hot encoding with drop_first=True (same as training)
        # This prevents multicollinearity by dropping the first category
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    
    # === STEP 4: Boolean to Integer Conversion ===
    # Convert any boolean columns to integers (XGBoost compatibility)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
    
    # === STEP 5: Feature Alignment with Training Schema ===
    # CRITICAL: Ensure features are in exact same order as training
    # Missing features get filled with 0, extra features are dropped
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    
    return df


def _get_confidence(probability: float, threshold: float = 0.35) -> str:
    """
    Convert a raw probability into a human-readable confidence label.
    
    Confidence is measured by the distance from the classification threshold.
    A prediction right at the threshold is 'Borderline', while predictions
    far from it are 'High Confidence'.
    """
    distance = abs(probability - threshold)
    if distance < 0.10:
        return "Borderline"
    elif distance < 0.25:
        return "Moderate"
    else:
        return "High Confidence"


def _get_shap_values(df_enc: pd.DataFrame) -> list:
    """
    Compute SHAP feature importance values for a single prediction row.
    
    Returns a sorted list of the top 8 features by absolute impact,
    with friendly display names and signed impact values (positive = 
    increases churn risk, negative = decreases churn risk).
    """
    if not SHAP_AVAILABLE or explainer is None:
        return []
    try:
        sv = explainer.shap_values(df_enc)
        # sv shape: (1, n_features) for XGBoost binary classification
        impacts = []
        for i, col in enumerate(FEATURE_COLS):
            impact = float(sv[0][i]) if hasattr(sv[0], "__len__") else float(sv[i])
            display = FEATURE_DISPLAY_NAMES.get(col, col)
            impacts.append({
                "feature": display,
                "raw_feature": col,
                "impact": round(impact, 4),
            })
        # Sort by absolute impact (most influential first)
        impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return impacts[:8]
    except Exception as e:
        print(f"⚠️ SHAP computation error: {e}")
        return []


def predict(input_dict: dict) -> dict:
    """
    Main prediction function for customer churn inference.
    
    Returns a rich result dict containing:
    - prediction: Human-readable label ("Likely to churn" / "Not likely to churn")
    - probability: Real churn probability from model.predict_proba() (0.0–1.0)
    - confidence: Confidence label based on distance from threshold
    - shap_values: Top 8 feature importance values for this prediction

    Args:
        input_dict: Dictionary containing raw customer data with keys matching
                   the CustomerData schema (18 features total)
    """
    # === STEP 1: Convert Input to DataFrame ===
    df = pd.DataFrame([input_dict])
    
    # === STEP 2: Apply Feature Transformations ===
    df_enc = _serve_transform(df)
    
    # === STEP 3: Get Binary Prediction ===
    try:
        preds = model.predict(df_enc)
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        result_int = preds[0] if isinstance(preds, (list, tuple)) else preds
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")

    # === STEP 4: Get Real Churn Probability ===
    probability = 0.5  # safe fallback
    if sklearn_model is not None:
        try:
            proba_arr = sklearn_model.predict_proba(df_enc)
            probability = float(proba_arr[0][1])  # probability of class 1 (churn)
        except Exception as e:
            print(f"⚠️ predict_proba failed: {e}")

    # === STEP 5: Compute SHAP Feature Importance ===
    shap_values = _get_shap_values(df_enc)

    # === STEP 6: Build Rich Response ===
    prediction_label = "Likely to churn" if result_int == 1 else "Not likely to churn"
    confidence = _get_confidence(probability)

    return {
        "prediction": prediction_label,
        "probability": round(probability, 4),
        "confidence": confidence,
        "shap_values": shap_values,
    }
