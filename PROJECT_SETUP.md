# Telco Customer Churn ML Project - Complete Setup Guide

## 📋 Project Overview
This is a production-ready MLOps pipeline for predicting customer churn in a telecom company using XGBoost, MLflow, FastAPI, and Gradio.

**Architecture:**
- **Training Pipeline**: Data loading → Validation → Preprocessing → Feature Engineering → XGBoost Training → MLflow Logging
- **Serving Pipeline**: FastAPI REST API + Gradio Web UI → MLflow Model Loading → Feature Transformation → Prediction
- **Orchestration**: MLflow for experiment tracking, model versioning, and artifact management

---

## 🔧 Prerequisites & Installation

### Step 1: Python Environment Setup
```bash
# Ensure Python 3.11+ is installed
python --version

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
# Install all required packages from requirements.txt
pip install -r requirements.txt

# Verify installation (test key packages)
python -c "import mlflow; import xgboost; import pandas; print('✅ All dependencies installed')"
```

### Step 3: Download Dataset
**You need the Telco Customer Churn dataset.**

**Option A: Download from Kaggle (Recommended)**
1. Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Save to: `data/raw/Telco-Customer-Churn.csv`

**Option B: Use any compatible CSV with these columns:**
```
Required columns for training:
- customerID (string)
- gender (Male/Female)
- SeniorCitizen (0/1)
- Partner (Yes/No)
- Dependents (Yes/No)
- tenure (numeric, months)
- PhoneService (Yes/No)
- MultipleLines (Yes/No/No phone service)
- InternetService (DSL/Fiber optic/No)
- OnlineSecurity (Yes/No/No internet service)
- OnlineBackup (Yes/No/No internet service)
- DeviceProtection (Yes/No/No internet service)
- TechSupport (Yes/No/No internet service)
- StreamingTV (Yes/No/No internet service)
- StreamingMovies (Yes/No/No internet service)
- Contract (Month-to-month/One year/Two year)
- PaperlessBilling (Yes/No)
- PaymentMethod (Electronic check/Mailed check/Bank transfer/Credit card)
- MonthlyCharges (numeric, USD)
- TotalCharges (numeric, USD)
- Churn (Yes/No) <- TARGET VARIABLE
```

---

## 🚀 Running the Project

### Phase 1: Training Pipeline
```bash
# Run complete training pipeline
python scripts/run_pipeline.py \
    --input data/raw/Telco-Customer-Churn.csv \
    --target Churn

# Optional parameters:
# --threshold 0.35      (classification threshold, default: 0.35)
# --test_size 0.2       (train/test split ratio, default: 0.2)
# --experiment "Telco Churn"  (MLflow experiment name)
# --mlflow_uri "file:///path/to/mlruns"  (custom MLflow tracking URI)
```

**Output:**
- ✅ Data loaded, validated, and preprocessed
- ✅ XGBoost model trained with optimized hyperparameters
- ✅ Metrics logged to MLflow (precision, recall, f1, roc_auc)
- ✅ Model artifacts saved in `mlruns/` directory
- ✅ Feature columns saved for serving consistency

### Phase 2: View Experiment Results
```bash
# Launch MLflow UI to see experiment tracking
mlflow ui --backend-store-uri file:./mlruns

# Then visit: http://localhost:5000
# View:
# - All training runs with their metrics
# - Model artifacts and parameters
# - Feature engineering artifacts
```

### Phase 3: Start Serving Application
```bash
# Method 1: Using FastAPI + Gradio web interface
python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000

# Then visit:
# - REST API docs: http://localhost:8000/docs
# - Gradio UI: http://localhost:8000/ui
```

**Endpoints:**
- `GET /` - Health check
- `POST /predict` - Make predictions (JSON API)
- `GET /ui` - Gradio web interface

### Phase 4: Run Tests
```bash
# Test data processing and feature engineering
python scripts/test_pipeline_phase1_data_features.py

# Test model training and evaluation
python scripts/test_pipeline_phase2_modeling.py

# Test FastAPI endpoints
python scripts/test_fastapi.py
```

---

## 📊 Data Flow Diagram

```
Raw Data (CSV)
    ↓
Load Data → Validate (Great Expectations) → Preprocess Data
    ↓
Feature Engineering (Binary Encoding + One-Hot Encoding)
    ↓
Train/Test Split (80/20 stratified)
    ↓
XGBoost Training (301 trees, depth=7, learning_rate=0.034)
    ↓
Model Evaluation (Precision, Recall, F1, ROC-AUC)
    ↓
Save to MLflow (Model + Feature Metadata + Metrics)
    ↓
Inference Pipeline:
  - Load MLflow Model
  - Transform Features (identical to training)
  - Make Predictions
  - Return Result (REST API or Web UI)
```

---

## 🔑 Critical Implementation Details

### Feature Engineering Consistency
**CRITICAL**: Training and serving use identical transformations to prevent train/serve skew.

**Training** (`src/features/build_features.py`):
- Binary features (Yes/No, Male/Female) → 0/1 mapping
- Multi-category features → one-hot encoding with `drop_first=True`
- Boolean columns → integers

**Serving** (`src/serving/inference.py`):
- Uses same `BINARY_MAP` dictionary
- Applies identical one-hot encoding
- Feature order enforced via `feature_columns.txt`

### Model Configuration
XGBoost hyperparameters (optimized in previous tuning):
```python
XGBClassifier(
    n_estimators=301,      # Number of trees
    learning_rate=0.034,   # Step size (eta)
    max_depth=7,           # Tree depth
    subsample=0.95,        # Row sampling
    colsample_bytree=0.98, # Feature sampling
    scale_pos_weight=2.5,  # Class imbalance handling (calculated)
    random_state=42,       # Reproducibility
)
```

### MLflow Integration
- **Tracking URI**: File-based at `{project_root}/mlruns`
- **Experiment**: "Telco Churn"
- **Artifacts**: Model + Feature columns + Preprocessing logic
- **Metrics Tracked**:
  - `precision`, `recall`, `f1`, `roc_auc` (model performance)
  - `train_time`, `pred_time` (inference latency)
  - `data_quality_pass` (validation status)

### Data Validation
Uses Great Expectations to validate:
- ✅ Required columns exist
- ✅ Categorical values within allowed sets
- ✅ Numeric ranges (tenure: 0-120 months, charges: 0-200)
- ✅ Data consistency (TotalCharges ≥ MonthlyCharges)
- ✅ No null values in critical features

---

## 📁 Project Structure

```
Telco-Customer-Churn-ML/
├── data/
│   ├── raw/                          # Original dataset
│   │   └── Telco-Customer-Churn.csv  # Download here
│   └── processed/                    # Cleaned data (auto-generated)
├── artifacts/                        # Feature metadata (auto-generated)
├── mlruns/                          # MLflow tracking (auto-generated)
├── src/
│   ├── __init__.py
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI + Gradio app
│   │   └── app.py                   # Alternative app entry
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py             # Data loading
│   │   └── preprocess.py            # Data cleaning
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py        # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── tune.py
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── inference.py             # Production inference
│   │   └── model/                   # MLflow model artifacts
│   └── utils/
│       ├── __init__.py
│       ├── utils.py
│       └── validate_data.py         # Data validation (Great Expectations)
├── scripts/
│   ├── run_pipeline.py              # Main training script
│   ├── prepare_processed_data.py    # Data prep only
│   ├── test_pipeline_phase1_data_features.py
│   ├── test_pipeline_phase2_modeling.py
│   └── test_fastapi.py
├── notebooks/
│   └── EDA.ipynb                    # Exploratory data analysis
├── requirements.txt                 # Python dependencies
├── dockerfile                       # Container build
├── CLAUDE.md                        # Claude AI guidance
├── PROJECT_SETUP.md                 # This file
└── README.md
```

---

## 🐳 Docker Deployment

```bash
# Build Docker image
docker build -t telco-churn-app .

# Run container
docker run -p 8000:8000 telco-churn-app

# Visit: http://localhost:8000/ui
```

**Note**: Docker includes pre-trained model. For custom model, update:
```dockerfile
COPY src/serving/model/[YOUR_RUN_ID]/artifacts/model /app/model
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'src'"
**Solution**: Ensure PYTHONPATH includes src:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Issue: "File not found: data/raw/Telco-Customer-Churn.csv"
**Solution**: Download the dataset and place it in `data/raw/` directory.

### Issue: "Failed to load model from /app/model"
**Solution**: This is expected in local development. The inference.py has a fallback that loads from `mlruns/` directory.

### Issue: Data validation fails
**Check**:
- ✅ Required columns present
- ✅ No unexpected null values
- ✅ Categorical values match expected sets
- ✅ Numeric ranges reasonable

---

## 🔄 MLflow Workflow

```bash
# View all experiments
mlflow experiments list

# View specific run details
mlflow run info <RUN_ID>

# Compare multiple runs
mlflow ui

# Register model (optional, for production)
# Via UI: Go to model → Register Model
# Via CLI: mlflow register-model "runs:/<RUN_ID>/model" "TelcoChurnModel"
```

---

## 📈 Performance Metrics Interpretation

After training, you'll see:
- **Precision**: Of predicted churners, how many actually churned? (minimize false positives)
- **Recall**: Of actual churners, how many did we catch? (minimize false negatives)
- **F1 Score**: Harmonic mean of precision & recall
- **ROC-AUC**: Threshold-independent performance (0.5=random, 1.0=perfect)

Lower threshold (default: 0.35) → Higher recall (catch more churners) but lower precision

---

## 📚 Key Files Modified/Created

### Fixed Issues:
1. ✅ Removed bad `from posthog import project_root` import
2. ✅ Created `__init__.py` for all packages
3. ✅ Created `data/raw/` and `data/processed/` directories
4. ✅ Created `artifacts/` directory
5. ✅ Verified main.py is complete

### Ready to Use:
- ✅ All training scripts
- ✅ All serving scripts  
- ✅ Data validation pipeline
- ✅ Feature engineering
- ✅ MLflow integration
- ✅ FastAPI endpoints
- ✅ Gradio UI

---

## 🎯 Next Steps

1. **Download Dataset**: Kaggle or compatible CSV to `data/raw/`
2. **Train Model**: `python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn`
3. **View Results**: `mlflow ui` (http://localhost:5000)
4. **Start Serving**: `python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000`
5. **Make Predictions**: Visit http://localhost:8000/ui or POST to `/predict`

---

## 📞 Support

For issues or questions:
1. Check MLflow UI for detailed run metrics
2. Review data validation output
3. Check `mlruns/` for experiment artifacts
4. Run test scripts to debug pipeline phases

**Happy predicting! 🎯**
