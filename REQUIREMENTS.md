# REQUIREMENTS & CONFIGURATION

## 🔧 System Requirements

### Python
- **Version**: Python 3.11 or higher
- **Why 3.11**: Modern type hints, better performance
- **Check version**:
  ```bash
  python --version
  ```

### Operating System
- ✅ Windows 10/11
- ✅ macOS (Intel or Apple Silicon)
- ✅ Linux (Ubuntu 20.04+, Debian, etc.)

### Hardware (Recommended)
- **CPU**: 4+ cores (for XGBoost training parallelization)
- **RAM**: 8GB minimum (dataset: ~20MB, processed: ~100MB)
- **Storage**: 2GB free space (for mlruns artifacts, trained model, etc.)

---

## 📦 Python Dependencies

All dependencies are in `requirements.txt`. Key packages:

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | 2.1.4 | Data manipulation |
| `numpy` | 1.26.4 | Numerical computing |
| `scikit-learn` | 1.5.2 | Machine learning utilities |
| `xgboost` | 3.0.3 | Gradient boosting model |
| `mlflow` | 2.14.1 | Experiment tracking & model registry |
| `fastapi` | 0.115.0 | REST API framework |
| `uvicorn` | 0.30.5 | ASGI server for FastAPI |
| `gradio` | latest | Web UI for model demo |
| `great_expectations` | 1.5.8 | Data validation |
| `pydantic` | 2.8.2 | Data validation schemas |
| `jupyter` | 1.0.0 | Notebooks (optional) |

**Installation:**
```bash
pip install -r requirements.txt
```

---

## 📊 Data Requirements

### Dataset Specifications

You need the **Telco Customer Churn** dataset with these 21 columns:

#### Customer Information
- `customerID`: Unique identifier (string)
- `gender`: "Male" or "Female" 
- `SeniorCitizen`: 0 or 1 (binary)
- `Partner`: "Yes" or "No"
- `Dependents`: "Yes" or "No"
- `tenure`: Months with company (0-120)

#### Services
- `PhoneService`: "Yes" or "No"
- `MultipleLines`: "Yes", "No", or "No phone service"
- `InternetService`: "DSL", "Fiber optic", or "No"
- `OnlineSecurity`: "Yes", "No", or "No internet service"
- `OnlineBackup`: "Yes", "No", or "No internet service"
- `DeviceProtection`: "Yes", "No", or "No internet service"
- `TechSupport`: "Yes", "No", or "No internet service"
- `StreamingTV`: "Yes", "No", or "No internet service"
- `StreamingMovies`: "Yes", "No", or "No internet service"

#### Account
- `Contract`: "Month-to-month", "One year", or "Two year"
- `PaperlessBilling`: "Yes" or "No"
- `PaymentMethod`: "Electronic check", "Mailed check", "Bank transfer (automatic)", or "Credit card (automatic)"

#### Financial
- `MonthlyCharges`: USD currency (0-200)
- `TotalCharges`: USD currency (0-10,000)

#### Target
- `Churn`: "Yes" or "No" (prediction target)

### File Location
```
data/raw/Telco-Customer-Churn.csv
```

### Data Size
- **Rows**: ~7,043 customer records
- **File Size**: ~1 MB
- **Missing Values**: Minimal (mostly in TotalCharges)

### Where to Get Data

**Option 1: Kaggle (Recommended)**
1. Visit: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Click "Download" (requires Kaggle account, free)
3. Extract `WA_Fn-UseC_-Telco-Customer-Churn.csv`
4. Rename and save to: `data/raw/Telco-Customer-Churn.csv`

**Option 2: IBM Sample Data**
1. Visit: https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-sample-data
2. Download and save to same location

**Option 3: Use Compatible CSV**
- Any CSV with the above columns and ~7K records
- Ensure data types match (binary as Yes/No, numeric as floats/ints)

---

## 🗂️ Directory Structure Required

```
project_root/
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv    [YOU MUST PROVIDE THIS]
│   └── processed/                       [Auto-generated after training]
├── artifacts/                           [Auto-generated after training]
├── mlruns/                              [Auto-generated after training]
├── src/
│   ├── __init__.py                      [Created ✅]
│   ├── app/
│   │   ├── __init__.py                  [Created ✅]
│   │   ├── main.py
│   │   └── app.py
│   ├── data/
│   │   ├── __init__.py                  [Created ✅]
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── features/
│   │   ├── __init__.py                  [Created ✅]
│   │   └── build_features.py
│   ├── models/
│   │   ├── __init__.py                  [Created ✅]
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── tune.py
│   ├── serving/
│   │   ├── __init__.py                  [Created ✅]
│   │   └── inference.py
│   └── utils/
│       ├── __init__.py                  [Created ✅]
│       ├── utils.py
│       └── validate_data.py
├── scripts/
│   ├── run_pipeline.py
│   ├── prepare_processed_data.py
│   ├── test_pipeline_phase1_data_features.py
│   ├── test_pipeline_phase2_modeling.py
│   ├── test_fastapi.py
│   └── validate_setup.py                [Created ✅]
├── notebooks/
│   └── EDA.ipynb
├── requirements.txt                     [Existing ✅]
├── dockerfile
├── CLAUDE.md
├── PROJECT_SETUP.md                     [Created ✅]
├── REQUIREMENTS.md                      [This file]
└── README.md
```

---

## ✅ Setup Checklist

Use this checklist to verify everything is ready:

- [ ] Python 3.11+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Dataset downloaded to `data/raw/Telco-Customer-Churn.csv`
- [ ] All package directories contain `__init__.py` files
- [ ] Data directories created: `data/raw/`, `data/processed/`, `artifacts/`
- [ ] Run validation: `python scripts/validate_setup.py`

---

## 🚀 Verify Installation

```bash
# Step 1: Check Python
python --version

# Step 2: Check packages
python scripts/validate_setup.py

# Step 3: If all pass, you're ready!
# Run training:
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn
```

---

## 🔄 What Was Fixed

The following issues were identified and corrected:

| Issue | Fix | Status |
|-------|-----|--------|
| Bad import `from posthog import project_root` | Removed, using `os.path` | ✅ Fixed |
| Missing `__init__.py` in packages | Created in all src/* directories | ✅ Fixed |
| Missing data directories | Created `data/raw`, `data/processed`, `artifacts` | ✅ Fixed |
| Missing `.gitkeep` files | Added to track empty directories | ✅ Fixed |
| Incomplete documentation | Created PROJECT_SETUP.md & REQUIREMENTS.md | ✅ Fixed |

---

## 📝 Environment Variables (Optional)

Create `.env` file for configuration:

```env
# MLflow
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_EXPERIMENT_NAME=Telco Churn

# FastAPI
API_HOST=0.0.0.0
API_PORT=8000

# Model
MODEL_THRESHOLD=0.35
TEST_SIZE=0.2
```

---

## 🐳 Docker Requirements (Optional)

To containerize the app:

- **Docker**: 20.10+ installed
- **Memory**: 2GB minimum
- **Base Image**: `python:3.11-slim`

```bash
docker build -t telco-churn-app .
docker run -p 8000:8000 telco-churn-app
```

---

## 🆘 Common Issues & Fixes

### Issue: "Module not found: src"
```bash
# Fix: Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"  # Linux/Mac
set PYTHONPATH=%CD%\src                       # Windows
```

### Issue: "File not found: data/raw/Telco-Customer-Churn.csv"
```bash
# Fix: Download dataset from Kaggle and place in data/raw/
```

### Issue: "No module named 'great_expectations'"
```bash
# Fix: Install dependencies
pip install -r requirements.txt
```

### Issue: "Port 8000 already in use"
```bash
# Fix: Use different port
python -m uvicorn src.app.main:app --port 8001
```

---

## 📊 Resource Usage

During execution, expect:

| Operation | CPU | Memory | Time |
|-----------|-----|--------|------|
| Data loading | 10% | 500MB | <1s |
| Validation | 20% | 300MB | 2-5s |
| Feature engineering | 30% | 800MB | 5-10s |
| Model training | 100% (all cores) | 2GB | 30-60s |
| Prediction (single) | 5% | 100MB | <100ms |

---

## 🎓 Learning Resources

- **MLflow**: https://mlflow.org/docs
- **XGBoost**: https://xgboost.readthedocs.io/
- **FastAPI**: https://fastapi.tiangolo.com/
- **Gradio**: https://gradio.app/
- **Great Expectations**: https://great-expectations.io/

---

## 📞 Support

If you encounter issues:

1. Run `python scripts/validate_setup.py` to diagnose
2. Check MLflow UI: `mlflow ui` → http://localhost:5000
3. Review error messages in console output
4. Check project documentation: PROJECT_SETUP.md

**You're all set! Ready to build your ML model. 🚀**
