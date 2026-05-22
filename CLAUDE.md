# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Training Pipeline
```bash
# Run the complete ML training pipeline
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn

# Prepare processed data only
python scripts/prepare_processed_data.py
```

### Testing
```bash
# Test data processing and feature engineering
python scripts/test_pipeline_phase1_data_features.py

# Test model training and evaluation
python scripts/test_pipeline_phase2_modeling.py

# Test FastAPI endpoints
python scripts/test_fastapi.py
```

### Local Development
```bash
# Run the FastAPI + Gradio application locally
python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000

# Alternative app entry point
python -m uvicorn src.app.app:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
# Build and run the containerized application
docker build -t telco-churn-app .
docker run -p 8000:8000 telco-churn-app
```

## Architecture Overview

### ML Pipeline Flow
This project implements a complete MLOps pipeline with two distinct phases:

**Training Pipeline** (`scripts/run_pipeline.py`):
1. **Data Loading** → **Data Validation** (Great Expectations) → **Preprocessing** → **Feature Engineering** → **XGBoost Training** → **MLflow Logging**
2. All artifacts (model, feature columns, preprocessing logic) are stored in MLflow for reproducibility

**Serving Pipeline** (`src/app/main.py` + `src/serving/inference.py`):
1. **FastAPI REST API** (`/predict` endpoint) + **Gradio Web UI** (`/ui` endpoint) → **MLflow Model Loading** → **Feature Transformation** → **Prediction**
2. Feature processing mirrors training-time transformations for consistency

### MLflow Integration Patterns
- **Experiment Name**: "Telco Churn" (default, can be overridden)
- **Tracking URI**: File-based at `{project_root}/mlruns`
- **Logged Artifacts**: `model/`, `feature_columns.txt`, `preprocessing.pkl`
- **Tracked Metrics**: precision, recall, f1, roc_auc, train_time, pred_time, data_quality_pass
- **Parameters**: model type, threshold (default 0.35), test_size (default 0.2)

### Feature Engineering Consistency
Critical pattern: Training and serving must use identical feature transformations.

**Training** (`src/features/build_features.py`):
- Binary features (Yes/No, Male/Female) → deterministic 0/1 mapping
- Multi-category features → one-hot encoding with `drop_first=True`
- Boolean columns → integers

**Serving** (`src/serving/inference.py`):
- Uses fixed `BINARY_MAP` dictionary for consistent binary encoding
- Applies `pd.get_dummies()` with same parameters as training
- Feature alignment via `FEATURE_COLS` from training artifacts

### Model Loading and Serving
- **Container Path**: Model loaded from `/app/model` (MLflow pyfunc format)
- **Feature Order**: Enforced using `feature_columns.txt` from training
- **Prediction Format**: Returns "Likely to churn" or "Not likely to churn" strings

### Data Validation
- **Tool**: Great Expectations with custom validation suite
- **Location**: `src/utils/validate_data.py`
- **Checks**: CustomerID presence, gender values, numeric ranges for tenure/charges
- **Integration**: Results logged to MLflow as `data_quality_pass` metric

### Docker Containerization
- **Base Image**: `python:3.11-slim`
- **Key Setting**: `PYTHONPATH=/app/src` for proper module imports
- **Model Artifacts**: Specific MLflow run copied to `/app/model` during build
- **Serving**: uvicorn with FastAPI app on port 8000

### CI/CD Pipeline
- **Trigger**: Push to main branch
- **Actions**: Build Docker image → Push to Docker Hub (`anasriad8/telco-fastapi:latest`)
- **Requirements**: `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets
- **Deployment**: Manual ECS service update (AWS Fargate + ALB)

## Key Implementation Details

### XGBoost Model Configuration
Optimized hyperparameters are hardcoded in `scripts/run_pipeline.py:100-110`:
- `n_estimators=301`, `learning_rate=0.034`, `max_depth=7`
- `scale_pos_weight` calculated dynamically for class imbalance handling

### API Endpoints
- `GET /` - Health check returning `{"status": "ok"}`
- `POST /predict` - Accepts `CustomerData` Pydantic model with 18 customer attributes
- `/ui` - Gradio interface mounted via `gr.mount_gradio_app()`

### File System Layout
- `data/raw/` - Original datasets
- `data/processed/` - Cleaned datasets
- `mlruns/` - MLflow experiment tracking database
- `artifacts/` - Shared preprocessing artifacts (`feature_columns.json`, `preprocessing.pkl`)
- `src/serving/model/` - Local MLflow run copies for development

### Development Notes
- No formal test suite exists; use manual test scripts in `scripts/test_*.py`
- MLflow UI can be accessed with: `mlflow ui --backend-store-uri file:./mlruns`
- The project uses file-based MLflow tracking (not a tracking server)
- Model serving expects exact feature column order from training time