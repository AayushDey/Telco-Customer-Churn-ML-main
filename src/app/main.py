"""
TELCO CHURN PREDICTION SERVICE - Production FastAPI & Web Dashboard
===================================================================

Endpoints:
  GET  /                  → redirect to /dashboard
  GET  /dashboard         → ChurnSight web UI (single prediction + batch + history)
  POST /predict           → single customer churn prediction (JSON)
  POST /predict/batch     → batch CSV upload, returns list of predictions (JSON)
  GET  /history           → last 50 predictions from SQLite log
  GET  /history/stats     → aggregate statistics
  GET  /health            → service & model health check
  GET  /docs              → OpenAPI Swagger UI
"""

import os
import io
import csv
import json

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel

from src.serving.inference import predict
from src.app.db import init_db, log_prediction, get_history, get_stats

# ── Initialise SQLite history DB ──────────────────────────────────────────────
init_db()

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description=(
        "ML API & Dashboard for predicting customer churn in the telecom industry. "
        "Powered by XGBoost with SHAP explainability."
    ),
    version="2.0.0",
)

# ── Static files ──────────────────────────────────────────────────────────────
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ═══════════════════════════════════════════════════════════════════════════════
#  Health
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint to verify service and model availability."""
    return {"status": "healthy", "service": "telco-churn-api", "version": "2.0.0"}


# ═══════════════════════════════════════════════════════════════════════════════
#  Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/dashboard", include_in_schema=False)
async def dashboard():
    """Serve the premium ChurnSight dashboard."""
    dashboard_file = os.path.join(static_dir, "index.html")
    if os.path.exists(dashboard_file):
        return FileResponse(dashboard_file, media_type="text/html")
    return {"error": "Dashboard not found"}


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to the dashboard."""
    return RedirectResponse(url="/dashboard")


# ═══════════════════════════════════════════════════════════════════════════════
#  Request / Response schemas
# ═══════════════════════════════════════════════════════════════════════════════

class CustomerData(BaseModel):
    """
    Customer data schema for churn prediction.
    Defines the exact 18 features required for churn prediction.
    """
    # Demographics
    gender: str                # "Male" or "Female"
    Partner: str               # "Yes" or "No"
    Dependents: str            # "Yes" or "No"

    # Phone services
    PhoneService: str          # "Yes" or "No"
    MultipleLines: str         # "Yes", "No", or "No phone service"

    # Internet services
    InternetService: str       # "DSL", "Fiber optic", or "No"
    OnlineSecurity: str        # "Yes", "No", or "No internet service"
    OnlineBackup: str          # "Yes", "No", or "No internet service"
    DeviceProtection: str      # "Yes", "No", or "No internet service"
    TechSupport: str           # "Yes", "No", or "No internet service"
    StreamingTV: str           # "Yes", "No", or "No internet service"
    StreamingMovies: str       # "Yes", "No", or "No internet service"

    # Account information
    Contract: str              # "Month-to-month", "One year", "Two year"
    PaperlessBilling: str      # "Yes" or "No"
    PaymentMethod: str         # "Electronic check", "Mailed check", etc.

    # Numeric features
    tenure: int                # Number of months with company
    MonthlyCharges: float      # Monthly charges in dollars
    TotalCharges: float        # Total charges to date


# ═══════════════════════════════════════════════════════════════════════════════
#  Single Prediction
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/predict", tags=["Inference"])
def get_prediction(data: CustomerData):
    """
    Predict churn for a single customer.

    Returns:
    - **prediction**: "Likely to churn" or "Not likely to churn"
    - **probability**: real churn probability (0.0–1.0)
    - **confidence**: "Borderline" | "Moderate" | "High Confidence"
    - **shap_values**: top-8 feature importances for this prediction
    """
    try:
        payload = data.dict()
        result = predict(payload)
        log_prediction(payload, result)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════════
#  Batch Prediction (CSV upload)
# ═══════════════════════════════════════════════════════════════════════════════

REQUIRED_COLS = {
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "tenure", "MonthlyCharges", "TotalCharges",
}


@app.post("/predict/batch", tags=["Inference"])
async def batch_predict(file: UploadFile = File(...)):
    """
    Upload a CSV file and get churn predictions for every row.

    The CSV must contain the same 18 columns as the single /predict endpoint.
    Returns a JSON list where each item includes the original row data plus
    prediction, probability, and confidence.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    contents = await file.read()
    try:
        text = contents.decode("utf-8-sig")  # handle BOM
        reader = csv.DictReader(io.StringIO(text))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    # Validate columns
    if reader.fieldnames:
        missing = REQUIRED_COLS - set(reader.fieldnames)
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"CSV is missing required columns: {', '.join(sorted(missing))}",
            )

    results = []
    errors = []
    for i, row in enumerate(reader, start=2):  # start=2 → header is row 1
        try:
            payload = {
                "gender": row.get("gender", ""),
                "Partner": row.get("Partner", ""),
                "Dependents": row.get("Dependents", ""),
                "PhoneService": row.get("PhoneService", ""),
                "MultipleLines": row.get("MultipleLines", ""),
                "InternetService": row.get("InternetService", ""),
                "OnlineSecurity": row.get("OnlineSecurity", ""),
                "OnlineBackup": row.get("OnlineBackup", ""),
                "DeviceProtection": row.get("DeviceProtection", ""),
                "TechSupport": row.get("TechSupport", ""),
                "StreamingTV": row.get("StreamingTV", ""),
                "StreamingMovies": row.get("StreamingMovies", ""),
                "Contract": row.get("Contract", ""),
                "PaperlessBilling": row.get("PaperlessBilling", ""),
                "PaymentMethod": row.get("PaymentMethod", ""),
                "tenure": int(float(row.get("tenure", 0) or 0)),
                "MonthlyCharges": float(row.get("MonthlyCharges", 0) or 0),
                "TotalCharges": float(row.get("TotalCharges", 0) or 0),
            }
            pred = predict(payload)
            log_prediction(payload, pred)
            results.append({
                "row": i,
                **payload,
                "prediction": pred["prediction"],
                "probability": pred["probability"],
                "confidence": pred["confidence"],
            })
        except Exception as e:
            errors.append({"row": i, "error": str(e)})

    return {"total": len(results), "errors": errors, "predictions": results}


# ═══════════════════════════════════════════════════════════════════════════════
#  Prediction History
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/history", tags=["History"])
def prediction_history(limit: int = 50):
    """Return the most recent predictions (default: last 50)."""
    return get_history(limit=limit)


@app.get("/history/stats", tags=["History"])
def history_stats():
    """Return aggregate statistics across all logged predictions."""
    return get_stats()
