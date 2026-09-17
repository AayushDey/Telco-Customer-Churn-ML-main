"""
Prediction History Database
============================
Manages a lightweight SQLite log of all predictions made through the app.
Enables the History tab in the dashboard.
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any

DB_PATH = os.path.join(os.path.dirname(__file__), "predictions.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the predictions table if it doesn't exist."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp         TEXT    NOT NULL,
            prediction        TEXT    NOT NULL,
            probability       REAL    NOT NULL,
            confidence        TEXT    NOT NULL,
            tenure            INTEGER,
            monthly_charges   REAL,
            total_charges     REAL,
            contract          TEXT,
            internet_service  TEXT,
            payment_method    TEXT,
            payload           TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(payload: dict, result: dict) -> int:
    """
    Persist one prediction to the database.

    Args:
        payload: Raw customer input dict.
        result:  Dict with keys: prediction, probability, confidence, shap_values.

    Returns:
        The auto-assigned row ID.
    """
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO predictions
           (timestamp, prediction, probability, confidence,
            tenure, monthly_charges, total_charges,
            contract, internet_service, payment_method, payload)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            result["prediction"],
            result["probability"],
            result["confidence"],
            payload.get("tenure", 0),
            payload.get("MonthlyCharges", 0.0),
            payload.get("TotalCharges", 0.0),
            payload.get("Contract", ""),
            payload.get("InternetService", ""),
            payload.get("PaymentMethod", ""),
            json.dumps(payload),
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Return the most recent predictions, newest first."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT id, timestamp, prediction, probability, confidence,
                  tenure, monthly_charges, total_charges,
                  contract, internet_service, payment_method
           FROM predictions
           ORDER BY id DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats() -> Dict[str, Any]:
    """Aggregate statistics for the history dashboard."""
    conn = _get_conn()
    row = conn.execute(
        """SELECT
               COUNT(*) AS total,
               SUM(CASE WHEN prediction = 'Likely to churn' THEN 1 ELSE 0 END) AS churners,
               ROUND(AVG(probability) * 100, 1) AS avg_probability
           FROM predictions"""
    ).fetchone()
    conn.close()
    if row and row["total"]:
        return dict(row)
    return {"total": 0, "churners": 0, "avg_probability": 0.0}
