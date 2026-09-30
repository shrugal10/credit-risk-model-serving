"""
app/main.py — FastAPI credit-risk prediction service.
"""
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ── Structured logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":%(message)s}',
    datefmt="%Y-%m-%dT%H:%M:%S"
)
logger = logging.getLogger("credit_risk_api")

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / "model.joblib"
try:
    model = joblib.load(MODEL_PATH)
    logger.info('"Model loaded successfully"')
except Exception as e:
    logger.error(f'"Failed to load model: {e}"')
    model = None

FEATURES = [
    "annual_income", "age", "debt_to_income", "credit_history_yrs",
    "num_open_accounts", "missed_payments", "loan_amount", "employment_years"
]

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predicts probability of loan default using a GradientBoosting model.",
    version="1.0.0"
)

# ── Request / Response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    annual_income      : float = Field(..., gt=0,    description="Annual income in USD")
    age                : float = Field(..., ge=18, le=100, description="Applicant age")
    debt_to_income     : float = Field(..., ge=0, le=1,   description="Debt-to-income ratio (0-1)")
    credit_history_yrs : float = Field(..., ge=0,   description="Years of credit history")
    num_open_accounts  : float = Field(..., ge=0,   description="Number of open credit accounts")
    missed_payments    : float = Field(..., ge=0,   description="Number of missed payments")
    loan_amount        : float = Field(..., gt=0,   description="Requested loan amount in USD")
    employment_years   : float = Field(..., ge=0,   description="Years of employment")

    model_config = {"json_schema_extra": {
        "example": {
            "annual_income": 55000,
            "age": 35,
            "debt_to_income": 0.3,
            "credit_history_yrs": 8,
            "num_open_accounts": 4,
            "missed_payments": 0,
            "loan_amount": 15000,
            "employment_years": 5
        }
    }}


class PredictResponse(BaseModel):
    default_probability : float
    risk_label          : str
    request_id          : str
    latency_ms          : float


# ── Middleware: request logging ───────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info(json.dumps({
        "path"       : request.url.path,
        "method"     : request.method,
        "status"     : response.status_code,
        "latency_ms" : round(duration_ms, 2)
    }))
    return response


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "credit-risk-api", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
def health():
    return {
        "status"      : "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "features"    : FEATURES
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(payload: PredictRequest, request: Request):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()

    # Build feature vector
    features = pd.DataFrame([{
        "annual_income"      : payload.annual_income,
        "age"                : payload.age,
        "debt_to_income"     : payload.debt_to_income,
        "credit_history_yrs" : payload.credit_history_yrs,
        "num_open_accounts"  : payload.num_open_accounts,
        "missed_payments"    : payload.missed_payments,
        "loan_amount"        : payload.loan_amount,
        "employment_years"   : payload.employment_years,
    }])[FEATURES]

    prob = float(model.predict_proba(features)[0, 1])

    # Risk label
    if prob < 0.2:
        risk_label = "LOW"
    elif prob < 0.5:
        risk_label = "MEDIUM"
    else:
        risk_label = "HIGH"

    latency_ms = (time.time() - start) * 1000

    # Structured log — hash income/loan to avoid logging PII
    logger.info(json.dumps({
        "event"              : "prediction",
        "request_id"         : hashlib.md5(str(time.time()).encode()).hexdigest()[:8],
        "income_hash"        : hashlib.sha256(str(payload.annual_income).encode()).hexdigest()[:8],
        "age"                : int(payload.age),
        "debt_to_income"     : round(payload.debt_to_income, 3),
        "missed_payments"    : int(payload.missed_payments),
        "default_probability": round(prob, 4),
        "risk_label"         : risk_label,
        "latency_ms"         : round(latency_ms, 2)
    }))

    return PredictResponse(
        default_probability=round(prob, 4),
        risk_label=risk_label,
        request_id=hashlib.md5(str(time.time()).encode()).hexdigest()[:8],
        latency_ms=round(latency_ms, 2)
    )


@app.get("/features", tags=["Info"])
def get_features():
    """Return expected feature names and descriptions."""
    return {
        "features": [
            {"name": "annual_income",       "type": "float", "description": "Annual income in USD"},
            {"name": "age",                  "type": "float", "description": "Applicant age (18-100)"},
            {"name": "debt_to_income",       "type": "float", "description": "Debt-to-income ratio (0-1)"},
            {"name": "credit_history_yrs",   "type": "float", "description": "Years of credit history"},
            {"name": "num_open_accounts",    "type": "float", "description": "Number of open credit accounts"},
            {"name": "missed_payments",      "type": "float", "description": "Number of missed payments"},
            {"name": "loan_amount",          "type": "float", "description": "Requested loan amount in USD"},
            {"name": "employment_years",     "type": "float", "description": "Years of employment"},
        ]
    }
