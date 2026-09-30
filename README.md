# Credit Risk Model API 🏦

A production-ready credit risk prediction service built with FastAPI, containerised with Docker, and monitored with a data drift report.

## Architecture

```
POST /predict → FastAPI → joblib model → JSON response
                    ↓
              Structured logs (JSON)
```

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/health` | GET | Model status |
| `/predict` | POST | Credit risk prediction |
| `/features` | GET | Feature descriptions |
| `/docs` | GET | Interactive API docs (Swagger) |

## curl Example

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "annual_income": 55000,
    "age": 35,
    "debt_to_income": 0.3,
    "credit_history_yrs": 8,
    "num_open_accounts": 4,
    "missed_payments": 0,
    "loan_amount": 15000,
    "employment_years": 5
  }'
```

**Response:**
```json
{
  "default_probability": 0.0823,
  "risk_label": "LOW",
  "request_id": "a1b2c3d4",
  "latency_ms": 12.4
}
```

## How to Run

### Option 1: Docker (recommended)

```bash
# Build
docker build -t credit-risk-api .

# Run
docker run -p 8000:8000 credit-risk-api

# Test
curl http://localhost:8000/health
```

### Option 2: Local

```bash
pip install -r requirements.txt
python train_model.py          # generates model.joblib + reference_data.csv
python generate_drift_report.py # generates drift_report.html
uvicorn app.main:app --reload
```

## Load Test

```bash
python tools/load_test.py --url http://localhost:8000/predict --n 100
```

p95 latency target: **< 200ms**

## Drift Report

Open `drift_report.html` in any browser to see the data drift analysis comparing training vs production distributions.

## Model

- **Algorithm:** GradientBoostingClassifier
- **Features:** annual_income, age, debt_to_income, credit_history_yrs, num_open_accounts, missed_payments, loan_amount, employment_years
- **Target:** Binary default prediction (0 = no default, 1 = default)
- **Metric:** ROC-AUC ~0.85

## Privacy

Raw PII fields (income, loan amount) are hashed (SHA-256) before logging. Age and ratios are logged as-is since they are not directly identifying.

## Reflection

The hardest part of productionising an ML model is not the model itself — it's the plumbing: consistent feature ordering, input validation, structured logging without PII, and a drift signal that non-technical stakeholders can understand. This project covers all four.
