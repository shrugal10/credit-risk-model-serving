"""
train_model.py — Train and save the credit risk model.
Run this once locally to generate model.joblib and reference_data.csv
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
import joblib

# ── Reproducible synthetic credit-risk dataset ────────────────────────────────
# Features mirror real credit risk: income, age, debt ratio, credit history etc.
np.random.seed(42)
N = 5000

df = pd.DataFrame({
    "annual_income"      : np.random.lognormal(10.8, 0.6, N),      # ~$50k median
    "age"                : np.random.randint(18, 75, N).astype(float),
    "debt_to_income"     : np.clip(np.random.beta(2, 5, N), 0, 1),
    "credit_history_yrs" : np.clip(np.random.exponential(7, N), 0, 40),
    "num_open_accounts"  : np.random.poisson(4, N).astype(float),
    "missed_payments"    : np.random.poisson(0.5, N).astype(float),
    "loan_amount"        : np.random.lognormal(9.5, 0.8, N),
    "employment_years"   : np.clip(np.random.exponential(5, N), 0, 40),
})

# Target: default probability influenced by features
log_odds = (
    -3.0
    + df["debt_to_income"] * 3.0
    + df["missed_payments"] * 1.2
    - df["credit_history_yrs"] * 0.1
    - np.log1p(df["annual_income"]) * 0.3
    + np.log1p(df["loan_amount"]) * 0.4
)
prob = 1 / (1 + np.exp(-log_odds))
df["default"] = (np.random.random(N) < prob).astype(int)

print(f"Default rate: {df['default'].mean():.1%}")

FEATURES = [c for c in df.columns if c != "default"]
X = df[FEATURES]
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save reference data for drift detection (training distribution)
X_train.to_csv("reference_data.csv", index=False)
print(f"Reference data saved: {len(X_train)} rows")

# Train model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, random_state=42
    ))
])
model.fit(X_train, y_train)

# Evaluate
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"Test ROC-AUC: {auc:.4f}")
print(classification_report(y_test, model.predict(X_test)))

# Save model
joblib.dump(model, "model.joblib")
print("Model saved: model.joblib")
print(f"Features: {FEATURES}")
