"""
generate_drift_report.py — Generate Evidently drift report.
Run: python generate_drift_report.py
Outputs: drift_report.html
"""
import numpy as np
import pandas as pd

# Load reference data
reference = pd.read_csv("reference_data.csv")

# Simulate current (production) data with slight drift
np.random.seed(99)
N = 1000
current = pd.DataFrame({
    "annual_income"      : np.random.lognormal(10.9, 0.65, N),   # slightly higher income
    "age"                : np.random.randint(22, 70, N).astype(float),
    "debt_to_income"     : np.clip(np.random.beta(2.5, 4.5, N), 0, 1),  # slightly higher DTI
    "credit_history_yrs" : np.clip(np.random.exponential(6, N), 0, 40),
    "num_open_accounts"  : np.random.poisson(4.5, N).astype(float),
    "missed_payments"    : np.random.poisson(0.7, N).astype(float),  # more missed payments
    "loan_amount"        : np.random.lognormal(9.7, 0.85, N),
    "employment_years"   : np.clip(np.random.exponential(4.5, N), 0, 40),
})

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    report.run(reference_data=reference, current_data=current)
    report.save_html("drift_report.html")
    print("✅ drift_report.html generated with Evidently!")

except ImportError:
    # Fallback: generate a clean HTML report manually
    print("Evidently not installed — generating manual HTML report...")
    _generate_manual_report(reference, current)

def _generate_manual_report(reference, current):
    import json

    features = reference.columns.tolist()
    rows = ""
    drifted = []

    for feat in features:
        ref_mean = reference[feat].mean()
        cur_mean = current[feat].mean()
        ref_std  = reference[feat].std()
        cur_std  = current[feat].std()
        pct_change = abs(cur_mean - ref_mean) / (ref_mean + 1e-9) * 100
        is_drift = pct_change > 10
        if is_drift:
            drifted.append(feat)
        status = "🔴 DRIFT" if is_drift else "🟢 OK"
        rows += f"""
        <tr class="{'drift' if is_drift else ''}">
            <td><b>{feat}</b></td>
            <td>{ref_mean:.4f}</td>
            <td>{cur_mean:.4f}</td>
            <td>{ref_std:.4f}</td>
            <td>{cur_std:.4f}</td>
            <td>{pct_change:.1f}%</td>
            <td>{status}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Credit Risk Model — Drift Report</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; }}
  h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
  h2 {{ color: #34495e; margin-top: 30px; }}
  .summary-box {{ background: #f8f9fa; border-left: 4px solid #3498db; padding: 15px 20px; margin: 20px 0; border-radius: 4px; }}
  .alert {{ background: #ffeeba; border-left: 4px solid #ffc107; padding: 15px 20px; margin: 20px 0; border-radius: 4px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
  th {{ background: #3498db; color: white; padding: 10px; text-align: left; }}
  td {{ padding: 9px 10px; border-bottom: 1px solid #dee2e6; }}
  tr.drift {{ background: #fff3cd; }}
  tr:hover {{ background: #f1f3f4; }}
  .badge {{ padding: 3px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; }}
  .badge-ok {{ background: #d4edda; color: #155724; }}
  .badge-drift {{ background: #f8d7da; color: #721c24; }}
  footer {{ margin-top: 40px; color: #888; font-size: 13px; border-top: 1px solid #eee; padding-top: 15px; }}
</style>
</head>
<body>
<h1>🏦 Credit Risk Model — Data Drift Report</h1>

<div class="summary-box">
  <b>Report generated:</b> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
  <b>Reference dataset:</b> {len(reference):,} rows (training data)<br>
  <b>Current dataset:</b> {len(current):,} rows (simulated production)<br>
  <b>Features analysed:</b> {len(features)}<br>
  <b>Features with drift (&gt;10% mean shift):</b> <span class="badge badge-{'drift' if drifted else 'ok'}">{len(drifted)}</span>
</div>

{'<div class="alert">⚠️ <b>Drift detected</b> in: ' + ', '.join(f'<b>{f}</b>' for f in drifted) + '. Consider retraining the model.</div>' if drifted else '<div class="summary-box">✅ No significant drift detected. Model distribution looks stable.</div>'}

<h2>Feature Distribution Comparison</h2>
<table>
  <thead>
    <tr>
      <th>Feature</th>
      <th>Ref Mean</th>
      <th>Curr Mean</th>
      <th>Ref Std</th>
      <th>Curr Std</th>
      <th>Mean Shift</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>

<h2>What This Means</h2>
<ul>
  <li><b>🟢 OK:</b> Feature distribution in production is similar to training data. Model predictions should be reliable.</li>
  <li><b>🔴 DRIFT:</b> Feature distribution has shifted &gt;10% from training. This may degrade model accuracy. Consider retraining.</li>
</ul>

<h2>Recommended Actions</h2>
<ol>
  <li>Monitor drifted features weekly using this report.</li>
  <li>If drift persists for 2+ weeks, collect new labels and retrain.</li>
  <li>Run A/B test between current model and retrained model before full rollout.</li>
</ol>

<footer>
  Generated by credit-risk-api drift monitoring pipeline · Model: GradientBoostingClassifier · 
  Reference: training split · Current: simulated production data
</footer>
</body>
</html>"""

    with open("drift_report.html", "w") as f:
        f.write(html)
    print("✅ drift_report.html generated (manual fallback)!")

# Call fallback if evidently not available
import sys
if __name__ == "__main__":
    try:
        from evidently.report import Report
    except ImportError:
        _generate_manual_report(reference, current)
