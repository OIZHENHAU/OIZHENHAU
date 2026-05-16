"""
evaluate_models.py
------------------
Evaluates Isolation Forest and LOF using engagement_rate > 1.0 as a
proxy ground-truth label (no manually labelled data available).

Run from the backend/ directory:
    python evaluate_models.py
"""

import sys
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

sys.path.insert(0, '.')
from analysis.preprocessing import load_data, get_post_features

# ── 1. Load & prepare data ────────────────────────────────────────────────────
print("Loading dataset...")
df = load_data()
X, feature_names = get_post_features(df)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Proxy ground-truth: engagement_rate > 1.0 → suspicious (1), else real (0)
y_true = (df['engagement_rate'] > 1.0).astype(int).values

print(f"Total posts      : {len(y_true):,}")
print(f"Suspicious (ER>1): {y_true.sum():,}  ({y_true.mean()*100:.2f}%)")
print(f"Real             : {(1-y_true).sum():,}  ({(1-y_true).mean()*100:.2f}%)")
print()

# ── 2. Train models ───────────────────────────────────────────────────────────
CONTAMINATION = 'auto'

print(f"Training Isolation Forest  (contamination={CONTAMINATION})...")
if_model = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    random_state=42,
    n_jobs=-1,
)
if_model.fit(X_scaled)

print(f"Training LOF               (contamination={CONTAMINATION})...")
lof_model = LocalOutlierFactor(
    n_neighbors=20,
    contamination=CONTAMINATION,
    novelty=True,
)
lof_model.fit(X_scaled)
print()

# ── 3. Predict (-1 = outlier/suspicious, 1 = inlier/real) ────────────────────
if_pred  = (if_model.predict(X_scaled)  == -1).astype(int)
lof_pred = (lof_model.predict(X_scaled) == -1).astype(int)

print(f"IF  accounts flagged : {if_pred.sum():,}  ({if_pred.mean()*100:.2f}%)")
print(f"LOF accounts flagged : {lof_pred.sum():,}  ({lof_pred.mean()*100:.2f}%)")
print()

# ── 4. Print metrics ──────────────────────────────────────────────────────────
def print_metrics(y_pred, model_name):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    print(f"{'='*45}")
    print(f"  {model_name}")
    print(f"{'='*45}")
    print(f"  Confusion Matrix:")
    print(f"    TP (correctly flagged suspicious) : {tp}")
    print(f"    FP (real flagged as suspicious)   : {fp}")
    print(f"    TN (correctly identified real)    : {tn}")
    print(f"    FN (suspicious missed)             : {fn}")
    print(f"  ----------------------------------------")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1-score  : {f1:.4f}")
    print()

print_metrics(if_pred,  "Isolation Forest")
print_metrics(lof_pred, "Local Outlier Factor (LOF)")
