"""
evaluate_models.py
------------------
Evaluates Isolation Forest and LOF for suspicious post detection.

Label strategy
--------------
engagement_rate > 1.0 is used as a PROXY ground-truth (a rate above 100 %
is physically impossible and strongly indicates inflated / bot-driven
engagement).  Crucially, engagement_rate is NO LONGER in POST_FEATURE_COLS,
so the models must infer suspicious behaviour from the remaining 10 features
(toxicity, rates, sentiment, growth, etc.) — eliminating the circular logic.

A silhouette score is also computed as a fully label-free quality check.

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
    f1_score, confusion_matrix, silhouette_score,
)

sys.path.insert(0, '.')
from analysis.preprocessing import load_data, get_post_features

# ── 1. Load & prepare data ────────────────────────────────────────────────────
print("Loading dataset...")
df = load_data()
X, feature_names = get_post_features(df)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Proxy ground-truth: engagement_rate > 1.0 → suspicious
# engagement_rate is NOT in the feature matrix, so this is a genuine test.
y_true = (df['engagement_rate'] > 1.0).astype(int).values
CONTAMINATION = float(y_true.mean())          # match actual suspicious rate

print(f"Features used    : {feature_names}")
print(f"Total posts      : {len(y_true):,}")
print(f"Suspicious (ER>1): {y_true.sum():,}  ({y_true.mean()*100:.2f}%)")
print(f"Real             : {(1-y_true).sum():,}  ({(1-y_true).mean()*100:.2f}%)")
print(f"Contamination    : {CONTAMINATION:.4f}")
print()

# ── 2. Train models ───────────────────────────────────────────────────────────
print(f"Training Isolation Forest  (contamination={CONTAMINATION:.4f})...")
if_model = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    random_state=42,
    n_jobs=-1,
)
if_model.fit(X_scaled)

print(f"Training LOF               (contamination={CONTAMINATION:.4f})...")
lof_model = LocalOutlierFactor(
    n_neighbors=20,
    contamination=CONTAMINATION,
    novelty=True,
)
lof_model.fit(X_scaled)
print()

# ── 3. Predict (-1 = outlier/suspicious, 1 = inlier/real) ────────────────────
if_pred   = (if_model.predict(X_scaled)  == -1).astype(int)
lof_pred  = (lof_model.predict(X_scaled) == -1).astype(int)

# Anomaly scores (lower = more anomalous for IF; more negative = more anomalous for LOF)
if_scores  = if_model.decision_function(X_scaled)
lof_scores = lof_model.decision_function(X_scaled)

print(f"IF  accounts flagged : {if_pred.sum():,}  ({if_pred.mean()*100:.2f}%)")
print(f"LOF accounts flagged : {lof_pred.sum():,}  ({lof_pred.mean()*100:.2f}%)")
print()

# ── 4. Silhouette score (fully unsupervised, label-free) ─────────────────────
# Use IF predictions as cluster labels for silhouette; sample for speed.
sample_size = min(10_000, len(X_scaled))
rng = np.random.default_rng(42)
idx = rng.choice(len(X_scaled), sample_size, replace=False)

sil_if  = silhouette_score(X_scaled[idx], if_pred[idx])
sil_lof = silhouette_score(X_scaled[idx], lof_pred[idx])

print("Silhouette Scores  (-1 worst -> +1 best; no labels needed)")
print(f"  IF  : {sil_if:.4f}")
print(f"  LOF : {sil_lof:.4f}")
print()

# ── 5. Proxy-label metrics ────────────────────────────────────────────────────
def print_metrics(y_pred, scores, model_name):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    # Anomaly score gap: suspicious vs real accounts
    score_susp = scores[y_pred == 1].mean()
    score_real = scores[y_pred == 0].mean()

    print(f"{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  Proxy-label evaluation (ER > 1.0  =  suspicious)")
    print(f"  TP (correctly flagged suspicious) : {tp:,}")
    print(f"  FP (real flagged as suspicious)   : {fp:,}")
    print(f"  TN (correctly identified real)    : {tn:,}")
    print(f"  FN (suspicious missed)            : {fn:,}")
    print(f"  --------------------------------------------------")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  --------------------------------------------------")
    print(f"  Avg anomaly score — suspicious : {score_susp:.4f}")
    print(f"  Avg anomaly score — real       : {score_real:.4f}")
    print()

print_metrics(if_pred,  if_scores,  "Isolation Forest")
print_metrics(lof_pred, lof_scores, "Local Outlier Factor (LOF)")

# ── 6. Top 20 most anomalous posts (for qualitative inspection) ───────────────
print("Top 20 most anomalous posts (Isolation Forest)")
print("-" * 60)
top20_idx = np.argsort(if_scores)[:20]
for rank, i in enumerate(top20_idx, 1):
    er   = df['engagement_rate'].iloc[i]
    tox  = df['toxicity_score'].iloc[i]
    uran = df['username_randomness'].iloc[i]
    uid  = df['user_id'].iloc[i] if 'user_id' in df.columns else i
    print(f"  #{rank:02d}  user={uid}  ER={er:.3f}  tox={tox:.3f}  username_rand={uran:.3f}  score={if_scores[i]:.4f}")
