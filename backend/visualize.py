"""
visualize.py
------------
Generates diagnostic plots for the Influencer Clout Detective project.

Plots saved to:  backend/plots/
Run from the backend/ directory:
    python visualize.py

Diagrams produced
-----------------
1. correlation_matrix.png      — Pearson correlation heat-map of key features
2. feature_histograms.png      — Distribution of every post feature (suspicious vs normal)
3. feature_importance.png      — Isolation Forest feature importance (mean depth proxy)
4. anomaly_score_dist.png      — IF & LOF anomaly-score distributions
5. confusion_matrices.png      — Confusion matrices for IF and LOF
6. pca_scatter.png             — PCA 2-D scatter: normal vs suspicious points
7. boxplots.png                — Box-plots of each feature split by suspicious label
8. roc_pr_curves.png           — ROC and Precision-Recall curves for both models
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # headless – no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
)

sys.path.insert(0, '.')
from analysis.preprocessing import load_data, get_post_features, POST_FEATURE_COLS

# ── Config ─────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE = {'normal': '#4C9BE8', 'suspicious': '#E85C4C'}
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.05)

# ── Data & models ──────────────────────────────────────────────────────────────
print("Loading data …")
df = load_data()
X, feature_names = get_post_features(df)

scaler    = StandardScaler()
X_scaled  = scaler.fit_transform(X)
y_true    = (df['engagement_rate'] > 1.0).astype(int).values
CONTAM    = float(y_true.mean())

print(f"Training Isolation Forest  (contamination={CONTAM:.4f}) …")
if_model = IsolationForest(n_estimators=200, contamination=CONTAM, random_state=42, n_jobs=-1)
if_model.fit(X_scaled)

print(f"Training LOF               (contamination={CONTAM:.4f}) …")
lof_model = LocalOutlierFactor(n_neighbors=20, contamination=CONTAM, novelty=True)
lof_model.fit(X_scaled)

if_pred   = (if_model.predict(X_scaled)  == -1).astype(int)
lof_pred  = (lof_model.predict(X_scaled) == -1).astype(int)
if_scores  = -if_model.decision_function(X_scaled)   # flip so higher = more anomalous
lof_scores = -lof_model.decision_function(X_scaled)

print("Generating plots …\n")

# ── Helpers ────────────────────────────────────────────────────────────────────
FRIENDLY = {
    'log_impressions':        'Log Impressions',
    'toxicity_score':         'Toxicity Score',
    'like_rate':              'Like Rate',
    'share_rate':             'Share Rate',
    'comment_rate':           'Comment Rate',
    'sentiment_score':        'Sentiment Score',
    'user_past_sentiment_avg':'Past Sentiment Avg',
    'user_engagement_growth': 'Engagement Growth',
    'buzz_change_rate':       'Buzz Change Rate',
    'username_randomness':    'Username Randomness',
}

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
corr_cols = [
    'engagement_rate', 'toxicity_score', 'sentiment_score',
    'likes_count', 'shares_count', 'comments_count', 'impressions',
    'like_rate', 'share_rate', 'comment_rate', 'username_randomness',
]
corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
    vmin=-1, vmax=1, linewidths=0.5, ax=ax,
    xticklabels=[c.replace('_', '\n') for c in corr_cols],
    yticklabels=[c.replace('_', '\n') for c in corr_cols],
)
ax.set_title('Pearson Correlation Matrix — Key Features', fontsize=14, fontweight='bold', pad=14)
fig.tight_layout()
save(fig, 'correlation_matrix.png')


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE HISTOGRAMS  (suspicious vs normal overlay)
# ══════════════════════════════════════════════════════════════════════════════
n_feat = len(feature_names)
ncols  = 2
nrows  = (n_feat + 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(13, nrows * 3.2))
axes = axes.flatten()

for i, feat in enumerate(feature_names):
    ax   = axes[i]
    vals = df[feat] if feat in df.columns else pd.Series(X[:, i])
    norm_vals = vals[y_true == 0]
    susp_vals = vals[y_true == 1]

    ax.hist(norm_vals, bins=40, alpha=0.6, color=PALETTE['normal'],
            density=True, label='Normal')
    ax.hist(susp_vals, bins=40, alpha=0.6, color=PALETTE['suspicious'],
            density=True, label='Suspicious')
    ax.set_title(FRIENDLY.get(feat, feat), fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    if i == 0:
        ax.legend(loc='upper right', fontsize=8)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Feature Distributions — Normal vs Suspicious Posts', fontsize=14,
             fontweight='bold', y=1.01)
fig.tight_layout()
save(fig, 'feature_histograms.png')


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE IMPORTANCE  (Isolation Forest mean path-depth proxy)
# ══════════════════════════════════════════════════════════════════════════════
# Permutation-style: how much does each feature shift the anomaly score
rng = np.random.default_rng(42)
baseline = if_model.decision_function(X_scaled)
importances = []
for col_i in range(X_scaled.shape[1]):
    X_perm = X_scaled.copy()
    X_perm[:, col_i] = rng.permutation(X_perm[:, col_i])
    perm_scores = if_model.decision_function(X_perm)
    importances.append(float(np.mean(np.abs(baseline - perm_scores))))

order = np.argsort(importances)[::-1]
sorted_names   = [FRIENDLY.get(feature_names[i], feature_names[i]) for i in order]
sorted_imp     = [importances[i] for i in order]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(sorted_names[::-1], sorted_imp[::-1],
               color=sns.color_palette('Blues_r', n_feat))
ax.set_xlabel('Mean |Score Δ| after permutation', fontweight='bold')
ax.set_title('Feature Importance — Isolation Forest\n(Permutation on Anomaly Score)',
             fontsize=13, fontweight='bold')
for bar, val in zip(bars, sorted_imp[::-1]):
    ax.text(val + 0.0002, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', fontsize=8)
fig.tight_layout()
save(fig, 'feature_importance.png')


# ══════════════════════════════════════════════════════════════════════════════
# 4. ANOMALY SCORE DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

for ax, scores, pred, title in [
    (axes[0], if_scores,  if_pred,  'Isolation Forest'),
    (axes[1], lof_scores, lof_pred, 'Local Outlier Factor (LOF)'),
]:
    ax.hist(scores[pred == 0], bins=60, alpha=0.65, color=PALETTE['normal'],
            density=True, label='Normal')
    ax.hist(scores[pred == 1], bins=60, alpha=0.65, color=PALETTE['suspicious'],
            density=True, label='Suspicious')
    ax.axvline(np.median(scores), color='black', linestyle='--', linewidth=1,
               label=f'Median ({np.median(scores):.3f})')
    ax.set_title(f'Anomaly Score Distribution\n{title}', fontweight='bold')
    ax.set_xlabel('Anomaly Score (higher = more anomalous)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)

fig.suptitle('Anomaly Score Distributions — IF vs LOF', fontsize=14,
             fontweight='bold')
fig.tight_layout()
save(fig, 'anomaly_score_dist.png')


# ══════════════════════════════════════════════════════════════════════════════
# 5. CONFUSION MATRICES
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, pred, title in [
    (axes[0], if_pred,  'Isolation Forest'),
    (axes[1], lof_pred, 'Local Outlier Factor (LOF)'),
]:
    cm = confusion_matrix(y_true, pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels = np.array([[f'{v}\n({p:.1%})' for v, p in zip(row_v, row_p)]
                       for row_v, row_p in zip(cm, cm_norm)])
    sns.heatmap(cm_norm, annot=labels, fmt='', cmap='Blues',
                xticklabels=['Normal', 'Suspicious'],
                yticklabels=['Normal', 'Suspicious'],
                vmin=0, vmax=1, ax=ax, linewidths=0.5)
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('Actual', fontweight='bold')
    ax.set_title(f'Confusion Matrix — {title}', fontweight='bold')

fig.suptitle('Confusion Matrices (Proxy Label: engagement_rate > 1.0)',
             fontsize=13, fontweight='bold')
fig.tight_layout()
save(fig, 'confusion_matrices.png')


# ══════════════════════════════════════════════════════════════════════════════
# 6. PCA 2-D SCATTER
# ══════════════════════════════════════════════════════════════════════════════
pca    = PCA(n_components=2, random_state=42)
X_pca  = pca.fit_transform(X_scaled)
var_ex = pca.explained_variance_ratio_

# Sample for readability
sample = min(8000, len(X_pca))
idx_s  = rng.choice(len(X_pca), sample, replace=False)
Xp     = X_pca[idx_s]
yt     = y_true[idx_s]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, pred_full, title in [
    (axes[0], if_pred,  'Isolation Forest'),
    (axes[1], lof_pred, 'LOF'),
]:
    pred_s = pred_full[idx_s]
    for label, color, marker in [
        (0, PALETTE['normal'],     'o'),
        (1, PALETTE['suspicious'], 'X'),
    ]:
        mask = pred_s == label
        ax.scatter(Xp[mask, 0], Xp[mask, 1],
                   c=color, marker=marker, s=12, alpha=0.55,
                   label='Normal' if label == 0 else 'Suspicious (flagged)')
    ax.set_xlabel(f'PC1 ({var_ex[0]:.1%} var)', fontweight='bold')
    ax.set_ylabel(f'PC2 ({var_ex[1]:.1%} var)', fontweight='bold')
    ax.set_title(f'PCA 2-D Scatter — {title}', fontweight='bold')
    ax.legend(markerscale=2, fontsize=9)

fig.suptitle('PCA Projection of Post Features (sampled)', fontsize=14,
             fontweight='bold')
fig.tight_layout()
save(fig, 'pca_scatter.png')


# ══════════════════════════════════════════════════════════════════════════════
# 7. BOX PLOTS — features by true label
# ══════════════════════════════════════════════════════════════════════════════
df_box = pd.DataFrame(X, columns=[FRIENDLY.get(f, f) for f in feature_names])
df_box['Label'] = np.where(y_true == 1, 'Suspicious', 'Normal')

fig, axes = plt.subplots(nrows, ncols, figsize=(13, nrows * 3.2))
axes = axes.flatten()

for i, col in enumerate(df_box.columns[:-1]):
    sns.boxplot(data=df_box, x='Label', y=col, ax=axes[i],
                palette={'Normal': PALETTE['normal'], 'Suspicious': PALETTE['suspicious']},
                linewidth=0.8, fliersize=2)
    axes[i].set_title(col, fontweight='bold')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Value')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Feature Box Plots — Normal vs Suspicious Posts', fontsize=14,
             fontweight='bold', y=1.01)
fig.tight_layout()
save(fig, 'boxplots.png')


# ══════════════════════════════════════════════════════════════════════════════
# 8. ROC + PRECISION-RECALL CURVES
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ROC
ax = axes[0]
for scores, name, color in [
    (if_scores,  'Isolation Forest', '#E85C4C'),
    (lof_scores, 'LOF',              '#4C9BE8'),
]:
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{name}  (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('False Positive Rate', fontweight='bold')
ax.set_ylabel('True Positive Rate', fontweight='bold')
ax.set_title('ROC Curve', fontweight='bold')
ax.legend(fontsize=9)

# Precision-Recall
ax = axes[1]
for scores, name, color in [
    (if_scores,  'Isolation Forest', '#E85C4C'),
    (lof_scores, 'LOF',              '#4C9BE8'),
]:
    prec, rec, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    ax.plot(rec, prec, color=color, lw=2, label=f'{name}  (AP = {ap:.3f})')
baseline_pr = y_true.mean()
ax.axhline(baseline_pr, color='grey', linestyle='--', lw=1,
           label=f'Random baseline ({baseline_pr:.3f})')
ax.set_xlabel('Recall', fontweight='bold')
ax.set_ylabel('Precision', fontweight='bold')
ax.set_title('Precision-Recall Curve', fontweight='bold')
ax.legend(fontsize=9)

fig.suptitle('Model Comparison — ROC & Precision-Recall\n'
             '(Proxy Label: engagement_rate > 1.0)',
             fontsize=13, fontweight='bold')
fig.tight_layout()
save(fig, 'roc_pr_curves.png')

print(f"\nAll plots saved to: {os.path.abspath(OUT_DIR)}")
