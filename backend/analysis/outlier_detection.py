import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from .preprocessing import load_and_preprocess, get_feature_matrix

SAMPLE_SIZE = 3000
CONTAMINATION = 0.5   # ~50% of data is non-real (Bot+Scam+Spam)


def run_outlier_detection():
    df = load_and_preprocess()
    sampled = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)

    X, _ = get_feature_matrix(sampled)
    labels = sampled['Labels'].tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Isolation Forest ---
    iso = IsolationForest(contamination=CONTAMINATION, n_estimators=100, random_state=42)
    iso_pred = iso.fit_predict(X_scaled)          # 1=normal, -1=outlier
    iso_scores = iso.score_samples(X_scaled)       # higher = more normal

    # --- LOF ---
    lof = LocalOutlierFactor(n_neighbors=20, contamination=CONTAMINATION)
    lof_pred = lof.fit_predict(X_scaled)           # 1=normal, -1=outlier
    lof_scores = lof.negative_outlier_factor_      # more negative = more outlier

    # Normalize both to [0, 1] (authenticity: 1 = very authentic)
    def norm(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-9)

    iso_norm = norm(iso_scores)
    lof_norm = norm(lof_scores)
    combined = 0.5 * iso_norm + 0.5 * lof_norm

    iso_outlier = (iso_pred == -1).astype(int)
    lof_outlier = (lof_pred == -1).astype(int)

    # Outlier count per label
    iso_by_label = defaultdict(lambda: {'total': 0, 'outliers': 0})
    lof_by_label = defaultdict(lambda: {'total': 0, 'outliers': 0})
    score_by_label = defaultdict(list)

    for i, label in enumerate(labels):
        iso_by_label[label]['total'] += 1
        lof_by_label[label]['total'] += 1
        if iso_outlier[i]:
            iso_by_label[label]['outliers'] += 1
        if lof_outlier[i]:
            lof_by_label[label]['outliers'] += 1
        score_by_label[label].append(float(combined[i]))

    score_stats = {}
    for label, scores in score_by_label.items():
        arr = np.array(scores)
        score_stats[label] = {
            'mean': round(float(arr.mean()), 4),
            'std': round(float(arr.std()), 4),
            'median': round(float(np.median(arr)), 4),
        }

    # Sample for scatter visualization (first 500)
    n = min(500, len(labels))
    return {
        'iso_outlier_by_label': {k: dict(v) for k, v in iso_by_label.items()},
        'lof_outlier_by_label': {k: dict(v) for k, v in lof_by_label.items()},
        'score_stats_by_label': score_stats,
        'sample': {
            'labels': labels[:n],
            'iso_scores': iso_norm[:n].tolist(),
            'lof_scores': lof_norm[:n].tolist(),
            'combined_scores': combined[:n].tolist(),
            'iso_outlier': iso_outlier[:n].tolist(),
            'lof_outlier': lof_outlier[:n].tolist(),
        },
    }
