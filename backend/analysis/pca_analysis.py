import numpy as np

_SUSP_PERCENTILE = 20   # bottom 20 % of authenticity score = suspicious


def _labels_from_scores(ensemble_scores: np.ndarray) -> np.ndarray:
    """Derive binary labels from model anomaly scores (no ER threshold used).
    Accounts below the 20th percentile of ensemble_score are flagged suspicious."""
    threshold = np.percentile(ensemble_scores, _SUSP_PERCENTILE)
    return (ensemble_scores <= threshold).astype(int)


def compute_scatter_data(df, ensemble_scores: np.ndarray, max_pts: int = 1000) -> dict:
    """
    Engagement Rate vs log(Impressions) scatter, coloured by model anomaly label.
    Labels are derived from ensemble_scores (bottom 20th percentile = suspicious),
    not from engagement_rate > 1.0.
    """
    labels  = _labels_from_scores(ensemble_scores)
    rng     = np.random.default_rng(0)
    log_imp = df['log_impressions'].values
    er      = df['engagement_rate'].values.clip(0, 2)

    def _sample(mask):
        idx = np.where(mask)[0]
        if len(idx) > max_pts:
            idx = rng.choice(idx, max_pts, replace=False)
        return [{'x': round(float(log_imp[i]), 4),
                 'y': round(float(er[i]), 5)} for i in idx]

    return {
        'real':       _sample(labels == 0),
        'suspicious': _sample(labels == 1),
    }


def compute_pca_data(components: np.ndarray, ensemble_scores: np.ndarray,
                     explained: np.ndarray, max_pts: int = 800) -> dict:
    """PC1 vs PC2 scatter coloured by model anomaly label + scree / variance data.
    Labels derived from ensemble_scores, not from engagement_rate > 1.0."""
    labels = _labels_from_scores(ensemble_scores)
    rng    = np.random.default_rng(1)

    def _sample(mask):
        idx = np.where(mask)[0]
        if len(idx) > max_pts:
            idx = rng.choice(idx, max_pts, replace=False)
        return [{'x': round(float(components[i, 0]), 4),
                 'y': round(float(components[i, 1]), 4)} for i in idx]

    exp_pct = (explained * 100).round(2).tolist()
    return {
        'real':       _sample(labels == 0),
        'suspicious': _sample(labels == 1),
        'explained':  exp_pct,
        'scree': {
            'labels': [f'PC{i+1}' for i in range(len(explained))],
            'values': exp_pct,
        },
        'cumulative': list(np.cumsum(explained * 100).round(2)),
    }


def compute_auth_dist(ensemble_scores: np.ndarray) -> dict:
    """Histogram of authenticity scores split by model's 20th-percentile threshold.
    Labels derived from ensemble_scores — no ER>1 proxy used."""
    labels = _labels_from_scores(ensemble_scores)
    bins   = list(range(0, 105, 5))

    real_hist, _ = np.histogram(ensemble_scores[labels == 0], bins=bins)
    susp_hist, _ = np.histogram(ensemble_scores[labels == 1], bins=bins)

    return {
        'labels':     [f'{b}-{b+5}' for b in bins[:-1]],
        'real':       real_hist.tolist(),
        'suspicious': susp_hist.tolist(),
    }
