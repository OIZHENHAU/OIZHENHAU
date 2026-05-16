import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def _rank_norm(scores: np.ndarray) -> np.ndarray:
    """Rank-based normalization → uniform [0, 100]. Score = authenticity percentile.
    Bottom 20 % gets 0–20, top 20 % gets 80–100, median = 50."""
    n = len(scores)
    if n <= 1:
        return np.full_like(scores, 50.0, dtype=float)
    ranks = rankdata(scores, method='average')  # 1 … N
    return (ranks - 1) / (n - 1) * 100.0


def _norm(scores: np.ndarray) -> np.ndarray:
    """Min-max normalization (used for calibration only)."""
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.full_like(scores, 50.0)
    return (scores - lo) / (hi - lo) * 100.0


# ─── Post model (dataset dashboard) ────────────────────────────────────────────

def train_post_models(X: np.ndarray):
    """Train IsolationForest + LOF on the post feature matrix."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if_model = IsolationForest(
        n_estimators=200, contamination=0.20, random_state=42, n_jobs=-1
    )
    if_model.fit(Xs)

    lof_model = LocalOutlierFactor(
        n_neighbors=20, contamination=0.20, novelty=True
    )
    lof_model.fit(Xs)

    return if_model, lof_model, scaler


def get_post_scores(X: np.ndarray, if_model, lof_model, scaler) -> tuple:
    """Return (if_auth, lof_auth, ensemble) arrays in [0, 100].
    Uses rank-based normalization so scores are uniformly distributed and
    the contamination percentile is a stable suspicious/real boundary.
    """
    Xs = scaler.transform(X)
    if_raw  = if_model.score_samples(Xs)
    lof_raw = lof_model.decision_function(Xs)
    if_auth  = _rank_norm(if_raw)
    lof_auth = _rank_norm(lof_raw)
    ensemble = _rank_norm(0.6 * _norm(if_raw) + 0.4 * _norm(lof_raw))
    return if_auth, lof_auth, ensemble


def run_pca(X: np.ndarray, scaler) -> tuple:
    """PCA on scaled features — return components, explained variance, loadings."""
    Xs = scaler.transform(X)
    n_comp = min(10, X.shape[1])
    pca = PCA(n_components=n_comp)
    components = pca.fit_transform(Xs)
    return components, pca.explained_variance_ratio_, pca.components_, pca


# ─── Account model (single lookup + batch CSV) ─────────────────────────────────

def train_account_model(X: np.ndarray) -> dict:
    """Train account-level IF + LOF; store calibration ranges."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if_model = IsolationForest(
        n_estimators=200, contamination=0.30, random_state=42, n_jobs=-1
    )
    if_model.fit(Xs)

    lof_model = LocalOutlierFactor(
        n_neighbors=15, contamination=0.30, novelty=True
    )
    lof_model.fit(Xs)

    # Calibrate on training distribution so new points score reasonably
    if_tr  = if_model.decision_function(Xs)
    lof_tr = lof_model.decision_function(Xs)

    return {
        'scaler':  scaler,
        'if':      if_model,
        'lof':     lof_model,
        'if_cal':  (float(np.percentile(if_tr,  5)), float(np.percentile(if_tr,  95))),
        'lof_cal': (float(np.percentile(lof_tr, 5)), float(np.percentile(lof_tr, 95))),
    }


def score_single_account(features: list, model: dict) -> tuple:
    """Score one account vector. Returns (if_score, lof_score, ensemble) in [0,100]."""
    X = np.array(features, dtype=float).reshape(1, -1)
    Xs = model['scaler'].transform(X)

    if_raw  = float(model['if'].decision_function(Xs)[0])
    lof_raw = float(model['lof'].decision_function(Xs)[0])

    def cal(v, lo, hi):
        if hi == lo:
            return 50
        return int(round(max(0.0, min(100.0, (v - lo) / (hi - lo) * 100.0))))

    if_score  = cal(if_raw,  *model['if_cal'])
    lof_score = cal(lof_raw, *model['lof_cal'])
    ensemble  = round(0.6 * if_score + 0.4 * lof_score)
    return if_score, lof_score, ensemble


def score_batch(X_batch: np.ndarray, model: dict) -> tuple:
    """Score a batch of account vectors. Returns (if_arr, lof_arr, ensemble_arr)."""
    Xs = model['scaler'].transform(X_batch)

    if_raw  = model['if'].decision_function(Xs)
    lof_raw = model['lof'].decision_function(Xs)

    lo_if,  hi_if  = model['if_cal']
    lo_lof, hi_lof = model['lof_cal']

    def cal_arr(arr, lo, hi):
        if hi == lo:
            return np.full(len(arr), 50.0)
        return np.clip((arr - lo) / (hi - lo) * 100.0, 0, 100)

    if_scores  = cal_arr(if_raw,  lo_if,  hi_if)
    lof_scores = cal_arr(lof_raw, lo_lof, hi_lof)
    ensemble   = 0.6 * if_scores + 0.4 * lof_scores
    return if_scores, lof_scores, ensemble
