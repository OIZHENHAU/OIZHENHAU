"""
Authenticity Confidence Score.

Strategy: Isolation Forest trained with contamination=0.25 on the full dataset
(expecting ~25% of accounts to be the most anomalous). The IF score is then
linearly scaled so that accounts resembling the organic cluster score highest.

Because the majority cluster in this dataset consists of bot/scam accounts with
very consistent (but suspicious) feature patterns, we additionally compute a
rule-based penalty that reduces the score when explicit fraud signals are present.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from .preprocessing import load_and_preprocess, get_feature_matrix, FEATURE_COLS

_cache = {}


def _get_models():
    if _cache:
        return _cache

    df = load_and_preprocess()
    X, _ = get_feature_matrix(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # contamination=0.25: treats 25% most anomalous as outliers
    iso = IsolationForest(contamination=0.25, n_estimators=200, random_state=42)
    iso.fit(X_scaled)

    train_scores = iso.score_samples(X_scaled)
    _cache['scaler'] = scaler
    _cache['iso'] = iso
    _cache['score_min'] = float(train_scores.min())
    _cache['score_max'] = float(train_scores.max())
    return _cache


def _rule_based_score(account: dict) -> float:
    """Returns 0-100; higher = more organic-looking."""
    score = 100.0
    followers = account.get('Followers', 0)
    following = account.get('Following', 0)
    posts = account.get('Posts', 0)
    ratio = account.get('Following/Followers', 0)

    # High following/followers ratio is the strongest bot signal
    if ratio > 200:
        score -= 55
    elif ratio > 50:
        score -= 35
    elif ratio > 10:
        score -= 15

    # Zero followers + high following
    if followers == 0 and following > 50:
        score -= 25

    # No posts at all
    if posts == 0:
        score -= 15

    # Missing profile basics
    if not account.get('Profile Picture', 0):
        score -= 10
    if not account.get('Bio', 0):
        score -= 8

    # No mutual friends is weak signal but still suspicious
    if account.get('Mutual Friends', 0) == 0 and followers < 100:
        score -= 5

    return max(0.0, min(100.0, score))


def compute_authenticity_score(account: dict) -> dict:
    models = _get_models()
    scaler = models['scaler']
    iso = models['iso']
    s_min = models['score_min']
    s_max = models['score_max']

    X = np.array([[account.get(col, 0) for col in FEATURE_COLS]])
    X_scaled = scaler.transform(X)

    raw = float(iso.score_samples(X_scaled)[0])
    # Normalize: higher raw → more normal in the dataset
    iso_score = (raw - s_min) / (s_max - s_min + 1e-9) * 100

    # Rule-based score captures explicit fraud signals
    rule_score = _rule_based_score(account)

    # Blend: rule-based dominates for clear fraud signals
    final = 0.35 * iso_score + 0.65 * rule_score
    final = round(max(0.0, min(100.0, final)), 1)

    if final >= 65:
        level, color = 'Authentic', '#22c55e'
    elif final >= 35:
        level, color = 'Suspicious', '#f59e0b'
    else:
        level, color = 'Fraudulent', '#ef4444'

    flags = []
    if account.get('Following/Followers', 0) > 50:
        flags.append('Extremely high Following/Followers ratio')
    if account.get('Followers', 0) == 0 and account.get('Following', 0) > 50:
        flags.append('Zero followers with high following count')
    if account.get('Posts', 0) == 0:
        flags.append('No posts published')
    if not account.get('Profile Picture', 0):
        flags.append('No profile picture')
    if not account.get('Bio', 0):
        flags.append('No bio')

    return {
        'authenticity_score': final,
        'level': level,
        'color': color,
        'flags': flags,
    }
