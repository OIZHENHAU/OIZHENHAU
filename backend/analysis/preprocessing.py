import os
import math
import numpy as np
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/Social Media Engagement Dataset.csv')

# Features used for the post-level anomaly model (dataset dashboard)
POST_FEATURE_COLS = [
    'log_impressions', 'engagement_rate', 'toxicity_score',
    'like_rate', 'share_rate', 'comment_rate',
    'sentiment_score', 'user_past_sentiment_avg',
    'user_engagement_growth', 'buzz_change_rate',
]

# Features used for the account-level model (single lookup + batch CSV)
ACCOUNT_FEATURE_COLS = [
    'log_followers', 'ff_ratio', 'post_freq',
    'bio_norm', 'has_profile_picture',
    'username_randomness', 'spam_comments_rate', 'generic_comment_rate',
]


def char_entropy(s: str) -> float:
    """Normalized Shannon entropy of characters — higher = more random."""
    s = str(s).lower()
    if len(s) <= 1:
        return 0.0
    freq: dict = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    unique = len(freq)
    if unique <= 1:
        return 0.0
    raw = -sum((f / n) * math.log2(f / n) for f in freq.values())
    return float(min(1.0, raw / math.log2(max(2, unique))))


def load_data() -> pd.DataFrame:
    """Load CSV, clean numeric columns, and engineer derived features."""
    df = pd.read_csv(DATA_PATH)

    numeric_cols = [
        'sentiment_score', 'toxicity_score',
        'likes_count', 'shares_count', 'comments_count',
        'impressions', 'engagement_rate',
        'user_past_sentiment_avg', 'user_engagement_growth', 'buzz_change_rate',
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=numeric_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Derived features
    df['log_impressions'] = np.log1p(df['impressions'])
    df['like_rate'] = df['likes_count'] / (df['impressions'] + 1)
    df['share_rate'] = df['shares_count'] / (df['impressions'] + 1)
    df['comment_rate'] = df['comments_count'] / (df['impressions'] + 1)
    df['username_randomness'] = df['user_id'].apply(char_entropy)

    return df


def get_post_features(df: pd.DataFrame):
    """Return feature matrix X and feature names for the post model."""
    X = df[POST_FEATURE_COLS].values.astype(float)
    return X, POST_FEATURE_COLS


def account_inputs_to_features(
    followers: float, following: float, posts: float,
    account_age_days: float, bio_length: float, has_profile_picture: float,
    username_randomness: float, spam_comments_rate: float, generic_comment_rate: float,
) -> list:
    """Convert raw account inputs to the 8-feature vector for the account model."""
    log_fol = math.log1p(max(followers, 0))
    ff_ratio = max(followers, 0) / max(following, 1)
    post_freq = max(posts, 0) / max(account_age_days, 1)
    bio_norm = min(max(bio_length, 0) / 160.0, 1.0)
    return [
        log_fol,
        ff_ratio,
        post_freq,
        bio_norm,
        float(has_profile_picture),
        float(username_randomness),
        float(spam_comments_rate),
        float(generic_comment_rate),
    ]


def batch_csv_to_features(df_batch: pd.DataFrame) -> np.ndarray:
    """
    Convert a batch-upload DataFrame to the account feature matrix.
    Required columns: username, followers, following, posts,
                      account_age_days, spam_comments_rate, generic_comment_rate
    """
    req = ['username', 'followers', 'following', 'posts',
           'account_age_days', 'spam_comments_rate', 'generic_comment_rate']
    for col in req:
        if col not in df_batch.columns:
            raise ValueError(f"Missing required column: {col}")

    for col in req[1:]:
        df_batch[col] = pd.to_numeric(df_batch[col], errors='coerce').fillna(0)

    rows = []
    for _, r in df_batch.iterrows():
        urand = char_entropy(str(r['username']))
        vec = account_inputs_to_features(
            r['followers'], r['following'], r['posts'],
            r['account_age_days'], 50, 1,       # bio_length=50, has_pic=1 (defaults)
            urand, r['spam_comments_rate'], r['generic_comment_rate'],
        )
        rows.append(vec)
    return np.array(rows, dtype=float)


def generate_account_training_data(n: int = 5000, seed: int = 42) -> np.ndarray:
    """
    Generate synthetic account training data for the unsupervised account model.
    65 % are plausible 'real' accounts, 35 % are bot-like — no labels used for training.
    """
    rng = np.random.default_rng(seed)
    n_real = int(n * 0.65)
    n_bot = n - n_real

    # ---- REAL accounts ----
    log_fol_r  = rng.normal(9.5, 2.0, n_real).clip(4, 15)
    ff_r       = rng.lognormal(1.5, 0.8, n_real).clip(0.5, 200)
    pfreq_r    = rng.lognormal(-0.5, 0.7, n_real).clip(0.01, 20)
    bio_r      = rng.beta(4, 1.5, n_real)
    pic_r      = rng.binomial(1, 0.95, n_real).astype(float)
    urand_r    = rng.beta(1.5, 5, n_real)
    spam_r     = rng.beta(1, 10, n_real)
    generic_r  = rng.beta(1.5, 7, n_real)

    # ---- BOT accounts ----
    log_fol_b  = rng.normal(6.0, 2.5, n_bot).clip(2, 12)
    ff_b       = (rng.beta(1, 10, n_bot) * 0.3).clip(0, 0.5)
    pfreq_b    = rng.exponential(0.05, n_bot).clip(0, 3)
    bio_b      = rng.beta(1, 6, n_bot)
    pic_b      = rng.binomial(1, 0.25, n_bot).astype(float)
    urand_b    = rng.beta(6, 2, n_bot)
    spam_b     = rng.beta(6, 2, n_bot)
    generic_b  = rng.beta(5, 2, n_bot)

    real_data = np.column_stack([log_fol_r, ff_r, pfreq_r, bio_r, pic_r, urand_r, spam_r, generic_r])
    bot_data  = np.column_stack([log_fol_b, ff_b, pfreq_b, bio_b, pic_b, urand_b, spam_b, generic_b])

    return np.vstack([real_data, bot_data])
