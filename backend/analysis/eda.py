import numpy as np
import pandas as pd


def compute_eda(df: pd.DataFrame) -> dict:
    """Compute all EDA statistics needed by the frontend."""

    # Platform distribution
    platform_counts = df['platform'].value_counts().to_dict()

    # Engagement rate by platform
    er_grp = df.groupby('platform')['engagement_rate'].agg(['mean', 'median'])
    er_by_platform = {
        'platforms': er_grp.index.tolist(),
        'means':     er_grp['mean'].round(4).tolist(),
        'medians':   er_grp['median'].round(4).tolist(),
    }

    # Toxicity distribution (20 bins, capped at 1)
    tox = df['toxicity_score'].clip(0, 1)
    tox_counts, tox_edges = np.histogram(tox, bins=20, range=(0, 1))
    tox_centers = ((tox_edges[:-1] + tox_edges[1:]) / 2).round(3).tolist()

    # Engagement rate distribution (log-clipped)
    er_clip = df['engagement_rate'].clip(0, 2)
    er_counts, er_edges = np.histogram(er_clip, bins=30, range=(0, 2))
    er_centers = ((er_edges[:-1] + er_edges[1:]) / 2).round(4).tolist()

    # Sentiment by platform
    sent_by_platform = df.groupby('platform')['sentiment_score'].mean().round(4).to_dict()

    # Day-of-week distribution (ordered)
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_vc = df['day_of_week'].value_counts()
    dow_counts = {d: int(dow_vc.get(d, 0)) for d in dow_order}

    # Correlation matrix (key numeric columns)
    corr_cols = [
        'engagement_rate', 'toxicity_score', 'sentiment_score',
        'likes_count', 'shares_count', 'comments_count', 'impressions',
    ]
    corr = df[corr_cols].corr().round(3)

    return {
        'platform_counts': platform_counts,
        'er_by_platform': er_by_platform,
        'toxicity_dist': {'centers': tox_centers, 'counts': tox_counts.tolist()},
        'engagement_dist': {'centers': er_centers, 'counts': er_counts.tolist()},
        'sentiment_by_platform': sent_by_platform,
        'dow_counts': dow_counts,
        'correlation': {
            'labels': corr_cols,
            'matrix': corr.values.tolist(),
        },
    }
