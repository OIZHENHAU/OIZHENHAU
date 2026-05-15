import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "new_dataset.csv"

FEATURE_COLS = [
    'Followers', 'Following', 'Following/Followers', 'Posts',
    'Posts/Followers', 'Bio', 'Profile Picture', 'External Link',
    'Mutual Friends', 'Threads'
]


def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)

    # Normalize binary categorical columns to 0/1
    binary_cols = ['Bio', 'Profile Picture', 'External Link', 'Threads']
    for col in binary_cols:
        df[col] = df[col].str.strip().str.lower().map(
            {'yes': 1, 'y': 1, 'n': 0, 'no': 0}
        ).fillna(0).astype(int)

    # Parse ratio columns (may be stored as strings)
    for col in ['Following/Followers', 'Posts/Followers']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df[col] = df[col].replace([np.inf, -np.inf], 0)
        cap = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=cap)

    return df


def get_feature_matrix(df):
    return df[FEATURE_COLS].values, FEATURE_COLS
