import numpy as np
from .preprocessing import load_and_preprocess

NUMERIC_COLS = ['Followers', 'Following', 'Following/Followers', 'Posts',
                'Posts/Followers', 'Mutual Friends']
BINARY_COLS = ['Bio', 'Profile Picture', 'External Link', 'Threads']


def get_eda_summary():
    df = load_and_preprocess()
    labels = df['Labels'].unique().tolist()

    label_counts = df['Labels'].value_counts().to_dict()

    # Mean of each numeric feature grouped by label
    feature_means_by_label = {}
    for label in labels:
        subset = df[df['Labels'] == label]
        feature_means_by_label[label] = {
            col: round(float(subset[col].mean()), 4) for col in NUMERIC_COLS
        }

    # Binary feature presence rate per label (%)
    binary_rates_by_label = {}
    for label in labels:
        subset = df[df['Labels'] == label]
        binary_rates_by_label[label] = {
            col: round(float(subset[col].mean()) * 100, 1) for col in BINARY_COLS
        }

    # Log-scale histogram for distribution charts
    distributions = {}
    for col in ['Followers', 'Following', 'Posts']:
        vals = df[col].clip(lower=1)
        hist, edges = np.histogram(np.log1p(vals), bins=20)
        distributions[col] = {
            'counts': hist.tolist(),
            'edges': [round(float(v), 2) for v in np.expm1(edges)]
        }

    # Correlation matrix
    corr = df[NUMERIC_COLS].corr().round(3)

    # Boxplot data (min, q1, median, q3, max per label per feature)
    boxplot_data = {}
    for col in NUMERIC_COLS:
        boxplot_data[col] = {}
        for label in labels:
            vals = df[df['Labels'] == label][col]
            boxplot_data[col][label] = {
                'min': round(float(vals.min()), 2),
                'q1': round(float(vals.quantile(0.25)), 2),
                'median': round(float(vals.median()), 2),
                'q3': round(float(vals.quantile(0.75)), 2),
                'max': round(float(vals.max()), 2),
                'mean': round(float(vals.mean()), 2),
            }

    return {
        'total_records': len(df),
        'label_counts': label_counts,
        'feature_means_by_label': feature_means_by_label,
        'binary_rates_by_label': binary_rates_by_label,
        'distributions': distributions,
        'correlation': {
            'columns': NUMERIC_COLS,
            'matrix': corr.values.tolist()
        },
        'boxplot_data': boxplot_data,
    }
