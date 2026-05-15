import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .preprocessing import load_and_preprocess, get_feature_matrix

SAMPLE_SIZE = 2000


def run_pca():
    df = load_and_preprocess()

    sampled = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
    X, feature_cols = get_feature_matrix(sampled)
    labels = sampled['Labels'].tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Loadings: contribution of each feature to each PC
    loadings = {
        'features': feature_cols,
        'PC1': [round(v, 4) for v in pca.components_[0].tolist()],
        'PC2': [round(v, 4) for v in pca.components_[1].tolist()],
    }

    return {
        'x': [round(v, 4) for v in X_pca[:, 0].tolist()],
        'y': [round(v, 4) for v in X_pca[:, 1].tolist()],
        'labels': labels,
        'explained_variance': [round(v, 4) for v in pca.explained_variance_ratio_.tolist()],
        'loadings': loadings,
    }
