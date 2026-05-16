import io
import os
import sys
import math

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# ── ensure local analysis package is importable when run as `python app.py` ──
sys.path.insert(0, os.path.dirname(__file__))

from analysis.preprocessing import (
    load_data, get_post_features,
    account_inputs_to_features, batch_csv_to_features,
    generate_account_training_data, char_entropy,
)
from analysis.outlier_detection import (
    train_post_models, get_post_scores, run_pca,
    train_account_model, score_single_account, score_batch,
)
from analysis.scoring import compute_df_stats, post_reasons, account_reasons
from analysis.eda import compute_eda
from analysis.pca_analysis import compute_scatter_data, compute_pca_data, compute_auth_dist

app = Flask(__name__)
CORS(app)

# ── Global in-memory cache populated at startup ───────────────────────────────
_C: dict = {}


def _init():
    print("[init] Loading dataset …")
    df = load_data()

    print("[init] Training post model (IF + LOF) …")
    X, feat_names = get_post_features(df)
    if_m, lof_m, scaler = train_post_models(X)
    if_sc, lof_sc, ens_sc = get_post_scores(X, if_m, lof_m, scaler)

    # Use the contamination percentile as the suspicious threshold so the label
    # distribution matches the model's contamination parameter (20 %).
    CONTAMINATION = 0.20
    susp_threshold = float(np.percentile(ens_sc, CONTAMINATION * 100))
    labels = (ens_sc <= susp_threshold).astype(int)   # 1 = suspicious, 0 = real
    df['if_score']       = np.round(if_sc).astype(int)
    df['lof_score']      = np.round(lof_sc).astype(int)
    df['ensemble_score'] = np.round(ens_sc).astype(int)
    df['label']          = labels

    print("[init] Running PCA …")
    components, explained, loadings, _ = run_pca(X, scaler)

    print("[init] Computing EDA & analytics …")
    df_stats = compute_df_stats(df)

    _C['df']             = df
    _C['df_stats']       = df_stats
    _C['feature_names']  = feat_names
    _C['post_if']        = if_m
    _C['post_lof']       = lof_m
    _C['post_sc']        = scaler
    _C['susp_threshold'] = susp_threshold

    # ── Overview ──────────────────────────────────────────────────────────────
    n_total  = len(df)
    n_susp   = int(labels.sum())
    n_real   = n_total - n_susp
    avg_sc   = float(ens_sc.mean())
    # High-risk = bottom 10 % of ensemble scores
    high_risk_thresh = float(np.percentile(ens_sc, 10))
    high_risk = int((ens_sc <= high_risk_thresh).sum())

    _C['overview'] = {
        'total':     n_total,
        'real':      n_real,
        'suspicious': n_susp,
        'avg_score': round(avg_sc, 1),
        'high_risk': high_risk,
    }

    # ── Chart data ────────────────────────────────────────────────────────────
    _C['scatter']   = compute_scatter_data(df, labels)
    _C['pca']       = compute_pca_data(components, labels, explained)
    _C['auth_dist'] = compute_auth_dist(ens_sc, labels)
    _C['eda']       = compute_eda(df)

    # ── Account list (cached first 200 for the table; search served from full df) ─
    _C['accounts_df'] = df[['user_id', 'platform', 'impressions',
                             'if_score', 'lof_score', 'ensemble_score', 'label',
                             'engagement_rate', 'toxicity_score',
                             'like_rate', 'buzz_change_rate',
                             'username_randomness']].copy()

    # ── Account model ─────────────────────────────────────────────────────────
    print("[init] Training account model …")
    X_acc = generate_account_training_data(n=5000)
    _C['account_model'] = train_account_model(X_acc)

    print(f"[init] Done — {n_total:,} posts | {n_real:,} real | {n_susp:,} suspicious")


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _row_to_account(row) -> dict:
    label_str = 'Suspicious' if row['label'] == 1 else 'Real'
    return {
        'username':       str(row['user_id']),
        'platform':       str(row['platform']),
        'followers':      int(row['impressions']),
        'if_score':       int(row['if_score']),
        'lof_score':      int(row['lof_score']),
        'auth_score':     int(row['ensemble_score']),
        'label':          label_str,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get('/api/overview')
def api_overview():
    return jsonify(_C['overview'])


@app.get('/api/scatter')
def api_scatter():
    return jsonify(_C['scatter'])


@app.get('/api/pca')
def api_pca():
    return jsonify(_C['pca'])


@app.get('/api/auth-dist')
def api_auth_dist():
    return jsonify(_C['auth_dist'])


@app.get('/api/eda')
def api_eda():
    return jsonify(_C['eda'])


@app.get('/api/accounts')
def api_accounts():
    """
    Query params:
      search   – filter by username (case-insensitive substring)
      page     – 1-based page number (default 1)
      per_page – rows per page (default 10, max 50)
    """
    search   = request.args.get('search', '').strip().lower()
    page     = max(1, int(request.args.get('page', 1)))
    per_page = min(50, max(1, int(request.args.get('per_page', 10))))

    adf = _C['accounts_df']
    if search:
        mask = adf['user_id'].str.lower().str.contains(search, na=False)
        adf  = adf[mask]

    total  = len(adf)
    start  = (page - 1) * per_page
    end    = start + per_page
    subset = adf.iloc[start:end]

    accounts = [_row_to_account(r) for _, r in subset.iterrows()]
    return jsonify({'total': total, 'page': page, 'per_page': per_page,
                    'accounts': accounts})


@app.get('/api/account/<username>')
def api_account_detail(username: str):
    """Return full detection detail for one account (by user_id)."""
    adf = _C['accounts_df']
    row = adf[adf['user_id'] == username]
    if row.empty:
        return jsonify({'error': 'Not found'}), 404

    r = row.iloc[0]
    reasons = post_reasons({
        'engagement_rate':    float(r['engagement_rate']),
        'toxicity_score':     float(r['toxicity_score']),
        'like_rate':          float(r['like_rate']),
        'buzz_change_rate':   float(r['buzz_change_rate']),
        'username_randomness': float(r['username_randomness']),
    }, _C['df_stats'])

    return jsonify({
        'username':    str(r['user_id']),
        'platform':    str(r['platform']),
        'followers':   int(r['impressions']),
        'if_score':    int(r['if_score']),
        'lof_score':   int(r['lof_score']),
        'auth_score':  int(r['ensemble_score']),
        'label':       'Suspicious' if r['label'] == 1 else 'Real',
        'reasons':     reasons,
    })


@app.post('/api/analyze')
def api_analyze():
    """Score a manually-entered account."""
    data = request.get_json(force=True)
    try:
        features = account_inputs_to_features(
            followers           = float(data.get('followers', 0)),
            following           = float(data.get('following', 1)),
            posts               = float(data.get('posts', 0)),
            account_age_days    = float(data.get('account_age_days', 1)),
            bio_length          = float(data.get('bio_length', 0)),
            has_profile_picture = float(data.get('has_profile_picture', 0)),
            username_randomness = float(data.get('username_randomness', 0.5)),
            spam_comments_rate  = float(data.get('spam_comments_rate', 0.0)),
            generic_comment_rate= float(data.get('generic_comment_rate', 0.0)),
        )
    except (KeyError, ValueError) as e:
        return jsonify({'error': str(e)}), 400

    if_s, lof_s, ens = score_single_account(features, _C['account_model'])
    label = 'Suspicious' if ens < 50 else 'Real'

    reasons = account_reasons({
        'followers':           data.get('followers', 0),
        'following':           data.get('following', 1),
        'spam_comments_rate':  data.get('spam_comments_rate', 0),
        'generic_comment_rate':data.get('generic_comment_rate', 0),
        'username_randomness': data.get('username_randomness', 0),
        'bio_length':          data.get('bio_length', 50),
        'has_profile_picture': data.get('has_profile_picture', 1),
        'account_age_days':    data.get('account_age_days', 100),
    })

    return jsonify({
        'if_score':    if_s,
        'lof_score':   lof_s,
        'ensemble':    ens,
        'label':       label,
        'reasons':     reasons,
    })


@app.post('/api/batch')
def api_batch():
    """
    Batch-score accounts from an uploaded CSV file.
    Required columns: username, followers, following, posts,
                      account_age_days, spam_comments_rate, generic_comment_rate
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    f = request.files['file']
    try:
        content = f.read().decode('utf-8', errors='replace')
        df_batch = pd.read_csv(io.StringIO(content))
    except Exception as e:
        return jsonify({'error': f'Could not parse CSV: {e}'}), 400

    try:
        X_batch = batch_csv_to_features(df_batch)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    if_arr, lof_arr, ens_arr = score_batch(X_batch, _C['account_model'])
    labels = (ens_arr < 50).astype(int)

    accounts = []
    for i, (_, row) in enumerate(df_batch.iterrows()):
        accounts.append({
            'username':  str(row.get('username', f'account_{i}')),
            'if_score':  int(round(if_arr[i])),
            'lof_score': int(round(lof_arr[i])),
            'ensemble':  int(round(ens_arr[i])),
            'verdict':   'Suspicious' if labels[i] == 1 else 'Real',
        })

    n_susp = int(labels.sum())
    return jsonify({
        'total':      len(accounts),
        'real':       len(accounts) - n_susp,
        'suspicious': n_susp,
        'accounts':   accounts,
    })


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    _init()
    app.run(debug=False, port=5000, host='0.0.0.0')
