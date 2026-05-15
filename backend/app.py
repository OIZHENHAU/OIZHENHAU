import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from analysis.eda import get_eda_summary
from analysis.pca_analysis import run_pca
from analysis.outlier_detection import run_outlier_detection
from analysis.scoring import compute_authenticity_score

app = Flask(__name__, static_folder=None)
CORS(app)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), '..', 'frontend')

# ── Static frontend ────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)

# ── API ────────────────────────────────────────────────────────────────────────

_eda_cache = None
_pca_cache = None
_outlier_cache = None


@app.route('/api/eda')
def api_eda():
    global _eda_cache
    if _eda_cache is None:
        _eda_cache = get_eda_summary()
    return jsonify(_eda_cache)


@app.route('/api/pca')
def api_pca():
    global _pca_cache
    if _pca_cache is None:
        _pca_cache = run_pca()
    return jsonify(_pca_cache)


@app.route('/api/outliers')
def api_outliers():
    global _outlier_cache
    if _outlier_cache is None:
        _outlier_cache = run_outlier_detection()
    return jsonify(_outlier_cache)


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    data = request.get_json(force=True)

    # Coerce types
    account = {
        'Followers': int(data.get('followers', 0)),
        'Following': int(data.get('following', 0)),
        'Posts': int(data.get('posts', 0)),
        'Mutual Friends': int(data.get('mutual_friends', 0)),
        'Bio': 1 if data.get('bio') else 0,
        'Profile Picture': 1 if data.get('profile_picture') else 0,
        'External Link': 1 if data.get('external_link') else 0,
        'Threads': 1 if data.get('threads') else 0,
    }
    followers = account['Followers']
    following = account['Following']
    posts = account['Posts']
    account['Following/Followers'] = round(following / followers, 4) if followers > 0 else following
    account['Posts/Followers'] = round(posts / followers, 4) if followers > 0 else posts

    result = compute_authenticity_score(account)
    result['account'] = account
    return jsonify(result)


@app.route('/api/report', methods=['POST'])
def api_report():
    try:
        from agent import generate_forensic_report
        report = generate_forensic_report()
        return jsonify({'report': report})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting Influencer Clout Detective...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=False, port=5000)
