/* api.js — thin wrapper around the Flask backend */
const API = (() => {
  const BASE = 'http://localhost:5000';

  async function _get(path) {
    const r = await fetch(BASE + path);
    if (!r.ok) throw new Error(`GET ${path} → ${r.status}`);
    return r.json();
  }

  async function _post(path, body) {
    const r = await fetch(BASE + path, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`POST ${path} → ${r.status}`);
    return r.json();
  }

  async function _upload(path, file) {
    const fd = new FormData();
    fd.append('file', file);
    const r = await fetch(BASE + path, {method: 'POST', body: fd});
    if (!r.ok) throw new Error(`UPLOAD ${path} → ${r.status}`);
    return r.json();
  }

  return {
    overview:      ()       => _get('/api/overview'),
    scatter:       ()       => _get('/api/scatter'),
    pca:           ()       => _get('/api/pca'),
    authDist:      ()       => _get('/api/auth-dist'),
    eda:           ()       => _get('/api/eda'),
    accounts:      (params) => _get('/api/accounts?' + new URLSearchParams(params)),
    accountDetail: (user)   => _get('/api/account/' + encodeURIComponent(user)),
    analyze:       (data)   => _post('/api/analyze', data),
    batch:         (file)   => _upload('/api/batch', file),
  };
})();
