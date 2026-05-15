const API = 'http://localhost:5000/api';

async function fetchEDA()      { return fetch(`${API}/eda`).then(r => r.json()); }
async function fetchPCA()      { return fetch(`${API}/pca`).then(r => r.json()); }
async function fetchOutliers() { return fetch(`${API}/outliers`).then(r => r.json()); }

async function analyzeAccount(data) {
  const res = await fetch(`${API}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return res.json();
}

async function generateReport() {
  const res = await fetch(`${API}/report`, { method: 'POST' });
  return res.json();
}
