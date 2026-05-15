/* ── Chart colour palette ─────────────────────────────────────────────────── */
const COLORS = {
  Real:  '#22c55e',
  Bot:   '#ef4444',
  Scam:  '#f59e0b',
  Spam:  '#a855f7',
};

const LABEL_ORDER = ['Real', 'Bot', 'Scam', 'Spam'];

Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = '#2e3358';
Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";

function makeChart(id, config) {
  const ctx = document.getElementById(id);
  if (!ctx) return null;
  if (ctx._chartInstance) ctx._chartInstance.destroy();
  const chart = new Chart(ctx, config);
  ctx._chartInstance = chart;
  return chart;
}

/* ── Overview charts ─────────────────────────────────────────────────────── */
function drawLabelPie(labelCounts) {
  const labels = LABEL_ORDER.filter(l => labelCounts[l] !== undefined);
  makeChart('chart-label-pie', {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data: labels.map(l => labelCounts[l]),
        backgroundColor: labels.map(l => COLORS[l]),
        borderWidth: 2,
        borderColor: '#1a1d2e',
      }],
    },
    options: {
      plugins: { legend: { position: 'bottom', labels: { padding: 16, boxWidth: 12 } } },
      cutout: '60%',
    },
  });
}

function drawFollowersBar(featureMeans) {
  const labels = LABEL_ORDER;
  makeChart('chart-followers-bar', {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Avg Followers',
        data: labels.map(l => featureMeans[l]?.Followers ?? 0),
        backgroundColor: labels.map(l => COLORS[l]),
        borderRadius: 6,
      }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, grid: { color: '#2e3358' } },
        x: { grid: { display: false } },
      },
    },
  });
}

function drawFollowingBar(featureMeans) {
  const labels = LABEL_ORDER;
  makeChart('chart-following-bar', {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Avg Following',
        data: labels.map(l => featureMeans[l]?.Following ?? 0),
        backgroundColor: labels.map(l => COLORS[l] + 'cc'),
        borderRadius: 6,
      }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, grid: { color: '#2e3358' } },
        x: { grid: { display: false } },
      },
    },
  });
}

function drawBinaryRadar(binaryRates) {
  const features = ['Bio', 'Profile Picture', 'External Link', 'Threads'];
  const labels = LABEL_ORDER;
  makeChart('chart-binary-radar', {
    type: 'radar',
    data: {
      labels: features,
      datasets: labels.map(l => ({
        label: l,
        data: features.map(f => binaryRates[l]?.[f] ?? 0),
        borderColor: COLORS[l],
        backgroundColor: COLORS[l] + '22',
        pointBackgroundColor: COLORS[l],
        borderWidth: 2,
      })),
    },
    options: {
      scales: {
        r: {
          beginAtZero: true,
          max: 100,
          ticks: { stepSize: 25, color: '#94a3b8' },
          grid: { color: '#2e3358' },
          pointLabels: { color: '#e2e8f0', font: { size: 11 } },
        },
      },
      plugins: { legend: { position: 'bottom', labels: { padding: 12, boxWidth: 10 } } },
    },
  });
}

/* ── EDA charts ──────────────────────────────────────────────────────────── */
function drawFollowersDist(distributions) {
  const d = distributions.Followers;
  const edgeLabels = d.edges.slice(0, -1).map((v, i) => {
    const k = Math.round(v / 1000);
    return k >= 1 ? `${k}k` : String(Math.round(v));
  });
  makeChart('chart-followers-dist', {
    type: 'bar',
    data: {
      labels: edgeLabels,
      datasets: [{
        label: 'Count',
        data: d.counts,
        backgroundColor: '#3b82f688',
        borderColor: '#3b82f6',
        borderWidth: 1,
        borderRadius: 4,
      }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        y: { grid: { color: '#2e3358' } },
        x: { grid: { display: false }, ticks: { maxRotation: 45, font: { size: 10 } } },
      },
    },
  });
}

function drawRatioBar(featureMeans) {
  const labels = LABEL_ORDER;
  makeChart('chart-ratio-bar', {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Avg Following/Followers Ratio',
        data: labels.map(l => featureMeans[l]?.['Following/Followers'] ?? 0),
        backgroundColor: labels.map(l => COLORS[l]),
        borderRadius: 6,
      }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, grid: { color: '#2e3358' } },
        x: { grid: { display: false } },
      },
    },
  });
}

function drawPostsBar(featureMeans) {
  const labels = LABEL_ORDER;
  makeChart('chart-posts-bar', {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Avg Posts',
        data: labels.map(l => featureMeans[l]?.Posts ?? 0),
        backgroundColor: labels.map(l => COLORS[l] + 'bb'),
        borderRadius: 6,
      }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, grid: { color: '#2e3358' } },
        x: { grid: { display: false } },
      },
    },
  });
}

function drawCorrelation(corrData) {
  const canvas = document.getElementById('chart-correlation');
  if (!canvas) return;
  const cols = corrData.columns;
  const matrix = corrData.matrix;
  const n = cols.length;

  const size = Math.min(canvas.parentElement.clientWidth - 40, 560);
  canvas.width = size;
  canvas.height = size;
  const cellSize = (size - 80) / n;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, size, size);

  const pad = 80;

  // Labels
  ctx.font = '11px Segoe UI';
  ctx.fillStyle = '#94a3b8';
  ctx.textAlign = 'right';
  cols.forEach((col, i) => {
    ctx.fillText(col.replace('/', '/\n'), pad - 6, pad + i * cellSize + cellSize / 2 + 4);
  });
  ctx.textAlign = 'center';
  cols.forEach((col, j) => {
    ctx.save();
    ctx.translate(pad + j * cellSize + cellSize / 2, pad - 6);
    ctx.rotate(-Math.PI / 4);
    ctx.fillText(col, 0, 0);
    ctx.restore();
  });

  // Cells
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const val = matrix[i][j];
      const r = val > 0
        ? `rgba(59,130,246,${Math.abs(val).toFixed(2)})`
        : `rgba(239,68,68,${Math.abs(val).toFixed(2)})`;
      ctx.fillStyle = r;
      ctx.fillRect(pad + j * cellSize, pad + i * cellSize, cellSize - 2, cellSize - 2);

      ctx.fillStyle = '#e2e8f0';
      ctx.font = '9px Segoe UI';
      ctx.textAlign = 'center';
      ctx.fillText(val.toFixed(2), pad + j * cellSize + cellSize / 2, pad + i * cellSize + cellSize / 2 + 3);
    }
  }
}

/* ── PCA charts ──────────────────────────────────────────────────────────── */
function drawPCAScatter(pcaData) {
  const labels = LABEL_ORDER;
  const byLabel = {};
  labels.forEach(l => { byLabel[l] = { x: [], y: [] }; });
  pcaData.labels.forEach((l, i) => {
    if (byLabel[l]) {
      byLabel[l].x.push(pcaData.x[i]);
      byLabel[l].y.push(pcaData.y[i]);
    }
  });

  const datasets = labels.map(l => ({
    label: l,
    data: byLabel[l].x.map((x, i) => ({ x, y: byLabel[l].y[i] })),
    backgroundColor: COLORS[l] + '88',
    pointRadius: 3,
    pointHoverRadius: 5,
  }));

  makeChart('chart-pca-scatter', {
    type: 'scatter',
    data: { datasets },
    options: {
      plugins: { legend: { position: 'bottom', labels: { padding: 14, boxWidth: 10 } } },
      scales: {
        x: { title: { display: true, text: `PC1 (${(pcaData.explained_variance[0]*100).toFixed(1)}% var)`, color: '#94a3b8' }, grid: { color: '#2e3358' } },
        y: { title: { display: true, text: `PC2 (${(pcaData.explained_variance[1]*100).toFixed(1)}% var)`, color: '#94a3b8' }, grid: { color: '#2e3358' } },
      },
    },
  });
}

function drawPCALoadings(pcaData) {
  const features = pcaData.loadings.features;
  const pc1 = pcaData.loadings.PC1;
  makeChart('chart-pca-loadings', {
    type: 'bar',
    data: {
      labels: features,
      datasets: [{
        label: 'PC1 Loading',
        data: pc1,
        backgroundColor: pc1.map(v => v >= 0 ? '#3b82f688' : '#ef444488'),
        borderColor: pc1.map(v => v >= 0 ? '#3b82f6' : '#ef4444'),
        borderWidth: 1,
        borderRadius: 4,
      }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        y: { grid: { color: '#2e3358' } },
        x: { grid: { display: false }, ticks: { font: { size: 10 } } },
      },
    },
  });
}

/* ── Outlier charts ──────────────────────────────────────────────────────── */
function drawOutlierRateChart(id, byLabel, title) {
  const labels = LABEL_ORDER.filter(l => byLabel[l]);
  const rates = labels.map(l => {
    const d = byLabel[l];
    return d.total ? Math.round(d.outliers / d.total * 100) : 0;
  });
  makeChart(id, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Outlier Rate (%)',
        data: rates,
        backgroundColor: labels.map(l => COLORS[l]),
        borderRadius: 6,
      }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, max: 100, grid: { color: '#2e3358' } },
        x: { grid: { display: false } },
      },
    },
  });
}

function drawScoreDist(scoreStats) {
  const labels = LABEL_ORDER.filter(l => scoreStats[l]);
  makeChart('chart-score-dist', {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Median Score',
          data: labels.map(l => (scoreStats[l].median * 100).toFixed(1)),
          backgroundColor: labels.map(l => COLORS[l]),
          borderRadius: 6,
        },
        {
          label: 'Mean Score',
          data: labels.map(l => (scoreStats[l].mean * 100).toFixed(1)),
          backgroundColor: labels.map(l => COLORS[l] + '55'),
          borderRadius: 6,
        },
      ],
    },
    options: {
      plugins: { legend: { position: 'bottom' } },
      scales: {
        y: { beginAtZero: true, max: 100, title: { display: true, text: 'Authenticity Score (0-100)', color: '#94a3b8' }, grid: { color: '#2e3358' } },
        x: { grid: { display: false } },
      },
    },
  });
}

function drawIsoLofScatter(sample) {
  const labels = LABEL_ORDER;
  const byLabel = {};
  labels.forEach(l => { byLabel[l] = []; });
  sample.labels.forEach((l, i) => {
    if (byLabel[l]) byLabel[l].push({
      x: sample.iso_scores[i],
      y: sample.lof_scores[i],
    });
  });
  makeChart('chart-iso-lof-scatter', {
    type: 'scatter',
    data: {
      datasets: labels.map(l => ({
        label: l,
        data: byLabel[l],
        backgroundColor: COLORS[l] + '99',
        pointRadius: 4,
      })),
    },
    options: {
      plugins: { legend: { position: 'bottom', labels: { padding: 14, boxWidth: 10 } } },
      scales: {
        x: { title: { display: true, text: 'Isolation Forest Score (normalised)', color: '#94a3b8' }, min: 0, max: 1, grid: { color: '#2e3358' } },
        y: { title: { display: true, text: 'LOF Score (normalised)', color: '#94a3b8' }, min: 0, max: 1, grid: { color: '#2e3358' } },
      },
    },
  });
}

/* ── Gauge chart ─────────────────────────────────────────────────────────── */
function drawGauge(score, color) {
  const canvas = document.getElementById('chart-gauge');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const cx = 110, cy = 120, r = 90, lw = 18;

  ctx.clearRect(0, 0, 220, 220);

  // Background arc
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, 2 * Math.PI);
  ctx.lineWidth = lw;
  ctx.strokeStyle = '#2e3358';
  ctx.stroke();

  // Score arc
  const angle = Math.PI + (score / 100) * Math.PI;
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, angle);
  ctx.lineWidth = lw;
  ctx.strokeStyle = color;
  ctx.lineCap = 'round';
  ctx.stroke();

  // Tick labels
  ctx.font = '11px Segoe UI';
  ctx.fillStyle = '#94a3b8';
  ctx.textAlign = 'center';
  [[0, 'L'], [50, 'M'], [100, 'R']].forEach(([pct, _]) => {
    const a = Math.PI + (pct / 100) * Math.PI;
    const tx = cx + (r + 14) * Math.cos(a);
    const ty = cy + (r + 14) * Math.sin(a);
    ctx.fillText(pct, tx, ty);
  });
}
