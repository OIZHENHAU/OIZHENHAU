/* ── Tab navigation ──────────────────────────────────────────────────────── */
const TAB_LOAD = { overview: false, eda: false, pca: false, outliers: false };

document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', () => {
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    item.classList.add('active');
    const tab = item.dataset.tab;
    document.getElementById(`tab-${tab}`).classList.add('active');
    loadTab(tab);
  });
});

function showLoader(text = 'Loading…') {
  document.getElementById('loader-text').textContent = text;
  document.getElementById('loader').style.display = 'flex';
}

function hideLoader() {
  document.getElementById('loader').style.display = 'none';
}

/* ── Lazy tab loader ─────────────────────────────────────────────────────── */
async function loadTab(tab) {
  if (TAB_LOAD[tab]) return;
  TAB_LOAD[tab] = true;

  try {
    if (tab === 'overview' || tab === 'eda') {
      showLoader('Running EDA…');
      const data = await fetchEDA();
      hideLoader();
      if (tab === 'overview') renderOverview(data);
      else renderEDA(data);
    } else if (tab === 'pca') {
      showLoader('Running PCA…');
      const data = await fetchPCA();
      hideLoader();
      renderPCA(data);
    } else if (tab === 'outliers') {
      showLoader('Running Isolation Forest & LOF…');
      const data = await fetchOutliers();
      hideLoader();
      renderOutliers(data);
    }
  } catch (e) {
    hideLoader();
    TAB_LOAD[tab] = false;
    alert('Error loading data: ' + e.message);
  }
}

/* ── Overview ────────────────────────────────────────────────────────────── */
function renderOverview(data) {
  document.getElementById('stat-total').textContent = data.total_records.toLocaleString();
  document.getElementById('stat-real').textContent  = (data.label_counts.Real  ?? 0).toLocaleString();
  document.getElementById('stat-bot').textContent   = (data.label_counts.Bot   ?? 0).toLocaleString();
  document.getElementById('stat-scam').textContent  = ((data.label_counts.Scam ?? 0) + (data.label_counts.Spam ?? 0)).toLocaleString();

  drawLabelPie(data.label_counts);
  drawFollowersBar(data.feature_means_by_label);
  drawFollowingBar(data.feature_means_by_label);
  drawBinaryRadar(data.binary_rates_by_label);
}

/* ── EDA ─────────────────────────────────────────────────────────────────── */
function renderEDA(data) {
  drawFollowersDist(data.distributions);
  drawRatioBar(data.feature_means_by_label);
  drawPostsBar(data.feature_means_by_label);
  drawCorrelation(data.correlation);
}

/* ── PCA ─────────────────────────────────────────────────────────────────── */
function renderPCA(data) {
  const pct1 = (data.explained_variance[0] * 100).toFixed(1);
  const pct2 = (data.explained_variance[1] * 100).toFixed(1);
  document.getElementById('pca-variance-info').textContent =
    `PC1 explains ${pct1}% of variance | PC2 explains ${pct2}% | Combined: ${(+pct1 + +pct2).toFixed(1)}%`;
  drawPCAScatter(data);
  drawPCALoadings(data);
}

/* ── Outlier Detection ───────────────────────────────────────────────────── */
function renderOutliers(data) {
  // Summary stats cards
  const grid = document.getElementById('outlier-stats');
  grid.innerHTML = '';
  ['Real', 'Bot', 'Scam', 'Spam'].forEach(label => {
    const s = data.score_stats_by_label[label];
    if (!s) return;
    const card = document.createElement('div');
    card.className = 'stat-card';
    card.innerHTML = `
      <div class="stat-value" style="color:${COLORS[label]};font-size:20px">${(s.mean*100).toFixed(1)}</div>
      <div class="stat-label">${label} — Avg Auth Score</div>
    `;
    grid.appendChild(card);
  });

  drawOutlierRateChart('chart-iso-outlier', data.iso_outlier_by_label, 'Isolation Forest');
  drawOutlierRateChart('chart-lof-outlier', data.lof_outlier_by_label, 'LOF');
  drawScoreDist(data.score_stats_by_label);
  drawIsoLofScatter(data.sample);
}

/* ── Account Analyzer ───────────────────────────────────────────────────── */
document.getElementById('btn-analyze').addEventListener('click', async () => {
  const payload = {
    followers:       +document.getElementById('a-followers').value,
    following:       +document.getElementById('a-following').value,
    posts:           +document.getElementById('a-posts').value,
    mutual_friends:  +document.getElementById('a-mutual').value,
    bio:              document.getElementById('a-bio').checked,
    profile_picture:  document.getElementById('a-pic').checked,
    external_link:    document.getElementById('a-link').checked,
    threads:          document.getElementById('a-threads').checked,
  };

  showLoader('Computing Authenticity Score…');
  try {
    const result = await analyzeAccount(payload);
    hideLoader();
    renderAnalyzerResult(result, payload);
  } catch (e) {
    hideLoader();
    alert('Error: ' + e.message);
  }
});

function renderAnalyzerResult(result, payload) {
  const panel = document.getElementById('analyzer-result');
  panel.style.display = 'flex';

  document.getElementById('score-number').textContent = result.authenticity_score + '%';
  document.getElementById('score-number').style.color = result.color;
  document.getElementById('score-label').textContent = result.level;

  drawGauge(result.authenticity_score, result.color);

  // Flags
  const flagsEl = document.getElementById('flags-section');
  if (result.flags && result.flags.length) {
    flagsEl.innerHTML = '<div style="font-size:12px;color:#94a3b8;margin-bottom:6px;font-weight:600;">RISK FLAGS</div>' +
      result.flags.map(f => `<div class="flag-item">⚠ ${f}</div>`).join('');
  } else {
    flagsEl.innerHTML = '<div style="font-size:13px;color:#22c55e;">✓ No risk flags detected</div>';
  }

  // Derived metrics
  const followers = payload.followers || 1;
  const ratio = payload.followers > 0 ? (payload.following / payload.followers).toFixed(2) : 'N/A';
  const pf    = payload.followers > 0 ? (payload.posts    / payload.followers).toFixed(4) : 'N/A';
  document.getElementById('derived-metrics').innerHTML = `
    <div class="derived-metric"><div class="dm-label">Following/Followers</div><div class="dm-value">${ratio}</div></div>
    <div class="derived-metric"><div class="dm-label">Posts/Followers</div><div class="dm-value">${pf}</div></div>
    <div class="derived-metric"><div class="dm-label">Followers</div><div class="dm-value">${payload.followers.toLocaleString()}</div></div>
    <div class="derived-metric"><div class="dm-label">Posts</div><div class="dm-value">${payload.posts.toLocaleString()}</div></div>
  `;
}

/* ── AI Report ───────────────────────────────────────────────────────────── */
document.getElementById('btn-report').addEventListener('click', async () => {
  const btn = document.getElementById('btn-report');
  const btnText = document.getElementById('btn-report-text');
  btn.disabled = true;
  btnText.textContent = 'Generating…';
  showLoader('AI agent is analysing the dataset… (~30 seconds)');

  try {
    const data = await generateReport();
    hideLoader();
    btn.disabled = false;
    btnText.textContent = 'Regenerate Report';

    const output = document.getElementById('report-output');
    output.style.display = 'block';
    output.innerHTML = marked.parse(data.report || data.error || 'No report generated.');
  } catch (e) {
    hideLoader();
    btn.disabled = false;
    btnText.textContent = 'Generate Report';
    alert('Error: ' + e.message);
  }
});

/* ── Boot: load overview immediately ─────────────────────────────────────── */
loadTab('overview');
