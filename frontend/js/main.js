/* main.js — application logic */

// ═══════════════════ NAVIGATION ═══════════════════
(function initNav() {
  const navItems = document.querySelectorAll('.nav-item');
  const pages    = document.querySelectorAll('.page');

  navItems.forEach(item => {
    item.addEventListener('click', e => {
      e.preventDefault();
      const target = item.dataset.page;
      navItems.forEach(n => n.classList.remove('active'));
      pages.forEach(p => p.classList.remove('active'));
      item.classList.add('active');
      document.getElementById('page-' + target).classList.add('active');
    });
  });
})();

// ═══════════════════ CHART TABS ═══════════════════
let _tabDataLoaded = {scatter: false, pca: false, authdist: false, eda: false};

(function initTabs() {
  const tabBtns = document.querySelectorAll('.tab-btn');
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.chart-tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const tab = btn.dataset.tab;
      document.getElementById('panel-' + tab).classList.add('active');
      loadTabData(tab);
    });
  });
})();

async function loadTabData(tab) {
  if (_tabDataLoaded[tab]) return;
  _tabDataLoaded[tab] = true;
  try {
    if (tab === 'pca') {
      const d = await API.pca();
      Charts.drawPCA(d);
    } else if (tab === 'authdist') {
      const d = await API.authDist();
      Charts.drawAuthDist(d);
    } else if (tab === 'eda') {
      const d = await API.eda();
      Charts.drawEdaER(d);
      Charts.drawEdaPlatform(d);
      Charts.drawEdaTox(d);
      Charts.drawEdaDow(d);
    }
  } catch (err) {
    console.error('Tab load error:', err);
  }
}

// ═══════════════════ DASHBOARD INIT ═══════════════════
let _allAccounts = [];
let _currentPage = 1;
const PER_PAGE = 10;
let _totalAccounts = 0;
let _searchTimer = null;

async function initDashboard() {
  try {
    // Overview stats
    const ov = await API.overview();
    document.getElementById('stat-total').textContent = ov.total.toLocaleString();
    document.getElementById('stat-real').textContent  = ov.real.toLocaleString();
    document.getElementById('stat-susp').textContent  = ov.suspicious.toLocaleString();
    document.getElementById('stat-avg').textContent   = ov.avg_score;
    document.getElementById('stat-risk').textContent  = ov.high_risk.toLocaleString();
    document.getElementById('badge-loaded').textContent = `${ov.total.toLocaleString()} accounts loaded`;

    // Default scatter chart
    const sc = await API.scatter();
    Charts.drawScatter(sc);

    // Load first page of accounts
    await loadAccounts(1, '');
  } catch (err) {
    console.error('Dashboard init error:', err);
    document.getElementById('badge-loaded').textContent = 'Backend not running';
    document.getElementById('badge-loaded').style.background = '#fde8e8';
    document.getElementById('badge-loaded').style.color = '#b03030';
  }
}

// ── Accounts table ──────────────────────────────────────────────────────────
async function loadAccounts(page, search) {
  _currentPage = page;
  try {
    const data = await API.accounts({page, per_page: PER_PAGE, search});
    _totalAccounts = data.total;
    renderTable(data.accounts);
    renderPagination(data.total, page);
  } catch (err) {
    console.error('Account load error:', err);
  }
}

function renderTable(accounts) {
  const tbody = document.getElementById('accounts-body');
  if (!accounts.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="table-empty">No accounts found.</td></tr>';
    return;
  }
  tbody.innerHTML = accounts.map((a, i) => {
    const isReal = a.label === 'Real';
    const barColor = isReal ? '' : ' low';
    const tag = isReal
      ? `<span class="tag real">&#10003; Real</span>`
      : `<span class="tag suspicious">&#9888; Suspicious</span>`;
    return `
      <tr data-username="${escHtml(a.username)}" class="${i === 0 ? 'selected' : ''}">
        <td>${escHtml(a.username)}</td>
        <td>${escHtml(a.platform)}</td>
        <td>${a.followers.toLocaleString()}</td>
        <td>
          <div class="score-bar-wrap">
            <div class="score-bar-track">
              <div class="score-bar-fill${barColor}" style="width:${a.auth_score}%"></div>
            </div>
            <span>${a.auth_score}</span>
          </div>
        </td>
        <td>${tag}</td>
      </tr>`;
  }).join('');

  // Click handlers
  tbody.querySelectorAll('tr').forEach(tr => {
    tr.addEventListener('click', () => {
      tbody.querySelectorAll('tr').forEach(r => r.classList.remove('selected'));
      tr.classList.add('selected');
      loadAccountDetail(tr.dataset.username);
    });
  });

  // Auto-load first row detail
  if (accounts.length) loadAccountDetail(accounts[0].username);
}

function renderPagination(total, current) {
  const totalPages = Math.ceil(total / PER_PAGE);
  const pag = document.getElementById('pagination');
  if (totalPages <= 1) { pag.innerHTML = ''; return; }

  const maxVisible = 5;
  let start = Math.max(1, current - 2);
  let end   = Math.min(totalPages, start + maxVisible - 1);
  if (end - start < maxVisible - 1) start = Math.max(1, end - maxVisible + 1);

  const search = document.getElementById('search-input').value.trim();
  let html = `<span>Showing ${((current-1)*PER_PAGE)+1}–${Math.min(current*PER_PAGE,total)} of ${total.toLocaleString()}</span>`;
  html += `<button class="pag-btn" ${current===1?'disabled':''} onclick="loadAccounts(${current-1},'${escAttr(search)}')">&#8249;</button>`;
  for (let p = start; p <= end; p++) {
    html += `<button class="pag-btn${p===current?' active':''}" onclick="loadAccounts(${p},'${escAttr(search)}')">${p}</button>`;
  }
  html += `<button class="pag-btn" ${current===totalPages?'disabled':''} onclick="loadAccounts(${current+1},'${escAttr(search)}')">&#8250;</button>`;
  pag.innerHTML = html;
}

// ── Account detail (sections 4 & 6) ────────────────────────────────────────
async function loadAccountDetail(username) {
  try {
    const d = await API.accountDetail(username);
    const isReal = d.label === 'Real';

    document.getElementById('bottom-grid').style.display = 'grid';

    // Score display
    const scoreEl = document.getElementById('det-score');
    scoreEl.textContent = d.auth_score;
    scoreEl.className   = 'detect-score' + (isReal ? ' high' : '');

    document.getElementById('det-label-badge').textContent = isReal ? '✓ Real' : '⚠ Suspicious';
    document.getElementById('det-label-badge').className   = 'verdict-badge ' + (isReal ? 'real' : 'susp');

    document.getElementById('det-if').textContent  = d.if_score;
    document.getElementById('det-lof').textContent = d.lof_score;
    document.getElementById('det-ens').textContent = d.auth_score;

    // Why flagged / why trusted
    const ICONS = {
      'chart-line': '📉', 'alert': '⚠️', 'bar-chart': '📊',
      'trending': '📈', 'user': '👤', 'radar': '🔍',
    };
    const why = document.getElementById('why-grid');
    if (isReal) {
      why.innerHTML = `
        <div class="why-item" style="grid-column:1/-1;text-align:center;padding:18px">
          <div class="why-icon">✅</div>
          <div class="why-title" style="color:var(--green)">Authentic account</div>
          <div class="why-desc">No significant anomaly signals detected. Metrics fall within the normal distribution of organic engagement.</div>
        </div>`;
    } else {
      why.innerHTML = (d.reasons || []).map(r => `
        <div class="why-item">
          <div class="why-icon">${ICONS[r.icon] || '🔍'}</div>
          <div class="why-title">${escHtml(r.title)}</div>
          <div class="why-desc">${escHtml(r.description)}</div>
        </div>`).join('');
    }
  } catch (err) {
    console.error('Account detail error:', err);
  }
}

// ── Search ───────────────────────────────────────────────────────────────────
document.getElementById('search-input').addEventListener('input', e => {
  clearTimeout(_searchTimer);
  _searchTimer = setTimeout(() => {
    loadAccounts(1, e.target.value.trim());
  }, 350);
});

// ═══════════════════ SINGLE ACCOUNT LOOKUP ═══════════════════
// Slider live values
['urand', 'spam', 'generic'].forEach(key => {
  const sl = document.getElementById('sl-' + key);
  const sv = document.getElementById('sv-' + key);
  sl.addEventListener('input', () => { sv.textContent = parseFloat(sl.value).toFixed(2); });
});

document.getElementById('analyse-btn').addEventListener('click', async () => {
  const btn = document.getElementById('analyse-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Analysing…';

  const payload = {
    followers:            parseFloat(document.getElementById('inp-followers').value) || 0,
    following:            parseFloat(document.getElementById('inp-following').value) || 1,
    posts:                parseFloat(document.getElementById('inp-posts').value) || 0,
    account_age_days:     parseFloat(document.getElementById('inp-age').value) || 1,
    bio_length:           parseFloat(document.getElementById('inp-bio').value) || 0,
    has_profile_picture:  parseFloat(document.getElementById('inp-pic').value),
    username_randomness:  parseFloat(document.getElementById('sl-urand').value),
    spam_comments_rate:   parseFloat(document.getElementById('sl-spam').value),
    generic_comment_rate: parseFloat(document.getElementById('sl-generic').value),
  };

  try {
    const res = await API.analyze(payload);
    renderSingleResult(res);
  } catch (err) {
    alert('Error: ' + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span class="btn-dot"></span> Analyse account';
  }
});

function renderSingleResult(res) {
  const isReal = res.label === 'Real';
  const verdictCls = isReal ? 'real' : 'susp';

  document.getElementById('single-results').style.display = 'block';

  ['if', 'lof', 'ens'].forEach((k, i) => {
    const score = [res.if_score, res.lof_score, res.ensemble][i];
    const el = document.getElementById('res-' + k);
    el.textContent = score;
    el.style.color = score >= 50 ? 'var(--green)' : 'var(--red)';
    const lbl = document.getElementById('res-' + k + '-label');
    lbl.textContent = score >= 50 ? '✓ Real' : '⚠ Suspicious';
    lbl.className = 'score-verdict ' + (score >= 50 ? 'real' : 'susp');
  });

  const banner = document.getElementById('single-verdict-banner');
  banner.className = 'verdict-banner ' + verdictCls;
  if (isReal) {
    banner.innerHTML = `<strong>✓ Real account</strong><br />Ensemble authenticity score: ${res.ensemble} / 100 — account metrics appear organic`;
  } else {
    banner.innerHTML = `<strong>⚠ Suspicious account</strong><br />Ensemble authenticity score: ${res.ensemble} / 100 — both models agree this account is anomalous`;
  }
}

// ═══════════════════ UPLOAD CSV ═══════════════════
let _uploadedFile = null;
let _batchRowCount = 0;

const dropzone  = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const browseBtn = document.getElementById('browse-btn');

browseBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

dropzone.addEventListener('dragover', e => {
  e.preventDefault();
  dropzone.classList.add('over');
});

dropzone.addEventListener('dragleave', () => dropzone.classList.remove('over'));

dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('over');
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  _uploadedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    const text = e.target.result;
    const lines = text.trim().split('\n');
    _batchRowCount = Math.max(0, lines.length - 1);

    // Preview table
    const headers = lines[0].split(',').map(h => h.trim());
    const previewRows = lines.slice(1, 4);

    document.getElementById('preview-info').textContent =
      `${file.name} · ${_batchRowCount} rows`;

    const thead = document.getElementById('preview-thead');
    thead.innerHTML = '<tr>' + headers.map(h => `<th>${escHtml(h)}</th>`).join('') + '</tr>';

    const tbody = document.getElementById('preview-tbody');
    tbody.innerHTML = previewRows.map(line => {
      const cells = line.split(',').map(c => c.trim());
      return '<tr>' + cells.map(c => `<td>${escHtml(c)}</td>`).join('') + '</tr>';
    }).join('');

    document.getElementById('file-preview').style.display = 'block';
    document.getElementById('run-count').textContent = _batchRowCount;
    document.getElementById('run-btn').style.display = 'flex';
    document.getElementById('batch-results').style.display = 'none';
    document.getElementById('progress-wrap').style.display = 'none';
  };
  reader.readAsText(file);
}

document.getElementById('run-btn').addEventListener('click', async () => {
  if (!_uploadedFile) return;

  const runBtn = document.getElementById('run-btn');
  runBtn.disabled = true;

  // Show progress
  const pw = document.getElementById('progress-wrap');
  pw.style.display = 'block';
  document.getElementById('batch-results').style.display = 'none';
  animateProgress(0, 70, 800);

  try {
    const res = await API.batch(_uploadedFile);
    animateProgress(70, 100, 400);

    setTimeout(() => {
      document.getElementById('progress-label').textContent =
        `${res.total} / ${res.total} complete`;
      document.getElementById('progress-status').textContent = '✓ Analysis complete';

      renderBatchResults(res);
      runBtn.disabled = false;
    }, 500);
  } catch (err) {
    document.getElementById('progress-status').textContent = '✗ Error: ' + err.message;
    document.getElementById('progress-status').style.color = 'var(--red)';
    runBtn.disabled = false;
  }
});

function animateProgress(from, to, duration) {
  const fill  = document.getElementById('progress-fill');
  const label = document.getElementById('progress-label');
  const total = _batchRowCount;
  const start = performance.now();

  function step(now) {
    const t = Math.min((now - start) / duration, 1);
    const pct = from + (to - from) * t;
    fill.style.width = pct + '%';
    label.textContent = `${Math.round(pct / 100 * total)} / ${total} complete`;
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function renderBatchResults(res) {
  document.getElementById('batch-result-title').textContent =
    `RESULTS — ${res.total} ACCOUNTS SCORED`;
  document.getElementById('batch-real-badge').textContent = `✓ ${res.real} real`;
  document.getElementById('batch-susp-badge').textContent = `⚠ ${res.suspicious} suspicious`;

  const tbody = document.getElementById('batch-body');
  tbody.innerHTML = res.accounts.map(a => {
    const isReal = a.verdict === 'Real';
    const tag = isReal
      ? `<span class="tag real">&#10003; Real</span>`
      : `<span class="tag suspicious">&#9888; Suspicious</span>`;
    return `<tr>
      <td>${escHtml(a.username)}</td>
      <td>${a.if_score}</td>
      <td>${a.lof_score}</td>
      <td>${a.ensemble}</td>
      <td>${tag}</td>
    </tr>`;
  }).join('');

  document.getElementById('batch-results').style.display = 'block';
}

// ═══════════════════ UTILITIES ═══════════════════
function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function escAttr(s) {
  return String(s).replace(/'/g, "\\'");
}

// ═══════════════════ BOOT ═══════════════════
initDashboard();
