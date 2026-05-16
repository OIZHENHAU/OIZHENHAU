/* charts.js — Chart.js rendering helpers */
const Charts = (() => {
  const REAL_COLOR = 'rgba(39, 174, 96, 0.55)';
  const SUSP_COLOR = 'rgba(231, 76, 60, 0.55)';
  const REAL_BORDER = 'rgba(39, 174, 96, 0.9)';
  const SUSP_BORDER = 'rgba(231, 76, 60, 0.9)';

  const _instances = {};

  function _destroy(id) {
    if (_instances[id]) {
      _instances[id].destroy();
      delete _instances[id];
    }
  }

  function _make(id, config) {
    _destroy(id);
    const ctx = document.getElementById(id).getContext('2d');
    _instances[id] = new Chart(ctx, config);
    return _instances[id];
  }

  /* ── Scatter: Engagement Rate vs log(Impressions) ── */
  function drawScatter(data) {
    _make('chart-scatter', {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Real',
            data: data.real,
            backgroundColor: REAL_COLOR,
            pointRadius: 3,
            pointHoverRadius: 5,
          },
          {
            label: 'Suspicious',
            data: data.suspicious,
            backgroundColor: SUSP_COLOR,
            pointRadius: 3,
            pointHoverRadius: 5,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {legend: {display: false}, tooltip: {
          callbacks: {
            label: (ctx) => `(${ctx.parsed.x.toFixed(3)}, ${ctx.parsed.y.toFixed(4)})`,
          },
        }},
        scales: {
          x: {
            title: {display: true, text: 'log(1 + Impressions)', font: {size: 11}},
            grid: {color: 'rgba(0,0,0,0.05)'},
          },
          y: {
            title: {display: true, text: 'Engagement Rate', font: {size: 11}},
            grid: {color: 'rgba(0,0,0,0.05)'},
          },
        },
      },
    });
  }

  /* ── PCA Scatter: PC1 vs PC2 ── */
  function drawPCA(data) {
    _make('chart-pca', {
      type: 'scatter',
      data: {
        datasets: [
          {label: 'Real',       data: data.real,       backgroundColor: REAL_COLOR, pointRadius: 3},
          {label: 'Suspicious', data: data.suspicious, backgroundColor: SUSP_COLOR, pointRadius: 3},
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {legend: {display: false}},
        scales: {
          x: {title: {display: true, text: `PC1 (${(data.explained[0] || 0).toFixed(1)}%)`, font: {size: 11}},
              grid: {color: 'rgba(0,0,0,0.05)'}},
          y: {title: {display: true, text: `PC2 (${(data.explained[1] || 0).toFixed(1)}%)`, font: {size: 11}},
              grid: {color: 'rgba(0,0,0,0.05)'}},
        },
      },
    });
  }

  /* ── Auth Score Distribution ── */
  function drawAuthDist(data) {
    _make('chart-authdist', {
      type: 'bar',
      data: {
        labels: data.labels,
        datasets: [
          {label: 'Real',       data: data.real,       backgroundColor: REAL_COLOR, borderColor: REAL_BORDER, borderWidth: 1},
          {label: 'Suspicious', data: data.suspicious, backgroundColor: SUSP_COLOR, borderColor: SUSP_BORDER, borderWidth: 1},
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {legend: {display: false}},
        scales: {
          x: {stacked: false, title: {display: true, text: 'Auth Score', font: {size: 11}},
              ticks: {maxTicksLimit: 10, font: {size: 10}}},
          y: {title: {display: true, text: 'Count', font: {size: 11}},
              grid: {color: 'rgba(0,0,0,0.05)'}},
        },
      },
    });
  }

  /* ── EDA: Engagement Rate by Platform (bar) ── */
  function drawEdaER(data) {
    const colors = ['#3498db','#e67e22','#9b59b6','#e74c3c','#1abc9c'];
    _make('chart-eda-er', {
      type: 'bar',
      data: {
        labels: data.er_by_platform.platforms,
        datasets: [{
          label: 'Mean ER',
          data: data.er_by_platform.means,
          backgroundColor: colors,
          borderRadius: 4,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {legend: {display: false}},
        scales: {
          x: {grid: {display: false}, ticks: {font: {size: 11}}},
          y: {grid: {color: 'rgba(0,0,0,0.05)'}, ticks: {font: {size: 10}}},
        },
      },
    });
  }

  /* ── EDA: Platform Post Count (doughnut) ── */
  function drawEdaPlatform(data) {
    const labels = Object.keys(data.platform_counts);
    const counts = Object.values(data.platform_counts);
    const colors = ['#3498db','#e67e22','#9b59b6','#e74c3c','#1abc9c'];
    _make('chart-eda-platform', {
      type: 'doughnut',
      data: {labels, datasets: [{data: counts, backgroundColor: colors, borderWidth: 2, borderColor: '#fff'}]},
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {legend: {position: 'right', labels: {font: {size: 11}, padding: 10}}},
      },
    });
  }

  /* ── EDA: Toxicity Distribution (area line) ── */
  function drawEdaTox(data) {
    _make('chart-eda-tox', {
      type: 'line',
      data: {
        labels: data.toxicity_dist.centers,
        datasets: [{
          label: 'Count',
          data: data.toxicity_dist.counts,
          fill: true,
          backgroundColor: 'rgba(231,76,60,0.15)',
          borderColor: 'rgba(231,76,60,0.8)',
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.4,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {legend: {display: false}},
        scales: {
          x: {title: {display: true, text: 'Toxicity Score', font: {size: 10}},
              ticks: {maxTicksLimit: 6, font: {size: 10}}},
          y: {grid: {color: 'rgba(0,0,0,0.04)'}, ticks: {font: {size: 10}}},
        },
      },
    });
  }

  /* ── EDA: Day of Week (bar) ── */
  function drawEdaDow(data) {
    const labels = Object.keys(data.dow_counts);
    const counts = Object.values(data.dow_counts);
    _make('chart-eda-dow', {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Posts',
          data: counts,
          backgroundColor: 'rgba(41,128,185,0.7)',
          borderRadius: 4,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {legend: {display: false}},
        scales: {
          x: {grid: {display: false}, ticks: {font: {size: 10},
              callback: (v, i) => labels[i].slice(0, 3)}},
          y: {grid: {color: 'rgba(0,0,0,0.04)'}, ticks: {font: {size: 10}}},
        },
      },
    });
  }

  return {drawScatter, drawPCA, drawAuthDist, drawEdaER, drawEdaPlatform, drawEdaTox, drawEdaDow};
})();
