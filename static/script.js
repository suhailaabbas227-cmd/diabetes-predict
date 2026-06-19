// ============================================================
//  DiabetesPredict — Frontend Logic
// ============================================================

const FEATURES = [
  'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
  'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits',
  'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
  'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex',
  'Age', 'Education', 'Income'
];

const GAUGE_CIRC = 2 * Math.PI * 80;   // r = 80
const GAUGE_COLORS = { green: '#34d399', orange: '#fbbf24', red: '#f87171' };

// ── Load model info (stats + feature importance) ────────────
async function loadModelInfo() {
  try {
    const res  = await fetch('/model-info');
    const info = await res.json();

    document.getElementById('statAcc').textContent      = info.accuracy + '%';
    document.getElementById('statSamples').textContent  = (info.train_samples / 1000).toFixed(0) + 'K';
    document.getElementById('statFeatures').textContent = info.n_features;
    document.getElementById('statAlgo').textContent     = 'RF';

    const chart = document.getElementById('featChart');
    chart.innerHTML = '';
    const max = Math.max(...info.top_features.map(f => f.importance));
    info.top_features.forEach((f, i) => {
      const row = document.createElement('div');
      row.className = 'feat-row';
      row.innerHTML = `
        <span class="feat-name">${f.name}</span>
        <div class="feat-bar-track"><div class="feat-bar"></div></div>
        <span class="feat-val">${f.importance}%</span>`;
      chart.appendChild(row);
      // animate
      setTimeout(() => {
        row.querySelector('.feat-bar').style.width = (f.importance / max * 100) + '%';
      }, 120 + i * 80);
    });
  } catch (err) {
    document.getElementById('featChart').innerHTML =
      '<div class="feat-loading">Could not load model insights.</div>';
  }
}

// ── Build payload from controls ─────────────────────────────
function collectPayload() {
  const form = document.getElementById('predForm');
  const payload = {};
  FEATURES.forEach(name => {
    const el = form.elements[name];
    if (!el) return;
    if (el.type === 'checkbox') {
      payload[name] = el.checked ? 1 : 0;
    } else if (el.length && el[0] && el[0].type === 'radio') {
      // radio group (Sex)
      payload[name] = parseFloat(form.querySelector(`input[name="${name}"]:checked`).value);
    } else {
      payload[name] = parseFloat(el.value);
    }
  });
  return payload;
}

// ── Submit ──────────────────────────────────────────────────
document.getElementById('predForm').addEventListener('submit', async function (e) {
  e.preventDefault();

  document.getElementById('btnText').classList.add('hidden');
  document.getElementById('btnLoader').classList.remove('hidden');
  document.getElementById('errorBox').classList.add('hidden');

  try {
    const res  = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(collectPayload())
    });
    const data = await res.json();

    if (data.success) {
      renderResult(data);
    } else {
      showError('Prediction error: ' + data.error);
    }
  } catch (err) {
    showError('Could not reach the server. Make sure app.py is running.');
  } finally {
    document.getElementById('btnText').classList.remove('hidden');
    document.getElementById('btnLoader').classList.add('hidden');
  }
});

// ── Render result ───────────────────────────────────────────
function renderResult(data) {
  document.getElementById('resultEmpty').classList.add('hidden');
  document.getElementById('resultBody').classList.remove('hidden');

  // Gauge
  const color = GAUGE_COLORS[data.color];
  const gauge = document.getElementById('gaugeFg');
  gauge.style.stroke = color;
  // reset then animate
  gauge.style.strokeDashoffset = GAUGE_CIRC;
  requestAnimationFrame(() => {
    gauge.style.strokeDashoffset = GAUGE_CIRC * (1 - data.risk_score / 100);
  });
  animateNumber(document.getElementById('gaugeScore'), data.risk_score, '%');

  // Label + advice
  const lbl = document.getElementById('resultLabel');
  lbl.textContent = data.label;
  lbl.className = 'result-label ' + data.color;
  document.getElementById('resultAdvice').textContent = data.advice;

  // Probability bars
  const keys = ['No Diabetes', 'Pre-Diabetes', 'Diabetes'];
  keys.forEach((k, i) => {
    const pct = data.probabilities[k];
    setTimeout(() => {
      document.getElementById('bar' + i).style.width = pct + '%';
    }, 100);
    document.getElementById('pct' + i).textContent = pct + '%';
  });

  document.getElementById('result').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Count-up animation for the gauge score
function animateNumber(el, target, suffix) {
  const start = performance.now(), dur = 1000;
  function step(now) {
    const p = Math.min((now - start) / dur, 1);
    el.textContent = (target * (0.5 - Math.cos(Math.PI * p) / 2)).toFixed(1) + suffix;
    if (p < 1) requestAnimationFrame(step);
    else el.textContent = target + suffix;
  }
  requestAnimationFrame(step);
}

function showError(msg) {
  const box = document.getElementById('errorBox');
  box.textContent = '⚠️ ' + msg;
  box.classList.remove('hidden');
}

// Init
loadModelInfo();
