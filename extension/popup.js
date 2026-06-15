const SERVER = 'http://localhost:8000';

// ── DOM refs ────────────────────────────────────────────────────────────────
const sNotYt    = document.getElementById('s-not-youtube');
const sNoServer = document.getElementById('s-no-server');
const sIndexing = document.getElementById('s-indexing');
const sSearch   = document.getElementById('s-search');

const videoHeader = document.getElementById('video-header');
const videoThumb  = document.getElementById('video-thumb');
const videoTitle  = document.getElementById('video-title');

const progressMsg  = document.getElementById('progress-msg');
const progressPct  = document.getElementById('progress-pct');
const progressFill = document.getElementById('progress-fill');

const searchInput  = document.getElementById('search-input');
const searchBtn    = document.getElementById('search-btn');
const resultsEl    = document.getElementById('results-container');
const resultsHint  = document.getElementById('results-hint');
const retryBtn     = document.getElementById('retry-btn');

// ── State ───────────────────────────────────────────────────────────────────
let currentVideoId = null;
let pollTimer = null;

// ── Init ────────────────────────────────────────────────────────────────────
async function init() {
  const tab = await getActiveTabVideo();
  if (!tab || !tab.videoId) { show(sNotYt); return; }

  currentVideoId = tab.videoId;
  showVideoHeader(tab.videoId, tab.title);

  // Ask background for current state
  const s = await bgMessage({ type: 'GET_STATE', videoId: tab.videoId });
  handleState(s);
}

function handleState(s) {
  if (!s) { show(sNoServer); return; }

  if (s.status === 'server_down') { show(sNoServer); return; }

  if (s.status === 'done') {
    show(sSearch);
    stopPoll();
    return;
  }

  if (s.status === 'error') {
    show(sNoServer);
    stopPoll();
    return;
  }

  // Still indexing (running / starting / queued / not_started)
  show(sIndexing);
  updateProgress(s);
  startPoll();
}

// ── Polling ─────────────────────────────────────────────────────────────────
function startPoll() {
  if (pollTimer) return;
  pollTimer = setInterval(async () => {
    if (!currentVideoId) { stopPoll(); return; }
    const s = await bgMessage({ type: 'GET_STATE', videoId: currentVideoId });
    handleState(s);
  }, 800);
}

function stopPoll() {
  clearInterval(pollTimer);
  pollTimer = null;
}

function updateProgress(s) {
  progressMsg.textContent  = s.message || 'Working…';
  progressPct.textContent  = `${s.pct || 0}%`;
  progressFill.style.width = `${s.pct || 0}%`;
}

// ── Search ───────────────────────────────────────────────────────────────────
searchBtn.addEventListener('click', doSearch);
searchInput.addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });

async function doSearch() {
  const q = searchInput.value.trim();
  if (!q) { searchInput.focus(); return; }

  searchBtn.disabled = true;
  searchBtn.innerHTML = '<span class="spin">⟳</span>';
  resultsEl.innerHTML = '<span class="no-results">Searching…</span>';

  try {
    const res  = await fetch(`${SERVER}/search?query=${encodeURIComponent(q)}`);
    const data = await res.json();
    if (res.ok) renderResults(data.results, q);
    else resultsEl.innerHTML = `<span class="no-results" style="color:#ff6e84">Error: ${data.detail}</span>`;
  } catch {
    resultsEl.innerHTML = '<span class="no-results" style="color:#ff6e84">Server not reachable.</span>';
  } finally {
    searchBtn.disabled = false;
    searchBtn.textContent = 'Search';
  }
}

function renderResults(results, query) {
  const flat = (results || []).flat().filter(r => r && r.timestamp != null);
  if (!flat.length) {
    resultsEl.innerHTML = `<span class="no-results">No matches for "${query}".</span>`;
    return;
  }

  resultsEl.innerHTML = '<div class="results-label">Relevant Moments</div>';
  flat.forEach(r => {
    const ts    = parseFloat(r.timestamp);
    const pct   = r.score != null ? Math.round(r.score * 100) : null;
    const color = pct == null ? '#767579' : pct >= 70 ? '#8ff5ff' : pct >= 45 ? '#ac89ff' : '#767579';

    const card = document.createElement('div');
    card.className = 'result-card';
    card.innerHTML = `
      <div class="ts-badge">${formatTime(ts)}</div>
      <div class="result-right">
        ${pct != null
          ? `<div class="score-row">
               <div class="score-track"><div class="score-fill" style="width:${pct}%;background:${color}"></div></div>
               <span class="score-pct" style="color:${color}">${pct}%</span>
             </div>`
          : `<span style="font-size:10px;color:#767579">${ts.toFixed(1)}s</span>`
        }
      </div>
      <button class="jump-btn" title="Jump to this moment">▶ Jump</button>
    `;

    card.querySelector('.jump-btn').addEventListener('click', (e) => {
      e.stopPropagation();
      bgMessage({ type: 'SEEK_IN_TAB', timestamp: ts });
      // Close popup so user sees the video
      window.close();
    });

    card.addEventListener('click', () => {
      bgMessage({ type: 'SEEK_IN_TAB', timestamp: ts });
      window.close();
    });

    resultsEl.appendChild(card);
  });
}

// ── Retry ───────────────────────────────────────────────────────────────────
retryBtn.addEventListener('click', () => {
  hideAll();
  init();
});

// ── Helpers ──────────────────────────────────────────────────────────────────

function show(el) {
  hideAll();
  el.classList.remove('hidden');
  if (el === sSearch || el === sIndexing) videoHeader.classList.remove('hidden');
}

function hideAll() {
  [sNotYt, sNoServer, sIndexing, sSearch, videoHeader].forEach(e => e.classList.add('hidden'));
}

function showVideoHeader(videoId, title) {
  videoThumb.src = `https://i.ytimg.com/vi/${videoId}/mqdefault.jpg`;
  videoTitle.textContent = title || 'YouTube Video';
  videoHeader.classList.remove('hidden');
}

function formatTime(sec) {
  const s = Math.floor(sec);
  const m = Math.floor(s / 60);
  const ss = s % 60;
  return `${String(m).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
}

function getActiveTabVideo() {
  return bgMessage({ type: 'GET_ACTIVE_TAB_VIDEO' });
}

function bgMessage(msg) {
  return new Promise((resolve) => {
    try {
      chrome.runtime.sendMessage(msg, resolve);
    } catch {
      resolve(null);
    }
  });
}

// Start
init();
