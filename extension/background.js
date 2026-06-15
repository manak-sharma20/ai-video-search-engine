// Service worker — coordinates indexing and polls progress.
// EventSource (SSE) is unavailable in MV3 workers, so we poll /status/.

const SERVER = 'http://localhost:8000';

// In-memory state (survives tab changes, lost on browser restart)
const state = {}; // videoId -> { status, message, pct, title }

// ── Message handler ───────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  switch (msg.type) {

    case 'INDEX_VIDEO':
      handleIndex(msg.videoId, msg.title || '');
      break;

    case 'GET_STATE': {
      sendResponse(state[msg.videoId] || { status: 'not_started', pct: 0 });
      break;
    }

    case 'GET_ACTIVE_TAB_VIDEO':
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        const tab = tabs[0];
        if (!tab) { sendResponse(null); return; }
        try {
          const id = new URL(tab.url).searchParams.get('v');
          sendResponse({ videoId: id, title: tab.title?.replace(' - YouTube', '').trim() || '' });
        } catch {
          sendResponse(null);
        }
      });
      return true; // keep channel open for async response

    case 'SEEK_IN_TAB':
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (!tabs[0]) return;
        chrome.tabs.sendMessage(tabs[0].id, { type: 'SEEK', timestamp: msg.timestamp });
      });
      break;
  }
});

// ── Indexing logic ────────────────────────────────────────────────────────────

async function handleIndex(videoId, title) {
  const current = state[videoId];
  if (current && (current.status === 'done' || current.status === 'running')) return;

  state[videoId] = { status: 'starting', message: 'Starting...', pct: 0, title };
  setBadge(videoId, 'starting');

  try {
    // Verify server is reachable
    await fetch(`${SERVER}/indexed_video`, { signal: AbortSignal.timeout(3000) });
  } catch {
    state[videoId] = { ...state[videoId], status: 'server_down', message: 'Server not running' };
    setBadge(videoId, 'error');
    return;
  }

  // Fire the indexing request
  try {
    const res = await fetch(`${SERVER}/index_youtube`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_id: videoId }),
    });
    if (!res.ok) throw new Error(await res.text());
  } catch (e) {
    state[videoId] = { ...state[videoId], status: 'error', message: String(e) };
    setBadge(videoId, 'error');
    return;
  }

  pollUntilDone(videoId);
}

async function pollUntilDone(videoId) {
  while (true) {
    await sleep(800);
    let data;
    try {
      const res = await fetch(`${SERVER}/status/${videoId}`, { signal: AbortSignal.timeout(3000) });
      data = await res.json();
    } catch {
      // Server went away temporarily — keep trying
      continue;
    }
    state[videoId] = { ...state[videoId], ...data };
    setBadge(videoId, data.status, data.pct);
    if (data.status === 'done' || data.status === 'error') break;
  }
}

function setBadge(videoId, status, pct) {
  // Only update badge when this video is the active tab's video
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (!tabs[0]) return;
    try {
      const id = new URL(tabs[0].url).searchParams.get('v');
      if (id !== videoId) return;
    } catch { return; }

    if (status === 'done') {
      chrome.action.setBadgeText({ text: '✓' });
      chrome.action.setBadgeBackgroundColor({ color: '#00eefc' });
    } else if (status === 'error' || status === 'server_down') {
      chrome.action.setBadgeText({ text: '!' });
      chrome.action.setBadgeBackgroundColor({ color: '#ff6e84' });
    } else if (pct) {
      chrome.action.setBadgeText({ text: `${pct}%` });
      chrome.action.setBadgeBackgroundColor({ color: '#8ff5ff' });
    } else {
      chrome.action.setBadgeText({ text: '…' });
      chrome.action.setBadgeBackgroundColor({ color: '#767579' });
    }
  });
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
