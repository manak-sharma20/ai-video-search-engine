document.addEventListener('DOMContentLoaded', () => {

  // ── Element References ──────────────────────────────────────────────
  const mainVideo        = document.getElementById('main-video');
  const videoPlaceholder = document.getElementById('video-placeholder');
  const videoOverlay     = document.getElementById('video-overlay');

  const panelUpload      = document.getElementById('panel-upload');
  const tabYt            = document.getElementById('tab-yt');
  const tabLocal         = document.getElementById('tab-local');
  const tabContentYt     = document.getElementById('tab-content-yt');
  const tabContentLocal  = document.getElementById('tab-content-local');
  const ytUrlInput       = document.getElementById('yt-url-input');
  const ytLoadBtn        = document.getElementById('yt-load-btn');
  const fileInput        = document.getElementById('file-input');
  const localLoadBtn     = document.getElementById('local-load-btn');
  const localFileName    = document.getElementById('local-file-name');
  const statusBox        = document.getElementById('status-box');
  const statusIcon       = document.getElementById('status-icon');
  const statusLabel      = document.getElementById('status-label');
  const statusSub        = document.getElementById('status-sub');
  const progressBar      = document.getElementById('progress-bar');
  const progressLabel    = document.getElementById('progress-label');

  const panelSearch      = document.getElementById('panel-search');
  const searchInput      = document.getElementById('search-input');
  const searchBtn        = document.getElementById('search-btn');
  const resetBtn         = document.getElementById('reset-btn');
  const resultsContainer = document.getElementById('results-container');

  // ── State ───────────────────────────────────────────────────────────
  let activeTab = 'yt';

  // ── Tab Switcher ────────────────────────────────────────────────────
  function switchTab(tab) {
    activeTab = tab;
    if (tab === 'yt') {
      tabYt.classList.add('active');
      tabLocal.classList.remove('active');
      tabContentYt.classList.remove('hidden');
      tabContentLocal.classList.add('hidden');
    } else {
      tabLocal.classList.add('active');
      tabYt.classList.remove('active');
      tabContentLocal.classList.remove('hidden');
      tabContentYt.classList.add('hidden');
    }
  }

  tabYt.addEventListener('click', () => switchTab('yt'));
  tabLocal.addEventListener('click', () => switchTab('local'));

  // ── Status Box Helpers ──────────────────────────────────────────────
  function showStatus(label, sub, pct, pLabel) {
    statusBox.classList.remove('hidden');
    statusBox.classList.add('flex');
    statusIcon.textContent = 'sync';
    statusIcon.classList.add('spin');
    statusIcon.style.color = '';
    statusLabel.textContent = label;
    statusSub.textContent   = sub;
    progressBar.style.width = pct + '%';
    progressLabel.textContent = pLabel;
  }

  function showSuccess(label) {
    statusIcon.textContent = 'check_circle';
    statusIcon.classList.remove('spin');
    statusIcon.style.color = '#8ff5ff';
    statusLabel.textContent = label;
    statusSub.textContent   = 'Indexing complete — ready to search';
    progressBar.style.width = '100%';
    progressLabel.textContent = 'Done';
  }

  function showError(label) {
    statusIcon.textContent = 'error';
    statusIcon.classList.remove('spin');
    statusIcon.style.color = '#ff6e84';
    statusLabel.textContent = label;
    statusSub.textContent   = 'Please try again';
    progressBar.style.width = '0%';
    progressLabel.textContent = 'Failed';
  }

  function hideStatus() {
    statusBox.classList.add('hidden');
    statusBox.classList.remove('flex');
    statusIcon.style.color = '';
  }

  function disableInputs(disabled) {
    ytUrlInput.disabled    = disabled;
    ytLoadBtn.disabled     = disabled;
    localLoadBtn.disabled  = disabled;
    tabYt.disabled         = disabled;
    tabLocal.disabled      = disabled;
  }

  // ── Index Video (shared logic after upload/download) ────────────────
  async function indexVideo(fetchFn) {
    disableInputs(true);
    showStatus('Uploading video…', 'Sending to server', 20, 'Step 1 of 3 — Uploading');

    let res;
    try {
      res = await fetchFn();
    } catch (err) {
      console.error('Upload/Download error:', err);
      showError('Upload failed — check connection');
      disableInputs(false);
      return;
    }

    showStatus('Extracting features…', 'Running vision & audio ML pipelines', 60, 'Step 2 of 3 — Indexing');

    let data;
    try {
      data = await res.json();
    } catch (err) {
      console.error('JSON parse error:', err);
      showError('Unexpected server response');
      disableInputs(false);
      return;
    }

    if (!res.ok) {
      showError(data.detail || 'Server error during indexing');
      disableInputs(false);
      return;
    }

    showStatus('Finalizing…', 'Almost there', 90, 'Step 3 of 3 — Loading video');

    mainVideo.src = '/video?' + Date.now();
    mainVideo.classList.remove('hidden');
    videoPlaceholder.classList.add('hidden');

    await new Promise(resolve => {
      mainVideo.onloadedmetadata = resolve;
      mainVideo.onerror = resolve;
      setTimeout(resolve, 3000);
    });

    showSuccess('Video indexed successfully!');

    setTimeout(() => {
      panelUpload.classList.add('hidden');
      panelSearch.classList.remove('hidden');
      panelSearch.classList.add('flex');
      searchInput.focus();
    }, 800);
  }

  // ── YouTube Load ────────────────────────────────────────────────────
  ytLoadBtn.addEventListener('click', () => {
    const url = ytUrlInput.value.trim();
    if (!url) {
      ytUrlInput.focus();
      ytUrlInput.style.borderColor = '#ff6e84';
      setTimeout(() => { ytUrlInput.style.borderColor = ''; }, 2000);
      return;
    }

    indexVideo(() => fetch('/youtube', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    }));
  });

  ytUrlInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') ytLoadBtn.click();
  });

  // ── Local File Load ─────────────────────────────────────────────────
  localLoadBtn.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', e => {
    const file = e.target.files[0];
    if (!file) return;

    localFileName.textContent = file.name;
    localFileName.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', file);

    indexVideo(() => fetch('/upload', {
      method: 'POST',
      body: formData
    }));
  });

  // ── Reset (New Video) ───────────────────────────────────────────────
  resetBtn.addEventListener('click', () => {
    panelSearch.classList.add('hidden');
    panelSearch.classList.remove('flex');
    panelUpload.classList.remove('hidden');

    mainVideo.pause();
    mainVideo.src = '';
    mainVideo.classList.add('hidden');
    videoPlaceholder.classList.remove('hidden');
    videoOverlay.classList.add('hidden');

    ytUrlInput.value = '';
    fileInput.value  = '';
    localFileName.classList.add('hidden');
    searchInput.value = '';
    resultsContainer.innerHTML = '<p class="text-on-surface-variant text-sm">Press Search to find moments in the video.</p>';
    hideStatus();
    disableInputs(false);
    switchTab('yt');
  });

  // ── Search ──────────────────────────────────────────────────────────
  async function performSearch() {
    const query = searchInput.value.trim();
    if (!query) {
      searchInput.focus();
      return;
    }

    searchBtn.disabled = true;
    searchBtn.innerHTML = `
      <span class="material-symbols-outlined spin text-[18px]">sync</span>
      Searching…
    `;
    resultsContainer.innerHTML = `
      <div class="flex items-center gap-3 text-on-surface-variant text-sm py-4">
        <span class="material-symbols-outlined spin text-[18px] text-primary">manage_search</span>
        Scanning video for "${query}"…
      </div>`;

    try {
      const res  = await fetch('/search?query=' + encodeURIComponent(query));
      const data = await res.json();

      if (res.ok) {
        renderResults(data.results, query);
      } else {
        resultsContainer.innerHTML = `<p class="text-error text-sm">Search failed: ${data.detail || 'Unknown error'}</p>`;
      }
    } catch (err) {
      console.error('Search error:', err);
      resultsContainer.innerHTML = '<p class="text-error text-sm">Could not reach the server.</p>';
    } finally {
      searchBtn.disabled = false;
      searchBtn.innerHTML = `
        <span class="material-symbols-outlined text-[18px]">manage_search</span>
        Search
      `;
    }
  }

  searchBtn.addEventListener('click', performSearch);
  searchInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') performSearch();
  });

  // ── Helpers ─────────────────────────────────────────────────────────
  function formatTime(sec) {
    const s = Math.floor(sec);
    const m = Math.floor(s / 60);
    const ss = s % 60;
    return String(m).padStart(2, '0') + ':' + String(ss).padStart(2, '0');
  }

  function scoreColor(score) {
    if (score >= 0.7) return '#8ff5ff';
    if (score >= 0.45) return '#ac89ff';
    return '#767579';
  }

  // ── Thumbnail capture via hidden video ──────────────────────────────
  async function captureThumbnails(cards, timestamps) {
    if (!mainVideo.src) return;

    const thumb = document.createElement('video');
    thumb.preload = 'metadata';
    thumb.crossOrigin = 'anonymous';
    thumb.src = mainVideo.src;
    thumb.style.display = 'none';
    document.body.appendChild(thumb);

    await new Promise(r => {
      thumb.onloadedmetadata = r;
      setTimeout(r, 4000);
    });

    const canvas = document.createElement('canvas');
    canvas.width  = 160;
    canvas.height = 90;
    const ctx = canvas.getContext('2d');

    for (let i = 0; i < cards.length; i++) {
      thumb.currentTime = timestamps[i];
      await new Promise(r => {
        const onSeeked = () => { thumb.removeEventListener('seeked', onSeeked); r(); };
        thumb.addEventListener('seeked', onSeeked);
        setTimeout(r, 1500);
      });

      try {
        ctx.drawImage(thumb, 0, 0, 160, 90);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.75);
        const placeholder = cards[i].querySelector('.thumb-placeholder');
        const img = cards[i].querySelector('.thumb-img');
        if (img && placeholder) {
          img.src = dataUrl;
          img.classList.remove('hidden');
          placeholder.classList.add('hidden');
        }
      } catch (_) { /* cross-origin or not-ready, leave placeholder */ }
    }

    thumb.remove();
  }

  // ── Render Results ──────────────────────────────────────────────────
  function renderResults(results, query) {
    resultsContainer.innerHTML = '';

    const flat = [];
    (results || []).forEach(item => {
      if (Array.isArray(item)) item.forEach(m => flat.push(m));
      else flat.push(item);
    });

    if (!flat.length) {
      resultsContainer.innerHTML = '<p class="text-on-surface-variant text-sm">No matching moments found. Try a different query.</p>';
      return;
    }

    const cards = [];
    const timestamps = [];

    flat.forEach(meta => {
      if (!meta || meta.timestamp === undefined) return;

      const timeSec     = parseFloat(meta.timestamp);
      const displayTime = formatTime(timeSec);
      const score       = meta.score != null ? meta.score : null;
      const scorePct    = score != null ? Math.round(score * 100) : null;
      const color       = score != null ? scoreColor(score) : '#767579';

      const card = document.createElement('div');
      card.className = 'result-card flex gap-3 p-3 rounded-xl bg-surface-container-high border border-transparent cursor-pointer';

      card.innerHTML = `
        <div class="w-24 h-16 rounded-lg overflow-hidden relative flex-shrink-0 bg-surface-container-lowest border border-outline-variant/10">
          <span class="thumb-placeholder material-symbols-outlined text-outline/40 absolute inset-0 flex items-center justify-center" style="font-size:28px">movie</span>
          <img class="thumb-img hidden absolute inset-0 w-full h-full object-cover" alt="" />
          <div class="absolute bottom-1 right-1 px-1 py-0.5 bg-black/80 rounded text-[9px] font-bold text-primary leading-none">${displayTime}</div>
        </div>
        <div class="flex flex-col justify-between flex-1 py-0.5 min-w-0">
          <p class="text-on-surface font-medium text-sm leading-snug">At ${displayTime}</p>
          ${scorePct != null
            ? `<div class="flex items-center gap-1.5 mt-1">
                 <div class="h-1 rounded-full flex-1 bg-surface-container-lowest overflow-hidden">
                   <div class="h-full rounded-full" style="width:${scorePct}%;background-color:${color}"></div>
                 </div>
                 <span class="text-[10px] font-bold" style="color:${color}">${scorePct}%</span>
               </div>`
            : `<span class="text-on-surface-variant text-[10px] mt-1">${timeSec.toFixed(2)}s</span>`
          }
        </div>
        <div class="flex items-center flex-shrink-0">
          <button class="download-btn p-2 rounded-lg hover:bg-primary/15 text-on-surface-variant hover:text-primary transition-colors" title="Download 5-second clip">
            <span class="material-symbols-outlined text-[18px]">download</span>
          </button>
        </div>
      `;

      card.addEventListener('click', e => {
        if (e.target.closest('.download-btn')) return;
        seekTo(card, timeSec, displayTime);
      });

      card.querySelector('.download-btn').addEventListener('click', e => {
        e.stopPropagation();
        const btn = e.currentTarget;
        btn.innerHTML = '<span class="material-symbols-outlined spin text-[18px]">sync</span>';
        window.location.href = `/download_clip?timestamp=${timeSec}`;
        setTimeout(() => {
          btn.innerHTML = '<span class="material-symbols-outlined text-[18px]">download</span>';
        }, 3000);
      });

      resultsContainer.appendChild(card);
      cards.push(card);
      timestamps.push(timeSec);
    });

    // Populate thumbnails in background without blocking the UI
    captureThumbnails(cards, timestamps).catch(() => {});
  }

  function seekTo(activeCard, timeSec, displayTime) {
    mainVideo.currentTime = timeSec;
    mainVideo.play().catch(() => {});

    videoOverlay.textContent = displayTime;
    videoOverlay.classList.remove('hidden');
    setTimeout(() => videoOverlay.classList.add('hidden'), 3000);

    document.querySelectorAll('#results-container .result-card').forEach(el => {
      el.classList.remove('bg-surface-bright', 'border-primary', 'active-card');
      el.classList.add('bg-surface-container-high', 'border-transparent');
    });
    activeCard.classList.remove('bg-surface-container-high', 'border-transparent');
    activeCard.classList.add('bg-surface-bright', 'border-primary', 'active-card');
  }

});
