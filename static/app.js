document.addEventListener('DOMContentLoaded', () => {
    const uploadBtn = document.getElementById('upload-btn');
    const fileInput = document.getElementById('file-input');
    const searchInput = document.getElementById('search-input');
    const resultsContainer = document.getElementById('results-container');
    const mainVideo = document.getElementById('main-video');
    const videoOverlay = document.getElementById('video-overlay');

    // 1. Handle Upload
    uploadBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', async (e) => {
        if (!e.target.files.length) return;
        
        const file = e.target.files[0];
        const formData = new FormData();
        formData.append('file', file);
        
        // Show video locally so the user can see it right away
        const videoURL = URL.createObjectURL(file);
        mainVideo.src = videoURL;
        
        const originalText = uploadBtn.innerText;
        uploadBtn.innerText = 'Uploading & Indexing...';
        uploadBtn.disabled = true;

        try {
            const res = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (res.ok) {
                alert('Video processed successfully!');
            } else {
                alert('Error processing video: ' + data.detail);
            }
        } catch (err) {
            console.error(err);
            alert('Failed to upload video');
        } finally {
            uploadBtn.innerText = originalText;
            uploadBtn.disabled = false;
        }
    });

    // 2. Handle Search
    searchInput.addEventListener('keydown', async (e) => {
        if (e.key === 'Enter' && searchInput.value.trim() !== '') {
            const query = searchInput.value.trim();
            resultsContainer.innerHTML = '<p class="text-on-surface-variant text-sm">Searching...</p>';
            
            try {
                const res = await fetch(`/search?query=${encodeURIComponent(query)}`);
                const data = await res.json();
                
                if (res.ok) {
                    renderResults(data.results, query);
                } else {
                    resultsContainer.innerHTML = '<p class="text-error text-sm">Error searching.</p>';
                }
            } catch (err) {
                console.error(err);
                resultsContainer.innerHTML = '<p class="text-error text-sm">Failed to search.</p>';
            }
        }
    });

    function formatTime(seconds) {
        const d = Number(seconds);
        const m = Math.floor(d % 3600 / 60);
        const s = Math.floor(d % 3600 % 60);
        const mDisplay = m < 10 ? '0' + m : m;
        const sDisplay = s < 10 ? '0' + s : s;
        return mDisplay + ":" + sDisplay;
    }

    function renderResults(results, query) {
        resultsContainer.innerHTML = '';
        if (!results || results.length === 0) {
            resultsContainer.innerHTML = '<p class="text-on-surface-variant text-sm">No results found.</p>';
            return;
        }

        results.forEach((metadatasArray) => {
            // results format comes from ChromaDB collection.query() which usually returns nested arrays.
            // If the endpoint strips outer array: [ {timestamp: ...}, {timestamp: ...} ]
            // We handle both nested and unnested gracefully.
            const metadataList = Array.isArray(metadatasArray) ? metadatasArray : [metadatasArray];
            
            metadataList.forEach(meta => {
                if (!meta || meta.timestamp === undefined) return;
                
                const timeSec = parseFloat(meta.timestamp);
                const displayTime = formatTime(timeSec);
                
                const card = document.createElement('div');
                card.className = "group flex gap-4 p-4 rounded-lg bg-surface-container-high hover:bg-surface-bright transition-all cursor-pointer hover:-translate-y-1";
                
                card.innerHTML = `
                    <div class="w-28 h-20 rounded-md overflow-hidden relative flex-shrink-0 bg-surface-container-lowest flex items-center justify-center">
                        <span class="material-symbols-outlined text-outline" style="font-size: 32px">movie</span>
                        <div class="absolute bottom-1 right-1 px-1.5 py-0.5 bg-black/70 rounded text-[10px] font-bold text-primary">${displayTime}</div>
                    </div>
                    <div class="flex flex-col justify-center">
                        <p class="text-on-surface font-medium text-sm leading-snug">Found moment matching "${query}"</p>
                        <span class="text-secondary text-[10px] font-bold mt-2 uppercase tracking-tighter">TIMESTAMP: ${timeSec.toFixed(2)}s</span>
                    </div>
                `;
                
                card.addEventListener('click', () => {
                    mainVideo.currentTime = timeSec;
                    mainVideo.play();
                    videoOverlay.innerText = displayTime;
                    videoOverlay.classList.remove('hidden');
                    
                    // remove active styling from others
                    document.querySelectorAll('#results-container > div').forEach(el => {
                        el.classList.remove('bg-surface-bright', 'border-2', 'border-primary', 'shadow-[0_0_20px_rgba(143,245,255,0.15)]');
                        el.classList.add('bg-surface-container-high');
                    });
                    
                    // add active styling
                    card.classList.remove('bg-surface-container-high');
                    card.classList.add('bg-surface-bright', 'border-2', 'border-primary', 'shadow-[0_0_20px_rgba(143,245,255,0.15)]');
                    
                    setTimeout(() => videoOverlay.classList.add('hidden'), 3000);
                });
                
                resultsContainer.appendChild(card);
            });
        });
    }
});
