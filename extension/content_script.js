// Runs on every youtube.com/watch page.
// Detects navigation (YouTube is a SPA), extracts the video ID,
// and tells the background service worker to start indexing.

const SERVER = 'http://localhost:8000';
let lastVideoId = null;

function videoIdFromUrl(href) {
  try {
    return new URL(href).searchParams.get('v');
  } catch {
    return null;
  }
}

function maybeIndex() {
  const id = videoIdFromUrl(location.href);
  if (!id || id === lastVideoId) return;
  lastVideoId = id;

  const title = document.querySelector('h1.ytd-video-primary-info-renderer')?.textContent?.trim()
    || document.title.replace(' - YouTube', '').trim();

  chrome.runtime.sendMessage({ type: 'INDEX_VIDEO', videoId: id, title });
}

// YouTube never fully reloads — watch for URL / title changes
const observer = new MutationObserver(maybeIndex);
observer.observe(document.head, { childList: true, subtree: true });
observer.observe(document.body, { childList: true, subtree: false });

// Initial check (user landed directly on a /watch URL)
setTimeout(maybeIndex, 1500);

// Listen for seek commands from the popup
chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type !== 'SEEK') return;
  const video = document.querySelector('video');
  if (!video) return;
  video.currentTime = msg.timestamp;
  video.play().catch(() => {});
});
