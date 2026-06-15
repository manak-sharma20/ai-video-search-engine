from youtube_transcript_api import YouTubeTranscriptApi
from audio_pipeline import embed_text

_api = YouTubeTranscriptApi()


def get_transcript_segments(video_id: str) -> list:
    """
    Return transcript as a list of {text, start, duration} dicts.
    Uses the best available track (manual preferred, auto-generated fallback).
    Raises Exception if no transcript exists for this video.
    """
    try:
        # Try a specific language list first, then fall back to whatever is available
        try:
            fetched = _api.fetch(video_id, languages=["en", "en-US", "en-GB"])
        except Exception:
            # list() returns all available tracks; pick the first one
            tracks = _api.list(video_id)
            fetched = next(iter(tracks)).fetch()

        return [
            {"text": s.text, "start": s.start, "duration": s.duration}
            for s in fetched
        ]
    except Exception as e:
        raise Exception(f"Transcript unavailable for '{video_id}': {e}")


def embed_transcript_segments(segments: list, window_size=4.0, step_size=3.0) -> list:
    """
    Sliding-window embedding over transcript segments.
    Returns list of (embedding_vector, timestamp) tuples.
    """
    if not segments:
        return []

    max_time = max(s["start"] + s["duration"] for s in segments)
    results = []
    t = 0.0

    while t < max_time:
        window = [s for s in segments if t <= s["start"] < t + window_size]
        if window:
            text = " ".join(s["text"] for s in window).strip()
            if text:
                results.append((embed_text(text), window[0]["start"]))
        t += step_size

    return results
