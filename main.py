from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import shutil
import os
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
import yt_dlp
import clip
import torch
from moviepy import VideoFileClip

from model_manager import get_clip, device
from audio_pipeline import extract_audio_segments, embed_text, embed_query
from vision_pipeline import extract_frames, generate_clip_embeddings
from vector_store import add_visual_embeddings, add_audio_embeddings, visual_collection, audio_collection
from search import search_visual, search_audio, merge_results
from youtube_transcript_pipeline import get_transcript_segments, embed_transcript_segments

app = FastAPI()

# Thread pool for blocking ML operations so they don't freeze the event loop
_executor = ThreadPoolExecutor(max_workers=1)

# Indexing progress store: video_id → {status, message, pct}
_progress: dict = {}
# Currently active video ID in ChromaDB
_current_video_id: str | None = None


# ── Request models ────────────────────────────────────────────────────────────

class YouTubeRequest(BaseModel):
    url: str
    quality: str = "720"

class IndexYouTubeRequest(BaseModel):
    video_id: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_progress(video_id: str, status: str, message: str, pct: int):
    _progress[video_id] = {"status": status, "message": message, "pct": pct}


def _clear_collection(coll):
    ids = coll.get(include=[])["ids"]
    if ids:
        coll.delete(ids=ids)


def yt_quality_to_format(quality: str) -> str:
    q = quality.lower().strip()
    if "audio" in q:
        return "bestaudio[ext=m4a]/bestaudio"
    digits = "".join(ch for ch in q if ch.isdigit())
    height = digits if digits else "720"
    return f"bestvideo[height<={height}][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<={height}]+bestaudio/best"


def process_audio_with_sliding_window(video_path, window_size=4.0, step_size=3.0):
    try:
        segments = extract_audio_segments(video_path)
    except Exception as e:
        print("Audio extraction failed (likely no audio stream):", e)
        return []

    all_words = []
    for segment in segments:
        if "words" in segment:
            all_words.extend(segment["words"])

    audio_embeddings = []
    if all_words:
        max_time = all_words[-1]["end"]
        current_start = 0.0
        while current_start < max_time:
            words_in_window = [w for w in all_words if current_start <= w["start"] < current_start + window_size]
            if words_in_window:
                text = " ".join([w["word"] for w in words_in_window]).strip()
                if text:
                    audio_embeddings.append((embed_text(text), words_in_window[0]["start"]))
            current_start += step_size
    else:
        for segment in segments:
            text = segment.get("text", "").strip()
            if not text:
                continue
            audio_embeddings.append((embed_text(text), segment.get("start", 0.0)))

    return audio_embeddings


def index_video_file(video_path: str):
    global _current_video_id
    _clear_collection(visual_collection)
    _clear_collection(audio_collection)

    print(f"Processing audio for {video_path}...")
    audio_embeddings = process_audio_with_sliding_window(video_path)
    if audio_embeddings:
        print(f"Storing {len(audio_embeddings)} audio embeddings...")
        add_audio_embeddings(audio_embeddings)

    print(f"Extracting frames for {video_path}...")
    frames = extract_frames(video_path)
    print(f"Generating embeddings for {len(frames)} frames...")
    frame_embeddings = generate_clip_embeddings(frames)

    print("Storing frame embeddings...")
    add_visual_embeddings(frame_embeddings)
    _current_video_id = None  # local file, not a YouTube ID
    print("Done indexing.")


# ── YouTube transcript indexing (async background) ────────────────────────────

async def _index_youtube_task(video_id: str):
    global _current_video_id
    loop = asyncio.get_running_loop()
    try:
        _set_progress(video_id, "running", "Fetching YouTube transcript...", 10)

        segments = await loop.run_in_executor(
            _executor, get_transcript_segments, video_id
        )
        _set_progress(video_id, "running", f"Embedding {len(segments)} transcript segments...", 45)

        audio_embeddings = await loop.run_in_executor(
            _executor, embed_transcript_segments, segments
        )
        _set_progress(video_id, "running", "Storing embeddings...", 85)

        _clear_collection(visual_collection)
        _clear_collection(audio_collection)
        if audio_embeddings:
            add_audio_embeddings(audio_embeddings)

        _current_video_id = video_id
        _set_progress(video_id, "done", f"Ready — {len(audio_embeddings)} moments indexed", 100)
        print(f"YouTube {video_id} indexed: {len(audio_embeddings)} moments")

    except Exception as e:
        _set_progress(video_id, "error", str(e), 0)
        print(f"YouTube indexing error for {video_id}: {e}")


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.post("/index_youtube")
async def index_youtube_transcript(req: IndexYouTubeRequest):
    video_id = req.video_id.strip()
    if not video_id:
        raise HTTPException(status_code=400, detail="video_id is required")
    _set_progress(video_id, "queued", "Queued...", 0)
    asyncio.create_task(_index_youtube_task(video_id))
    return {"video_id": video_id}


@app.get("/status/{video_id}")
async def get_status(video_id: str):
    """Polling endpoint for extension progress checks."""
    return _progress.get(video_id, {"status": "not_started", "message": "Not started", "pct": 0})


@app.get("/progress/{video_id}")
async def progress_sse(video_id: str):
    """SSE stream for web-app progress bar."""
    async def stream():
        while True:
            p = _progress.get(video_id, {"status": "waiting", "message": "Waiting...", "pct": 0})
            yield f"data: {json.dumps(p)}\n\n"
            if p["status"] in ("done", "error"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/indexed_video")
async def indexed_video():
    """Returns the YouTube video ID currently held in ChromaDB, if any."""
    return {"video_id": _current_video_id}


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    temp_path = "current_video.mp4"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        index_video_file(temp_path)
        return {"message": "Video processed and indexed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")


@app.post("/youtube")
async def youtube_upload(req: YouTubeRequest):
    if not req.url:
        raise HTTPException(status_code=400, detail="YouTube URL cannot be empty")

    ydl_opts = {
        "format": yt_quality_to_format(req.quality),
        "outtmpl": "current_video.%(ext)s",
        "merge_output_format": "mp4",
        "overwrites": True,
    }

    print(f"Downloading YouTube video from {req.url}...")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([req.url])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download YouTube video: {str(e)}")

    temp_path = "current_video.mp4"
    if not os.path.exists(temp_path):
        raise HTTPException(status_code=500, detail="Download failed to produce output file.")

    try:
        index_video_file(temp_path)
        return {"message": "YouTube video downloaded, processed, and indexed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process YouTube video: {str(e)}")


@app.get("/video")
async def get_video():
    if not os.path.exists("current_video.mp4"):
        raise HTTPException(status_code=404, detail="No video found")
    return FileResponse(
        "current_video.mp4",
        media_type="video/mp4",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/search")
async def search_video(query: str):
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    clip_model, _ = get_clip()
    # "a photo of X" prompt template improves CLIP zero-shot retrieval accuracy
    clip_prompt = f"a photo of {query}"
    tokens = clip.tokenize([clip_prompt], truncate=True).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    visual_results = search_visual(text_features.squeeze().tolist(), n=10)

    audio_results = search_audio(embed_query(query))

    results = merge_results(visual_results + audio_results)
    return {"results": results}


@app.get("/download_clip")
async def download_clip(timestamp: float):
    if not os.path.exists("current_video.mp4"):
        raise HTTPException(status_code=404, detail="No video found. Please upload one first.")

    start_time = max(0, timestamp - 2)
    end_time = timestamp + 3
    output_filename = f"clip_{int(timestamp)}.mp4"

    try:
        with VideoFileClip("current_video.mp4") as video:
            end_time = min(end_time, video.duration)
            clip_video = video.subclipped(start_time, end_time)
            clip_video.write_videofile(output_filename, codec="libx264", audio_codec="aac")
        return FileResponse(output_filename, filename=output_filename, media_type="video/mp4")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
