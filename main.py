from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
import shutil
import os
from pydantic import BaseModel
import clip
import torch

# Original imports
from audio_pipeline import extract_audio_segments
from vision_pipeline import extract_frames, generate_clip_embeddings
from vector_store import add_embeddings, collection
from search import search_embeddings

app = FastAPI()

# Load CLIP for search query encoding once on startup
clip_model, preprocess = clip.load("ViT-B/32")

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    # Save file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Clear old data (from original main.py logic)
        try:
            collection.delete(where={})
        except Exception:
            pass # collection might be empty
            
        print(f"Processing audio for {temp_path}...")
        # Audio (not used yet but ready - original logic)
        try:
            segments = extract_audio_segments(temp_path)
        except Exception as e:
            print("Audio extraction failed or skipped:", e)
            
        print(f"Extracting frames for {temp_path}...")
        # Vision
        frames = extract_frames(temp_path)
        print(f"Generating embeddings for {len(frames)} frames...")
        frame_embeddings = generate_clip_embeddings(frames)
        
        # Store
        print("Storing embeddings to ChromaDB...")
        add_embeddings(frame_embeddings)
        print("Done storing.")
        
        return {"message": "Video processed and indexed successfully."}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/search")
async def search_video(query: str):
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    clip_text = clip.tokenize([query])
    with torch.no_grad():
        text_features = clip_model.encode_text(clip_text)
    
    query_embedding = text_features.squeeze().tolist()
    results = search_embeddings(query_embedding)
    
    return {"results": results}

# Mount static files to serve the frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)