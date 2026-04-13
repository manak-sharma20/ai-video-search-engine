from audio_pipeline import extract_audio_segments
from vision_pipeline import extract_frames, generate_clip_embeddings
from vector_store import add_embeddings, collection
from search import search_embeddings
import clip
import torch

video_path = input("Enter video path: ")

# Clear old data
collection.delete(where={})

# Audio (not used yet but ready)
segments = extract_audio_segments(video_path)

# Vision
frames = extract_frames(video_path)
frame_embeddings = generate_clip_embeddings(frames)

#  Store
add_embeddings(frame_embeddings)

# Load CLIP for query
clip_model, preprocess = clip.load("ViT-B/32")

query = input("Enter your search query: ")

clip_text = clip.tokenize([query])

with torch.no_grad():
    text_features = clip_model.encode_text(clip_text)

query_embedding = text_features.squeeze().tolist()

results = search_embeddings(query_embedding)

print(results)