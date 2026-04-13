from audio_pipeline import extract_audio_segments
from vision_pipeline import extract_frames,generate_clip_embeddings
from vector_store import add_embeddings
from search import search_embeddings
import clip
import torch


video_path="data/video.mp4"
segments= extract_audio_segments(video_path)
frames=extract_frames(video_path)
frame_embedding=generate_clip_embeddings(frames)
add_embeddings(frame_embedding)
clip_model,preprocess=clip.load("ViT-B/32")
query = input("Enter your search query: ")
clip_text=clip.tokenize([query])
with torch.no_grad():
    text_features = clip_model.encode_text(clip_text)
query_embedding = text_features.encode_text(query).tolist()
results = search_embeddings(query_embedding)
print(results)