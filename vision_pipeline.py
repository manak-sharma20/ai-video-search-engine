import cv2
import clip
import torch
from PIL import Image

clip_model, preprocess = clip.load("ViT-B/32")

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    count = 0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % int(fps) == 0:
            timestamp = count / fps
            frames.append((frame, timestamp))

        count += 1

    return frames


def generate_clip_embeddings(frames):
    results = []

    for frame, timestamp in frames:
        image = Image.fromarray(frame)
        image_input = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            image_features = image_features.squeeze().tolist()

        results.append((image_features, timestamp))

    return results