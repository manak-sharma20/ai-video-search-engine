import cv2
import torch
from PIL import Image

BATCH_SIZE = 16


def extract_frames(video_path, interval_seconds=1.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps <= 0:
        fps = 30.0

    duration_sec = total_frames / fps
    if duration_sec > 0 and (duration_sec / interval_seconds) > 600:
        interval_seconds = duration_sec / 600

    frames = []
    count = 0
    next_target_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = count / fps
        if current_time >= next_target_time:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((rgb_frame, current_time))
            next_target_time += interval_seconds

        count += 1

    cap.release()
    return frames


def generate_clip_embeddings(frames):
    from model_manager import get_clip
    clip_model, preprocess = get_clip()

    results = []
    timestamps = [ts for _, ts in frames]
    images = [Image.fromarray(f) for f, _ in frames]

    for batch_start in range(0, len(images), BATCH_SIZE):
        batch_imgs = images[batch_start: batch_start + BATCH_SIZE]
        batch_ts   = timestamps[batch_start: batch_start + BATCH_SIZE]

        inputs = torch.stack([preprocess(img) for img in batch_imgs])

        with torch.no_grad():
            features = clip_model.encode_image(inputs)
            features /= features.norm(dim=-1, keepdim=True)

        for feat, ts in zip(features, batch_ts):
            results.append((feat.tolist(), ts))

    return results
