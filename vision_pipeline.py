import cv2
import CLIP


def extract_frames(video_path):
    cap=cv2.VideoCapture(video_path)
    fps=cap.get(cv2.CAP_PROP_FPS)
    count=0
    frames=[]
    while True:
        ret,frame=cap.read()
        if not ret:
            break
        if count % int(fps)==0:
            timestamp=count/fps
            frames.append((frame, timestamp))
        count += 1
    return frames

            
