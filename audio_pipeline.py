import whisper

def extract_audio_segments(video_path):
    model=whisper.load_model("base")
    result= model.transcribe(video_path)
    segments=result["segments"]
    return segments
