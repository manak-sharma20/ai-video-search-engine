import whisper

_whisper_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def extract_audio_segments(video_path):
    model = _get_whisper()
    result = model.transcribe(video_path, word_timestamps=True)
    return result.get("segments", [])


def embed_text(text: str) -> list:
    from model_manager import get_text_model
    model = get_text_model()
    emb = model.encode([text], normalize_embeddings=True)
    return emb[0].tolist()


# BGE models use a query instruction prefix at search time (not at index time)
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

def embed_query(text: str) -> list:
    from model_manager import get_text_model
    model = get_text_model()
    emb = model.encode([_BGE_QUERY_PREFIX + text], normalize_embeddings=True)
    return emb[0].tolist()
