import clip
import torch
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

_clip_model = None
_clip_preprocess = None
_text_model = None


def get_clip():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)
    return _clip_model, _clip_preprocess


def get_text_model():
    global _text_model
    if _text_model is None:
        _text_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _text_model
