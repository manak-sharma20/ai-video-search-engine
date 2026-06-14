import chromadb
import uuid

client = chromadb.Client()

visual_collection = client.get_or_create_collection(
    name="visual_frames",
    metadata={"hnsw:space": "cosine"}
)
audio_collection = client.get_or_create_collection(
    name="audio_frames",
    metadata={"hnsw:space": "cosine"}
)

# backward-compat alias
collection = visual_collection


def _add(coll, data):
    if not data:
        return
    embeddings = [item[0] for item in data]
    metadatas  = [{"timestamp": item[1]} for item in data]
    ids        = [str(uuid.uuid4()) for _ in data]
    coll.add(embeddings=embeddings, ids=ids, metadatas=metadatas)


def add_visual_embeddings(data):
    _add(visual_collection, data)


def add_audio_embeddings(data):
    _add(audio_collection, data)


def add_embeddings(data):
    _add(visual_collection, data)
