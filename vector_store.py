import chromadb
client = chromadb.Client()
collection = client.create_collection(name="frames")
def add_embeddings(data):
    for i, (embedding, timestamp) in enumerate(data):
        collection.add(
            embeddings=[embedding],
            ids=[str(i)],
            metadatas=[{"timestamp": timestamp, "type": "frame"}]
        )