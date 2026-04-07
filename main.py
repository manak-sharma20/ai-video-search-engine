from sentence_transformers import SentenceTransformer
import chromadb
import whisper

model=SentenceTransformer("all-MiniLM-L6-v2")
sentences=["I love football","I enjoy soccer"]
embedding=model.encode(sentences)
client=chromadb.Client()
collection=client.create_collection(name="videos")
collection.add(
    documents=sentences,
    embeddings=embedding,
    ids=["1","2"]
)
query="creation"
query_embedding=model.encode(query)
result=collection.query(
    query_embeddings=[query_embedding],
    n_results=1
)
whisper_model=whisper.load_model("base")
result=whisper_model.transcribe("video.mp4")
segments=result["segments"]
i=0
for segment in segments:
    text=segment["text"]
    start=segment["start"]
    emb=model.encode(text)
    collection.add(
        documents=[text],
        embeddings=[emb],
        ids=[str(i)],
        metadatas=[{"start": start}]
    )
    i+=1
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

print(results["documents"])
print(results["metadatas"])
