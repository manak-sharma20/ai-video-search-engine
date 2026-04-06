from sentence_transformers import SentenceTransformer

import chromadb
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
query="football match"
query_embedding=model.encode(query)
result=collection.query(
    query_embeddings=[query_embedding],
    n_results=1
)
print(result)