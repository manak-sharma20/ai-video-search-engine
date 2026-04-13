import chromadb
from vector_store import collection
def search_embeddings(query_embedding):
    results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)
    return results["metadatas"]
    