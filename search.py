from vector_store import visual_collection, audio_collection


def _query(coll, embedding, n=5):
    count = coll.count()
    if count == 0:
        return []
    results = coll.query(
        query_embeddings=[embedding],
        n_results=min(count, n)
    )
    out = []
    for dist, meta in zip(results["distances"][0], results["metadatas"][0]):
        # cosine space: dist = 1 - cos_sim, so recover cos_sim directly
        score = round(max(0.0, 1.0 - dist), 3)
        out.append({"timestamp": meta["timestamp"], "score": score})
    return out


def search_visual(query_embedding, n=5):
    return _query(visual_collection, query_embedding, n)


def search_audio(query_embedding, n=5):
    return _query(audio_collection, query_embedding, n)


def merge_results(results, proximity=2.0, min_score=0.10):
    """Deduplicate by timestamp proximity, filter low-confidence, cap at 8."""
    filtered = [r for r in results if r["score"] >= min_score]
    filtered.sort(key=lambda x: -x["score"])

    merged = []
    for r in filtered:
        ts = r["timestamp"]
        if not any(abs(ts - m["timestamp"]) < proximity for m in merged):
            merged.append(r)
        if len(merged) >= 8:
            break
    return merged


def search_embeddings(query_embedding):
    """Legacy visual-only search."""
    return merge_results(_query(visual_collection, query_embedding))
