import time
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# -----------------------------
# Dummy documents (113)
# -----------------------------
documents = [
    {"id": i, "content": f"API documentation about authentication method {i}"}
    for i in range(113)
]

doc_texts = [d["content"] for d in documents]
doc_ids = [d["id"] for d in documents]

# -----------------------------
# TF-IDF Vectorizer (LIGHT)
# -----------------------------
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(doc_texts)

# -----------------------------
# Vector search
# -----------------------------
def vector_search(query: str, k: int = 5):
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, doc_vectors)[0]

    # normalize to 0–1 safely
    sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-9)

    top_idx = np.argsort(sims)[::-1][:k]

    results = []
    for idx in top_idx:
        # deterministic tiny boost to break ties
        rank_boost = (len(doc_ids) - idx) / len(doc_ids) * 0.01
        final_score = float(max(0.001, min(1.0, sims[idx] + rank_boost)))
    
        results.append({
            "id": int(doc_ids[idx]),
            "score": final_score,
            "content": doc_texts[idx],
            "metadata": {"source": "vector_search"}
        })

    return results
# -----------------------------
# Light reranking
# -----------------------------
def rerank_results(query: str, candidates: list, top_k: int = 3):
    if not candidates:
        return []

    query_words = set(query.lower().split())

    for doc in candidates:
        doc_words = set(doc["content"].lower().split())
        overlap = len(query_words & doc_words)

        boost = overlap / (len(query_words) + 1e-6)

        # combine safely and normalize
        new_score = doc["score"] * 0.7 + boost * 0.3
        doc["score"] = float(min(1.0, max(0.001, new_score)))

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]

# -----------------------------
# Request model
# -----------------------------
class SearchRequest(BaseModel):
    query: str
    k: int = 5
    rerank: bool = True
    rerankK: int = 3

# -----------------------------
# API endpoint
# -----------------------------
@app.post("/search")
def semantic_search(req: SearchRequest):
    start_time = time.time()

    candidates = vector_search(req.query, req.k)

    if req.rerank:
        final_results = rerank_results(req.query, candidates, req.rerankK)
        reranked_flag = True
    else:
        final_results = candidates[:req.rerankK]
        reranked_flag = False

    latency = int((time.time() - start_time) * 1000)

    return {
        "results": final_results,
        "reranked": reranked_flag,
        "metrics": {
            "latency": latency,
            "totalDocs": len(documents)
        }
    }


