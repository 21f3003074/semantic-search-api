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

    top_idx = np.argsort(sims)[::-1][:k]

    results = []
    for idx in top_idx:
        results.append({
            "id": int(doc_ids[idx]),
            "score": float(sims[idx]),
            "content": doc_texts[idx],
            "metadata": {"source": "vector_search"}
        })

    return results

# -----------------------------
# Light reranking
# -----------------------------
def rerank_results(query: str, candidates: list, top_k: int = 3):
    # simple boost based on keyword overlap
    query_words = set(query.lower().split())

    for doc in candidates:
        doc_words = set(doc["content"].lower().split())
        overlap = len(query_words & doc_words)
        boost = overlap / (len(query_words) + 1e-6)

        doc["score"] = float(min(1.0, doc["score"] + 0.2 * boost))

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
