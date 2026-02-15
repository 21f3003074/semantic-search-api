import time
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer, CrossEncoder

app = FastAPI()

# -----------------------------
# Load FREE local models
# -----------------------------
print("Loading embedding model...")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Loading reranker model...")
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# -----------------------------
# Dummy documents (113)
# -----------------------------
documents = [
    {"id": i, "content": f"API documentation about authentication method {i}"}
    for i in range(113)
]

# -----------------------------
# Build embeddings once
# -----------------------------
doc_texts = [d["content"] for d in documents]
doc_ids = [d["id"] for d in documents]

print("Creating document embeddings...")
doc_embeddings = embed_model.encode(doc_texts, normalize_embeddings=True)

# -----------------------------
# Vector search
# -----------------------------
def vector_search(query: str, k: int = 5):
    query_emb = embed_model.encode([query], normalize_embeddings=True)[0]

    scores = np.dot(doc_embeddings, query_emb)
    top_idx = np.argsort(scores)[::-1][:k]

    results = []
    for idx in top_idx:
        results.append({
            "id": int(doc_ids[idx]),
            "score": float(scores[idx]),
            "content": doc_texts[idx],
            "metadata": {"source": "vector_search"}
        })

    return results

# -----------------------------
# Re-ranking (FREE local)
# -----------------------------
def rerank_results(query: str, candidates: list, top_k: int = 3):
    if not candidates:
        return []

    pairs = [[query, doc["content"]] for doc in candidates]
    scores = reranker_model.predict(pairs)

    for doc, score in zip(candidates, scores):
        # normalize to 0–1
        norm_score = float(1 / (1 + np.exp(-score)))
        doc["score"] = norm_score

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
