# vector_store.py
import numpy as np
import time
from typing import List, Dict, Any

def text_to_vector(text: str, dim=128):
    """
    Simple deterministic hash-based text embedder.
    (Not an actual neural embedder — sufficient for this assignment.)
    """
    vec = np.zeros(dim, dtype=float)
    for i, ch in enumerate(text.lower()):
        vec[i % dim] += (ord(ch) % 97) / 97.0
    # normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

def cosine_sim(a: np.ndarray, b: np.ndarray):
    if a is None or b is None:
        return 0.0
    da = np.linalg.norm(a)
    db = np.linalg.norm(b)
    if da == 0 or db == 0:
        return 0.0
    return float(np.dot(a, b) / (da * db))

class InMemoryVectorStore:
    def __init__(self, dim=128):
        self.dim = dim
        self.records = []  # list of dicts {id, vector, metadata, ts}

    def add(self, id: str, text: str, metadata: Dict[str,Any]):
        vec = text_to_vector(text, dim=self.dim)
        rec = {"id": id, "vector": vec, "text": text, "metadata": metadata.copy(), "ts": time.time()}
        self.records.append(rec)
        return rec

    def query(self, text: str, top_k=5):
        qv = text_to_vector(text, dim=self.dim)
        sims = []
        for r in self.records:
            s = cosine_sim(qv, r["vector"])
            sims.append((s, r))
        sims = sorted(sims, key=lambda x: x[0], reverse=True)[:top_k]
        return [{"score": float(round(s,3)), "id": r["id"], "text": r["text"], "metadata": r["metadata"]} for s,r in sims]
