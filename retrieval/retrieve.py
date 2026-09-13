#This file handles retrieving similar past cases at chat time
import pickle
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(".")
import config

_model = None
_vecs = None
_records = None


# loads the model + saved index once, reused across calls instead of reloading every time
def _load():
    global _model, _vecs, _records
    if _model is not None:
        return
    _model = SentenceTransformer(config.EMBEDDING_MODEL)
    _vecs = np.load(f"{config.INDEX_DIR}/vectors.npy")
    with open(f"{config.INDEX_DIR}/records.pkl", "rb") as f:
        _records = pickle.load(f)


# embeds the new message, finds the k closest past cases by cosine similarity
def retrieve_similar(text: str, k: int = 3) -> list[dict]:
    _load()
    q = _model.encode([text], normalize_embeddings=True)[0]
    sims = _vecs @ q  # dot product of normalized vectors = cosine similarity
    top_k = np.argsort(-sims)[:k]  # indices of the k highest similarities
    return [{**_records[i], "similarity": float(sims[i])} for i in top_k]


# quick manual check
if __name__ == "__main__":
    for r in retrieve_similar("my driver was speeding and it scared me"):
        print(f"[{r['similarity']:.2f}] {r['customer_text'][:60]} -> {r['brand_reply'][:60]}")