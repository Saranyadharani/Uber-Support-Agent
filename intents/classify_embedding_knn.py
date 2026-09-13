import sys
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(".")
import config
from intents.taxonomy import INTENTS, INTENT_NAMES

# "simple" baseline - embed the taxonomy's own example tweets, embed the
# input, vote by nearest neighbors. no LLM, no labeled training set, just what's already sitting in taxonomy.py
_model = None
_intent_vecs = None
_intent_labels = None


def _load():
    global _model, _intent_vecs, _intent_labels
    if _model is not None:
        return
    _model = SentenceTransformer(config.EMBEDDING_MODEL)
    texts, labels = [], []
    for name, spec in INTENTS.items():
        for ex in spec["examples"]:
            texts.append(ex)
            labels.append(name)
    _intent_vecs = _model.encode(texts, normalize_embeddings=True)
    _intent_labels = labels


def classify(text: str, k: int = 3) -> str:
    _load()
    vec = _model.encode([text], normalize_embeddings=True)[0]
    sims = _intent_vecs @ vec
    top_k = np.argsort(-sims)[:k]
    votes = {}
    for i in top_k:
        label = _intent_labels[i]
        votes[label] = votes.get(label, 0) + sims[i]  # weighted by similarity, not just count
    return max(votes, key=votes.get)


if __name__ == "__main__":
    tests = [
        "my driver was speeding and it scared me",
        "why was I charged $50 for a 5 min ride",
        "how do I add a promo code",
    ]
    for t in tests:
        print(t, "->", classify(t))