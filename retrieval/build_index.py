"""
Build a local embedding index over historically RESOLVED threads
(data/processed/<brand>_resolved_threads.jsonl), keyed by the customer's
opening message. At reply time we retrieve the k most similar past resolved
cases and show the brand's actual historical reply as grounding context.

Note (decision log): resolution is a heuristic (see 01_reconstruct_threads.py)
-- some "resolved" threads are actually just abandoned. This is a known
source of noise in the grounding corpus, discussed in the report.
"""
import json
import os
import pickle
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(".")
import config


def main():
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    records = []
    with open(config.RESOLVED_THREADS_JSONL, encoding="utf-8") as f:
        for line in f:
            thread = json.loads(line)
            turns = thread["turns"]
            first_customer = next((t for t in turns if t["author"] == "customer"), None)
            last_brand = next((t for t in reversed(turns) if t["author"] == "brand"), None)
            if not first_customer or not last_brand:
                continue
            records.append({
                "thread_id": thread["thread_id"],
                "customer_text": first_customer["text"],
                "brand_reply": last_brand["text"],
            })

    print(f"Indexing {len(records)} resolved (customer_msg -> brand_reply) pairs")
    texts = [r["customer_text"] for r in records]
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    os.makedirs(config.INDEX_DIR, exist_ok=True)
    np.save(f"{config.INDEX_DIR}/vectors.npy", vecs)
    with open(f"{config.INDEX_DIR}/records.pkl", "wb") as f:
        pickle.dump(records, f)
    print(f"Wrote index -> {config.INDEX_DIR}/")


if __name__ == "__main__":
    main()