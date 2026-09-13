# So basically what the code does is it takes the resolved threads from the JSONL file, extracts the first customer message and the last brand reply, and then uses a sentence transformer model to encode the customer messages into vector representations. These vectors are then saved to a NumPy file, and the corresponding records (thread ID, customer text, brand reply) are saved to a pickle file for later retrieval. This allows for efficient similarity searches or retrieval of relevant brand replies based on new customer messages.
import json
import os
import pickle 
import sys

import numpy as np
from sentence_transformers import SentenceTransformer  # it transforms the text into a vector representation 

sys.path.append(".")
import config

def main():
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    records = []
    with open(config.RESOLVED_THREADS_JSONL) as f:
        for line in f:
            thread = json.loads(line)
            turns = thread["turns"]
            first_customer = next((t for t in turns if t["author"] == "customer"), None)
            last_brand = next((t for t in reversed(turns) if t["author"] == "brand"), None)
            if not first_customer or not last_brand:
                continue  # skip threads with no real back-and-forth
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