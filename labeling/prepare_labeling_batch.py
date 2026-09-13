import json
import random
import sys

sys.path.append(".")
import config
from intents.classify_baseline_keyword import classify as classify_keyword
from intents.taxonomy import INTENT_NAMES

random.seed(config.RANDOM_SEED)


# pulls every customer message out of the reconstructed threads, keeps thread
# context too so the labeling tool can show prior turns for ambiguous cases
def load_customer_turns(threads_path):
    candidates = []
    with open(threads_path) as f:
        for line in f:
            thread = json.loads(line)
            turns = thread["turns"]
            for i, t in enumerate(turns):
                if t["author"] == "customer" and len(t["text"].split()) >= 4:
                    candidates.append({
                        "thread_id": thread["thread_id"],
                        "turn_index": i,
                        "text": t["text"],
                        "context": [x["text"] for x in turns[:i]],
                    })
    return candidates


# stratified sample using the keyword classifier just to spread coverage across
# intents (not trusted as ground truth) -- rest overflows into a general pool
def main():
    candidates = load_customer_turns(config.THREADS_JSONL)
    random.shuffle(candidates)
    print(f"{len(candidates)} candidate customer turns available")

    bucketed = {name: [] for name in INTENT_NAMES}
    extra_pool = []
    batch = []

    for c in candidates:
        if len(batch) >= config.GOLDEN_SET_SIZE:
            break
        kw_intent = classify_keyword(c["text"])
        item = {**c, "suggested": None}  # no LLM