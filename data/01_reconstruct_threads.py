"""
Step 1: reconstruct customer<->brand threads from twcs.csv for one brand.

twcs.csv columns (standard Kaggle "Customer Support on Twitter" schema):
    tweet_id, author_id, inbound, created_at, text,
    response_tweet_id, in_response_to_tweet_id

`inbound` == True means the tweet is FROM a customer TO a company.
We walk each inbound tweet's `response_tweet_id` chain forward and
`in_response_to_tweet_id` chain backward to assemble a full thread.

Output: one JSON object per line in data/processed/<brand>_threads.jsonl:
{
  "thread_id": "<first tweet_id>",
  "turns": [
    {"tweet_id": ..., "author": "customer"|"brand", "text": ..., "created_at": ...},
    ...
  ],
  "resolved": bool   # heuristic: thread ends on a brand turn with no further
                      # customer reply within the data, OR contains a
                      # thank-you / satisfaction phrase from the customer
}
"""
import json
import re
import sys
import pandas as pd
from tqdm import tqdm

sys.path.append(".")
import config

RESOLUTION_PHRASES = re.compile(
    r"\b(thank you|thanks|thx|appreciate it|got it sorted|all set|resolved|"
    r"that (fixed|solved|worked)|perfect,? thanks)\b", re.I
)


def load_brand_df(csv_path: str, brand_handle: str) -> pd.DataFrame:
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, dtype=str)
    df["inbound"] = df["inbound"].astype(str).str.lower() == "true"

    # tweets FROM the brand
    brand_tweets = df[(~df["inbound"]) & (df["author_id"] == brand_handle)]
    brand_tweet_ids = set(brand_tweets["tweet_id"])

    # tweets that are in_response_to a brand tweet, OR whose response_tweet_id
    # touches a brand tweet -> pulls in the surrounding thread
    relevant_ids = set(brand_tweet_ids)
    relevant_ids |= set(
        brand_tweets["in_response_to_tweet_id"].dropna().tolist()
    )
    # customer tweets that got a brand reply
    for _, row in brand_tweets.iterrows():
        if pd.notna(row.get("in_response_to_tweet_id")):
            relevant_ids.add(row["in_response_to_tweet_id"])

    sub = df[df["tweet_id"].isin(relevant_ids) | df["in_response_to_tweet_id"].isin(relevant_ids)]
    print(f"  brand tweets: {len(brand_tweets)}, related subset: {len(sub)}")
    return sub


def build_threads(df: pd.DataFrame, brand_handle: str, max_threads: int):
    by_id = {row["tweet_id"]: row for _, row in df.iterrows()}

    # find thread roots: inbound customer tweets with no in_response_to (start of conversation)
    roots = df[df["inbound"] & df["in_response_to_tweet_id"].isna()]

    threads = []
    for _, root in tqdm(roots.iterrows(), total=min(len(roots), max_threads)):
        if len(threads) >= max_threads:
            break
        turns = []
        cur = root
        seen = set()
        while cur is not None and cur["tweet_id"] not in seen:
            seen.add(cur["tweet_id"])
            author = "customer" if cur["inbound"] else "brand"
            turns.append({
                "tweet_id": cur["tweet_id"],
                "author": author,
                "text": cur["text"],
                "created_at": cur.get("created_at"),
            })
            # move to the first response, if any
            resp_ids = cur.get("response_tweet_id")
            if pd.isna(resp_ids):
                break
            next_id = str(resp_ids).split(",")[0].strip()
            cur = by_id.get(next_id)

        # keep only threads that actually involve the target brand
        if not any(t["author"] == "brand" for t in turns):
            continue
        if len(turns) < 2:
            continue

        last_customer_text = ""
        for t in reversed(turns):
            if t["author"] == "customer":
                last_customer_text = t["text"]
                break
        resolved = bool(RESOLUTION_PHRASES.search(last_customer_text)) or (
            turns[-1]["author"] == "brand"
        )

        threads.append({
            "thread_id": turns[0]["tweet_id"],
            "turns": turns,
            "resolved": resolved,
        })

    return threads


def main():
    df = load_brand_df(config.RAW_CSV, config.BRAND_HANDLE)
    threads = build_threads(df, config.BRAND_HANDLE, config.MAX_THREADS_TO_LOAD)

    with open(config.THREADS_JSONL, "w") as f:
        for t in threads:
            f.write(json.dumps(t) + "\n")

    resolved = [t for t in threads if t["resolved"]]
    with open(config.RESOLVED_THREADS_JSONL, "w") as f:
        for t in resolved:
            f.write(json.dumps(t) + "\n")

    print(f"Wrote {len(threads)} threads -> {config.THREADS_JSONL}")
    print(f"Wrote {len(resolved)} resolved threads (RAG corpus) -> {config.RESOLVED_THREADS_JSONL}")
    print(f"NOTE: 'resolved' is a heuristic (thank-you phrase OR brand-had-last-word). "
          f"This is a known source of label noise -- see report 'misleading headline number' section.")


if __name__ == "__main__":
    main()
