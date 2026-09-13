import sys

import pandas as pd

sys.path.append(".")
from intents.classify_llm import get_client, classify
from agent.escalation_policy import decide


# it reruns only the known escalation misses through the improved prompt 
def main():
    df = pd.read_csv("eval/results/predictions.csv")
    df["human_escalate"] = df["human_escalate"].astype(str).str.strip() == "True"
    df["pred_escalate_system"] = df["pred_escalate_system"].astype(str).str.strip() == "True"

    false_negatives = df[(df["human_escalate"]) & (~df["pred_escalate_system"])].copy()
    print(f"Re-testing {len(false_negatives)} known escalation false-negatives "
          f"with the improved distress-signal prompt...\n")

    client = get_client()
    results = []
    fixed_count = 0

    for _, row in false_negatives.iterrows():
        text = row["text"]
        cls = classify(client, text, context=[])
        new_intent = cls["intent"]
        new_signals = cls.get("signals", {})
        new_decision = decide(new_intent, new_signals)

        fixed = new_decision["escalate"]
        if fixed:
            fixed_count += 1

        results.append({
            "text": text,
            "old_intent": row["pred_intent_llm"],
            "new_intent": new_intent,
            "old_escalate": False,
            "new_escalate": fixed,
            "new_reasons": "; ".join(new_decision["reasons"]),
        })
        status = "FIXED" if fixed else "still missed"
        print(f"  [{status:12s}] {text[:80]}")

    out_df = pd.DataFrame(results)
    out_df.to_csv("eval/results/escalation_retest.csv", index=False)

    n = len(false_negatives)
    print(f"\n{'='*60}")
    print(f"BEFORE: {n}/{n} escalation false negatives (0 correctly escalating)")
    print(f"AFTER improved prompt: {fixed_count}/{n} now correctly escalate "
          f"({fixed_count/n:.1%} fixed)")
    print(f"Remaining false negatives: {n - fixed_count}")
    print(f"\nFull detail written to eval/results/escalation_retest.csv")


if __name__ == "__main__":
    main()