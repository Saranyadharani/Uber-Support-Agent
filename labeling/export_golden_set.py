import json
import sys

import pandas as pd

sys.path.append(".")
import config


# turns the browser-exported labeling_output.json into the final golden_set.csv
# labeling was done fully by hand, no LLM pre-suggestions, so nothing to compare against
def main():
    with open(config.GOLDEN_LABELING_OUTPUT, encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    missing_intent = df["human_intent"].isna().sum()
    missing_esc = df["human_escalate"].isna().sum()
    if missing_intent or missing_esc:
        print(f"WARNING: {missing_intent} rows missing human_intent, "
              f"{missing_esc} missing human_escalate. go back and finish labeling those.")

    df.to_csv(config.GOLDEN_SET_CSV, index=False)

    n = len(df)
    print(f"Wrote {n} labeled examples -> {config.GOLDEN_SET_CSV}")
    print(f"Reply-quality gold notes present for: {(df['reply_quality_note'].str.len() > 0).sum()} / {n}")
    print("\nIntent distribution:")
    print(df["human_intent"].value_counts())
    print("\nEscalation distribution:")
    print(df["human_escalate"].value_counts())


if __name__ == "__main__":
    main()