import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.append(".")
import config


# samples n judged replies for hand-scoring -- score blind, don't look at
# the judge column first or you're not really testing agreement
def make_human_scoring_template(judged_csv: str, out_csv: str, n: int = 40, seed: int = None):
    seed = seed or config.RANDOM_SEED
    df = pd.read_csv(judged_csv)
    sample = df.sample(n=min(n, len(df)), random_state=seed).copy()
    sample["human_overall_score"] = ""
    keep_cols = ["text", "intent", "draft_reply", "human_overall_score"]
    sample[keep_cols].to_csv(out_csv, index=False)
    print(f"Wrote {len(sample)} rows -> {out_csv}. Fill in human_overall_score (1-5), "
          f"then run compute_agreement().")


# exact match / within-1 / kappa, plus the actual disagreement list --
# the disagreement list is what makes this credible, not the aggregate number
def compute_agreement(judged_csv: str, human_scored_csv: str):
    judged = pd.read_csv(judged_csv)
    human = pd.read_csv(human_scored_csv)
    merged = human.merge(judged[["text", "overall"]], on="text", how="left", suffixes=("", "_judge"))
    merged = merged.dropna(subset=["human_overall_score", "overall"])
    merged["human_overall_score"] = merged["human_overall_score"].astype(int)
    merged["overall"] = merged["overall"].astype(int)

    exact_match = (merged["human_overall_score"] == merged["overall"]).mean()
    within_1 = (abs(merged["human_overall_score"] - merged["overall"]) <= 1).mean()
    kappa = cohen_kappa_score(merged["human_overall_score"], merged["overall"], weights="linear")

    print(f"N compared: {len(merged)}")
    print(f"Exact match rate: {exact_match:.1%}")
    print(f"Within +/-1 point rate: {within_1:.1%}")
    print(f"Linear-weighted Cohen's kappa: {kappa:.3f}")

    big_disagreements = merged[abs(merged["human_overall_score"] - merged["overall"]) >= 2]
    print(f"\n{len(big_disagreements)} cases with disagreement >=2 points:")
    for _, row in big_disagreements.iterrows():
        print(f"  human={row['human_overall_score']} judge={row['overall']}  reply=\"{row['draft_reply'][:100]}\"")

    return {"exact_match": exact_match, "within_1": within_1, "kappa": kappa,
            "n_big_disagreements": len(big_disagreements)}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "template":
        make_human_scoring_template("eval/results/judged_replies.csv", "eval/results/human_scoring_template.csv")
    else:
        compute_agreement("eval/results/judged_replies.csv", "eval/results/human_scoring_template.csv")