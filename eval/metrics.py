import sys

import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support

sys.path.append(".")


def intent_metrics(golden_df: pd.DataFrame, pred_col: str, gold_col: str = "human_intent"):
    report = classification_report(golden_df[gold_col], golden_df[pred_col],
                                     zero_division=0, output_dict=True)
    return report


def escalation_metrics(golden_df: pd.DataFrame, pred_col: str, gold_col: str = "human_escalate"):
    gold = golden_df[gold_col].astype(bool)
    pred = golden_df[pred_col].astype(bool)
    precision, recall, f1, _ = precision_recall_fscore_support(gold, pred, average="binary", zero_division=0)

    false_negatives = golden_df[(gold) & (~pred)]  
    false_positives = golden_df[(~gold) & (pred)] 

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_negative_rate": len(false_negatives) / max(gold.sum(), 1),
        "false_negative_count": len(false_negatives),
        "false_negative_examples": false_negatives["text"].tolist()[:5],
        "false_positive_count": len(false_positives),
    }


def print_report(golden_df, pred_intent_col, pred_escalate_col, label=""):
    print(f"\n===== {label} =====")
    im = intent_metrics(golden_df, pred_intent_col)
    print(f"Intent accuracy: {im['accuracy']:.3f}")
    print(f"Intent macro F1: {im['macro avg']['f1-score']:.3f}")

    em = escalation_metrics(golden_df, pred_escalate_col)
    print(f"Escalation precision: {em['precision']:.3f}  recall: {em['recall']:.3f}  f1: {em['f1']:.3f}")
    print(f"Escalation FALSE NEGATIVES (missed, should've escalated): "
          f"{em['false_negative_count']} ({em['false_negative_rate']:.1%} of true-escalate cases)")
    if em["false_negative_examples"]:
        print("  examples:", em["false_negative_examples"])
    return {"intent": im, "escalation": em}