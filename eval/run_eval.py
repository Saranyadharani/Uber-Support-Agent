import ast
import json
import sys

import pandas as pd
from tqdm import tqdm

sys.path.append(".")
import config
from eval.metrics import print_report
from eval.llm_judge import get_client as get_judge_client, judge_reply
from intents.classify_baseline_keyword import classify as classify_keyword
from intents.classify_embedding_knn import classify as classify_knn
from intents.classify_llm import get_client, classify as classify_llm
from agent.escalation_policy import decide
from agent.reply_generator import generate_reply
from intents.taxonomy import AUTO_HANDLE_INTENTS_DEFAULT


# it escalate everything -- useless as a real policy, just shows what recall=1
def trivial_escalate_baseline(intent):
    return True


#it  escalate purely off intent, ignoring safety/money/distress signals entirely
def simple_escalate_baseline(intent):
    return intent not in AUTO_HANDLE_INTENTS_DEFAULT


def main():
    golden = pd.read_csv(config.GOLDEN_SET_CSV)
    client = get_client()
    judge_client = get_judge_client()

    rows = []
    judged_rows = []

    for _, r in tqdm(golden.iterrows(), total=len(golden), desc="Running system"):
        text = r["text"]
        raw_context = r.get("context")
        context = []
        if isinstance(raw_context, str) and raw_context.strip():
            try:
                context = json.loads(raw_context)
            except json.JSONDecodeError:
                try:
                    context = ast.literal_eval(raw_context)  # pandas writes lists w/ single quotes, not valid json
                except (ValueError, SyntaxError):
                    context = []

        kw_intent = classify_keyword(text)
        knn_intent = classify_knn(text)
        llm_out = classify_llm(client, text, context)
        llm_intent = llm_out["intent"]
        signals = llm_out.get("signals", {})

        esc_trivial = trivial_escalate_baseline(llm_intent)
        esc_simple = simple_escalate_baseline(llm_intent)
        esc_system = decide(llm_intent, signals)["escalate"]

        draft = generate_reply(client, text, llm_intent)
        judged = judge_reply(judge_client, text, llm_intent, draft["reply"], draft["grounding_examples"])

        rows.append({
            "text": text,
            "human_intent": r["human_intent"],
            "human_escalate": r["human_escalate"],
            "pred_intent_keyword": kw_intent,
            "pred_intent_knn": knn_intent,
            "pred_intent_llm": llm_intent,
            "pred_escalate_trivial": esc_trivial,
            "pred_escalate_simple": esc_simple,
            "pred_escalate_system": esc_system,
        })
        judged_rows.append({
            "text": text,
            "intent": llm_intent,
            "draft_reply": draft["reply"],
            **judged,
        })

    pred_df = pd.DataFrame(rows)
    judged_df = pd.DataFrame(judged_rows)
    pred_df.to_csv("eval/results/predictions.csv", index=False)
    judged_df.to_csv("eval/results/judged_replies.csv", index=False)

    summary = {}
    summary["keyword_baseline"] = print_report(pred_df, "pred_intent_keyword", "pred_escalate_trivial",
                                                 label="TRIVIAL baseline (keyword intent, escalate-everything)")
    summary["simple_baseline"] = print_report(pred_df, "pred_intent_knn", "pred_escalate_simple",
                                                label="SIMPLE baseline (kNN intent, intent-only escalation)")
    summary["system"] = print_report(pred_df, "pred_intent_llm", "pred_escalate_system",
                                       label="SYSTEM (LLM intent + rule-based escalation policy)")

    quality_cols = ["grounded", "safe", "actionable", "tone", "overall"]
    summary["reply_quality_means"] = judged_df[quality_cols].mean().to_dict()
    print("\n===== Reply quality (LLM judge means) =====")
    print(judged_df[quality_cols].mean())

    with open("eval/results/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\nWrote eval/results/summary.json, predictions.csv, judged_replies.csv")


if __name__ == "__main__":
    main()