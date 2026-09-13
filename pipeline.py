import sys

sys.path.append(".")
import config
from intents.classify_llm import get_client, classify
from agent.escalation_policy import decide
from agent.reply_generator import generate_reply


# this functiotakes a raw customer message, classifies it, decides escalate or not, drafts a reply
def run_agent(text: str, context: list[str] | None = None, repeat_unresolved: bool = False,
              client=None) -> dict:
    client = client or get_client()
    context = context or []

    cls = classify(client, text, context)
    intent = cls["intent"]
    signals = cls.get("signals", {})

    esc = decide(intent, signals, repeat_unresolved=repeat_unresolved)

    result = {
        "text": text,
        "intent": intent,
        "confidence": cls.get("confidence"),
        "signals": signals,
        "escalate": esc["escalate"],
        "escalation_reasons": esc["reasons"],
    }

    # it still draft a reply either way and if escalating, human still gets a starting point
    draft = generate_reply(client, text, intent)
    result["draft_reply"] = draft["reply"]
    result["draft_reply_status"] = "SUGGESTION_FOR_HUMAN_REVIEW" if esc["escalate"] else "AUTO_SEND"
    result["grounded_on"] = draft["grounded_on"]

    return result


# Used for a quick manual tests.
if __name__ == "__main__":
    examples = [
        "my driver was speeding and it scared me, I want this reported",
        "why was I charged $50 for a 5 min ride, that's insane",
        "how does surge pricing actually work?",
    ]
    client = get_client()
    for e in examples:
        out = run_agent(e, client=client)
        print(f"TEXT: {e}")
        print(f"INTENT: {out['intent']}  ESCALATE: {out['escalate']}")
        print(f"REASONS: {out['escalation_reasons']}")
        print(f"REPLY ({out['draft_reply_status']}): {out['draft_reply']}")