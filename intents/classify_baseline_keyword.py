import re
import sys
sys.path.append(".")
from intents.taxonomy import INTENT_NAMES

# floor baseline -- if the LLM classifier can't beat this by a wide margin,
# something's wrong. order matters, first match wins.
KEYWORD_RULES = [
    ("trip_safety_incident", r"\b(unsafe|accident|crash|assault|harass|scared|dangerous|reckless|speeding)\b"),
    ("fare_dispute", r"\b(overcharg|surge|cancellation fee|wrong (amount|fare)|billed|charged)\b"),
    ("lost_item", r"\b(left my|lost my|forgot my|missing (item|phone|bag|wallet))\b"),
    ("driver_behavior_complaint", r"\b(rude|unprofessional|bad attitude|cancelled on me|refused to)\b"),
    ("account_payment_issue", r"\b(login|log in|password|promo code|payment method|app (crash|bug|error))\b"),
    ("trip_cancellation_refund", r"\b(cancel my (ride|trip)|refund|no.?show|never (showed|arrived))\b"),
]


def classify(text: str) -> str:
    t = text.lower()
    for intent, pattern in KEYWORD_RULES:
        if re.search(pattern, t):
            return intent
    return "general_inquiry"  # nothing no keyword matches , it classify it has general inquiry


if __name__ == "__main__":
    tests = [
        "my driver was speeding and it scared me",
        "why was I charged $50 for a 5 min ride",
        "how do I add a promo code",
    ]
    for t in tests:
        print(t, "->", classify(t))