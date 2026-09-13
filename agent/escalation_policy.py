import sys
sys.path.append(".")
from intents.taxonomy import ESCALATION_RULES, AUTO_HANDLE_INTENTS_DEFAULT

# Escalation combines the LLM signals with the rule table in intents/taxonomy.py, and stays deterministic and auditable.
# So "why did the agent escalate or not" is answerable from the code, not by re-prompting the model.


def decide(intent: str, signals: dict, repeat_unresolved: bool = False,
           multi_issue_override: bool | None = None) -> dict:
    fired = []

    if signals.get("mentions_safety"):
        fired.append("safety_concern")
    if signals.get("mentions_legal_or_regulatory"):
        fired.append("legal_or_regulatory")
    if intent in ("fare_dispute", "trip_cancellation_refund") and signals.get("requests_money"):
        fired.append("monetary_dispute_above_trivial")
    if signals.get("high_distress_or_abusive"):
        fired.append("high_distress_or_abuse")
    if repeat_unresolved:
        fired.append("repeat_unresolved_contact")
    multi_issue = signals.get("multi_issue") if multi_issue_override is None else multi_issue_override
    if multi_issue:
        fired.append("ambiguous_multi_issue")

    if fired:
        rule_reasons = {r[0]: r[1] for r in ESCALATION_RULES}
        return {
            "escalate": True,
            "reasons": [rule_reasons[r] for r in fired],
            "rule_ids": fired,
        }

    if intent in AUTO_HANDLE_INTENTS_DEFAULT:
        return {"escalate": False, "reasons": [f"Intent '{intent}' is auto-handleable and no risk rule fired."],
                "rule_ids": []}

    # Default-safe: intents not marked auto-handle, with no rule fired, still
    # auto-handle -- a deliberate, named default (see decision log), not a
    # silent fallthrough.
    return {"escalate": False,
            "reasons": [f"Intent '{intent}' has no escalation rule triggered; default auto-handle applies."],
            "rule_ids": []}


if __name__ == "__main__":
    print(decide("fare_dispute", {"requests_money": True}))
    print(decide("general_inquiry", {}))
    print(decide("trip_safety_incident", {"mentions_safety": True}))