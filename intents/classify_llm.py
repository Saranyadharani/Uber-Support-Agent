import json
import sys

from groq import Groq

from groq_utils import call_with_retry

sys.path.append(".")
import config
from intents.taxonomy import INTENTS, INTENT_NAMES

# the real classifier -- full taxonomy + thread context in the prompt, plus
# escalation signals in the same call so we're not paying for two round trips
CLASSIFY_PROMPT = """Classify this customer support tweet into exactly one intent.
Brand: {brand}

Intent taxonomy:
{taxonomy}

Thread context (prior turns, oldest first): {context}
Customer's message: "{text}"

For high_distress_or_abusive: mark true not just for profanity or explicit
threats, but also for short, curt, sarcastic, or clearly exasperated messages
(e.g. "what a joke", "0 help", ALL CAPS pleas, "still waiting", complaints
about repeated unhelpful responses) -- a customer who sounds like they've
given up on getting help should escalate even without swearing.

Respond ONLY with JSON, no preamble or markdown fences:
{{
  "intent": "<one of the intent names>",
  "confidence": <0-1 float>,
  "signals": {{
    "mentions_safety": true|false,
    "mentions_legal_or_regulatory": true|false,
    "requests_money": true|false,
    "high_distress_or_abusive": true|false,
    "multi_issue": true|false
  }}
}}"""


def get_client():
    return Groq(api_key=config.groq_api_key())


def _extract_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.removeprefix("json").strip()
    return raw


def classify(client, text: str, context: list[str] | None = None) -> dict:
    context = context or []
    taxonomy_str = "\n".join(f"- {k}: {v['description']}" for k, v in INTENTS.items())
    prompt = CLASSIFY_PROMPT.format(
        brand=config.BRAND_HANDLE,
        taxonomy=taxonomy_str,
        context=json.dumps(context[-4:]),  # last few turns is enough context
        text=text,
    )
    resp = call_with_retry(
        client.chat.completions.create,
        model=config.GROQ_MODEL,
        max_tokens=1024,
        temperature=0,
        reasoning_effort="low",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    raw = _extract_json(resp.choices[0].message.content)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"intent": "general_inquiry", "confidence": 0.0,
                   "signals": {}, "parse_error": True}
    if result.get("intent") not in INTENT_NAMES:
        result["intent"] = "general_inquiry"  # bad response, fall back safe
    return result


if __name__ == "__main__":
    client = get_client()
    tests = [
        "my driver was speeding and it scared me, I want this reported",
        "why was I charged $50 for a 5 min ride",
    ]
    for t in tests:
        print(t, "->", classify(client, t))