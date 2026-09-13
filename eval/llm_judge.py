import json
import sys
import time

from groq import Groq

sys.path.append(".")
import config
from groq_utils import call_with_retry

JUDGE_MODEL = config.GROQ_MODEL

JUDGE_PROMPT = """You are evaluating a draft customer-support reply for quality. Score each axis 1-5.
Be strict -- a 5 means genuinely excellent, no notes. Most decent replies should score 3-4.

Customer message: "{text}"
Intent: {intent}
Draft reply: "{reply}"
(Historical grounding examples the reply was supposed to draw on): {grounding}

Score on:
1. grounded (1-5): does it match how this brand has actually resolved similar issues, vs inventing policy?
2. safe (1-5): does it avoid over-promising refunds/guarantees it's not authorized to make?
3. actionable (1-5): does the customer clearly know what happens next?
4. tone (1-5): appropriate empathy and brand voice, not robotic or dismissive?

Respond ONLY with JSON, no preamble or markdown fences:
{{"grounded": <1-5>, "safe": <1-5>, "actionable": <1-5>, "tone": <1-5>,
  "overall": <1-5>, "justification": "<1-2 sentences>"}}"""


def get_client():
    return Groq(api_key=config.groq_api_key())


def _extract_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.removeprefix("json").strip()
    return raw


def judge_reply(client, text: str, intent: str, reply: str, grounding: list[dict]) -> dict:
    grounding_str = "; ".join(
        f'"{g["customer_text"][:80]}" -> "{g["brand_reply"][:80]}"' for g in grounding
    ) if grounding else "none retrieved"

    prompt = JUDGE_PROMPT.format(text=text, intent=intent, reply=reply, grounding=grounding_str)
    resp = call_with_retry(
        client.chat.completions.create,
        model=JUDGE_MODEL,
        max_tokens=1024,
        temperature=0,
        reasoning_effort="low",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    time.sleep(1.5)  # stay under free-tier per-minute token cap
    raw = _extract_json(resp.choices[0].message.content)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"grounded": None, "safe": None, "actionable": None, "tone": None,
                "overall": None, "justification": "PARSE_ERROR"}


if __name__ == "__main__":
    client = get_client()
    out = judge_reply(
        client,
        "my driver was speeding and it scared me",
        "trip_safety_incident",
        "We're so sorry to hear this. Please DM us your trip ID so our safety team can investigate right away.",
        [],
    )
    print(out)