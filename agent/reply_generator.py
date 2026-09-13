import sys
import time

from groq import Groq

sys.path.append(".")
import config
from groq_utils import call_with_retry
from retrieval.retrieve import retrieve_similar

REPLY_PROMPT = """You are drafting a reply as the official {brand} support account, replying to a
customer tweet. Match the brand's real historical tone and resolution pattern shown below.
Keep it under 280 characters, be specific and actionable, and do not promise a refund amount
or compensation you are not authorized to commit to -- if unsure, direct them to DM with details.

Intent: {intent}
Customer's message: "{text}"

Similar past cases and how {brand} actually replied (most similar first):
{examples}

Write ONLY the reply text, nothing else -- no quotes, no preamble.
"""


def get_client():
    return Groq(api_key=config.groq_api_key())


def format_examples(examples: list[dict]) -> str:
    if not examples:
        return "(no similar historical cases found)"
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f'{i}. Customer: "{ex["customer_text"][:150]}"\n   {config.BRAND_HANDLE} replied: "{ex["brand_reply"][:150]}"')
    return "\n".join(lines)


def generate_reply(client, text: str, intent: str, k: int = 3) -> dict:
    examples = retrieve_similar(text, k=k)
    prompt = REPLY_PROMPT.format(
        brand=config.BRAND_HANDLE,
        intent=intent,
        text=text,
        examples=format_examples(examples),
    )
    resp = call_with_retry(
        client.chat.completions.create,
        model=config.GROQ_MODEL,
        max_tokens=600,
        temperature=0.4,
        reasoning_effort="low",
        messages=[{"role": "user", "content": prompt}],
    )
    time.sleep(1.5)  # pace calls to stay under the free-tier 8K TPM cap
    reply_text = resp.choices[0].message.content.strip().strip('"')
    return {
        "reply": reply_text,
        "grounded_on": [e["thread_id"] for e in examples],
        "grounding_examples": examples,
    }


if __name__ == "__main__":
    client = get_client()
    out = generate_reply(client, "my driver was speeding and it scared me", "trip_safety_incident")
    print(out["reply"])
    print("Grounded on threads:", out["grounded_on"])