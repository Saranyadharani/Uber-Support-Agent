# Uber Support Agent 

Take-home project for the Hiver SDE Intern assignment.
An AI support agent for **Uber_Support** (Twitter customer-support handle),
built on a subsample of the Kaggle "Customer Support on Twitter" dataset.

The agent: classifies an incoming customer message into one of 7 intents,
drafts a reply grounded in how Uber has historically resolved similar cases
(RAG over resolved threads), and decides auto-handle vs. escalate-to-human
with a stated, rule-based reason.

https://github.com/user-attachments/assets/63e23635-f4dc-481e-8de2-659d042a4faa

Architecture :

<img width="3596" height="1611" alt="image" src="https://github.com/user-attachments/assets/54483391-561a-4c57-a7b5-ce2b5d8746ae" />


## Why Uber

Picked Uber over an airline (my first instinct) mainly because the dataset has
good volume for it and there's a real, defensible line between what should be
auto-handled and what shouldn't — safety incidents always need a human, FAQ
questions don't. That distinction is what makes the escalation logic more than
a coin flip.

One thing I'll flag upfront rather than let you discover it: Uber's actual
historical replies on Twitter are pretty repetitive ("please DM your trip ID").
That limits how much the RAG grounding can really differentiate one reply from
another — worth knowing before reading too much into the "grounded" quality
score.

## How it works

**Data.** `data/01_reconstruct_threads.py` reads the raw `twcs.csv` and walks
the reply chains to rebuild full customer↔brand threads for Uber_Support. A
thread counts as "resolved" if it ends with a thank-you-style phrase from the
customer, or the brand had the last word — this is a heuristic, not verified,
and it's a real source of noise (more on this in the report).

**Intents.** I read about 120 real tweets by hand before touching any model and
landed on 7 categories: `trip_safety_incident`, `fare_dispute`, `lost_item`,
`driver_behavior_complaint`, `account_payment_issue`, `trip_cancellation_refund`,
`general_inquiry`. Full definitions are in `intents/taxonomy.py`.

**Classification.** Three approaches, so results are comparable:
- keyword/regex matching (`intents/classify_baseline_keyword.py`) — the floor
- embedding + nearest-neighbor vote against the taxonomy's own examples
  (`intents/classify_embedding_knn.py`) — a step up, no LLM calls
- the actual system: an LLM call with the full taxonomy and thread context
  (`intents/classify_llm.py`), which also pulls out safety/money/distress
  signals in the same call

**Escalation.** This is a rule table (`agent/escalation_policy.py`), not left
up to the LLM to decide on its own. The LLM extracts signals (mentions safety,
requests money, sounds distressed, etc.), and a fixed set of rules decides
escalate or not from those signals. I wanted this auditable — "why did it
escalate" should be answerable by reading code, not by re-prompting the model
and hoping it explains itself the same way twice.

**Reply generation.** `agent/reply_generator.py` embeds the incoming message,
retrieves the 3 most similar historically-resolved cases, and asks the LLM to
draft a reply matching Uber's actual tone/pattern from those examples — not
just "be a helpful support agent."

**Golden set.** 180 hand-labeled examples in `eval/golden_set.csv`. I started
out trying LLM-assisted pre-labeling (have the model suggest, I just
accept/override) but ran into repeated Groq rate limits mid-labeling, and more
importantly realized it created a circularity problem — if the same kind of
model pre-labels my ground truth and also runs my system, I'm not really
testing anything independent. So I switched to fully manual labeling, using
the local keyword classifier only to stratify sampling across intents (not as
a suggestion). See `labeling/` for the sampling script, the tool I used to
label (`label_tool.html`, a small keyboard-driven page), and the raw
input/output.

**Eval harness.** `eval/run_eval.py` runs all three classifiers plus the
system's escalation and reply generation against the golden set, and computes
intent accuracy/F1, escalation precision/recall/false-negative rate, and
LLM-judge reply quality scores. `eval/judge_agreement.py` checks how well that
judge agrees with me scoring the same replies by hand.

**Frontend.** `frontend/` is a small Flask backend plus a single-page chat UI
that calls the real pipeline — lets you type a message and
see the classification, escalation decision, and grounded reply live, with a
toggle to show/hide the underlying reasoning.


# Especially what makes my agent trustable

AI is only trustable only when it covers both transparency and accountability my agent acheives both by the following reason,
#Transparency - It gains the trust users and developers since the decision made by agent is explainable (based on what criteria does it has made the decision )
#Accountability - When something goes wrong, the organization search for the responsible person for the issue using my agent we can identify whether the agent is responsible or escalated human is responsible based upon who handled the customer issue .

