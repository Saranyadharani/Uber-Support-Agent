# Uber Customer Support Agent

An AI agent that reads an incoming customer support tweet, classifies what
it's actually about, decides whether it's safe to auto-handle or needs a
human, and drafts a reply grounded in how the brand has actually responded
to similar issues before — not a generic chatbot wrapper.

Built on a real dataset (Kaggle's "Customer Support on Twitter"), using
Uber's support handle as the test case.

https://github.com/user-attachments/assets/63e23635-f4dc-481e-8de2-659d042a4faa

**Architecture:**

<img width="3596" height="1611" alt="image" src="https://github.com/user-attachments/assets/54483391-561a-4c57-a7b5-ce2b5d8746ae" />

## What I set out to do

Most "AI support bot" demos stop at a working chat window. I wanted to prove
three specific things instead:

1. **The classification is grounded in real data, not vibes** — 7 intents
   defined by actually reading ~120 real tweets by hand, not asking an LLM
   to invent categories.
2. **Escalation is a real, auditable decision** — not just the LLM deciding
   on its own whim. A deterministic rule table reads signals the LLM
   extracts (safety mentioned, money requested, distress detected) and
   decides from there, so "why did this escalate" is answerable by reading
   code, not by re-prompting and hoping for a consistent answer.
3. **The system is actually evaluated, not just demoed** — a 180-example
   hand-labeled golden set, two baselines, an LLM-judge rubric with a
   human-agreement check, and a measured before/after on a real bug I found
   and fixed.

## Why Uber specifically

Picked it over an airline (my first instinct) because the dataset has solid
volume and there's a genuinely defensible line between auto-handle and
escalate — safety incidents always need a human, FAQ questions don't. That
distinction is what makes the escalation logic more than a coin flip.

Worth flagging upfront: Uber's actual historical replies on Twitter are
pretty repetitive ("please DM your trip ID"). That limits how much the
retrieval-based grounding can really differentiate one reply from another —
better to know this going in than discover it halfway through reading the
results.

## How it works

**Data pipeline.** `data/01_reconstruct_threads.py` reads the raw dataset
and walks reply chains to rebuild full customer↔brand conversation threads.
A thread counts as "resolved" if it ends with a thank-you-style phrase or
the brand had the last word — a heuristic, not verified ground truth, and a
real source of noise I call out explicitly rather than hide.

**Intent taxonomy.** 7 categories — `trip_safety_incident`, `fare_dispute`,
`lost_item`, `driver_behavior_complaint`, `account_payment_issue`,
`trip_cancellation_refund`, `general_inquiry` — defined by hand-reading real
tweets *before* running any model, so the categories aren't shaped by the
same system being evaluated. Full definitions in `intents/taxonomy.py`.

**Classification — three approaches, so results are comparable:**
- keyword/regex matching (`intents/classify_baseline_keyword.py`) — the floor
- embedding + nearest-neighbor vote against the taxonomy's own examples
  (`intents/classify_embedding_knn.py`) — no LLM calls
- the actual system: an LLM call with full taxonomy + thread context
  (`intents/classify_llm.py`), which also extracts safety/money/distress
  signals in the same call

**Escalation policy.** A fixed rule table (`agent/escalation_policy.py`)
reading LLM-extracted signals — deliberately not left up to the LLM to
decide freely, so the decision stays consistent and inspectable.

**Reply generation.** `agent/reply_generator.py` retrieves the 3 most
similar historically-resolved cases and asks the LLM to match the brand's
actual tone/pattern. Help-center links are mapped per intent to real,
current Uber help articles and appended deterministically in code — the
model never has to reproduce a URL character-for-character (it will
truncate long ones under a length constraint, which I found and fixed).

**Golden evaluation set.** 180 hand-labeled real examples
(`eval/golden_set.csv`). I initially tried LLM-assisted pre-labeling
(model suggests, I accept/override) but ran into repeated API rate limits
mid-labeling, and more importantly realized it created a circularity
problem — if the model pre-labeling my ground truth is the same kind of
model running my system, I'm not testing anything independent. Switched to
fully manual labeling, using only a local keyword classifier to stratify
sampling across intents (never as a label suggestion).

**Evaluation harness.** `eval/run_eval.py` runs all three classifiers plus
the system's escalation and reply generation against the golden set, and
reports intent accuracy/F1, escalation precision/recall/false-negative rate,
and LLM-judge reply quality scores. `eval/judge_agreement.py` checks how
well that judge agrees with me scoring the same replies by hand — including
honestly reporting a low agreement score and digging into *why* instead of
reframing it.

**Frontend.** `frontend/` — a Flask backend plus a single-page chat UI that
calls the real pipeline, not a mocked demo. Shows the classification,
escalation decision, and grounded reply live, with a toggle to show/hide the
underlying reasoning.

## What makes this trustworthy, not just functional

Two things, deliberately:

**Transparency** — every decision is explainable after the fact. The
classification, the escalation reason, and which historical cases a reply
was grounded on are all visible, not hidden inside a black box.

**Accountability** — because escalation is a rule-based decision reading
explicit signals (not an opaque LLM judgment call), it's possible to trace
back *why* something was auto-handled or escalated, and audit whether the
system or the human in the loop was responsible for a given outcome.

## Tech stack

- **LLM:** Groq API (`openai/gpt-oss-20b`) for classification, reply
  generation, and judging
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`), run locally
- **Backend:** Python, Flask
- **Data:** pandas, scikit-learn
- **Frontend:** vanilla HTML/CSS/JS

