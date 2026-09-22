# Report — Uber Support Agent

## 1. Problem framing

For Uber support on Twitter, most incoming messages are low-stakes — account
questions, FAQs — but a real minority are genuinely urgent: safety incidents,
explicit refund disputes. "Good" for this agent means getting the stakes right
more than getting every reply perfect. Concretely:

- Never quietly auto-handle a safety incident or a specific monetary dispute.
  Recall on escalation for those categories matters more than overall accuracy.
- When auto-handling, ground the reply in what Uber has actually said before,
  not an invented policy.
- Prefer an honest "I don't know, escalating" over a confident wrong answer.

**What I didn't build, on purpose:**
- No fine-tuned classifier. With 180 golden examples and a taxonomy I wrote by
  hand, prompted classification was faster to iterate on and gave a fairer
  comparison against the baselines.
- No real conversation state machine. The agent scores each new message using
  thread context as input, not as a persistent session — a real deployment
  would need this, but it's a separate problem from proving the core pipeline
  works.
- No dedup of near-identical spam/duplicate tweets in the raw data. I didn't
  clean this aggressively, which probably inflates how "diverse" the dataset
  looks.
- No cost/latency tuning. Every call — classify, generate, judge — goes
  through the same model. A production system would almost certainly use a
  cheaper model for the high-volume classification step.

## 2. Results vs. baselines

All numbers are from the full 180-item golden set (`eval/results/summary.json`).

**Intent classification**

| | Accuracy | Macro F1 |
|---|---|---|
| Trivial (keyword rules) | 58.9% | 0.592 |
| Simple (kNN over taxonomy examples) | 36.1% | 0.355 |
| System (LLM) | **71.7%** | **0.698** |

The kNN baseline being worse than plain keyword matching surprised me at
first, but it makes sense — embedding similarity against a handful of
hand-picked examples per intent is a weaker signal than explicit keyword
rules when the categories overlap in vocabulary (e.g. "charged" shows up in
both fare_dispute and trip_cancellation_refund examples).

**Escalation decision**

| | Precision | Recall | F1 | Missed (false negatives) |
|---|---|---|---|---|
| Trivial (escalate everything) | 0.633 | 1.000 | 0.776 | 0 |
| Simple (intent-only) | 0.800 | 0.667 | 0.727 | 38 (33.3%) |
| System (rule-based) | **0.944** | 0.746 | **0.833** | 29 (25.4%) |

The trivial baseline gets perfect recall by construction — that's not a real
strategy, it's a reminder that recall alone isn't the bar. The system clearly
wins on precision, but a 25.4% false-negative rate on escalation is a real
weakness, not something to gloss over. See failure analysis below.

**Reply quality** (LLM judge, 1–5 scale, means over all system replies)

| grounded | safe | actionable | tone | overall |
|---|---|---|---|---|
| 3.55 | 4.82 | 4.01 | 3.58 | 3.67 |

Safety is high — the model rarely over-promises anything. Groundedness is the
weakest axis, which lines up with what I flagged early on: Uber's real replies
are fairly templated, so there's less for the retrieval step to meaningfully
differentiate on.

## 3. Judge-human agreement

I hand-scored 40 of the system's drafted replies blind (without looking at the
judge's score first), then compared.

- Exact match: 32.5%
- Within ±1 point: 82.5%
- Linear-weighted Cohen's kappa: **-0.009**

That kappa number looks bad, and I want to be straightforward about it rather
than bury it. Two things are going on. First, both my scores and the judge's
cluster heavily in the 3–5 range, and kappa is known to behave badly (including
going negative) with compressed, low-variance distributions like this — a
handful of disagreements have outsized effect. Second, and more interesting,
looking at the 7 cases with a 2+ point gap showed a real pattern: I gave
higher scores to replies that were generic but functionally correct (e.g.
"DM us your trip ID" for a straightforward lost-item request), while the judge
penalized genericness more heavily even when the reply correctly solved the
problem. We seem to be measuring somewhat different things — I was rewarding
"does this work," the judge was rewarding "does this feel specific to what was
said."

I'd treat the judge's absolute scores as directional, not validated against
human judgment, at this sample size. A larger re-scoring pass with a tighter,
more explicit rubric (maybe splitting "correctness" from "specificity" into
separate axes) would be needed before trusting this as ground truth.

## 4. Failure analysis

**1. Escalation misses skew toward passive/sarcastic frustration, not overt
anger.** Looking at the 29 missed escalations, most weren't profanity-laden —
things like "what a joke," "0 help," "CAN YOU HELP ME," "ZERO customer
service!" The original distress signal in the prompt was tuned toward
explicit anger/threats and missed curt, exhausted, or sarcastic phrasing.

I tested a fix for this: broadened the distress-signal instruction in the
classifier prompt to explicitly call out short/curt/sarcastic/exasperated
phrasing, not just profanity. Re-running the same 29 previously-missed cases
through the updated prompt (`eval/retest_escalation_fixes.py`), 22 of 29
(75.9%) now correctly escalate. Worth noting this is a targeted re-test on
the known failure set, not a clean full re-run of all 180 — a full re-run
would be needed to confirm the fix doesn't introduce new false positives
elsewhere, and I didn't have time/budget left to do that cleanly. Still, it's
a real, measured improvement, not just a guess.

**2. `fare_dispute` vs. `trip_cancellation_refund` genuinely overlap.**
"I got charged a cancellation fee I disagree with" fits both categories
reasonably. This shows up repeatedly in the confusion between these two
intents. Partly a taxonomy design issue rather than a pure model error — a
production system might merge these or allow multi-label.

**3. Multi-issue tweets only get one label.** A tweet complaining about a
missing delivery item *and* rude courier behavior in one message only gets
classified under one intent. The taxonomy has an `ambiguous_multi_issue`
escalation trigger as a stopgap, but it's not a real fix for the underlying
single-label limitation.

**4. The resolved-thread heuristic adds noise to the RAG corpus.** "Resolved"
is currently just "thank-you phrase OR brand had the last word" — the second
condition is weak, since the brand usually does have the last word regardless
of whether the issue actually got fixed. Some retrieved "example resolutions"
are probably threads the customer just gave up on.

**5. Single-turn classification loses context for follow-up messages.**
Some tweets in the raw data are just "I just sent a DM over" or similar —
clearly a follow-up to an earlier, unlabeled conversation. Classified as
`general_inquiry` by default, but the real intent is unknowable from the text
alone. This is a structural limitation of scoring one message at a time.

## 5. What's misleading about the headline numbers

- The golden set is stratified (~20-25 per intent), not sampled proportional
  to real traffic. Actual live traffic almost certainly skews toward
  `general_inquiry` and `account_payment_issue`, which are the easier
  categories — real accuracy on live traffic is probably higher than 71.7%,
  but I don't have a way to confirm that without a proportionally-sampled set.
- The LLM classifier and the LLM judge share a model family (both run on
  `openai/gpt-oss-20b`). Any blind spot the model has — certain phrasing,
  sarcasm, specific domains — is a blind spot both share, which could make the
  judge look more favorable toward the system's own outputs than an
  independent judge would.
- The escalation false-negative rate (25.4%) sounds precise but is based on a
  small number of actual true-positive cases in a 180-item set — a handful of
  edge cases moving one way or the other shifts this rate noticeably. Treat it
  as a rough estimate, not a stable measurement.
- The 75.9% "fixed" rate on the escalation prompt improvement is measured only
  on the known failure set, not validated against a full clean re-run — it's
  real evidence of improvement, but not a confirmed final number.
- "Resolved" threads used for grounding are heuristically labeled, not
  verified. Some fraction of the RAG corpus is probably noise, and I don't
  have a clean way to quantify how much.

## 6. What I'd do next with a week

1. Re-run the full 180-item eval with the improved escalation prompt to get a
   real, not just targeted, before/after number.
2. Build a second golden subset sampled proportional to actual traffic
   frequency, to get an honest "expected live accuracy" figure alongside the
   stratified one.
3. Swap the judge to a different model family than the classifier/generator,
   and re-run the human-agreement check to see if that changes the picture.
4. Hand-verify a sample of "resolved" threads to calibrate how noisy that
   heuristic actually is, rather than just flagging it as a known issue.
5. Support multi-label intent classification for the subset of tweets that
   clearly bundle more than one issue.

## 7. Decision log

- Picked Uber over an airline (my original plan) — similar volume and a
  similarly real safety-escalation boundary, though I noted upfront that
  Uber's actual historical replies are more repetitive, which limits how much
  retrieval grounding can differentiate.
- Defined the 7 intents by reading ~120 real tweets by hand before running any
  model, so the taxonomy wasn't shaped by the same model being evaluated.
- Escalation is decided by a fixed rule table reading LLM-extracted signals,
  not by asking the LLM to decide escalate/auto-handle directly — wanted this
  auditable and consistent across runs.
- Started with LLM-assisted pre-labeling for the golden set, switched to fully
  manual labeling partway through — partly because of repeated Groq rate
  limits mid-run, but more importantly because pre-labeling with the same kind
  of model being evaluated creates a circularity problem for an "independent"
  ground truth set.
- Used the local keyword classifier only to stratify sampling for the golden
  set, never as a label suggestion.
- "Resolved" thread status is a heuristic (thank-you phrase or brand-had-last-
  word), not verified — accepted as a known limitation given time, flagged
  explicitly rather than treated as clean ground truth.
- Reported the escalation false-negative rate as its own separate metric
  instead of folding it into F1, since a missed safety escalation is
  categorically worse than an unnecessary one and F1 alone hides that.
- Only collected reply-quality gold notes for a subset (~40) of the golden
  set rather than all 180 — sized for a meaningful judge-agreement check, not
  full coverage, given time constraints.
- Kept intent classification and escalation-signal extraction in one LLM call
  rather than two separate calls, to cut cost/latency, accepting that this
  couples their error modes together.
- Still generate a draft reply even when escalating, but mark it clearly as
  `SUGGESTION_FOR_HUMAN_REVIEW` rather than auto-sendable — gives a human
  agent a starting point without the system claiming it as final.
- Randomized the labeling batch order rather than grouping by intent, to avoid
  anchoring/fatigue bias where visually similar nearby examples get
  rubber-stamped the same way.
- Chose the trivial baseline as "escalate everything" rather than "escalate
  nothing" — the safer of the two trivial strategies, since "escalate
  nothing" isn't something any real team would seriously consider.
- Switched Groq models twice over the course of the project —
  `llama-3.3-70b-versatile` got deprecated mid-project, and
  `openai/gpt-oss-120b` ran out of daily token budget during the eval run —
  landed on `openai/gpt-oss-20b`, a separate model with its own budget. Added
  retry/backoff and checkpointing to `run_eval.py` and the labeling script
  after losing partial progress to a crash once, so a rate limit or network
  blip doesn't cost re-running everything from scratch.
- Reported the judge-human kappa (-0.009) honestly rather than reframing it,
  and dug into *why* it was low (score compression + a real scoring-philosophy
  difference) instead of just citing the number.
