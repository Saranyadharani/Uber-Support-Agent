# Report — Uber Support Agent

*(Fill in bracketed [NUMBERS] after running `eval/run_eval.py` and
`eval/judge_agreement.py` on your machine — everything else is drafted.)*

## 1. Problem framing

**What "good" means for this brand.** Uber support on Twitter is high-volume
and mostly low-stakes (account/FAQ questions), but contains a real minority
of safety-critical and money-critical messages. "Good" for this agent means:
(a) never silently auto-handle a safety incident or an explicit monetary
dispute — recall on escalation for those categories matters far more than
overall accuracy; (b) for auto-handled replies, be grounded in what Uber has
actually said before rather than inventing policy; (c) be honest about
uncertainty — an "I don't know, escalating" is strictly better than a
confident wrong answer for this brand.

**What I chose not to build.**
- No fine-tuning of a classifier — with ~200 golden examples and a
  well-specified taxonomy, prompted classification is more reliable and far
  faster to iterate on than training a model, and it's a fair comparison
  point against future work.
- No multi-turn conversation *state machine* — the agent scores each new
  customer message using thread context as input, not as a persistent
  session; a real deployment would need this, but it's orthogonal to proving
  classify/ground/escalate quality.
- No live paraphrase-detection of duplicate/spam tweets — the raw dataset has
  near-duplicate complaints; I didn't dedupe aggressively, which likely
  inflates apparent example diversity somewhat (see §4).
- No cost/latency optimization — every call goes through the same model
  (`claude-sonnet-4-6`) for classification, reply, and judging. A production
  system would likely use a cheaper/faster model for classification.

## 2. Results vs. baselines

| | Intent accuracy | Intent macro-F1 | Escalation precision | Escalation recall | Escalation FN rate |
|---|---|---|---|---|---|
| Trivial (keyword intent, escalate-everything) | [X] | [X] | [X] | 1.00 | 0.00 |
| Simple (kNN intent, intent-only escalation) | [X] | [X] | [X] | [X] | [X] |
| System (LLM intent + rule-based escalation) | [X] | [X] | [X] | [X] | [X] |

Reply quality (LLM judge, 1-5 mean): grounded=[X], safe=[X], actionable=[X], tone=[X], overall=[X]

Judge-human agreement (n=[X]): exact match [X]%, within ±1 point [X]%,
linear-weighted Cohen's kappa = [X]. [X] cases disagreed by ≥2 points —
see `eval/results/judged_replies.csv` cross-referenced with
`eval/results/human_scoring_template.csv` for the specific examples.

**Reading these numbers honestly:** the trivial "escalate everything"
baseline gets perfect escalation recall by construction and should NOT be
beaten on recall alone — the system needs to beat it on precision while
keeping recall high, or it isn't actually adding value over "when in doubt,
escalate."

## 3. Failure analysis — top 5 failure modes

*(Replace with your actual observed failures from `predictions.csv` /
`judged_replies.csv` — these are the categories to look for, based on the
taxonomy design, not fabricated examples.)*

1. **Sarcasm misread as a different intent.** e.g. "wow great job charging me
   for a ride I never took 👍" — keyword baseline and possibly the LLM
   classifier can miss that this is a fare_dispute, not a compliment.
   Hypothesis: sarcasm markers (emoji + positive words + negative context)
   aren't explicitly modeled.
2. **Multi-issue tweets get only the first-mentioned intent.** A tweet
   combining a lost item AND a rude driver complaint may only trigger one
   escalation rule. Hypothesis: single-label classification is a
   simplification that under-serves compound complaints — the
   `ambiguous_multi_issue` escalation rule is a stopgap, not a fix.
3. **"Resolved" thread noise in the RAG corpus.** Some threads tagged
   resolved (via the thank-you-phrase heuristic) are actually abandoned
   conversations where the customer just stopped replying. Hypothesis: this
   occasionally surfaces an unhelpful "example" reply to ground on.
4. **Repeat-unresolved-contact detection is thread-local only.** The
   `repeat_unresolved_contact` rule only looks within one thread; a customer
   who tweets a NEW thread about the same underlying issue (common on
   Twitter) won't trigger it. Hypothesis: under-escalates persistent
   complainers.
5. **Escalation reason wording can be generic.** The rule-based reasons
   ("The tweet is trip_safety_incident...") are consistent but not
   tailored to the specific tweet — a human agent reviewing the queue gets
   less context than a free-text explanation would give, trading
   auditability for specificity.

## 4. "What is misleading about my headline number?"

- **Golden set is not a random sample of live traffic.** It's stratified
  ~20/intent + oversampled edge cases, so headline accuracy is NOT the
  accuracy you'd see on the raw incoming stream, which skews heavily toward
  general_inquiry/account_payment_issue. A weighted accuracy by true
  intent-frequency would likely look different (probably higher, since the
  easy majority classes dominate real traffic).
- **The LLM classifier's "signals" (safety/money/distress) feed directly into
  the same model family used for the golden-set pre-labels.** Even though a
  human reviewed and could override every pre-label, there's residual risk
  that ambiguous cases got resolved in the direction the LLM already leaned,
  understating true error rate on hard cases. The override rate (see
  `labeling/export_golden_set.py` output) is the honest check on this — a low
  override rate could mean either "the LLM is good" or "the human
  rubber-stamped it."
- **"Resolved" thread heuristic biases the RAG corpus and evaluation
  alike.** Both the retrieval corpus and any downstream judgment of
  "grounded in how the brand has resolved" quietly depend on a noisy
  resolved/unresolved heuristic (thank-you phrase OR brand-had-last-word).
  Brand-had-last-word is a weak proxy — it's true for most threads
  regardless of actual resolution.
- **LLM-judge and LLM-classifier share a training lineage.** Structural
  blind spots the classifier has (certain phrasing, sarcasm, code-switching)
  may be blind spots the judge shares, inflating quality scores on exactly
  the inputs where the system is weakest. The human-agreement check
  (§2) is the mitigation, but n=[X] is a small sample.
- **Escalation false-negative rate is the number that matters most and is
  also the noisiest**, since safety/legal-triggering tweets are rare even in
  the oversampled edge-case bucket — a handful of missed cases move this
  rate a lot. Treat the point estimate with wide error bars.

## 5. What I'd do next with one more week

1. Expand the golden set to include intent-frequency-weighted sampling
   (in addition to the stratified set) to get an honest "expected live
   accuracy" number, not just a per-class one.
2. Replace the thank-you-phrase resolution heuristic with a small
   human-labeled sample used to validate/calibrate it, rather than trusting
   it directly as ground truth for the RAG corpus.
3. Add multi-label intent support for the ~[X]% of tweets flagged
   `multi_issue`, instead of forcing single-label classification.
4. Build a second, cheaper/faster model path for the classification step
   (most of the traffic is easy) and reserve the larger model for
   generation/judging — cost matters at Twitter-support scale.
5. Run a larger judge-agreement study (n=100+) stratified by intent, since
   the current n is likely too small to trust kappa per-intent.

## 6. Decision log

Plain list of non-obvious decisions and why:

1. **Chose Uber over an airline** despite airlines having richer historical
   resolution patterns, because Uber's safety-escalation boundary is just as
   real and the brand is well-represented in the dataset; explicitly flagged
   that Uber's replies are more templated, which limits how much retrieval
   grounding can differentiate replies (see Problem Framing).
2. **Hand-defined the intent taxonomy by open-coding ~120 tweets before
   running any model**, rather than asking an LLM to propose categories from
   scratch, to avoid the taxonomy itself being an artifact of the same model
   being evaluated.
3. **Escalation is rule-based on top of LLM-extracted signals, not decided
   by the LLM directly** — so the decision is auditable in code and doesn't
   silently drift between runs of the same prompt.
4. **Used LLM-suggested labels as a first pass for the golden set, human
   reviewed/overrode every one** — disclosed override rate as a transparency
   metric rather than hiding the assistance.
5. **"Resolved" is a heuristic (thank-you phrase OR brand-had-last-word),
   not verified ground truth** — accepted as a known limitation rather than
   manually verifying all threads, given time constraints; flagged
   explicitly in §4 rather than presented as clean data.
6. **Escalation false-negative rate is reported separately from F1**,
   because for this brand a missed safety escalation is categorically worse
   than an unnecessary one, and F1 alone would hide that asymmetry.
7. **Reply-quality gold notes only collected for a subset (~40) of the
   golden set, not all 200** — full free-text quality annotation for 200
   examples wasn't a good time tradeoff versus getting broader intent/escalation
   coverage; sized the subset to be enough for a meaningful judge-agreement
   check instead.
8. **Kept intent and escalation-signal extraction in ONE LLM call**
   (`classify_llm.py`) rather than two, to cut latency/cost, accepting that
   this couples the two tasks' error modes together.
9. **Still draft a reply even when escalating**, but marked
   `SUGGESTION_FOR_HUMAN_REVIEW` rather than `AUTO_SEND` — gives the human
   agent a starting point without the system claiming it as final.
10. **Randomized the labeling batch order** (not grouped by intent) to avoid
    anchoring/fatigue bias where nearby similar examples get rubber-stamped
    the same way.
11. **Used a local embedding model (MiniLM) for retrieval/kNN baseline
    instead of API embeddings** — no extra API cost or latency for what's a
    supporting, not headline, component.
12. **Trivial baseline is "escalate everything," not "escalate nothing"** —
    chose the safer trivial baseline on purpose, since "escalate nothing" is
    a strategy no real team would consider and comparing against it would be
    a strawman.
13. **Multi-issue detection is a boolean flag from the classifier, not a
    separate multi-label classification pass** — cheaper, but likely
    under-detects compound issues (see Failure Mode #2); flagged as
    future work rather than solved.
14. **Used Groq (Llama 3.3 70B) instead of a larger frontier model** for all
    three LLM roles (classify, generate, judge) — free tier and low latency
    made rapid iteration on the golden set realistic on a laptop; tradeoff is
    a smaller/weaker model than a frontier option, which likely lowers the
    ceiling on subtle cases (sarcasm, compound intents) relative to what a
    larger model would catch. Worth re-running the eval with a stronger model
    to see how much of the reported error rate is "hard problem" vs.
    "model capacity" — noted as a natural ablation for next steps.
15. **Did not use a separate, independent model family for the judge** —
    the judge uses the same Groq model as the classifier/generator by
    default (`JUDGE_MODEL = config.GROQ_MODEL` in `eval/llm_judge.py`, one
    line to change). This is a known source of correlated blind spots
    (see §4, "what's misleading") and is easy to fix by pointing
    `JUDGE_MODEL` at a different provider/model if judge independence
    turns out to matter after checking human-agreement numbers.
