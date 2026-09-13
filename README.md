# Uber Support Agent — Hiver SDE Intern Take-Home

An AI support agent for **Uber_Support** (Twitter customer-support handle),
built on a subsample of the Kaggle "Customer Support on Twitter" dataset.

The agent: classifies an incoming customer message into one of 7 intents,
drafts a reply grounded in how Uber has historically resolved similar cases
(RAG over resolved threads), and decides auto-handle vs. escalate-to-human
with a stated, rule-based reason.

Uses **Groq** (`llama-3.3-70b-versatile`) for all LLM calls — fast, has a
free tier, and OpenAI-compatible, so swapping models/providers later is a
one-line change in `config.py`.

## Why Uber, why these intents

See `intents/taxonomy.py` for the full taxonomy + escalation rules and
`REPORT.md` for the reasoning. Short version: Uber support has a genuine
safety-escalation boundary (driver incidents must always go to a human),
which makes the escalate/auto-handle decision defensible rather than
arbitrary — the same property that would've made airlines work too.

## Repo structure

```
config.py                          # brand, paths, model config — edit here first
data/
  01_reconstruct_threads.py        # twcs.csv -> per-brand threads (jsonl)
intents/
  taxonomy.py                      # the 7 intents + escalation rule table
  classify_baseline_keyword.py     # TRIVIAL baseline
  classify_embedding_knn.py        # SIMPLE baseline
  classify_llm.py                  # SYSTEM classifier
labeling/
  prepare_labeling_batch.py        # stratified sample + LLM pre-suggested labels
  label_tool.html                  # browser labeling UI (accept/override, keyboard-driven)
  export_golden_set.py             # labeling_output.json -> eval/golden_set.csv
retrieval/
  build_index.py                   # embeds resolved threads for RAG
  retrieve.py                      # query-time nearest-neighbor retrieval
agent/
  escalation_policy.py             # deterministic rule engine (not just LLM judgment)
  reply_generator.py               # RAG-grounded reply drafting
pipeline.py                        # ties classify -> escalate -> reply into one call
eval/
  metrics.py                       # accuracy/precision/recall/F1, escalation FN rate
  llm_judge.py                     # 4-axis reply quality rubric
  judge_agreement.py               # judge vs. human agreement (kappa + disagreement list)
  run_eval.py                      # runs everything against golden_set.csv
REPORT.md                          # problem framing, results, failure analysis, decision log
```

## Setup (~5 min)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...   # free key at https://console.groq.com/keys
```

Download `twcs.csv` from Kaggle (`thoughtvector/customer-support-on-twitter`)
and place it at `data/raw/twcs.csv`.

## Reproduce headline results (~15 min total, most of it is API calls)

```bash
# 1. Reconstruct Uber threads from the raw 3M-row dataset (subsampled, ~2 min)
python data/01_reconstruct_threads.py

# 2. Build the RAG index over historically resolved threads (~1 min, local embeddings)
python retrieval/build_index.py

# 3. (Already done for you — golden set is checked in at eval/golden_set.csv.)
#    To rebuild it from scratch instead:
#      python labeling/prepare_labeling_batch.py     # samples + LLM pre-labels ~200 examples
#      open labeling/label_tool.html in a browser, load labeling_batch.json, label, export
#      python labeling/export_golden_set.py

# 4. Run the full eval: two baselines + system, intent + escalation + reply quality (~8-10 min)
python eval/run_eval.py

# 5. (Optional, already summarized in REPORT.md) Judge-human agreement check:
python eval/judge_agreement.py template     # samples 40 replies for you to hand-score
# ... fill in eval/results/human_scoring_template.csv by hand ...
python eval/judge_agreement.py
```

Outputs land in `eval/results/`: `predictions.csv`, `judged_replies.csv`, `summary.json`.

## Try a single message interactively

```bash
python pipeline.py
```

## What this does NOT do (see REPORT.md for the full list)

- No fine-tuning — everything is prompted/retrieval-based, on purpose (see
  Problem Framing in REPORT.md).
- No multi-turn dialogue management — each customer message is scored
  independently using thread context, not a stateful conversation manager.
- No live Twitter/production integration — this is an offline pipeline
  against the historical dataset.

## Citations / borrowed work

- Dataset: Kaggle `thoughtvector/customer-support-on-twitter`.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (local, no API cost).
- LLM: Groq API (`llama-3.3-70b-versatile` by default) for classification, reply generation, and judging.
  Groq's chat completions endpoint is OpenAI-compatible; the `groq` Python SDK is a thin wrapper.
- No borrowed code beyond standard library usage of pandas/sklearn/sentence-transformers.
