# Uber Support Agent 

An AI support agent for **Uber_Support** (Twitter customer-support handle),
built on a subsample of the Kaggle "Customer Support on Twitter" dataset.

The agent: classifies an incoming customer message into one of 7 intents,
drafts a reply grounded in how Uber has historically resolved similar cases
(RAG over resolved threads), and decides auto-handle vs. escalate-to-human
with a stated, rule-based reason.


