# s1 — an open System-One decision model (Jev replica)

State in, typed calibrated decisions out, **one forward pass, no decoding**. Three primitives, same shape as
TypeSafe's Jev API: `choice` (distribution over options + confidence), `score` (distribution over ordered levels +
expected level), `noul` (P(true)). Up to 52 options in a single pass (A–Z, a–z are single tokens in Gemma's
vocabulary); larger option sets are scored in chunks with a "none of the above" slot and a final round.

Everything runs on Modal. Nothing is installed locally except the `modal` CLI.

## Layout
- `s1/schema.py`   prompt format (`<state>…</state>` + one `Answer k: (` slot per question), tokenisation, collate
- `s1/data_real.py` 92 public HF decision datasets (intents, routing, moderation, NLI, fact-check, relevance, QA,
  rubric scores, pairwise judging, tool selection, agent next-action); 23 are held-out (never trained)
- `s1/data_synth.py` rule-based generators for the demo families: doom game state, smart home, support triage,
  security alerts, AP invoice packets, agent-trace observability, catalog alignment (exact teacher labels)
- `s1/modal_common.py` shared Modal image/volume; `S1_GPU` picks the GPU (or `none` for CPU)
- `s1/doom_sim.py` closed-loop arena: run the model as a live policy vs teacher and random
- `s1/engine.py`   slot-logit scoring, `decide(state, questions)` API, option chunking, temperature
- `build_data.py`  builds the mixture (one CPU container per task) -> volume `/vol/data/v1`
- `typesafe_eval.py` rebuilds TypeSafe's public eval (20 cases / 372 reference pairs) from evals.typesafe.ai
- `train.py`       CE + Brier training, periodic in-task / held-out eval, temperature scaling, save
- `evaluate.py`    in-task, held-out, synthetic, TypeSafe head-to-head (strict common subset vs Opus/Sol/Jev), latency
- `serve.py`       one-origin FastAPI app: `GET /` interactive demo (4 preset scenarios, 28 fields each), `POST /decide` (Jev-style or schema-style JSON), `GET /info`
- `s1/demo_presets.py`, `s1/demo_page.py`  demo scenarios (adapted from jev-on-a-laptop, MIT) and the demo page

## Run order
```bash
modal run build_data.py                                   # ~5 min, CPU only
modal run typesafe_eval.py                                # seconds
S1_GPU=H100 modal run train.py --name lite-270m --model unsloth/gemma-3-270m-it --tier lite
S1_GPU=H100 modal run evaluate.py --run lite-270m
S1_GPU=H100 modal run train.py --name e2b-full --model google/gemma-4-E2B-it --tier full --lora 1
S1_GPU=H100 modal run evaluate.py --run e2b-full
S1_GPU=L4 S1_RUN=e2b-full modal deploy serve.py
./demo_client.sh https://<workspace>--jev-serve-e2b-full-server-decide.modal.run
```
`S1_GPU=none` runs any app on CPU (smoke tests).

## Live demo
https://mithalouni--jev-serve-e2b-full-server-web.modal.run/  (E2B on an L4; scales to zero after 2 min idle, first request after idle takes ~60 s)
