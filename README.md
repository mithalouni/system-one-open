# System One, open — a Jev-style decision model you can run yourself

**State in, typed calibrated decisions out, one forward pass, no decoding.** An open replica of TypeSafe's [Jev](https://typesafe.ai) built on
Gemma 4 E2B (attention LoRA) and Gemma 3 270M, trained and served on [Modal](https://modal.com). MIT licensed.

| | Jev (published) | This replica (Gemma 4 E2B) |
|---|---|---|
| TypeSafe public eval, strict common subset (343 pairs) | **86.9%** | 76.7% |
| Same eval, stock Qwen 7B (jev-on-a-laptop study) | | 73.8% |
| 27-question launch-demo ticket, one call | 114 ms | 97 ms (H100) |
| 1,000 real emails, 4 decisions each | | 13.4 s · 74.6 emails/s · 95.4% spam accuracy |
| Demo task families (Doom, smart home, support, security, invoice, agent trace, catalog) | | 98.8% · ECE 0.003 |
| Held-out task types (never trained) | | 74.8% |
| Weights | closed, API only | open (merged safetensors on the Modal volume; HF upload pending) |
| Price | $0.042 / M input tokens | GPU seconds on your own L4/H100, or CPU |

Full report with baselines and caveats: https://claude.ai/artifact/8xJE6T4ovza9JwAFpMbQay

## Nine demos, all real recordings of the live model
| # | Page / script | Replica of | What is real |
|---|---|---|---|
| 1 | `/` | TypeSafe presets | 4 scenarios, 28 typed fields each, one call |
| 2 | `/race` | launch-video terminal race | the 27 launch questions, 97 ms on an H100 |
| 3 | `/emails` | @ryanvogel inbox classifier | 1,500 Enron emails, 74.6/s, 95.4% spam accuracy |
| 4 | `/viral` | @rileybrown viral-post analyzer | judged 500 ms after you stop typing |
| 5 | `games.py --game doom` | TypeSafe's Doom | ViZDoom (freedoom), model picks every action from labels + depth + game vars |
| 6 | `games.py --game mario` | @faadilhshaik's Mario | NES emulator, RAM-derived state, model picks controller input |
| 7 | `rec/agent.mjs` | @gregpr07 browser-use agent | real Google Flights, model picks element + operation each step |
| 8 | `/drive` | @Neel490 self-driving sim | lidar sectors, code reflexes, model makes the tactical call at ~3 Hz |
| 9 | `/home` | TypeSafe smart-home demo | typed commands -> device, action, ambiguity, confirmation |

Base URL: https://mithalouni--jev-serve-e2b-full-server-web.modal.run  (scales to zero; first request after idle takes ~60 s).
Videos in `media/` were recorded from these pages on one H100.


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
