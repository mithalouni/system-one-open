"""Upload the trained models from the Modal volume to Hugging Face.
  modal secret create hf-write HF_TOKEN=hf_xxx          # once
  modal run hf_upload.py --user <hf-username>
"""
import modal, os, json
from s1.modal_common import vol
app = modal.App("jev-hf-upload")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("huggingface_hub[hf_xet]").env({"HF_HUB_ENABLE_HF_TRANSFER": "0"})
CARD = """---
license: apache-2.0
base_model: {base}
tags: [decision-model, calibrated, structured-output, system-one, one-pass, gemma]
pipeline_tag: text-classification
---
# {name}: typed calibrated decisions in one forward pass (open Jev replica)

State in, typed decisions out. No text generation: the model is read at one answer slot per question, restricted to the
option-letter tokens, so a whole list of choice / score / yes-no questions is answered in a single forward pass with a
probability distribution and a confidence per answer.

* Base: `{base}` ({how})
* Trained on 92 public decision datasets + 7 rule-generated task families (intents, routing, moderation, NLI, relevance,
  QA, rubric scores, judging, tool selection, agent next-action, game states) with cross-entropy + Brier, then temperature-scaled.
* Results and honest baselines: https://github.com/mithalouni/system-one-open
* TypeSafe public eval, strict common subset: Jev 86.9% vs this model {ts}% (untrained Qwen 7B: 73.8%).

## Usage
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("{repo}"); m = AutoModelForCausalLM.from_pretrained("{repo}", dtype=torch.bfloat16).cuda().eval()
prompt = ("<state>\\nMy card was charged twice for the same purchase.\\n</state>\\n\\n"
          "Question (choice): Which department should handle this?\\nOptions:\\n(A) billing\\n(B) technical support\\n(C) sales\\nAnswer: (")
ids = tok(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    logits = m(**ids).logits[0, -1]
letters = [tok.encode(L, add_special_tokens=False)[0] for L in "ABC"]
print(torch.softmax(logits[letters].float() / {temp}, -1))   # P(billing), P(technical support), P(sales)
```
For several questions in one pass, append further `Question k (...): ... Answer k: (` blocks and read the logits at each `(`.
The `s1` package in the GitHub repo does this, plus option chunking beyond 52 options, `score` expected values and a FastAPI server.
"""


@app.function(image=image, volumes={"/vol": vol}, secrets=[modal.Secret.from_name("hf-write")], timeout=3 * 3600, cpu=4, memory=16384)
def upload(user: str, which: str):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"]); me = api.whoami()["name"]; user = user or me
    runs = {"e2b-full": ("system-one-e2b", "google/gemma-4-E2B-it", "LoRA r=64 on attention, merged", "76.7"), "full-270m": ("system-one-270m", "unsloth/gemma-3-270m-it", "full fine-tune", "42.6")}
    out = {}
    for run, (name, base, how, ts) in runs.items():
        if which != "all" and run != which:
            continue
        src = f"/vol/runs/{run}/model"; repo = f"{user}/{name}"
        temp = json.load(open(f"{src}/s1_config.json")).get("temperature", 1.0)
        api.create_repo(repo, repo_type="model", exist_ok=True)
        open(f"{src}/README.md", "w").write(CARD.format(name=name, base=base, how=how, ts=ts, repo=repo, temp=temp))
        api.upload_folder(folder_path=src, repo_id=repo, repo_type="model", commit_message=f"Upload {name} (open Jev replica)")
        out[run] = f"https://huggingface.co/{repo}"
    return out


@app.local_entrypoint()
def main(user: str = "", which: str = "all"):
    print(json.dumps(upload.remote(user, which), indent=1))
