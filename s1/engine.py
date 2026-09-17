"""Inference engine: state + typed questions -> calibrated distributions, all questions in one forward pass.

Backbone-agnostic (Gemma 3 text, Gemma 4 multimodal-with-text-only, or any HF causal LM): we take the final hidden
state at each answer slot and project it with the LM-head rows of the option-letter tokens. Nothing is decoded.
"""
from __future__ import annotations
import json, math, os, time
from .schema import Example, Q, LETTERS, MAX_OPTIONS, NONE_OPT, build, collate, letter_ids


def find_backbone(lm):
    inner = getattr(lm, "model", None)
    if inner is None:
        return lm
    return getattr(inner, "language_model", inner)


class DecisionModel:
    """Thin wrapper used by both training and inference."""

    def __init__(self, name, dtype=None, device=None, grad_ckpt=False, lora=None, lora_path=None, attn=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)
        self.tok = AutoTokenizer.from_pretrained(name)
        kw = dict(dtype=dtype)
        if attn:
            kw["attn_implementation"] = attn
        self.lm = AutoModelForCausalLM.from_pretrained(name, **kw)
        self.backbone = find_backbone(self.lm)
        cfg = self.lm.config; tc = getattr(cfg, "text_config", cfg)
        self.softcap = getattr(tc, "final_logit_softcapping", None)
        if lora_path:
            from peft import PeftModel
            self.lm = PeftModel.from_pretrained(self.lm, lora_path)
            self.backbone = find_backbone(self.lm.base_model.model)
        elif lora:
            from peft import LoraConfig, get_peft_model
            self.lm = get_peft_model(self.lm, LoraConfig(**lora))
            self.backbone = find_backbone(self.lm.base_model.model)
        if grad_ckpt:
            self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(self.lm, "enable_input_require_grads"):
                self.lm.enable_input_require_grads()
        self.lm.to(self.device)
        self.letters = torch.tensor(letter_ids(self.tok), device=self.device)
        self.temperature = 1.0
        cfg_path = os.path.join(name, "s1_config.json") if os.path.isdir(name) else None
        if cfg_path and os.path.exists(cfg_path):
            self.temperature = float(json.load(open(cfg_path)).get("temperature", 1.0))

    def lm_head_weight(self):
        lm = self.lm
        if hasattr(lm, "base_model") and hasattr(lm.base_model, "model"):
            lm = lm.base_model.model
        return lm.lm_head.weight

    def slot_logits(self, input_ids, attention_mask, slot_idx, slot_batch, nopts):
        import torch, torch.nn.functional as F
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        hs = h[slot_batch, slot_idx]
        W = self.lm_head_weight()[self.letters]
        logits = F.linear(hs.to(W.dtype), W).float()
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        ar = torch.arange(MAX_OPTIONS, device=logits.device)[None, :]
        return logits.masked_fill(ar >= nopts[:, None], float("-inf"))

    def forward_batch(self, batch):
        b = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in batch.items()}
        return self.slot_logits(b["input_ids"], b["attention_mask"], b["slot_idx"], b["slot_batch"], b["nopts"])

    def trainable_parameters(self):
        return [p for p in self.lm.parameters() if p.requires_grad]

    def save(self, path, merge_lora=True):
        os.makedirs(path, exist_ok=True)
        lm = self.lm
        if hasattr(lm, "merge_and_unload") and merge_lora:
            lm = lm.merge_and_unload()
        lm.save_pretrained(path, safe_serialization=True)
        self.tok.save_pretrained(path)
        json.dump({"temperature": self.temperature}, open(os.path.join(path, "s1_config.json"), "w"))


# ----------------------------------------------------------------------------------- scoring examples (eval)
def score_examples(model, examples, max_state_tokens=2048, max_tokens=24000, max_rows=32, temperature=None, progress=False):
    """Returns list (per example) of list (per question) of prob arrays in ORIGINAL option order (None if skipped)."""
    import torch, numpy as np
    T = temperature if temperature is not None else model.temperature
    items = [build(e, model.tok, max_state_tokens=max_state_tokens, train=False) for e in examples]
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    results = [None] * len(items)
    i = 0; t0 = time.time()
    with torch.no_grad():
        while i < len(order):
            L = len(items[order[i]]["ids"]); Tpad = ((L + 63) // 64) * 64
            B = max(1, min(max_rows, max_tokens // Tpad))
            # rows sorted by length: take next B rows, but bound T by the longest in chunk
            chunk = order[i:i + B]
            Lmax = max(len(items[j]["ids"]) for j in chunk); Tpad = ((Lmax + 63) // 64) * 64
            B = max(1, min(len(chunk), max_tokens // Tpad)); chunk = chunk[:B]
            batch = collate([items[j] for j in chunk], model.tok.pad_token_id)
            logits = model.forward_batch(batch) / T
            probs = torch.softmax(logits, -1).cpu().numpy()
            k = 0
            for j in chunk:
                it = items[j]; per_q = []
                for qi in range(len(it["slots"])):
                    p = probs[k]; k += 1
                    perm = it["perms"][qi]; full = np.zeros(len(examples[j].qs[qi].options))
                    for pos, oi in enumerate(perm):
                        full[oi] = p[pos]
                    per_q.append(full)
                results[j] = per_q
            i += B
            if progress and (i // B) % 20 == 0:
                print(f"  scored {i}/{len(order)} ({time.time() - t0:.0f}s)", flush=True)
    return results


# ----------------------------------------------------------------------------------- API-style decisions
def _api_to_examples(state, questions, max_q_per_pass=24, chunk_opts=40):
    """Convert Jev-style question dicts to Examples (possibly several passes), tracking how to reassemble."""
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False, indent=1)
    qs = []; plan = []          # plan: per api question -> list of (example_idx, q_idx, option_subset or None)
    for qi, q in enumerate(questions):
        kind = q.get("type", "choice")
        text = q.get("instructions") or q.get("question") or q.get("text") or ""
        if kind == "noul":
            crit = q.get("criteria") or {}
            descs = [crit.get("false"), crit.get("true")] if isinstance(crit, dict) and any(crit.get(k) for k in ("true", "false")) else None
            qs.append((qi, Q(text, ["no", "yes"], -1, kind="noul", descs=descs), None))
        elif kind == "score":
            levels = q.get("levels") or q.get("criteria") or []
            if isinstance(levels, dict):
                keys = list(levels.keys()); descs = [levels[k] for k in keys]
            else:
                keys = [str(i) for i in range(len(levels))]; descs = list(levels)
            qs.append((qi, Q(text, keys, -1, kind="score", descs=descs), None))
        else:
            opts = q.get("options") or q.get("criteria") or []
            if isinstance(opts, dict):
                keys = list(opts.keys()); descs = [opts[k] for k in keys]
            else:
                keys = [str(o) for o in opts]; descs = None
            if len(keys) <= MAX_OPTIONS:
                qs.append((qi, Q(text, keys, -1, kind="choice", descs=descs), None))
            else:
                for c in range(0, len(keys), chunk_opts):
                    sub = list(range(c, min(len(keys), c + chunk_opts)))
                    o = [keys[s] for s in sub] + [NONE_OPT]
                    d = ([descs[s] for s in sub] + ["none of the options in this list applies"]) if descs else None
                    qs.append((qi, Q(text, o, -1, kind="choice", descs=d), sub))
    examples = []
    for c in range(0, len(qs), max_q_per_pass):
        group = qs[c:c + max_q_per_pass]
        examples.append(Example(state, [g[1] for g in group]))
        for k, (qi, q, sub) in enumerate(group):
            plan.append((qi, len(examples) - 1, k, sub))
    return examples, plan


def _confidence(p):
    p = sorted(p, reverse=True)
    return float(p[0] - (p[1] if len(p) > 1 else 0.0))


def decide(model, state, questions, max_state_tokens=8192, max_q_per_pass=24, temperature=None):
    import numpy as np
    examples, plan = _api_to_examples(state, questions, max_q_per_pass)
    t0 = time.time()
    res = score_examples(model, examples, max_state_tokens=max_state_tokens, temperature=temperature)
    # first pass: gather; for chunked choices, run a second pass over the top candidates
    gathered = {}
    for qi, ei, k, sub in plan:
        gathered.setdefault(qi, []).append((res[ei][k], sub))
    second = []; second_plan = []
    for qi, parts in gathered.items():
        if parts[0][1] is None:
            continue
        cands = []
        for p, sub in parts:
            p_none = p[-1]
            for j, oi in enumerate(sub):
                cands.append((float(p[j]) * (1 - p_none), oi))
        cands.sort(reverse=True); top = [oi for _, oi in cands[:MAX_OPTIONS]]
        q = questions[qi]; opts = q.get("options") or q.get("criteria") or []
        keys = list(opts.keys()) if isinstance(opts, dict) else [str(o) for o in opts]
        descs = [opts[k] for k in keys] if isinstance(opts, dict) else None
        second.append(Example(examples[0].state, [Q(q.get("instructions") or q.get("question") or "", [keys[i] for i in top], -1, kind="choice", descs=[descs[i] for i in top] if descs else None)]))
        second_plan.append((qi, top, keys))
    if second:
        res2 = score_examples(model, second, max_state_tokens=max_state_tokens, temperature=temperature)
        for (qi, top, keys), r in zip(second_plan, res2):
            full = np.zeros(len(keys))
            for pos, oi in enumerate(top):
                full[oi] = r[0][pos]
            gathered[qi] = [(full, None)]
    answers = []
    for qi, q in enumerate(questions):
        p = gathered[qi][0][0]; kind = q.get("type", "choice"); qid = q.get("id", f"q{qi + 1}")
        if kind == "noul":
            answers.append({"id": qid, "type": "noul", "noul": float(p[1]), "confidence": _confidence(p)})
        elif kind == "score":
            levels = q.get("levels") or q.get("criteria") or []
            keys = list(levels.keys()) if isinstance(levels, dict) else [str(i) for i in range(len(levels))]
            try:
                num = [float(k) for k in keys]
            except ValueError:
                num = list(range(len(keys)))
            exp = float(sum(pi * ni for pi, ni in zip(p, num)))
            answers.append({"id": qid, "type": "score", "score": exp, "level": keys[int(np.argmax(p))], "probabilities": {k: float(v) for k, v in zip(keys, p)}, "confidence": _confidence(p)})
        else:
            opts = q.get("options") or q.get("criteria") or []
            keys = list(opts.keys()) if isinstance(opts, dict) else [str(o) for o in opts]
            answers.append({"id": qid, "type": "choice", "choice": keys[int(np.argmax(p))], "probabilities": {k: float(v) for k, v in zip(keys, p)}, "confidence": _confidence(p)})
    return {"answers": answers, "latency_ms": round((time.time() - t0) * 1000, 1), "passes": len(examples) + len(second)}
