"""Train the System-One decision model on Modal.

  S1_GPU=H100 modal run train.py --name e2b-full --model google/gemma-4-E2B-it --lora 1 --tier full
  S1_GPU=none modal run train.py --name smoke --model unsloth/gemma-3-270m-it --tier smoke      # CPU smoke test
"""
import modal, os, json
from s1.modal_common import vol, image, gpu_kwargs

app = modal.App("jev-train")
DATA = "/vol/data/v1"

TIERS = {
    "smoke": dict(per_task_cap=6, eval_cap=4, max_state_tokens=384, max_tokens=4096, accum=1, epochs=1.0, max_steps=6, eval_every=3, warmup=2, tasks=["banking77", "boolq", "doom", "support", "helpsteer2", "glaive_tools"]),
    "lite": dict(per_task_cap=2500, eval_cap=120, max_state_tokens=1024, max_tokens=32768, accum=1, epochs=1.0, max_steps=0, eval_every=400, warmup=100),
    "full": dict(per_task_cap=12000, eval_cap=150, max_state_tokens=2048, max_tokens=24576, accum=2, epochs=1.0, max_steps=0, eval_every=800, warmup=200),
}


@app.function(image=image, volumes={"/vol": vol}, timeout=8 * 3600, **gpu_kwargs())
def train(cfg: dict):
    import gzip, math, random, time, glob
    import numpy as np, torch, torch.nn.functional as F
    from s1.schema import Example, Q, build, collate, batches_by_tokens, NONE_OPT
    from s1.engine import DecisionModel, score_examples
    from s1.metrics import summarize

    name = cfg["name"]; out = f"/vol/runs/{name}"; os.makedirs(out, exist_ok=True)
    logf = open(f"{out}/train.log", "a")
    def log(*s):
        msg = " ".join(str(x) for x in s); print(msg, flush=True); logf.write(msg + "\n"); logf.flush()
    log("[cfg]", json.dumps(cfg))
    torch.manual_seed(cfg["seed"]); rng = random.Random(cfg["seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------ data
    manifest = json.load(open(f"{DATA}/manifest.json"))
    tasks = cfg.get("tasks") or [t for t, m in manifest.items() if "error" not in m]
    train_tasks = [t for t in tasks if not manifest[t].get("heldout")]
    held_tasks = [t for t in manifest if manifest[t].get("heldout") and "error" not in manifest[t] and (not cfg.get("tasks") or t in cfg["tasks"])]
    def read(path, cap):
        exs = []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                exs.append(Example.from_json(line))
                if cap and len(exs) >= cap:
                    break
        return exs
    train = []; per_task = {}
    for t in train_tasks:
        exs = read(f"{DATA}/{t}.train.jsonl.gz", cfg["per_task_cap"])
        if cfg.get("train_frac", 1.0) < 1.0:
            exs = exs[: max(1, int(len(exs) * cfg["train_frac"]))]
        per_task[t] = len(exs); train += exs
    evals = {t: read(f"{DATA}/{t}.eval.jsonl.gz", cfg["eval_cap"]) for t in train_tasks}
    held = {t: read(f"{DATA}/{t}.eval.jsonl.gz", cfg["eval_cap"]) for t in held_tasks}
    log(f"[data] train tasks={len(train_tasks)} examples={len(train)} heldout tasks={len(held_tasks)}; per-task: " + json.dumps(per_task))

    # ------------------------------------------------------------------ model
    lora = None
    if cfg.get("lora"):
        if cfg.get("lora_targets", "all") == "attn":
            tm = r".*(language_model|model)\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
        else:
            tm = r".*(language_model|model)\.layers\.\d+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
        lora = dict(r=cfg["lora_r"], lora_alpha=cfg["lora_r"] * 2, lora_dropout=0.0, bias="none", target_modules=tm)
    model = DecisionModel(cfg["model"], grad_ckpt=cfg.get("grad_ckpt", True) and dev == "cuda", lora=lora)
    if not lora and cfg.get("freeze_embeddings", False):
        for n, p in model.lm.named_parameters():
            if "embed" in n:
                p.requires_grad_(False)
    params = model.trainable_parameters()
    log(f"[model] {cfg['model']} class={type(model.lm).__name__} backbone={type(model.backbone).__name__} trainable={sum(p.numel() for p in params)/1e6:.1f}M total={sum(p.numel() for p in model.lm.parameters())/1e6:.1f}M device={dev}")
    tok = model.tok

    # ------------------------------------------------------------------ tokenize
    def none_aug(e):
        qs = []
        for q in e.qs:
            if q.kind == "choice" and len(q.options) >= 3 and rng.random() < cfg["none_prob"] and not any(NONE_OPT in o for o in q.options):
                if rng.random() < 0.5 and q.gold >= 0:
                    opts = [o for i, o in enumerate(q.options) if i != q.gold] + [NONE_OPT]
                    descs = ([d for i, d in enumerate(q.descs) if i != q.gold] + ["none of the listed options applies"]) if q.descs else None
                    qs.append(Q(q.text, opts, len(opts) - 1, kind=q.kind, descs=descs))
                else:
                    qs.append(Q(q.text, list(q.options) + [NONE_OPT], q.gold, kind=q.kind, descs=(list(q.descs) + ["none of the listed options applies"]) if q.descs else None))
            else:
                qs.append(q)
        return Example(e.state, qs, e.task)
    t0 = time.time(); items = []
    for e in train:
        it = build(none_aug(e), tok, rng, max_state_tokens=cfg["max_state_tokens"]); items.append(it)
    ntok = sum(len(it["ids"]) for it in items); nq = sum(len(it["slots"]) for it in items)
    log(f"[data] {len(items)} items, {nq} questions, {ntok/1e6:.1f}M tokens, tokenized in {time.time()-t0:.0f}s")
    del train

    # ------------------------------------------------------------------ eval helper
    def run_eval(sets, cap=None):
        model.lm.eval(); res = {}
        for t, exs in sets.items():
            exs = exs[:cap] if cap else exs
            if not exs:
                continue
            r = score_examples(model, exs, max_state_tokens=cfg["max_state_tokens"], max_tokens=cfg["max_tokens"])
            rows, golds, nopts = [], [], []
            for e, pq in zip(exs, r):
                for q, p in zip(e.qs, pq):
                    rows.append(np.asarray(p)); golds.append(q.gold); nopts.append(len(q.options))
            K = max(len(p) for p in rows); probs = np.zeros((len(rows), K))
            for i, p in enumerate(rows):
                probs[i, :len(p)] = p
            res[t] = summarize(probs, np.array(golds), np.array(nopts))
        model.lm.train()
        agg = {}
        for k in ["acc", "ece", "nll", "brier", "aurc", "acc_at_80"]:
            vals = [v[k] for v in res.values() if v.get("n")]
            agg[k] = float(np.mean(vals)) if vals else None
        return res, agg

    # ------------------------------------------------------------------ train loop
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("wd", 0.0), betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(len(batches_by_tokens(items, cfg["max_tokens"], random.Random(0))) / cfg["accum"])
    total = int(steps_per_epoch * cfg["epochs"])
    if cfg.get("max_steps"):
        total = min(total, cfg["max_steps"])
    log(f"[sched] {steps_per_epoch} optimizer steps/epoch, {total} total, warmup {cfg['warmup']}")
    def lr_at(s):
        if s < cfg["warmup"]:
            return cfg["lr"] * (s + 1) / cfg["warmup"]
        return cfg["lr"] * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - cfg["warmup"]) / max(1, total - cfg["warmup"])))))
    step = micro = ep = 0; hist = []; model.lm.train()
    t0 = time.time(); t_last = t0; ce_acc = 0.0; n_acc = 0; tok_acc = 0
    def loss_fn(logits, golds):
        ok = golds >= 0; logits, golds = logits[ok], golds[ok]
        ce = F.cross_entropy(logits, golds)
        loss = ce
        if cfg["brier_w"] > 0:
            p = torch.nan_to_num(torch.softmax(logits, -1)); onehot = F.one_hot(golds, p.shape[1]).float()
            loss = loss + cfg["brier_w"] * ((p - onehot) ** 2).sum(1).mean()
        return loss, ce.detach()
    done = False
    while not done:
        for bidx in batches_by_tokens(items, cfg["max_tokens"], rng):
            if step >= total:
                done = True; break
            b = collate([items[i] for i in bidx], tok.pad_token_id)
            logits = model.forward_batch(b)
            loss, ce = loss_fn(logits, b["golds"].to(logits.device))
            (loss / cfg["accum"]).backward()
            ce_acc += ce.item(); n_acc += 1; micro += 1; tok_acc += b["input_ids"].numel()
            if micro % cfg["accum"] == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 20 == 0 or step == total or step <= 3:
                    now = time.time(); tps = tok_acc / max(1e-6, now - t_last)
                    mem = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0
                    log(f"[train] step {step}/{total} ep {ep} ce {ce_acc/max(1,n_acc):.4f} gn {float(gn):.2f} lr {lr_at(step):.2e} {(now-t0)/60:.1f}min eta {(total-step)*(now-t_last)/max(1,min(20,step))/60:.0f}min {tps:.0f}tok/s mem {mem:.1f}GB")
                    ce_acc = 0.0; n_acc = 0; t_last = now; tok_acc = 0
                if (cfg["eval_every"] and step % cfg["eval_every"] == 0) or step == total:
                    r_in, a_in = run_eval(evals, cap=cfg["eval_cap"]); r_h, a_h = run_eval(held, cap=cfg["eval_cap"])
                    log(f"[eval] step {step} in-task {json.dumps(a_in)} | held-out {json.dumps(a_h)}")
                    hist.append(dict(step=step, in_task=a_in, heldout=a_h, per_task_in=r_in, per_task_held=r_h))
                    json.dump(hist, open(f"{out}/hist.json", "w"), indent=1)
        ep += 1
        if step >= total:
            done = True
    # ------------------------------------------------------------------ temperature scaling on in-task eval (letter logits)
    model.lm.eval()
    with torch.no_grad():
        all_logits, all_golds = [], []
        for t, exs in evals.items():
            for e in exs[: cfg["eval_cap"]]:
                it = build(e, tok, max_state_tokens=cfg["max_state_tokens"], train=False)
                b = collate([it], tok.pad_token_id)
                lg = model.forward_batch(b).cpu()
                all_logits.append(lg); all_golds.append(torch.tensor(it["golds"]))
        L = torch.cat(all_logits); G = torch.cat(all_golds); ok = G >= 0; L, G = L[ok], G[ok]
        best_T, best_nll = 1.0, 1e9
        for T in np.linspace(0.3, 5.0, 95):
            nll = F.cross_entropy(L / float(T), G).item()
            if nll < best_nll:
                best_T, best_nll = float(T), nll
        model.temperature = best_T
        log(f"[calib] temperature={best_T:.3f} nll={best_nll:.4f} (T=1 nll={F.cross_entropy(L, G).item():.4f}) on {len(G)} questions")
    model.save(f"{out}/model")
    r_in, a_in = run_eval(evals, cap=cfg["eval_cap"]); r_h, a_h = run_eval(held, cap=cfg["eval_cap"])
    log(f"[final] in-task {json.dumps(a_in)} | held-out {json.dumps(a_h)}")
    summary = dict(name=name, model=cfg["model"], steps=total, tokens=ntok, temperature=best_T, in_task=a_in, heldout=a_h, per_task_in=r_in, per_task_held=r_h, minutes=round((time.time() - t0) / 60, 1))
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
    vol.commit()
    log("[done] saved", f"{out}/model")
    return summary


@app.local_entrypoint()
def main(name: str, model: str = "unsloth/gemma-3-270m-it", tier: str = "lite", lora: int = 0, lora_r: int = 64, lr: float = 0.0,
         brier_w: float = 0.3, none_prob: float = 0.08, train_frac: float = 1.0, seed: int = 0, tasks: str = "", grad_ckpt: int = 1,
         max_state_tokens: int = 0, max_tokens: int = 0, epochs: float = 0.0, max_steps: int = 0, freeze_embeddings: int = 0, accum: int = 0, lora_targets: str = "all"):
    cfg = dict(TIERS[tier]); cfg.update(name=name, model=model, lora=bool(lora), lora_r=lora_r, brier_w=brier_w, none_prob=none_prob, train_frac=train_frac,
                                       seed=seed, grad_ckpt=bool(grad_ckpt), freeze_embeddings=bool(freeze_embeddings), lora_targets=lora_targets)
    if tasks:
        cfg["tasks"] = tasks.split(",")
    cfg["lr"] = lr or (1e-4 if lora else (1e-4 if "270m" in model.lower() else 1e-5))
    for k, v in dict(max_state_tokens=max_state_tokens, max_tokens=max_tokens, epochs=epochs, max_steps=max_steps, accum=accum).items():
        if v:
            cfg[k] = v
    s = train.remote(cfg)
    print(json.dumps({k: v for k, v in s.items() if k not in ("per_task_in", "per_task_held")}, indent=1))
