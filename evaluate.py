"""Full evaluation of a trained run (or a zero-shot base model):
  in-task test splits, held-out tasks, synthetic demo tasks, the TypeSafe public head-to-head (20 cases / 372 pairs), latency.

  S1_GPU=H100 modal run evaluate.py --run e2b-full
  S1_GPU=H100 modal run evaluate.py --run zeroshot-270m --model unsloth/gemma-3-270m-it     # untrained baseline
"""
import modal, os, json
from s1.modal_common import vol, image, gpu_kwargs
app = modal.App("jev-evaluate")
DATA = "/vol/data/v1"; SYNTH = ["doom", "smart_home", "support", "security", "invoice", "agent_trace", "catalog"]


@app.function(image=image, volumes={"/vol": vol}, timeout=4 * 3600, **gpu_kwargs())
def evaluate(run: str, model_path: str | None, which: list, cap: int, ts_state_tokens: int):
    import gzip, time, numpy as np, torch
    from s1.schema import Example
    from s1.engine import DecisionModel, score_examples, decide
    from s1.metrics import summarize
    out = f"/vol/runs/{run}/eval"; os.makedirs(out, exist_ok=True)
    path = model_path or f"/vol/runs/{run}/model"
    model = DecisionModel(path); model.lm.eval()
    report = {"run": run, "model": path, "temperature": model.temperature, "device": model.device}
    manifest = json.load(open(f"{DATA}/manifest.json"))
    def read(p, cap):
        exs = []
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                exs.append(Example.from_json(line))
                if cap and len(exs) >= cap:
                    break
        return exs
    def metrics_for(exs, max_state_tokens=2048):
        r = score_examples(model, exs, max_state_tokens=max_state_tokens, max_tokens=24000)
        rows, golds, nopts, kinds = [], [], [], []
        for e, pq in zip(exs, r):
            for q, p in zip(e.qs, pq):
                rows.append(np.asarray(p)); golds.append(q.gold); nopts.append(len(q.options)); kinds.append(q.kind)
        K = max(len(p) for p in rows); probs = np.zeros((len(rows), K))
        for i, p in enumerate(rows):
            probs[i, :len(p)] = p
        m = summarize(probs, np.array(golds), np.array(nopts))
        by_kind = {}
        for k in set(kinds):
            idx = [i for i, kk in enumerate(kinds) if kk == k]
            by_kind[k] = summarize(np.array(probs)[idx], np.array(golds)[idx], np.array(nopts)[idx])
        m["by_kind"] = by_kind
        return m
    def agg(res):
        keys = ["acc", "ece", "nll", "brier", "aurc", "acc_at_80", "chance"]
        return {k: float(np.mean([v[k] for v in res.values() if v.get("n")])) for k in keys}

    if "intask" in which:
        t0 = time.time(); res = {}
        for t, m in manifest.items():
            if "error" in m or m.get("heldout") or t in SYNTH:
                continue
            exs = read(f"{DATA}/{t}.eval.jsonl.gz", cap)
            if exs:
                res[t] = metrics_for(exs); print(f"  in-task {t:22s} acc={res[t]['acc']:.3f} ece={res[t]['ece']:.3f} n={res[t]['n']}", flush=True)
        report["intask"] = dict(per_task=res, agg=agg(res), secs=round(time.time() - t0)); print("IN-TASK", json.dumps(report["intask"]["agg"]))
    if "heldout" in which:
        t0 = time.time(); res = {}
        for t, m in manifest.items():
            if "error" in m or not m.get("heldout"):
                continue
            exs = read(f"{DATA}/{t}.eval.jsonl.gz", cap)
            if exs:
                res[t] = metrics_for(exs, max_state_tokens=4096 if t == "quality" else 2048); print(f"  held-out {t:22s} acc={res[t]['acc']:.3f} ece={res[t]['ece']:.3f} n={res[t]['n']}", flush=True)
        report["heldout"] = dict(per_task=res, agg=agg(res), secs=round(time.time() - t0)); print("HELD-OUT", json.dumps(report["heldout"]["agg"]))
    if "synth" in which:
        t0 = time.time(); res = {}
        for t in SYNTH:
            if t not in manifest or "error" in manifest[t]:
                continue
            exs = read(f"{DATA}/{t}.eval.jsonl.gz", cap)
            res[t] = metrics_for(exs); print(f"  synth {t:12s} acc={res[t]['acc']:.3f} ece={res[t]['ece']:.3f} by_kind={ {k: round(v['acc'],3) for k,v in res[t]['by_kind'].items()} }", flush=True)
        report["synth"] = dict(per_task=res, agg=agg(res), secs=round(time.time() - t0)); print("SYNTH", json.dumps(report["synth"]["agg"]))
    if "typesafe" in which:
        ts = json.load(open("/vol/eval/typesafe_full.json")); t0 = time.time()
        rows = []; per_wf = {}
        for wf, d in ts["workflows"].items():
            n_ok = n = 0; lat = []
            for c in d["cases"]:
                qs = []
                for q in c["questions"]:
                    if q["kind"] == "noul":
                        crit = {"false": q["descs"][0], "true": q["descs"][1]} if q["descs"] else None
                        qs.append({"id": q["qid"], "type": "noul", "instructions": q["text"], "criteria": crit})
                    elif q["kind"] == "score":
                        qs.append({"id": q["qid"], "type": "score", "instructions": q["text"], "levels": q["descs"]})
                    else:
                        opts = dict(zip(q["options"], q["descs"])) if q["descs"] else q["options"]
                        qs.append({"id": q["qid"], "type": "choice", "instructions": q["text"], "options": opts})
                t1 = time.time(); r = decide(model, c["state"], qs, max_state_tokens=ts_state_tokens); lat.append(time.time() - t1)
                for q, a in zip(c["questions"], r["answers"]):
                    if q["kind"] == "noul":
                        pred = "yes" if a["noul"] >= 0.5 else "no"; conf = max(a["noul"], 1 - a["noul"])
                    elif q["kind"] == "score":
                        pred = a["level"]; conf = max(a["probabilities"].values())
                    else:
                        pred = a["choice"]; conf = max(a["probabilities"].values())
                    ok = pred == q["ref_value"]; n += 1; n_ok += ok
                    rows.append(dict(wf=wf, case=c["case_id"], qid=q["qid"], kind=q["kind"], pred=pred, ref=q["ref_value"], ok=bool(ok), conf=float(conf), published={m: c["published"].get(m, {}).get(q["qid"]) for m in ("opus", "sol", "typesafe")}))
            per_wf[wf] = dict(agree=n_ok, n=n, acc=round(n_ok / max(1, n), 4), mean_case_latency_s=round(float(np.mean(lat)), 2), passes=len(d["cases"]))
            print(f"  typesafe {wf:28s} {n_ok}/{n} = {n_ok/max(1,n):.3f}  latency/case {np.mean(lat):.2f}s", flush=True)
        # strict common subset: pairs answered by all published models
        common = [r for r in rows if all(r["published"][m] is not None for m in ("opus", "sol", "typesafe"))]
        def acc_of(fn, rs):
            return (sum(fn(r) for r in rs), len(rs))
        table = {"ours": acc_of(lambda r: r["ok"], common)}
        for m in ("opus", "sol", "typesafe"):
            table[m] = acc_of(lambda r, m=m: r["published"][m] == r["ref"], common)
        table_all = {"ours": acc_of(lambda r: r["ok"], rows)}
        for m in ("opus", "sol", "typesafe"):
            answered = [r for r in rows if r["published"][m] is not None]
            table_all[m] = acc_of(lambda r, m=m: r["published"][m] == r["ref"], answered)
        by_kind = {}
        for k in ("noul", "score", "choice"):
            rs = [r for r in rows if r["kind"] == k]; by_kind[k] = acc_of(lambda r: r["ok"], rs)
        conf = np.array([r["conf"] for r in rows]); ok = np.array([r["ok"] for r in rows], dtype=float)
        from s1.metrics import ece as _ece
        # baselines: per-type majority (most frequent reference value of that kind), per-question majority (same qid across cases), always-first-option
        from collections import Counter
        maj_type = {k: Counter(r["ref"] for r in rows if r["kind"] == k).most_common(1)[0][0] for k in ("noul", "score", "choice")}
        maj_q = {}
        for qid in set(r["qid"] for r in rows):
            maj_q[qid] = Counter(r["ref"] for r in rows if r["qid"] == qid).most_common(1)[0][0]
        baselines = {"per_type_majority": acc_of(lambda r: maj_type[r["kind"]] == r["ref"], rows),
                     "per_question_majority": acc_of(lambda r: maj_q[r["qid"]] == r["ref"], rows),
                     "always_no_for_noul": acc_of(lambda r: r["ref"] == "no", [r for r in rows if r["kind"] == "noul"])}
        baselines_common = {"per_type_majority": acc_of(lambda r: maj_type[r["kind"]] == r["ref"], common), "per_question_majority": acc_of(lambda r: maj_q[r["qid"]] == r["ref"], common)}
        report["typesafe"] = dict(per_workflow=per_wf, overall=acc_of(lambda r: r["ok"], rows), common_subset=table, common_n=len(common), all_answered=table_all, by_kind=by_kind, ece=_ece(conf, ok), baselines=baselines, baselines_common=baselines_common, secs=round(time.time() - t0))
        print("BASELINES", json.dumps(baselines), "common:", json.dumps(baselines_common))
        json.dump(rows, open(f"{out}/typesafe_rows.json", "w"), indent=1); json.dump(report, open(f"{out}/report.json", "w"), indent=1); vol.commit()
        print("TYPESAFE overall", report["typesafe"]["overall"], "common-subset", table)
    if "latency" in which:
        exs = read(f"{DATA}/doom.eval.jsonl.gz", 32)
        qs = [{"id": f"q{i}", "type": q.kind, "instructions": q.text, "options": q.options, "levels": q.options if q.kind == "score" else None} for i, q in enumerate(exs[0].qs)]
        for q in qs:
            if q["type"] == "score":
                q["levels"] = q.pop("options")
        decide(model, exs[0].state, qs)  # warm
        ts_ = []
        for e in exs[:16]:
            t1 = time.time(); decide(model, e.state, qs); ts_.append((time.time() - t1) * 1000)
        report["latency"] = dict(doom_state_6q_ms_median=float(np.median(ts_)), doom_state_6q_ms_p90=float(np.percentile(ts_, 90)), n_tokens_state=len(model.tok.encode(exs[0].state)))
        print("LATENCY", json.dumps(report["latency"]))
    if "doom" in which:
        import random as _r
        from s1 import doom_sim as D
        t0 = time.time(); lines = []
        log = lambda m: (print(m, flush=True), lines.append(m))
        pol = D.model_policy(model, decide)
        r_model = D.run_episodes(pol, n=12, seed=500, log=log)
        r_teacher = D.run_episodes(D.teacher_policy, n=12, seed=500)
        r_random = D.run_episodes(D.random_policy(_r.Random(0)), n=12, seed=500)
        # agreement with the teacher on the same states as a supplementary number
        agree = n = 0
        for ep in range(6):
            a = D.Arena(seed=900 + ep); done = False
            while not done and n < 300:
                st = a.state(); act = pol(st); agree += int(act == D.teacher_policy(st)); n += 1; done = a.step(act)
        report["doom"] = dict(model=r_model, teacher=r_teacher, random=r_random, teacher_agreement=round(agree / max(1, n), 3), n_states=n, secs=round(time.time() - t0))
        print("DOOM model", json.dumps(r_model), "| teacher", json.dumps(r_teacher), "| random", json.dumps(r_random), "| agreement", report["doom"]["teacher_agreement"])
    json.dump(report, open(f"{out}/report.json", "w"), indent=1)
    vol.commit()
    return {k: (v if k in ("run", "model", "temperature", "device", "latency", "doom") else {kk: vv for kk, vv in v.items() if kk != "per_task"}) for k, v in report.items()}


@app.local_entrypoint()
def main(run: str, model: str = "", which: str = "intask,heldout,synth,typesafe,latency,doom", cap: int = 300, ts_state_tokens: int = 14000):
    r = evaluate.remote(run, model or None, which.split(","), cap, ts_state_tokens)
    print(json.dumps(r, indent=1, default=str))
