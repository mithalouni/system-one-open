"""Build the decision-task mixture on Modal: one CPU container per task, shards written to the shared volume.

  modal run build_data.py                 # all tasks
  modal run build_data.py --only banking77,doom
Output: /vol/data/v1/<task>.train.jsonl.gz, <task>.eval.jsonl.gz, and /vol/data/v1/manifest.json
"""
import modal, os, json

VERSION = "v1"
app = modal.App("jev-build-data")
vol = modal.Volume.from_name("jev-replica", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("datasets==5.0.1", "hf_transfer", "pyarrow", "numpy")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_DATASETS_TRUST_REMOTE_CODE": "0", "TOKENIZERS_PARALLELISM": "false"})
         .add_local_python_source("s1"))


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=8192, timeout=3600, max_containers=24)
def build_task(name: str):
    import gzip, time, traceback, random
    from s1 import data_real as R, data_synth as S
    t0 = time.time(); out_dir = f"/vol/data/{VERSION}"; os.makedirs(out_dir, exist_ok=True)
    try:
        if name in S.SYNTH:
            tr, ev = S.load_synth(name); heldout = False; group = "synth"
        else:
            tr, ev = R.load_task(name); heldout = R.TASKS[name]["heldout"]; group = R.TASKS[name]["group"]
    except Exception as e:
        return dict(name=name, error=f"{type(e).__name__}: {str(e)[:300]}", tb=traceback.format_exc()[-1500:], secs=round(time.time() - t0))
    random.Random(0).shuffle(tr)
    for split, exs in [("train", tr), ("eval", ev)]:
        with gzip.open(f"{out_dir}/{name}.{split}.jsonl.gz", "wt", encoding="utf-8") as f:
            for e in exs:
                f.write(e.to_json() + "\n")
    vol.commit()
    nq = sum(len(e.qs) for e in tr); kinds = {}
    for e in (tr or ev):
        for q in e.qs:
            kinds[q.kind] = kinds.get(q.kind, 0) + 1
    if not (tr or ev):
        return dict(name=name, error="loader returned no examples", tb="", secs=round(time.time() - t0))
    sample = (tr or ev)[0]
    return dict(name=name, group=group, heldout=heldout, train=len(tr), eval=len(ev), train_questions=nq, kinds=kinds,
                avg_state_chars=int(sum(len(e.state) for e in (tr or ev)[:2000]) / max(1, len((tr or ev)[:2000]))),
                n_options=len(sample.qs[0].options), secs=round(time.time() - t0))


@app.function(image=image, volumes={"/vol": vol}, cpu=1, memory=2048, timeout=600)
def write_manifest(results: list):
    out_dir = f"/vol/data/{VERSION}"; os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/manifest.json"
    old = json.load(open(path)) if os.path.exists(path) else {}
    for r in results:
        old[r["name"]] = r
    json.dump(old, open(path, "w"), indent=1)
    vol.commit()
    return old


@app.local_entrypoint()
def main(only: str = ""):
    from s1 import data_real as R, data_synth as S
    names = [n for n in only.split(",") if n] or (list(R.TASKS) + list(S.SYNTH))
    print(f"building {len(names)} tasks ...")
    results = []
    for r in build_task.map(names, return_exceptions=True, order_outputs=False):
        if isinstance(r, Exception):
            print("EXC", repr(r)[:300]); continue
        results.append(r)
        if "error" in r:
            print(f"FAIL {r['name']:22s} {r['error']}")
        else:
            print(f"ok   {r['name']:22s} train={r['train']:6d} eval={r['eval']:5d} q={r['train_questions']:7d} kinds={r['kinds']} chars={r['avg_state_chars']} held={r['heldout']} {r['secs']}s")
    m = write_manifest.remote(results)
    ok = [r for r in results if "error" not in r]
    print(f"\nDONE: {len(ok)} ok / {len(results) - len(ok)} failed; total train examples={sum(r['train'] for r in ok):,} questions={sum(r['train_questions'] for r in ok):,}")
    for r in results:
        if "error" in r:
            print("---", r["name"], "\n", r["tb"][-600:])
