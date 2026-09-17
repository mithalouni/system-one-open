"""Build a labeled email set for the inbox-classifier demo: 1,500 real Enron emails (750 ham / 750 spam)."""
import modal, json, os
app = modal.App("jev-build-emails")
vol = modal.Volume.from_name("jev-replica", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("datasets==5.0.1", "pyarrow")

@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=4096, timeout=900)
def build(n_each: int = 750):
    from datasets import load_dataset
    import random, re
    ds = load_dataset("SetFit/enron_spam", split="test").shuffle(seed=7)
    out = []; counts = {0: 0, 1: 0}
    for r in ds:
        y = int(r["label"])
        if counts[y] >= n_each:
            continue
        subj = (r.get("subject") or "").strip() or "(no subject)"; text = (r.get("text") or "").strip()
        text = re.sub(r"\s+", " ", text)[:1200]
        if len(text) < 20:
            continue
        counts[y] += 1
        out.append({"id": r.get("message_id") or f"m{len(out)}", "subject": subj[:140], "text": text, "date": str(r.get("date") or ""), "spam_label": y})
        if all(c >= n_each for c in counts.values()):
            break
    random.Random(3).shuffle(out)
    os.makedirs("/vol/demo", exist_ok=True); json.dump(out, open("/vol/demo/emails.json", "w")); vol.commit()
    return {"n": len(out), "spam": sum(e["spam_label"] for e in out), "avg_chars": sum(len(e["text"]) for e in out) // len(out), "sample": out[0]["subject"]}

@app.local_entrypoint()
def main():
    print(build.remote())
