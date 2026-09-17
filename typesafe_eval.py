"""Build the TypeSafe public-eval head-to-head set (20 cases, 373 reference pairs) from evals.typesafe.ai viewer data.
Output: /vol/eval/typesafe_full.json  {workflows: {wf: {cases: [{case_id, state, questions:[{qid,kind,text,options,descs,ref_value}], published:{opus:{qid:val},...}}]}}}
Run:  modal run typesafe_eval.py
"""
import modal, json, re, os
app = modal.App("jev-typesafe-eval")
vol = modal.Volume.from_name("jev-replica", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("requests")
WORKFLOWS = ["security_incidents", "agent_trace_observability", "invoice_processing", "customer_service"]


def render(doc):
    return doc if isinstance(doc, str) else json.dumps(doc, indent=1, ensure_ascii=False)


def canonical(value, qtype):
    if value is None:
        return None
    if qtype == "noul":
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, (int, float)):
            return "yes" if float(value) >= 0.5 else "no"
        return "yes" if str(value).lower() in ("true", "yes", "1") else "no"
    if qtype == "score":
        try:
            return str(int(round(float(value))))
        except (TypeError, ValueError):
            return None
    return str(value)


def consensus(entry, qtype):
    sets = entry.get("sets", [])
    acc = {}; values = []
    for s in sets:
        for k, v in (s.get("probabilities") or {}).items():
            acc[k] = acc.get(k, 0.0) + float(v)
        if s.get("value") is not None:
            values.append(s["value"])
    if acc:
        probs = {k: v / len(sets) for k, v in acc.items()}
        best = max(probs, key=probs.get)
        return canonical(best if qtype != "noul" else (best in ("true", "yes", "True")), qtype), probs
    if values:
        canon = [c for c in (canonical(v, qtype) for v in values) if c is not None]
        if canon:
            return max(set(canon), key=canon.count), {}
    return None, {}


def build_from_viewer(raw_by_wf):
    out = {"workflows": {}}
    for wf in WORKFLOWS:
        raw = raw_by_wf[wf]
        m = re.search(r"__VIEWER_DATA__\((.*)\)\s*;?\s*$", raw, re.S) or re.search(r"__VIEWER_DATA__\((.*)\)", raw, re.S)
        ev = json.loads(m.group(1))["eval"]
        docs = ev["documents"]; catalog = ev["questions"]
        wf_qmap = {}
        for ex in ev["examples"]:
            case = ev["cases"][ex["case_id"]]
            for mv in case["models"].values():
                for n in mv.get("nodes", []):
                    for qid, idx in (n.get("questions") or {}).items():
                        wf_qmap.setdefault(qid, int(idx))
        cases = []
        for ex in ev["examples"]:
            cid = ex["case_id"]; case = ev["cases"][cid]
            doc_idx = set()
            for mv in case["models"].values():
                for n in mv.get("nodes", []):
                    if n.get("doc") is not None:
                        doc_idx.add(int(n["doc"]))
            reference = {}
            for node_answers in case.get("reference_answers", {}).values():
                for qid, entry in node_answers.items():
                    val, probs = consensus(entry, entry["type"])
                    reference[qid] = {"value": val, "probs": probs, "type": entry["type"]}
            published = {}
            for mkey, mv in case["models"].items():
                answers = {}
                for n in mv.get("nodes", []):
                    for qid, ans in (n.get("answers") or {}).items():
                        raw_v = ans.get(ans["type"])
                        answers[qid] = canonical(raw_v, ans["type"])
                published[mkey] = answers
            questions = []
            for qid, ref in reference.items():
                idx = wf_qmap.get(qid)
                if idx is None or not (0 <= idx < len(catalog)) or ref["value"] is None:
                    continue
                q = catalog[idx]; crit = q.get("criteria")
                if q["type"] == "noul":
                    opts = ["no", "yes"]; descs = None
                    if isinstance(crit, dict):
                        descs = [crit.get("false") or None, crit.get("true") or None]
                        if not any(descs):
                            descs = None
                elif q["type"] == "score":
                    lv = crit if isinstance(crit, list) else []
                    opts = [str(i) for i in range(len(lv))]; descs = list(lv)
                else:
                    if isinstance(crit, dict):
                        opts = list(crit.keys()); descs = [crit[k] for k in opts]
                    elif isinstance(crit, list):
                        opts = list(crit); descs = None
                    else:
                        continue
                if ref["value"] not in opts:
                    continue
                questions.append({"qid": qid, "kind": q["type"], "text": q["instructions"], "options": opts, "descs": descs, "ref_value": ref["value"], "ref_probs": ref["probs"]})
            state = "\n\n".join(f"## Document {i}\n{render(docs[i])}" for i in sorted(doc_idx) if 0 <= i < len(docs))
            cases.append({"case_id": cid, "name": ex.get("name", cid), "state": state, "state_chars": len(state), "questions": questions, "published": published})
        out["workflows"][wf] = {"cases": cases, "catalog_size": len(catalog)}
    return out


@app.function(image=image, volumes={"/vol": vol}, timeout=600)
def build():
    import requests
    raw = {wf: requests.get(f"https://evals.typesafe.ai/{wf}-cases.js", timeout=60).text for wf in WORKFLOWS}
    out = build_from_viewer(raw)
    os.makedirs("/vol/eval", exist_ok=True)
    json.dump(out, open("/vol/eval/typesafe_full.json", "w"))
    vol.commit()
    summary = {}
    for wf, d in out["workflows"].items():
        pairs = sum(len(c["questions"]) for c in d["cases"]); chars = [c["state_chars"] for c in d["cases"]]
        kinds = {}
        for c in d["cases"]:
            for q in c["questions"]:
                kinds[q["kind"]] = kinds.get(q["kind"], 0) + 1
        nopt = max((len(q["options"]) for c in d["cases"] for q in c["questions"]), default=0)
        summary[wf] = dict(cases=len(d["cases"]), pairs=pairs, chars=f"{min(chars)}-{max(chars)}", kinds=kinds, max_options=nopt,
                           published_cov={m: sum(1 for c in d["cases"] for q in c["questions"] if c["published"].get(m, {}).get(q["qid"]) is not None) for m in ["opus", "sol", "typesafe"]})
    return summary


@app.local_entrypoint()
def main():
    s = build.remote()
    tot = 0
    for wf, v in s.items():
        print(wf, json.dumps(v)); tot += v["pairs"]
    print("TOTAL reference pairs:", tot)
