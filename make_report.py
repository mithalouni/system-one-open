"""Render report.html from results/*.report.json + *.summary.json."""
import json, os, html, sys
R = "results"
RUNS = [("zeroshot-270m", "Gemma 3 270M, untrained"), ("zeroshot-e2b", "Gemma 4 E2B, untrained"), ("lite-270m", "270M, lite run"), ("full-270m", "270M, full mix"), ("e2b-full", "E2B, attention LoRA")]


def load(name):
    p = f"{R}/{name}.report.json"
    return json.load(open(p)) if os.path.exists(p) else None


def summ(name):
    p = f"{R}/{name}.summary.json"
    return json.load(open(p)) if os.path.exists(p) else None


rep = {n: load(n) for n, _ in RUNS}
sm = {n: summ(n) for n, _ in RUNS}
pub = None
for n, _ in RUNS:
    if rep[n] and "typesafe" in rep[n]:
        pub = rep[n]["typesafe"]; break
common_n = pub["common_n"] if pub else 343


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "—"


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def cell(x):
    return f'<td class="num">{x}</td>'


# ---------------- head-to-head table + chart
h2h_rows = []
bars = []
if pub:
    for key, label in [("opus", "Claude Opus 5 (published)"), ("sol", "GPT Sol (published)"), ("typesafe", "Jev / TypeSafe (published)")]:
        a, b = pub["common_subset"][key]; h2h_rows.append((label, a, b, "ref")); bars.append((label, a / b, "ref"))
    for n, label in RUNS:
        r = rep[n]
        if r and "typesafe" in r:
            a, b = r["typesafe"]["common_subset"]["ours"]; h2h_rows.append((label, a, b, "ours")); bars.append((label, a / b, "ours"))
    for key, label in [("per_question_majority", "Per-question majority baseline"), ("per_type_majority", "Per-type majority baseline")]:
        a, b = pub["baselines_common"][key]; h2h_rows.append((label, a, b, "base")); bars.append((label, a / b, "base"))

# per-workflow table (all answered pairs)
wf_names = {"security_incidents": "Security incidents", "agent_trace_observability": "Agent-trace observability", "invoice_processing": "Invoice processing", "customer_service": "Customer service"}
JEV_PUB_WF = {"security_incidents": (20, 26), "agent_trace_observability": (35, 49), "invoice_processing": (169, 184), "customer_service": (81, 92)}


def wf_table():
    cols = [(n, l) for n, l in RUNS if rep[n] and "typesafe" in rep[n]]
    head = "".join(f"<th>{html.escape(l)}</th>" for _, l in cols)
    out = [f"<table><thead><tr><th>Workflow</th><th>Jev (published)</th>{head}</tr></thead><tbody>"]
    for wf, name in wf_names.items():
        j = JEV_PUB_WF[wf]
        row = f"<tr><td>{name}</td>{cell(f'{j[0]}/{j[1]} · {pct(*j)}')}"
        for n, _ in cols:
            v = rep[n]["typesafe"]["per_workflow"][wf]; row += cell(f"{v['agree']}/{v['n']} · {pct(v['agree'], v['n'])}")
        out.append(row + "</tr>")
    row = "<tr class=\"total\"><td>All answered pairs</td>" + cell("305/351 · 86.9%")
    for n, _ in cols:
        a, b = rep[n]["typesafe"]["overall"]; row += cell(f"{a}/{b} · {pct(a, b)}")
    out.append(row + "</tr></tbody></table>")
    return "\n".join(out)


def kind_table():
    cols = [(n, l) for n, l in RUNS if rep[n] and "typesafe" in rep[n]]
    head = "".join(f"<th>{html.escape(l)}</th>" for _, l in cols)
    out = [f"<table><thead><tr><th>Question type</th>{head}</tr></thead><tbody>"]
    for k, name in [("noul", "Noul (yes/no)"), ("score", "Score (rubric level)"), ("choice", "Choice")]:
        row = f"<tr><td>{name}</td>"
        for n, _ in cols:
            a, b = rep[n]["typesafe"]["by_kind"][k]; row += cell(f"{a}/{b} · {pct(a, b)}")
        out.append(row + "</tr>")
    row = "<tr><td>Calibration error (ECE) on these pairs</td>"
    for n, _ in cols:
        row += cell(f3(rep[n]["typesafe"]["ece"]))
    out.append(row + "</tr></tbody></table>")
    return "\n".join(out)


def gen_table():
    cols = [(n, l) for n, l in RUNS if rep[n]]
    head = "".join(f"<th>{html.escape(l)}</th>" for _, l in cols)
    rows = []
    for sec, name in [("intask", "In-task test splits (trained families)"), ("heldout", "Held-out tasks (never trained)"), ("synth", "Demo families (Doom, smart home, support, security, invoice, agent trace, catalog)")]:
        for metric, mname in [("acc", "accuracy"), ("ece", "ECE"), ("acc_at_80", "accuracy at 80% coverage")]:
            row = f"<tr><td>{name} — {mname}</td>"
            for n, _ in cols:
                v = rep[n].get(sec, {}).get("agg", {}).get(metric) if rep[n] else None
                row += cell("—" if v is None else (pct(v, 1) if metric != "ece" else f3(v)))
            rows.append(row + "</tr>")
    chance = f"<tr><td class=\"muted\">chance level (mean 1/options)</td>"
    for n, _ in cols:
        v = rep[n].get("heldout", {}).get("agg", {}).get("chance") if rep[n] else None
        chance += cell("—" if v is None else pct(v, 1))
    return f"<table><thead><tr><th>Split</th>{head}</tr></thead><tbody>{''.join(rows)}{chance}</tr></tbody></table>"


def synth_table():
    cols = [(n, l) for n, l in RUNS if rep[n] and "synth" in rep[n]]
    head = "".join(f"<th>{html.escape(l)}</th>" for _, l in cols)
    out = [f"<table><thead><tr><th>Demo family</th>{head}</tr></thead><tbody>"]
    for t, name in [("doom", "Doom game state → action, danger, threat, target"), ("smart_home", "Smart home → device, action, ambiguity, confirm"), ("support", "Support chat → issue, frustration, refund, next action"), ("security", "Security alert → true positive, evidence, scope, type"), ("invoice", "AP packet → PO match, price basis, duplicate, action"), ("agent_trace", "Agent run → goal met, first bad step, satisfaction"), ("catalog", "Two product records → same product?")]:
        row = f"<tr><td>{name}</td>"
        for n, _ in cols:
            v = rep[n]["synth"]["per_task"].get(t); row += cell("—" if not v else pct(v["acc"], 1))
        out.append(row + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def doom_table():
    cols = [(n, l) for n, l in RUNS if rep[n] and "doom" in rep[n]]
    if not cols:
        return "<p class=\"muted\">Closed-loop run pending.</p>"
    n0 = cols[-1][0]; d = rep[n0]["doom"]
    rows = [("Rule-based teacher", d["teacher"]), ("Random agent", d["random"])] + [(l, rep[n]["doom"]["model"]) for n, l in cols]
    out = ["<table><thead><tr><th>Policy</th><th>Survival rate</th><th>Kill rate</th><th>Mean ticks alive</th><th>Mean final health</th></tr></thead><tbody>"]
    for name, r in rows:
        out.append(f"<tr><td>{name}</td>{cell(pct(r['survival_rate'], 1))}{cell(pct(r['kill_rate'], 1))}{cell(f'{r['mean_ticks']:.0f}')}{cell(f'{r['mean_final_health']:.0f}')}</tr>")
    out.append("</tbody></table>")
    agree = ", ".join(f"{l}: {rep[n]['doom']['teacher_agreement']*100:.0f}%" for n, l in cols)
    return "\n".join(out) + f"<p class=\"note\">Agreement with the teacher's action on the same states — {agree}.</p>"


def latency_table():
    cols = [(n, l) for n, l in RUNS if rep[n] and "latency" in rep[n]]
    out = ["<table><thead><tr><th>Model</th><th>Device</th><th>6 questions on a 168-token game state, median</th><th>p90</th></tr></thead><tbody>"]
    for n, l in cols:
        r = rep[n]; L = r["latency"]
        out.append(f"<tr><td>{html.escape(l)}</td><td>{'H100' if r['device']=='cuda' else '16 CPU cores'}</td>{cell(f'{L['doom_state_6q_ms_median']:.0f} ms')}{cell(f'{L['doom_state_6q_ms_p90']:.0f} ms')}</tr>")
    out.append(f"<tr><td>Jev (TypeSafe, published)</td><td>hosted API</td>{cell('70–500 ms')}{cell('—')}</tr></tbody></table>")
    return "\n".join(out)


def train_table():
    out = ["<table><thead><tr><th>Run</th><th>Base model</th><th>Trainable params</th><th>Training tokens</th><th>Wall time</th><th>Fitted temperature</th></tr></thead><tbody>"]
    meta = {"lite-270m": ("Gemma 3 270M, full fine-tune", "268M"), "full-270m": ("Gemma 3 270M, full fine-tune", "268M"), "e2b-full": ("Gemma 4 E2B, LoRA r=64 on attention", "21.4M")}
    for n, l in RUNS:
        s = sm[n]
        if not s:
            continue
        b, tp = meta.get(n, ("", ""))
        out.append(f"<tr><td>{html.escape(l)}</td><td>{b}</td>{cell(tp)}{cell(f'{s['tokens']/1e6:.0f}M')}{cell(f'{s['minutes']:.0f} min')}{cell(f3(s['temperature']))}</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


# ---------------- chart (SVG)
def chart():
    if not bars:
        return ""
    W, H = 720, 30 * len(bars) + 20; lw = 300
    rows = []
    for i, (label, v, kind) in enumerate(bars):
        y = 10 + i * 30; cls = {"ref": "bar-ref", "ours": "bar-ours", "base": "bar-base"}[kind]
        rows.append(f'<text x="{lw - 10}" y="{y + 14}" text-anchor="end" class="lbl">{html.escape(label)}</text>'
                    f'<rect x="{lw}" y="{y}" width="{(W - lw - 70) * v:.1f}" height="20" class="{cls}"/>'
                    f'<text x="{lw + (W - lw - 70) * v + 6:.1f}" y="{y + 14}" class="val">{v * 100:.1f}%</text>')
    return f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Agreement with the frontier consensus on the strict common subset">{"".join(rows)}</svg>'


best = None
for n in ["e2b-full", "full-270m", "lite-270m"]:
    if rep.get(n) and "typesafe" in rep[n]:
        best = n; break
best_label = dict(RUNS).get(best, "")
best_cs = rep[best]["typesafe"]["common_subset"]["ours"] if best else (0, common_n)
jev_cs = pub["common_subset"]["typesafe"] if pub else (298, 343)
best_synth = rep[best]["synth"]["agg"]["acc"] if best and "synth" in rep[best] else None
best_lat = rep[best]["latency"]["doom_state_6q_ms_median"] if best and "latency" in rep[best] else None
pending = [l for n, l in RUNS if not rep[n]]

page = f"""<title>System One, Open</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#F5F7F9;--ink:#16202A;--muted:#5F6E7C;--rule:#D6DEE5;--accent:#0E7C86;--accent-soft:#D9EEF0;--amber:#B8730A;--amber-soft:#F6E9D2;--ref:#8A97A4;--card:#FFFFFF}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#0F1519;--ink:#E6ECF1;--muted:#93A1AE;--rule:#26323B;--accent:#3FB4BE;--accent-soft:#143338;--amber:#E0A13A;--amber-soft:#332812;--ref:#5C6B78;--card:#161E24}}}}
:root[data-theme="dark"]{{--bg:#0F1519;--ink:#E6ECF1;--muted:#93A1AE;--rule:#26323B;--accent:#3FB4BE;--accent-soft:#143338;--amber:#E0A13A;--amber-soft:#332812;--ref:#5C6B78;--card:#161E24}}
body{{background:var(--bg);color:var(--ink);font-family:"Source Sans 3",system-ui,sans-serif;font-size:17px;line-height:1.55;padding-inline:20px;padding-block:32px 64px;margin:0}}
.wrap{{max-width:900px;margin:0 auto}}
h1,h2,h3{{font-family:Fraunces,Georgia,serif;font-weight:600;letter-spacing:-.01em;text-wrap:balance;margin:0}}
h1{{font-size:2.4rem;line-height:1.1}} h2{{font-size:1.45rem;margin-top:3rem;padding-top:1rem;border-top:1px solid var(--rule)}} h3{{font-size:1.1rem;margin-top:1.6rem}}
p{{max-width:68ch}} .lede{{font-size:1.15rem;color:var(--muted);max-width:60ch;margin-top:.8rem}}
.eyebrow{{font-family:"JetBrains Mono",monospace;font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;color:var(--accent)}}
.board{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:2rem 0 0}}
.tile{{background:var(--card);border:1px solid var(--rule);padding:14px 16px}}
.tile .k{{font-family:"JetBrains Mono",monospace;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}}
.tile .v{{font-family:"JetBrains Mono",monospace;font-size:1.6rem;font-weight:500;font-variant-numeric:tabular-nums;margin-top:6px}}
.tile .s{{font-size:.9rem;color:var(--muted)}}
.tile.ours .v{{color:var(--accent)}} .tile.base .v{{color:var(--amber)}}
table{{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.95rem}} .tbl{{overflow-x:auto}}
th{{text-align:left;font-weight:600;font-size:.8rem;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--ink);padding:8px 10px 6px}}
td{{padding:8px 10px;border-bottom:1px solid var(--rule);vertical-align:top}} td.num{{font-family:"JetBrains Mono",monospace;font-variant-numeric:tabular-nums;white-space:nowrap;font-size:.9rem}}
tr.total td{{font-weight:600;border-bottom:2px solid var(--ink)}} td.muted{{color:var(--muted)}}
.note{{font-size:.9rem;color:var(--muted);max-width:68ch}}
.callout{{border-left:3px solid var(--amber);background:var(--amber-soft);padding:12px 16px;max-width:68ch;margin:1.2rem 0}}
svg .lbl{{font-family:"Source Sans 3",sans-serif;font-size:13px;fill:var(--ink)}} svg .val{{font-family:"JetBrains Mono",monospace;font-size:12px;fill:var(--ink)}}
svg .bar-ref{{fill:var(--ref)}} svg .bar-ours{{fill:var(--accent)}} svg .bar-base{{fill:var(--amber)}}
pre{{background:var(--card);border:1px solid var(--rule);padding:14px 16px;overflow-x:auto;font-family:"JetBrains Mono",monospace;font-size:.82rem;line-height:1.45}}
code{{font-family:"JetBrains Mono",monospace;font-size:.9em}}
ul{{max-width:68ch;padding-left:1.2rem}} li{{margin:.3rem 0}}
.cols{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}}
a{{color:var(--accent)}} a:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
@media (prefers-reduced-motion: reduce){{*{{animation:none!important;transition:none!important}}}}
</style>
<div class="wrap">
<div class="eyebrow">Replication report · September 2026</div>
<h1>System One, Open</h1>
<p class="lede">An open replica of TypeSafe's Jev: state in, typed calibrated decisions out, one forward pass, no decoding. Built on Modal from Gemma 3 270M and Gemma 4 E2B, evaluated on Jev's own public cases.</p>

<div class="board">
 <div class="tile ours"><div class="k">Best run · common subset</div><div class="v">{pct(*best_cs)}</div><div class="s">{html.escape(best_label)} · {best_cs[0]}/{best_cs[1]} pairs</div></div>
 <div class="tile"><div class="k">Jev · same pairs</div><div class="v">{pct(*jev_cs)}</div><div class="s">published answers · {jev_cs[0]}/{jev_cs[1]}</div></div>
 <div class="tile base"><div class="k">Majority baseline</div><div class="v">{pct(*pub['baselines_common']['per_question_majority']) if pub else '—'}</div><div class="s">per-question majority · the floor to beat</div></div>
 <div class="tile ours"><div class="k">Demo families</div><div class="v">{pct(best_synth, 1) if best_synth is not None else '—'}</div><div class="s">accuracy across 7 task families</div></div>
 <div class="tile ours"><div class="k">Latency · 6 questions</div><div class="v">{f'{best_lat:.0f} ms' if best_lat else '—'}</div><div class="s">one pass on an H100 · Jev: 70–500 ms</div></div>
</div>
{('<p class="note">Pending: ' + ', '.join(pending) + '.</p>') if pending else ''}

<h2>What Jev is, and what was rebuilt</h2>
<p>Jev takes a block of state and a list of typed questions and returns each answer with a probability distribution and a confidence value, in a single pass, with no text generation. Three primitives: <strong>Choice</strong> (pick one of up to 255 options), <strong>Score</strong> (an ordered rubric level) and <strong>Noul</strong> (a yes/no probability). Output tokens are free because there are none.</p>
<p>The replica keeps that contract exactly. The prompt is the state inside <code>&lt;state&gt;</code> tags followed by one block per question, each ending in an answer slot <code>Answer k: (</code>. The model is read only at those slots, restricted to the option-label tokens. Every upper- and lower-case letter is a single token in Gemma's vocabulary, so one pass covers 52 options; larger sets are scored in chunks with a "none of the above" slot and a final round. Training minimises cross-entropy plus a Brier term on those slot distributions, then fits one temperature on held-in test data. Confidence is the margin between the top two probabilities.</p>
<div class="cols">
 <div><h3>Training mixture</h3><ul><li>92 public decision datasets: intents and ticket routing, moderation, NLI and fact-checking, passage relevance, reading comprehension, rubric scores, pairwise judging, tool selection, agent next-action.</li><li>7 rule-generated families mirroring the demo and eval workflows, with exact teacher labels.</li><li>23 datasets held out entirely to measure generalisation to unseen question types.</li></ul></div>
 <div><h3>Evaluation</h3><ul><li>TypeSafe's public eval viewer, rebuilt: 20 cases, 372 reference pairs, reference = mean of GPT-6 Astra and Claude Fable 5.1.</li><li>Strict common subset: only the {common_n} pairs that Opus, Sol and Jev all answered, so coverage differences flatter no one.</li><li>Majority baselines reported next to every model, because the invoice workflow is dominated by "no".</li></ul></div>
</div>

<h2>Head-to-head on Jev's own cases</h2>
<p>Agreement with the frontier consensus on the strict common subset. Grey bars are published answers, teal bars are this replica, amber bars are what you get by always answering the most common reference value.</p>
<div class="tbl">{chart()}</div>
<div class="callout">Read the amber bars first. An untrained model that leans on one letter lands between the two majority baselines, so any number under the per-question majority is not evidence of understanding. The reference itself is a two-model consensus, so agreement measures closeness to the frontier average, not correctness.</div>
<h3>Per workflow, all answered pairs</h3>
<div class="tbl">{wf_table()}</div>
<h3>By question type</h3>
<div class="tbl">{kind_table()}</div>

<h2>The demo tasks</h2>
<p>Each family is generated from latent variables with a deterministic teacher, so labels are exact and the untrained baselines are honest. Accuracy over all questions in the family's held-out generation seed.</p>
<div class="tbl">{synth_table()}</div>
<h3>Doom, closed loop</h3>
<p>The model plays as the live policy in a small arena: enemies close in and attack, items restore health and ammo, the episode ends on death, a clear, or 200 ticks. Same seeds for every policy.</p>
<div class="tbl">{doom_table()}</div>

<h2>Generalisation and calibration</h2>
<p>Held-out tasks were never seen in training; they are the fair test of "one model for any typed question". Accuracy at 80% coverage is what you get by acting on the 80% most confident answers and routing the rest to a person, the pattern TypeSafe's docs recommend.</p>
<div class="tbl">{gen_table()}</div>

<h2>Latency</h2>
<div class="tbl">{latency_table()}</div>

<h2>Training runs</h2>
<div class="tbl">{train_table()}</div>
<p class="note">Everything ran on Modal. Data build on CPU containers, training and evaluation on single H100s, serving on an L4 or CPU. Gemma 4 E2B is compute-heavy for its "2.3B effective" label: all-module LoRA with gradient checkpointing trained at 8.8K tokens/s, attention-only LoRA at 13.6K tokens/s, so the E2B run used the latter on 35% of each task's cap.</p>

<h2>The API</h2>
<p>One request, mixed primitives, one forward pass. Deployed as a Modal web endpoint; the response shape is Jev's.</p>
<pre>POST /decide
{{"state": {{"conversation": [{{"speaker": "customer", "text": "Charged $29.99 AGAIN after you said it was cancelled. Refund it and cancel it for real."}}],
           "customer": {{"tenure_months": 19, "plan": "premium"}}}},
 "questions": [
  {{"id": "primary_issue", "type": "choice", "instructions": "What is the customer's primary issue?",
   "options": {{"billing_dispute": "questions a fee or a double charge", "cancel_account": "wants to cancel", "refund_request": "wants money back for a purchase they agreed to"}}}},
  {{"id": "frustration", "type": "score", "instructions": "How frustrated is the customer?", "levels": ["calm", "mildly irritated", "clearly frustrated", "angry", "hostile"]}},
  {{"id": "refund", "type": "noul", "instructions": "Does the customer ask for money back?"}}]}}

→ {{"answers": [
  {{"id": "primary_issue", "type": "choice", "choice": "billing_dispute", "probabilities": {{"billing_dispute": 0.71, "cancel_account": 0.22, "refund_request": 0.07}}, "confidence": 0.49}},
  {{"id": "frustration", "type": "score", "score": 3.1, "level": "3", "probabilities": {{"0": 0.01, "1": 0.04, "2": 0.18, "3": 0.52, "4": 0.25}}, "confidence": 0.27}},
  {{"id": "refund", "type": "noul", "noul": 0.93, "confidence": 0.86}}],
 "latency_ms": 18.4, "passes": 1}}</pre>
<p class="note">The response above is illustrative of the shape; live numbers come from the deployed model.</p>
</div>
"""
open("report.html", "w").write(page)
print("wrote report.html", len(page), "bytes; runs present:", [n for n, _ in RUNS if rep[n]], "pending:", pending)
