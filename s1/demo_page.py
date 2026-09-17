PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>System One decisions</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
:root{--bg:#F5F7F9;--ink:#16202A;--muted:#5F6E7C;--rule:#D6DEE5;--accent:#0E7C86;--accent-soft:#E3F3F4;--card:#fff;--ok:#1D8A4E;--ok-soft:#E6F5EC}
@media (prefers-color-scheme:dark){:root{--bg:#0F1519;--ink:#E6ECF1;--muted:#93A1AE;--rule:#26323B;--accent:#3FB4BE;--accent-soft:#143338;--card:#161E24;--ok:#4CC27A;--ok-soft:#14301F}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:"Source Sans 3",system-ui,sans-serif;font-size:16px;padding:18px 20px 40px}
.top{display:flex;flex-wrap:wrap;gap:10px;align-items:center;max-width:1100px;margin:0 auto 14px}
select,button,textarea{font:inherit}select{padding:8px 12px;border:1px solid var(--rule);border-radius:6px;background:var(--card);color:var(--ink);min-width:320px;max-width:100%}
button{background:var(--ink);color:var(--bg);border:0;border-radius:6px;padding:9px 16px;font-weight:600;cursor:pointer}button:disabled{opacity:.5;cursor:default}button:focus-visible,select:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.banner{max-width:1100px;margin:0 auto 14px;background:var(--ok-soft);border:1px solid var(--ok);border-radius:8px;padding:10px 16px;font-family:"JetBrains Mono",monospace;text-align:center;color:var(--ok);font-weight:500}
.banner[hidden]{display:none}
.panel{max-width:1100px;margin:0 auto;background:var(--card);border:1px solid var(--rule);border-radius:10px;overflow:hidden}
.head{display:flex;justify-content:space-between;align-items:center;padding:14px 20px;border-bottom:1px solid var(--rule);gap:10px;flex-wrap:wrap}
.head h1{font-size:1.15rem;margin:0;font-weight:600}.badge{font-family:"JetBrains Mono",monospace;font-size:1.05rem;padding:6px 12px;border-radius:6px;border:1px solid var(--rule);background:var(--bg)}
.badge.ok{border-color:var(--ok);color:var(--ok);background:var(--ok-soft)}
pre{margin:0;padding:18px 20px;font-family:"JetBrains Mono",monospace;font-size:.86rem;line-height:1.55;overflow-x:auto;min-height:200px;white-space:pre}
.k{color:var(--accent)}.v{color:var(--ink)}.p{color:var(--muted)}.hi{color:var(--ok)}.lo{color:#B8730A}
.muted{color:var(--muted)}details{max-width:1100px;margin:14px auto 0}summary{cursor:pointer;color:var(--muted)}
textarea{width:100%;min-height:180px;margin-top:8px;padding:10px;border:1px solid var(--rule);border-radius:6px;background:var(--card);color:var(--ink);font-family:"JetBrains Mono",monospace;font-size:.82rem}
.foot{max-width:1100px;margin:16px auto 0;color:var(--muted);font-size:.85rem}
</style></head><body>
<div class="top">
 <select id="preset" aria-label="Scenario"></select>
 <button id="run">⚡ Run decisions</button>
 <span class="muted" id="meta"></span>
</div>
<div class="banner" id="banner" hidden></div>
<div class="panel">
 <div class="head"><h1>Typed decisions · <span id="model"></span></h1><span class="badge" id="lat">0.0 ms</span></div>
 <pre id="out"><span class="muted">Pick a scenario and click "Run decisions"…</span></pre>
</div>
<details><summary>Input state (editable) and schema</summary>
 <textarea id="ctx" aria-label="Input state"></textarea>
 <pre id="schema" style="min-height:0;padding:10px 0"></pre>
</details>
<p class="foot">One forward pass per request: every field is read from its own answer slot, no tokens are generated. <code>prob</code> is the model's probability for the chosen value after temperature scaling. Scenarios adapted from the jev-on-a-laptop study (MIT).</p>
<script>
const $=s=>document.querySelector(s);let PRESETS=[];let MODEL="";
async function init(){const r=await fetch("info");const j=await r.json();MODEL=j.model;$("#model").textContent=j.model;PRESETS=j.presets;
 for(const p of PRESETS){const o=document.createElement("option");o.value=p.id;o.textContent=p.title;$("#preset").appendChild(o)}
 select();$("#preset").addEventListener("change",select);$("#run").addEventListener("click",run)}
function cur(){return PRESETS.find(p=>p.id===$("#preset").value)}
function select(){const p=cur();$("#ctx").value=p.context;$("#schema").textContent=JSON.stringify(p.schema,null,1);$("#meta").textContent=Object.keys(p.schema).length+" fields";$("#banner").hidden=true;$("#lat").textContent="0.0 ms";$("#lat").className="badge";$("#out").innerHTML='<span class="muted">Ready.</span>'}
function esc(s){return String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}
async function run(){const p=cur();$("#run").disabled=true;$("#out").innerHTML='<span class="muted">Deciding…</span>';const t0=performance.now();
 try{const r=await fetch("decide",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({context:$("#ctx").value,schema:p.schema})});const j=await r.json();const wall=performance.now()-t0;
  const lines=["{"];const ans=j.answers;ans.forEach((a,i)=>{let val,prob;if(a.type==="noul"){val=a.noul>=0.5;prob=val?a.noul:1-a.noul}else if(a.type==="score"){val=a.level;prob=Math.max(...Object.values(a.probabilities))}else{val=a.choice;prob=a.probabilities[a.choice]}
   const vs=typeof val==="boolean"?val:JSON.stringify(val);const cls=prob>=0.8?"hi":(prob<0.55?"lo":"p");lines.push(`  <span class="k">"${esc(a.id)}"</span>: { "value": <span class="v">${esc(vs)}</span>, "prob": <span class="${cls}">${prob.toFixed(4)}</span> }${i<ans.length-1?",":""}`)});lines.push("}");
  $("#out").innerHTML=lines.join("\n");$("#lat").textContent=j.latency_ms.toFixed(1)+" ms";$("#lat").className="badge ok";
  $("#banner").hidden=false;$("#banner").textContent=`${ans.length} typed decisions · ${j.latency_ms.toFixed(1)} ms model time · ${wall.toFixed(0)} ms round trip · ${j.passes} forward pass${j.passes>1?"es":""} · 0 tokens generated`}
 catch(e){$("#out").innerHTML='<span class="lo">Request failed: '+esc(e)+'</span>'}
 $("#run").disabled=false}
init();
</script></body></html>"""
