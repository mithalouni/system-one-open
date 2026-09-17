CSS = r"""
:root{--bg:#F5F7F9;--ink:#16202A;--muted:#5F6E7C;--rule:#D6DEE5;--accent:#0E7C86;--card:#fff;--ok:#1D8A4E;--ok-soft:#E6F5EC;--warn:#B8730A;--bad:#B3261E}
@media (prefers-color-scheme:dark){:root{--bg:#0F1519;--ink:#E6ECF1;--muted:#93A1AE;--rule:#26323B;--accent:#3FB4BE;--card:#161E24;--ok:#4CC27A;--ok-soft:#14301F;--warn:#E0A13A;--bad:#E5675F}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:"Source Sans 3",system-ui,sans-serif;font-size:15px}
.mono{font-family:"JetBrains Mono",monospace}button{font:inherit;background:var(--ink);color:var(--bg);border:0;border-radius:6px;padding:8px 14px;font-weight:600;cursor:pointer}button:disabled{opacity:.5}
select,input{font:inherit;padding:6px 10px;border:1px solid var(--rule);border-radius:6px;background:var(--card);color:var(--ink)}
"""
FONTS = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;500&display=swap">'

RACE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>System One race</title>""" + FONTS + """<style>""" + CSS + r"""
body{background:#E48AA8;padding:28px;min-height:100vh}
.win{max-width:1180px;margin:0 auto;background:#2B2226;border-radius:14px;box-shadow:0 20px 60px rgba(0,0,0,.35);overflow:hidden;color:#EAE2E6}
.bar{display:flex;align-items:center;gap:8px;padding:12px 16px;position:relative}.dot{width:12px;height:12px;border-radius:50%}.t{position:absolute;left:0;right:0;text-align:center;font-family:"JetBrains Mono",monospace;font-weight:500;letter-spacing:.02em}
pre{margin:0;padding:8px 26px 22px;font-family:"JetBrains Mono",monospace;font-size:15px;line-height:1.5;white-space:pre-wrap;color:#EAE2E6;min-height:640px}
.c{color:#F6F1F3}.g{color:#9B8F95}.k{color:#F29AB8}.n{color:#8FE0A3}.s{color:#F6F1F3}
.foot{padding:0 26px 22px;font-family:"JetBrains Mono",monospace;color:#9B8F95;display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:12px}
.tag{background:#EFE6DC;color:#2B2226;padding:6px 18px;border-radius:4px;font-weight:600}
.cmp{max-width:1180px;margin:14px auto 0;font-family:"JetBrains Mono",monospace;color:#3a1d28;font-size:13px}
</style></head><body>
<div class="win"><div class="bar"><span class="dot" style="background:#ff5f57"></span><span class="dot" style="background:#febc2e"></span><span class="dot" style="background:#28c840"></span><span class="t">SYSTEM ONE (OPEN)</span></div>
<pre id="term"></pre>
<div class="foot"><div id="stats"></div><div class="tag" id="model">gemma-4-e2b · s1</div></div></div>
<p class="cmp">Same 27 questions as TypeSafe's launch demo. Jev, published: completed in 0.114 s, cost $0.000081. This run: open weights on one GPU, cost = GPU seconds.</p>
<script>
const term=document.getElementById("term");const sleep=ms=>new Promise(r=>setTimeout(r,ms));
async function typeLine(s,cls,delay=18){const span=document.createElement("span");span.className=cls;term.appendChild(span);for(const ch of s){span.textContent+=ch;await sleep(delay)}term.appendChild(document.createTextNode("\n"))}
function esc(s){return String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}
async function run(){const auto=new URLSearchParams(location.search).get("autorun");await sleep(auto?900:200);
 await typeLine("$ s1 race ask system-one","c",40);await sleep(300);await typeLine("Sending request to parallel System One API (open replica)...","g",8);
 const t0=performance.now();const r=await fetch("race_run",{method:"POST"});const j=await r.json();const rt=performance.now()-t0;
 term.appendChild(document.createTextNode("\n"));await typeLine("{","g",1);
 for(let i=0;i<j.answers.length;i++){const a=j.answers[i];let body;
  if(a.type==="noul")body=`{<span class="s">"noul"</span>: <span class="n">${a.noul.toFixed(2)}</span>, <span class="s">"type"</span>: <span class="s">"noul"</span>}`;
  else if(a.type==="score")body=`{<span class="s">"score"</span>: <span class="n">${a.score.toFixed(2)}</span>, <span class="s">"confidence"</span>: <span class="n">${a.confidence.toFixed(2)}</span>, <span class="s">"legend"</span>: {"0": …}`;
  else body=`{<span class="s">"choice"</span>: <span class="s">"${esc(a.choice)}"</span>, <span class="s">"confidence"</span>: <span class="n">${a.confidence.toFixed(2)}</span>, <span class="s">"probabilities"</span>: {…}`;
  const line=document.createElement("span");line.innerHTML=`  <span class="k">"${esc(a.id)}"</span>: ${body}${i<j.answers.length-1?",":""}`;term.appendChild(line);term.appendChild(document.createTextNode("\n"));await sleep(22)}
 await typeLine("}","g",1);
 document.getElementById("stats").innerHTML=`cost $${j.cost_usd.toFixed(6)} (GPU time on ${esc(j.gpu)})<br>completed in ${(j.latency_ms/1000).toFixed(3)}s · round trip ${(rt/1000).toFixed(3)}s · ${j.answers.length} decisions · 1 forward pass · 0 tokens generated`;
 document.getElementById("model").textContent=j.model}
run();
</script></body></html>"""

EMAILS = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Inbox decisions</title>""" + FONTS + """<style>""" + CSS + r"""
body{padding:14px 18px}.wrap{display:grid;grid-template-columns:1fr 340px;gap:16px;max-width:1500px;margin:0 auto}
@media (max-width:900px){.wrap{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--rule);border-radius:10px}
table{width:100%;border-collapse:collapse;font-size:13px}th{position:sticky;top:0;background:var(--card);text-align:left;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);padding:10px 10px 8px;border-bottom:1px solid var(--rule)}
td{padding:6px 10px;border-bottom:1px solid var(--rule);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:0}td.sub{max-width:520px}td.n{font-family:"JetBrains Mono",monospace;font-size:12px;text-align:right}
.tbl{max-height:calc(100vh - 60px);overflow:auto}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:600;background:var(--bg);border:1px solid var(--rule)}
.pill.high{color:var(--bad);border-color:var(--bad)}.pill.normal{color:var(--warn);border-color:var(--warn)}.pill.low{color:var(--muted)}
.spam{color:var(--bad);font-weight:600}.ok{color:var(--ok)}.wrong{background:rgba(179,38,30,.12)}
.side{padding:16px;display:flex;flex-direction:column;gap:14px;position:sticky;top:14px;align-self:start}
.row{display:flex;justify-content:space-between;align-items:center;gap:8px}.lbl{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.big{font-family:"JetBrains Mono",monospace;font-size:22px;font-weight:500}.sm{font-family:"JetBrains Mono",monospace;font-size:14px}
.bar{height:8px;background:var(--rule);border-radius:4px;overflow:hidden}.bar i{display:block;height:100%;background:var(--ok);width:0;transition:width .2s}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:8px 12px}canvas{width:100%;height:56px}
h1{font-size:15px;margin:0}
</style></head><body><div class="wrap">
<div class="card tbl"><table><thead><tr><th style="width:40%">Subject</th><th>Preview</th><th>Category</th><th>Priority</th><th>Spam</th><th>Reply</th><th>Label</th><th>Time</th></tr></thead><tbody id="rows"></tbody></table></div>
<div class="card side">
 <div class="row"><h1>Run</h1><span id="status" class="lbl">Ready</span></div>
 <div class="grid2"><div><div class="lbl">Emails</div><select id="n"><option>100</option><option selected>1000</option><option>1500</option></select></div><div><div class="lbl">Batch × workers</div><select id="bw"><option value="16,2">16 × 2</option><option value="32,2" selected>32 × 2</option><option value="32,4">32 × 4</option><option value="1,8">1 × 8 (one email per call)</option></select></div></div>
 <button id="start">Start</button>
 <div><div class="row"><span class="lbl">Progress</span><span class="sm" id="prog">0 of 0</span></div><div class="bar"><i id="pbar"></i></div></div>
 <div class="grid2"><div><div class="lbl">Emails / second</div><div class="big" id="rate">0.0</div></div><div><div class="lbl">Elapsed</div><div class="big" id="elapsed">0.0 s</div></div>
 <div><div class="lbl">Avg latency / email</div><div class="sm" id="avg">–</div></div><div><div class="lbl">p95 / email</div><div class="sm" id="p95">–</div></div>
 <div><div class="lbl">Tokens in</div><div class="sm" id="tin">0</div></div><div><div class="lbl">Tokens out</div><div class="sm" id="tout">0</div></div>
 <div><div class="lbl">Spam accuracy (Enron labels)</div><div class="big ok" id="acc">–</div></div><div><div class="lbl">GPU cost so far</div><div class="sm" id="cost">$0.0000</div></div></div>
 <div><div class="lbl">Latency per batch</div><canvas id="spark" width="600" height="56"></canvas></div>
 <div class="lbl" id="foot">1,500 real emails from the Enron corpus (750 spam / 750 ham). Decisions per email: category, priority, spam, needs reply.</div>
</div></div>
<script>
const $=s=>document.querySelector(s);let EMAILS=[];let running=false;
async function init(){EMAILS=await (await fetch("emails_data")).json();$("#start").addEventListener("click",start);render(EMAILS.slice(0,parseInt($("#n").value)));$("#n").addEventListener("change",()=>render(EMAILS.slice(0,parseInt($("#n").value))));
 if(new URLSearchParams(location.search).get("autorun")){setTimeout(start,1200)}}
function esc(s){return String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}
function render(list){$("#rows").innerHTML=list.map((e,i)=>`<tr id="r${i}"><td class="sub" title="${esc(e.subject)}">${esc(e.subject)}</td><td class="sub" style="color:var(--muted)">${esc(e.text.slice(0,90))}</td><td></td><td></td><td></td><td></td><td class="lbl">${e.spam_label?"spam":"ham"}</td><td class="n"></td></tr>`).join("")}
const lat=[];function spark(){const c=$("#spark"),g=c.getContext("2d");g.clearRect(0,0,c.width,c.height);if(!lat.length)return;const m=Math.max(...lat);g.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue("--ok");g.lineWidth=2;g.beginPath();lat.forEach((v,i)=>{const x=i/(Math.max(1,lat.length-1))*c.width,y=c.height-4-(v/m)*(c.height-8);i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke()}
async function start(){if(running)return;running=true;$("#start").disabled=true;$("#status").textContent="Running";
 const N=parseInt($("#n").value);const [B,W]=$("#bw").value.split(",").map(Number);const list=EMAILS.slice(0,N);render(list);lat.length=0;
 let next=0,done=0,tin=0,tout=0,ok=0,gpu_ms=0;const per=[];const t0=performance.now();
 const tick=()=>{const el=(performance.now()-t0)/1000;$("#elapsed").textContent=el.toFixed(1)+" s";$("#rate").textContent=(done/el).toFixed(1);$("#prog").textContent=`${done.toLocaleString()} of ${N.toLocaleString()}`;$("#pbar").style.width=(100*done/N)+"%";
  if(per.length){const s=[...per].sort((a,b)=>a-b);$("#avg").textContent=(per.reduce((a,b)=>a+b,0)/per.length).toFixed(0)+" ms";$("#p95").textContent=s[Math.floor(s.length*.95)].toFixed(0)+" ms"}
  $("#tin").textContent=tin.toLocaleString();$("#tout").textContent=tout.toLocaleString();$("#acc").textContent=done?(100*ok/done).toFixed(1)+"%":"–";$("#cost").textContent="$"+(gpu_ms/1000*GPU_RATE/3600).toFixed(4);spark()};
 const GPU_RATE=window.GPU_RATE||3.95;
 async function worker(){while(next<list.length){const s=next;next+=B;const items=list.slice(s,s+B);const t1=performance.now();
   try{const r=await fetch("decide_batch",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({states:items.map(e=>`Subject: ${e.subject}\n\n${e.text}`),task:"email"})});const j=await r.json();const dt=performance.now()-t1;lat.push(dt);gpu_ms+=j.latency_ms;tin+=j.tokens_in;tout+=0;window.GPU_RATE=j.gpu_rate_per_hour||GPU_RATE;
    j.results.forEach((ans,k)=>{const e=items[k];const i=s+k;const tr=document.getElementById("r"+i);if(!tr)return;const cat=ans[0].choice,pr=ans[1].level,sp=ans[2].noul,rp=ans[3].noul;const prName=["low","normal","high"][parseInt(pr)]||pr;
     const cells=tr.children;cells[2].innerHTML=`<span class="pill">${esc(cat)}</span>`;cells[3].innerHTML=`<span class="pill ${prName}">${prName}</span>`;cells[4].innerHTML=`<span class="${sp>=0.5?"spam":""}">${(100*sp).toFixed(0)}%</span>`;cells[5].textContent=(100*rp).toFixed(0)+"%";cells[7].textContent=(dt/items.length).toFixed(0)+" ms";
     const correct=(sp>=0.5)===!!e.spam_label;ok+=correct;if(!correct)tr.classList.add("wrong");done++;per.push(dt/items.length)});
    if(done<=N){const tr=document.getElementById("r"+Math.min(list.length-1,s+B));if(tr)tr.scrollIntoView({block:"center"})}}
   catch(err){console.error(err);done+=items.length}
   tick()}}
 await Promise.all(Array.from({length:W},worker));tick();$("#status").textContent="Complete";$("#start").disabled=false;running=false}
init();
</script></body></html>"""

VIRAL = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Viral potential</title>""" + FONTS + """<style>""" + CSS + r"""
body{background:#0b0e13;color:#e8ecf1;padding:24px;font-size:15px;min-height:100vh}
.box{max-width:720px;margin:0 auto;display:flex;flex-direction:column;gap:14px}
.composer{background:#161b22;border:1px solid #2a323c;border-radius:16px;padding:18px}
.who{display:flex;gap:12px;align-items:flex-start}.av{width:40px;height:40px;border-radius:50%;background:#f2a33a;display:grid;place-items:center;font-weight:700;color:#111}
textarea{flex:1;background:transparent;border:0;color:#e8ecf1;font:inherit;font-size:20px;line-height:1.35;resize:none;min-height:96px;outline:none}
.meta{display:flex;justify-content:space-between;align-items:center;margin-top:10px;flex-wrap:wrap;gap:8px}.chips{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.chip{background:#232b36;border-radius:999px;padding:4px 12px;font-size:13px}.chip b{color:#fff}.chip.dim{color:#7d8794}
.post{background:#eef2f6;color:#111;border-radius:999px;padding:8px 18px;font-weight:700}.lat{font-family:"JetBrains Mono",monospace;font-size:12px;color:#7d8794}
.panel{background:#161b22;border:1px solid #2a323c;border-radius:16px;padding:18px}
.score{display:flex;gap:16px;align-items:center}.ring{width:64px;height:64px;border-radius:50%;display:grid;place-items:center;font-family:"JetBrains Mono",monospace;font-size:22px;font-weight:600;border:4px solid #2a323c}
.verdict{font-size:18px;font-weight:600}.sub{color:#7d8794;font-size:13px}
.bars{display:grid;grid-template-columns:1fr 1fr;gap:8px 28px;margin-top:14px}.b{display:grid;grid-template-columns:86px 1fr 34px;align-items:center;gap:10px;font-size:13px;color:#aab3bf}.b i{display:block;height:6px;border-radius:3px;background:#2a323c;overflow:hidden}.b i b{display:block;height:100%;width:0;transition:width .3s}.b span{font-family:"JetBrains Mono",monospace;text-align:right}
ul{margin:14px 0 0;padding-left:18px;color:#c9d1da;font-size:13px}li{margin:4px 0}
</style></head><body><div class="box">
<div class="composer"><div class="who"><div class="av">S</div><textarea id="t" placeholder="What's happening?" spellcheck="false"></textarea></div>
 <div class="meta"><div class="chips"><span class="chip dim">Type</span><span class="chip" id="type">Start typing</span><span class="chip dim" id="type2"></span></div><div class="chips"><span class="lat" id="lat"></span><span class="post">Post</span></div></div></div>
<div class="panel"><div class="score"><div class="ring" id="ring">–</div><div><div class="verdict" id="verdict">Viral potential</div><div class="sub" id="sub">Judged 500 ms after you stop typing</div></div></div>
 <div class="bars" id="bars"></div><ul id="tips"></ul></div>
<p class="sub" style="text-align:center">Open System One replica · gemma-4-e2b · one forward pass per judgment</p>
</div>
<script>
const $=s=>document.querySelector(s);const DIMS=[["hook","Hook"],["feeling","Feeling"],["repostable","Repostable"],["fit","Fit"],["specific","Specific"],["fresh","Fresh"],["clear","Clear"],["replies","Replies"]];
$("#bars").innerHTML=DIMS.map(([k,l])=>`<div class="b"><label>${l}</label><i><b id="b_${k}"></b></i><span id="v_${k}">–</span></div>`).join("");
const TIPS={hook:"Lead with the payoff. On X the first line is the whole post; put the result or the claim there, not the setup.",specific:"Add a number, a name, or a concrete result. Vague claims don't travel.",fresh:"This angle is familiar. What's the part only you can say, from something you actually did?",repostable:"Give readers a takeaway they'd look smart reposting: a rule, a stat, or a one-line lesson.",clear:"Shorter sentences, one idea per line.",feeling:"Raise the stakes. What here is surprising, or at odds with what people assume?",fit:"Cut the throat-clearing; start at the claim.",replies:"End on something people will want to argue with or add to."};
let timer=null,seq=0;$("#t").addEventListener("input",()=>{clearTimeout(timer);timer=setTimeout(judge,500)});
const col=v=>v>=3?"#4cc27a":(v>=2?"#f2a33a":"#e5675f");
async function judge(){const text=$("#t").value.trim();if(text.length<6)return;const my=++seq;const t0=performance.now();
 const r=await fetch("decide",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({task:"viral",state:text})});const j=await r.json();if(my!==seq)return;
 const a=Object.fromEntries(j.answers.map(x=>[x.id,x]));const ty=a.type;const probs=Object.entries(ty.probabilities).sort((x,y)=>y[1]-x[1]);
 $("#type").innerHTML=`${esc(probs[0][0].replace("_","-"))} <b>${(100*probs[0][1]).toFixed(0)}%</b>`;$("#type2").textContent=probs.slice(1,3).map(([k,v])=>`${k.replace("_","-")} ${(100*v).toFixed(0)}%`).join(" · ");
 let tot=0,conf=0;DIMS.forEach(([k])=>{const s=a[k].score;const v=Math.round(s*25);tot+=s;conf+=a[k].confidence;$("#b_"+k).style.width=v+"%";$("#b_"+k).style.background=col(s);$("#v_"+k).textContent=v});
 const score=Math.round(tot/DIMS.length*25);const cert=Math.round(100*conf/DIMS.length);const verdict=score<30?"Scroll past":(score<55?"Decent":(score<75?"Strong":"Banger"));
 $("#ring").textContent=score;$("#ring").style.borderColor=col(score/25);$("#verdict").textContent=verdict;$("#sub").textContent=`${probs[0][0].replace("_","-")} · certainty ${cert}%`;
 const weak=DIMS.map(([k])=>[k,a[k].score]).sort((x,y)=>x[1]-y[1]).filter(([k,s])=>s<2.5).slice(0,4);$("#tips").innerHTML=weak.map(([k])=>`<li>${TIPS[k]}</li>`).join("");
 $("#lat").textContent=`${j.latency_ms.toFixed(0)} ms model · ${(performance.now()-t0).toFixed(0)} ms round trip`}
function esc(s){return String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}
</script></body></html>"""
