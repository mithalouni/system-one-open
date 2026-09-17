from s1.demo_pages2 import CSS, FONTS

DRIVE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>System One drives</title>""" + FONTS + """<style>""" + CSS + r"""
body{background:#0c1116;color:#dfe6ee;padding:14px 18px;font-size:13px}
.top{display:flex;gap:14px;align-items:baseline;flex-wrap:wrap;max-width:1500px;margin:0 auto 10px}.top h1{font-size:16px;margin:0}.top .d{color:#8b98a6;max-width:900px}
.grid{display:grid;grid-template-columns:2fr 1fr;gap:12px;max-width:1500px;margin:0 auto}@media(max-width:900px){.grid{grid-template-columns:1fr}}
canvas{width:100%;background:#0f161d;border:1px solid #223040;border-radius:8px}
.card{background:#121a22;border:1px solid #223040;border-radius:8px;padding:12px}.lbl{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:#8b98a6}
.bars{display:grid;gap:5px;margin-top:8px}.b{display:grid;grid-template-columns:112px 1fr 44px;align-items:center;gap:8px}.b i{display:block;height:7px;background:#223040;border-radius:4px;overflow:hidden}.b i b{display:block;height:100%;background:#4f8df7;width:0;transition:width .15s}.b span{font-family:"JetBrains Mono",monospace;text-align:right;color:#dfe6ee}.b.win b{background:#76b7ff}
.kv{display:grid;grid-template-columns:1fr auto;gap:3px 12px;font-family:"JetBrains Mono",monospace;font-size:12px;margin-top:6px}.kv span:nth-child(odd){color:#8b98a6;font-family:"Source Sans 3",sans-serif}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;background:#223040;margin-right:6px}.pill.warn{background:#3a2d10;color:#f2a33a}.pill.bad{background:#3a1512;color:#f2665c}
.log{font-family:"JetBrains Mono",monospace;font-size:11px;color:#aab6c2;max-height:150px;overflow:auto;white-space:pre-wrap;margin-top:6px}
pre{font-family:"JetBrains Mono",monospace;font-size:10.5px;color:#8b98a6;max-height:170px;overflow:auto;margin:6px 0 0;white-space:pre-wrap}
button{background:#dfe6ee;color:#0c1116;padding:5px 12px}
</style></head><body>
<div class="top"><h1>System One drives</h1><div class="d">A self-driving car on a 5-station obstacle course. Code does perception (lidar → sectors) and safety (a reflex that always wins); the open model makes the tactical call — around, through, wait, or back out — at ~3 Hz.</div><button id="restart">Restart run</button><span class="pill" id="state">starting</span></div>
<div class="grid">
 <div><canvas id="c" width="1200" height="620"></canvas></div>
 <div style="display:grid;gap:12px">
  <div class="card"><div class="lbl">Model's last judgment · <span id="lat">–</span></div><div class="bars" id="bars"></div><div style="margin-top:8px"><span class="pill" id="risk">risk –</span><span class="pill" id="wait">wait –</span><span class="pill" id="dead">dead end –</span></div><div id="why" style="margin-top:6px;color:#aab6c2"></div></div>
  <div class="card"><div class="lbl">Run</div><div class="kv"><span>elapsed</span><span id="el">0.0 s</span><span>distance to goal</span><span id="dist">–</span><span>speed</span><span id="spd">0.0 m/s</span><span>station</span><span id="st">–</span><span>collisions</span><span id="col">0</span><span>reflex overrides</span><span id="ref">0</span><span>time stuck</span><span id="stuck">0.0 s</span></div>
   <div class="lbl" style="margin-top:10px">Model calls</div><div class="kv"><span>calls</span><span id="calls">0</span><span>reused (scene unchanged)</span><span id="reused">0</span><span>median latency</span><span id="med">–</span><span>GPU cost so far</span><span id="cost">$0.00000</span><span>tokens / call</span><span id="tok">–</span></div></div>
  <div class="card"><div class="lbl">Event log</div><div class="log" id="log"></div></div>
  <div class="card"><div class="lbl">What the model sees (the whole state, last call)</div><pre id="seen"></pre></div>
 </div></div>
<script>
const $=s=>document.querySelector(s);const ACTIONS=["hold_course","steer_left","steer_right","slow_down","stop_and_wait","reverse_and_turn"];
const DESC={hold_course:"keep heading toward the target at current speed",steer_left:"bear left to go around an obstacle",steer_right:"bear right to go around an obstacle",slow_down:"reduce speed but keep moving",stop_and_wait:"stop and wait for the gap to open or the pedestrian to pass",reverse_and_turn:"back out of a dead end and turn around"};
$("#bars").innerHTML=ACTIONS.map(a=>`<div class="b" id="b_${a}"><label>${a.replace(/_/g," ")}</label><i><b></b></i><span>0%</span></div>`).join("");
// ---------------- world
const W=1200,H=620;let world,car,t0,calls=0,reused=0,lats=[],gpu_ms=0,tokPer=0,GPU=3.95;const STATIONS=["slalom","fork / pockets","sliding gate","cluster","goal"];
function mkWorld(seed){const rnd=mulberry32(seed);const peds=[];for(let i=0;i<12;i++)peds.push({x:150+rnd()*820,y:110+rnd()*400,vx:(rnd()-.5)*.25,vy:(rnd()-.5)*.25,r:11});
 const walls=[{x:0,y:0,w:W,h:8},{x:0,y:H-8,w:W,h:8},{x:0,y:0,w:8,h:H},{x:W-8,y:0,w:8,h:H},{x:300,y:60,w:14,h:200},{x:300,y:360,w:14,h:200},{x:560,y:60,w:14,h:130},{x:560,y:440,w:14,h:120},{x:820,y:60,w:14,h:200},{x:820,y:360,w:14,h:200}];
 const gate={x:700,y:190,w:12,h:120,vy:.9,minY:60,maxY:430};const pillars=[{x:450,y:215,r:22},{x:460,y:410,r:18},{x:980,y:200,r:18},{x:980,y:420,r:18}];
 return {peds,walls,gate,pillars,goal:{x:1120,y:310,r:16},t:0}}
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function reset(){world=mkWorld(Date.now()%100000);car={x:70,y:310,a:0,v:0,col:0,ref:0,stuck:0,lastMove:0,done:false};t0=performance.now();calls=0;reused=0;lats=[];gpu_ms=0;lastKey="";$("#log").textContent="";$("#state").textContent="running · open model brain";log(`run started with System One brain`)}
function log(m){$("#log").textContent=`t=${((performance.now()-t0)/1000).toFixed(1)}s ${m}\n`+$("#log").textContent}
// lidar: 16 sectors, 200px range
function lidar(){const out=[];for(let i=0;i<16;i++){const ang=car.a+(i-8)*(Math.PI/8)+Math.PI/16;let d=200;for(let s=0;s<200;s+=4){const px=car.x+Math.cos(ang)*s,py=car.y+Math.sin(ang)*s;if(hit(px,py)){d=s;break}}out.push({sector:i,bearing_deg:Math.round((i-8)*22.5+11.25),range:d})}return out}
function hit(px,py){for(const w of world.walls)if(px>=w.x&&px<=w.x+w.w&&py>=w.y&&py<=w.y+w.h)return true;const g=world.gate;if(px>=g.x&&px<=g.x+g.w&&py>=g.y&&py<=g.y+g.h)return true;for(const p of world.pillars)if(Math.hypot(px-p.x,py-p.y)<p.r)return true;for(const p of world.peds)if(Math.hypot(px-p.x,py-p.y)<p.r)return true;return false}
function station(){return car.x<300?0:(car.x<560?1:(car.x<820?2:(car.x<1090?3:4)))}
function stateJSON(){const L=lidar();const g=world.goal;const dg=Math.hypot(g.x-car.x,g.y-car.y);const bg=((Math.atan2(g.y-car.y,g.x-car.x)-car.a)*180/Math.PI+540)%360-180;
 const peds=world.peds.map(p=>({dist:Math.round(Math.hypot(p.x-car.x,p.y-car.y)/10*10)/10,bearing_deg:Math.round(((Math.atan2(p.y-car.y,p.x-car.x)-car.a)*180/Math.PI+540)%360-180),moving:Math.hypot(p.vx,p.vy)>.1})).filter(p=>p.dist<220).sort((a,b)=>a.dist-b.dist).slice(0,5);
 const front=Math.min(...L.slice(6,10).map(l=>l.range)),left=Math.min(...L.slice(2,6).map(l=>l.range)),right=Math.min(...L.slice(10,14).map(l=>l.range)),back=Math.min(L[0].range,L[15].range);
 return {vehicle:{speed_mps:+(car.v*3).toFixed(1),heading_deg:Math.round(car.a*180/Math.PI),station:STATIONS[station()],time_stuck_s:+car.stuck.toFixed(1)},target:{distance_m:+(dg/10).toFixed(1),bearing_deg:Math.round(bg)},lidar:{front_clearance_m:+(front/10).toFixed(1),left_clearance_m:+(left/10).toFixed(1),right_clearance_m:+(right/10).toFixed(1),rear_clearance_m:+(back/10).toFixed(1),sectors:L.map(l=>({bearing_deg:l.bearing_deg,range_m:+(l.range/10).toFixed(1)}))},pedestrians_nearby:peds,sliding_gate:{distance_m:+(Math.abs(world.gate.x-car.x)/10).toFixed(1),opening_offset_m:+(((world.gate.y+world.gate.h/2)-car.y)/10).toFixed(1),moving:true}}}
let lastKey="",lastAns=null,pending=false;
async function decide(){const s=stateJSON();const key=JSON.stringify([Math.round(s.lidar.front_clearance_m*2),Math.round(s.lidar.left_clearance_m*2),Math.round(s.lidar.right_clearance_m*2),Math.round(s.target.bearing_deg/10),s.pedestrians_nearby.length,Math.min(3,Math.floor(car.stuck/1.5))]);
 if(key===lastKey&&lastAns){reused++;return lastAns}
 if(pending)return lastAns;pending=true;const t1=performance.now();
 try{const r=await fetch("decide",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({state:s,questions:[{id:"action",type:"choice",instructions:"You are the tactical brain of a small autonomous car. Given the lidar clearances, pedestrians and the bearing to the target, what should it do for the next second?",options:DESC},{id:"risk",type:"score",instructions:"How risky is the car's situation right now?",levels:["0 - clear","1 - caution","2 - hazardous"]},{id:"dead_end",type:"noul",instructions:"Is the car facing a dead end that requires backing out?"}]})});const j=await r.json();const dt=performance.now()-t1;lats.push(dt);calls++;gpu_ms+=j.latency_ms;tokPer=Math.round(JSON.stringify(s).length/3.6);
  lastAns=j;lastKey=key;$("#seen").textContent=JSON.stringify(s,null,1);$("#lat").textContent=`${dt.toFixed(0)} ms`;render_bars(j);return j}catch(e){console.error(e);return lastAns}finally{pending=false}}
function render_bars(j){const a=j.answers[0];let best=null,bp=0;for(const [k,v] of Object.entries(a.probabilities)){const el=document.getElementById("b_"+k);el.querySelector("b").style.width=(100*v)+"%";el.querySelector("span").textContent=(100*v).toFixed(0)+"%";el.classList.toggle("win",false);if(v>bp){bp=v;best=k}}document.getElementById("b_"+best).classList.add("win");
 const risk=j.answers[1].score;$("#risk").textContent=`risk ${risk.toFixed(2)} / 2`;$("#risk").className="pill"+(risk>1.2?" bad":(risk>.6?" warn":""));$("#wait").textContent=`wait ${(100*a.probabilities.stop_and_wait).toFixed(0)}%`;$("#dead").textContent=`dead end ${(100*j.answers[2].noul).toFixed(0)}%`;$("#why").textContent=DESC[best]}
// ---------------- physics + reflex
let lastDecision=0,lastAction="hold_course";
function step(dt){if(car.done)return;world.t+=dt;const g=world.gate;g.y+=g.vy;if(g.y<g.minY||g.y>g.maxY)g.vy*=-1;for(const p of world.peds){p.x+=p.vx;p.y+=p.vy;if(p.x<20||p.x>W-20)p.vx*=-1;if(p.y<20||p.y>H-20)p.vy*=-1}
 const s=stateJSON();const front=s.lidar.front_clearance_m;let act=lastAction;
 // reflex (code): imminent obstacle -> stop, always wins
 if(front<3.0&&car.v>0){act="stop_and_wait";car.ref++}
 if(car.stuck>2.0&&!car.backout){car.backout=1.4;car.ref++;log("reflex: stuck, backing out toward the open side")}
 if(car.backout>0){car.backout-=dt;act="reverse_and_turn"}
 if(act==="steer_left"&&s.lidar.left_clearance_m<1.0){act="hold_course";car.ref++}if(act==="steer_right"&&s.lidar.right_clearance_m<1.0){act="hold_course";car.ref++}
 const goalB=s.target.bearing_deg*Math.PI/180;let steer=0,target=1.6;
 if(act==="hold_course"){steer=Math.max(-.06,Math.min(.06,goalB*.5));target=0.85}else if(act==="steer_left"){steer=-.07;target=.7}else if(act==="steer_right"){steer=.07;target=.7}else if(act==="slow_down"){steer=Math.max(-.05,Math.min(.05,goalB*.5));target=.4}else if(act==="stop_and_wait"){target=0}else if(act==="reverse_and_turn"){target=-.8;steer=(s.lidar.left_clearance_m>s.lidar.right_clearance_m?-.08:.08)}
 car.v+=(target-car.v)*.12;car.a+=steer*(Math.abs(car.v)>.05?1:0)*(car.v<0?-1:1);const nx=car.x+Math.cos(car.a)*car.v*3,ny=car.y+Math.sin(car.a)*car.v*3;
 if(hit(nx+Math.cos(car.a)*14,ny+Math.sin(car.a)*14)||hit(nx,ny)){if(!car.touching){car.col++;log(`COLLISION at station ${STATIONS[station()]}`)}car.touching=true;car.v=0;car.x-=Math.cos(car.a)*2;car.y-=Math.sin(car.a)*2}else{car.x=nx;car.y=ny;car.touching=false}
 if(Math.abs(car.v)<.3)car.stuck+=dt;else car.stuck=0;
 if(Math.hypot(world.goal.x-car.x,world.goal.y-car.y)<20){car.done=true;$("#state").textContent="reached goal ✓";log(`RUN OVER: reached goal in ${((performance.now()-t0)/1000).toFixed(1)}s, ${car.col} collisions`)}
 $("#el").textContent=((performance.now()-t0)/1000).toFixed(1)+" s";$("#dist").textContent=s.target.distance_m+" m";$("#spd").textContent=(car.v*3).toFixed(1)+" m/s";$("#st").textContent=`${station()+1} ${STATIONS[station()]}`;$("#col").textContent=car.col;$("#ref").textContent=car.ref;$("#stuck").textContent=car.stuck.toFixed(1)+" s";
 $("#calls").textContent=calls;$("#reused").textContent=reused;if(lats.length){const sl=[...lats].sort((a,b)=>a-b);$("#med").textContent=sl[Math.floor(sl.length/2)].toFixed(1)+" ms"}$("#cost").textContent="$"+(gpu_ms/1000*GPU/3600).toFixed(5);$("#tok").textContent=tokPer||"–"}
async function brain(){if(car.done)return;const j=await decide();if(j){const a=j.answers[0];let best=null,bp=0;for(const [k,v] of Object.entries(a.probabilities))if(v>bp){bp=v;best=k}if(best!==lastAction){log(`model: ${best} (${(100*bp).toFixed(0)}%) risk ${j.answers[1].score.toFixed(1)} · ${lats[lats.length-1].toFixed(0)} ms`)}lastAction=best}}
function draw(){const c=$("#c"),g=c.getContext("2d");g.clearRect(0,0,W,H);g.fillStyle="#233240";for(const w of world.walls)g.fillRect(w.x,w.y,w.w,w.h);g.fillStyle="#f2a33a";g.fillRect(world.gate.x,world.gate.y,world.gate.w,world.gate.h);
 g.fillStyle="#3a4a5a";for(const p of world.pillars){g.beginPath();g.arc(p.x,p.y,p.r,0,7);g.fill()}g.fillStyle="#9aa8b6";for(const p of world.peds){g.beginPath();g.arc(p.x,p.y,p.r,0,7);g.fill()}
 g.strokeStyle="#4cc27a";g.lineWidth=3;g.beginPath();g.arc(world.goal.x,world.goal.y,world.goal.r,0,7);g.stroke();
 const L=lidar();g.strokeStyle="rgba(120,160,220,.25)";g.lineWidth=1;for(const l of L){const ang=car.a+(l.sector-8)*(Math.PI/8)+Math.PI/16;g.beginPath();g.moveTo(car.x,car.y);g.lineTo(car.x+Math.cos(ang)*l.range,car.y+Math.sin(ang)*l.range);g.stroke()}
 g.save();g.translate(car.x,car.y);g.rotate(car.a);g.fillStyle="#76b7ff";g.fillRect(-16,-9,32,18);g.fillStyle="#0c1116";g.fillRect(8,-6,6,12);g.restore()}
async function loop(){let last=performance.now();setInterval(()=>{const now=performance.now();step((now-last)/1000);last=now;draw()},50);while(true){await brain();await new Promise(r=>setTimeout(r,330))}}
$("#restart").addEventListener("click",reset);reset();loop();
</script></body></html>"""

HOME = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>System One home</title>""" + FONTS + """<style>""" + CSS + r"""
body{padding:16px 18px}.wrap{max-width:1200px;margin:0 auto;display:grid;grid-template-columns:1fr 380px;gap:14px}@media(max-width:900px){.wrap{grid-template-columns:1fr}}
h1{font-size:16px;margin:0 0 10px}.rooms{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px}
.room{background:var(--card);border:1px solid var(--rule);border-radius:10px;padding:12px;transition:background .3s}.room.hit{outline:2px solid var(--accent)}
.room h3{margin:0 0 8px;font-size:14px}.dev{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-top:1px solid var(--rule);font-size:13px}.dev .v{font-family:"JetBrains Mono",monospace;font-size:12px}
.on{color:var(--ok);font-weight:600}.off{color:var(--muted)}.lock{color:var(--bad);font-weight:600}
.side{background:var(--card);border:1px solid var(--rule);border-radius:10px;padding:14px;display:flex;flex-direction:column;gap:10px}
input{width:100%}.lbl{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.ans{display:grid;grid-template-columns:auto 1fr;gap:4px 10px;font-size:13px}.ans .k{color:var(--muted)}.ans .v{font-family:"JetBrains Mono",monospace;font-size:12px}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:600;border:1px solid var(--rule)}.pill.warn{color:var(--warn);border-color:var(--warn)}.pill.ok{color:var(--ok);border-color:var(--ok)}
.hist{font-size:12px;color:var(--muted);max-height:220px;overflow:auto}.hist div{padding:3px 0;border-top:1px solid var(--rule)}
.presets{display:flex;flex-wrap:wrap;gap:6px}.presets button{background:var(--bg);color:var(--ink);border:1px solid var(--rule);padding:4px 10px;font-weight:400;font-size:12px}
</style></head><body><div class="wrap">
<div><h1>System One home · say what you want, the model decides which device and what to do</h1><div class="rooms" id="rooms"></div></div>
<div class="side"><div class="lbl">Command</div><input id="cmd" placeholder="e.g. it's too dark in the kitchen" autocomplete="off"><div class="presets" id="presets"></div>
 <div class="lbl">Decision · <span id="lat">–</span></div><div class="ans" id="ans"><span class="k">device</span><span class="v" id="a_dev">–</span><span class="k">action</span><span class="v" id="a_act">–</span><span class="k">ambiguous</span><span class="v" id="a_amb">–</span><span class="k">needs confirmation</span><span class="v" id="a_conf">–</span><span class="k">is a home command</span><span class="v" id="a_is">–</span></div>
 <div id="verdict"></div><div class="lbl">History</div><div class="hist" id="hist"></div></div></div>
<script>
const $=s=>document.querySelector(s);
const HOME={devices:[{id:"light.living_room",type:"light",room:"living room",on:true,brightness:80},{id:"light.kitchen",type:"light",room:"kitchen",on:false,brightness:0},{id:"light.bedroom",type:"light",room:"bedroom",on:false,brightness:0},{id:"light.office",type:"light",room:"office",on:true,brightness:50},{id:"blinds.living_room",type:"blinds",room:"living room",position:"open"},{id:"blinds.bedroom",type:"blinds",room:"bedroom",position:"closed"},{id:"speaker.kitchen",type:"speaker",room:"kitchen",playing:false},{id:"speaker.living_room",type:"speaker",room:"living room",playing:true},{id:"thermostat.main",type:"thermostat",room:"hallway",temperature:19,target:20,mode:"heat"},{id:"lock.front_door",type:"lock",room:"front door",locked:true},{id:"lock.garage_door",type:"lock",room:"garage door",locked:false}],context:{time:"21:40",people_home:2,user_location:"living room"}};
const ACTIONS=["turn_on","turn_off","set_brightness","set_temperature","lock","unlock","open","close","play","pause","no_action"];
const PRESETS=["it's too dark in the kitchen","lights off in the office, I'm done for today","unlock the garage door","make it 22 degrees","close the bedroom blinds","pause the music in the living room","turn it off","what's the weather tomorrow?","dim the living room to 30%"];
$("#presets").innerHTML=PRESETS.map(p=>`<button data-p="${p.replace(/"/g,'&quot;')}">${p}</button>`).join("");
document.querySelectorAll("#presets button").forEach(b=>b.addEventListener("click",()=>{$("#cmd").value=b.dataset.p;go()}));
function render(hit){const rooms={};for(const d of HOME.devices)(rooms[d.room]=rooms[d.room]||[]).push(d);
 $("#rooms").innerHTML=Object.entries(rooms).map(([r,ds])=>`<div class="room ${ds.some(d=>d.id===hit)?"hit":""}"><h3>${r}</h3>${ds.map(d=>`<div class="dev"><span>${d.type}</span><span class="v">${dv(d)}</span></div>`).join("")}</div>`).join("")}
function dv(d){if(d.type==="light")return d.on?`<span class="on">on · ${d.brightness}%</span>`:`<span class="off">off</span>`;if(d.type==="blinds")return d.position;if(d.type==="speaker")return d.playing?`<span class="on">playing</span>`:`<span class="off">paused</span>`;if(d.type==="thermostat")return `${d.temperature}° → ${d.target}° ${d.mode}`;if(d.type==="lock")return d.locked?`<span class="lock">locked</span>`:`<span class="on">unlocked</span>`;return ""}
function apply(devId,action,cmd){const d=HOME.devices.find(x=>x.id===devId);if(!d)return "no device";const m=cmd.match(/(\d{2})\s*(%|percent|degrees|°)?/);const num=m?parseInt(m[1]):null;
 switch(action){case "turn_on":if(d.type==="light"){d.on=true;d.brightness=d.brightness||80}else if(d.type==="speaker")d.playing=true;break;case "turn_off":if(d.type==="light"){d.on=false}else if(d.type==="speaker")d.playing=false;break;case "set_brightness":d.on=true;d.brightness=num??50;break;case "set_temperature":d.target=num??21;break;case "lock":d.locked=true;break;case "unlock":d.locked=false;break;case "open":d.position="open";break;case "close":d.position="closed";break;case "play":d.playing=true;break;case "pause":d.playing=false;break;default:return "nothing to do"}return "applied"}
$("#cmd").addEventListener("keydown",e=>{if(e.key==="Enter")go()});
async function go(){const cmd=$("#cmd").value.trim();if(!cmd)return;const state={devices:HOME.devices,context:HOME.context,user_request:cmd};const t0=performance.now();
 const r=await fetch("decide",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({state,questions:[{id:"device",type:"choice",instructions:"Which device does the request refer to?",options:[...HOME.devices.map(d=>d.id),"none (no device involved)"]},{id:"action",type:"choice",instructions:"What action should the home controller take?",options:ACTIONS},{id:"ambiguous",type:"noul",instructions:"Is the request ambiguous about which device or room is meant?",criteria:{"true":"the request could refer to more than one device, or gives no room or device","false":"the target device and action are clear from the request and the device list"}},{id:"confirm",type:"noul",instructions:"Should the controller ask for confirmation before acting?",criteria:{"true":"security-sensitive (unlocking) or ambiguous, so confirm first","false":"safe, reversible action that can be executed right away"}},{id:"is_command",type:"noul",instructions:"Is the request a home-control command at all?"}]})});
 const j=await r.json();const a=Object.fromEntries(j.answers.map(x=>[x.id,x]));const dev=a.device.choice,act=a.action.choice;
 $("#lat").textContent=`${j.latency_ms.toFixed(0)} ms model · ${(performance.now()-t0).toFixed(0)} ms round trip`;$("#a_dev").textContent=`${dev} (${(100*a.device.probabilities[dev]).toFixed(0)}%)`;$("#a_act").textContent=`${act} (${(100*a.action.probabilities[act]).toFixed(0)}%)`;$("#a_amb").textContent=(100*a.ambiguous.noul).toFixed(0)+"%";$("#a_conf").textContent=(100*a.confirm.noul).toFixed(0)+"%";$("#a_is").textContent=(100*a.is_command.noul).toFixed(0)+"%";
 let verdict,cls;if(a.is_command.noul<0.5){verdict="Not a home command — nothing changed.";cls="pill"}else if(a.ambiguous.noul>=0.5){verdict="Ambiguous — asking which device you mean.";cls="pill warn"}else if(a.confirm.noul>=0.5){verdict=`Needs confirmation before ${act.replace("_"," ")} on ${dev}.`;cls="pill warn"}else{const res=apply(dev,act,cmd);verdict=`Done: ${act.replace("_"," ")} → ${dev} (${res})`;cls="pill ok"}
 $("#verdict").innerHTML=`<span class="${cls}">${verdict}</span>`;render(dev);$("#hist").insertAdjacentHTML("afterbegin",`<div>"${cmd}" → ${dev} · ${act} · ${verdict.split(" ")[0]}</div>`)}
render();
</script></body></html>"""
