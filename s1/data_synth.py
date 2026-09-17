"""Rule-based synthetic generators for the Jev demo task families. No LLM needed: every state is generated from
latent variables and the labels are computed by a deterministic teacher over those variables, so labels are exact.

Families (each -> multi-question examples with JSON states, option descriptions where Jev would have them):
  doom        : first-person shooter game state -> action / danger / threat level / target   (the launch demo)
  smart_home  : device states + user request -> device, action, needs-confirmation, ambiguity
  support     : customer conversation + account -> primary issue, frustration, refund asked, unauthorized charge, next action
  security    : SIEM alert + joined context -> true positive, evidence strength, scope, attack type, contained
  invoice     : AP packet (invoice, PO, contract, receipts, prior invoices) -> matches, price basis, duplicate, action
  agent_trace : LLM-agent run (instructions, tools, messages, tool calls) -> goal achieved, first bad step, satisfaction, unbacked claim
  catalog     : two product records -> same product?  (entity alignment)
"""
from __future__ import annotations
import json, math, random, zlib
from .schema import Example, Q

Y = ["no", "yes"]


def _j(d):
    return json.dumps(d, ensure_ascii=False, indent=1)


# ------------------------------------------------------------------------------------------ doom
DOOM_ACTIONS = ["move_forward", "move_backward", "strafe_left", "strafe_right", "turn_left", "turn_right", "attack", "use", "pick_up"]
ENEMY_TYPES = {"imp": (60, 2), "zombieman": (20, 1), "demon": (150, 3), "cacodemon": (400, 4), "shotgun_guy": (30, 2), "baron": (1000, 5)}
ITEMS = ["medkit", "stimpack", "ammo_clip", "shells", "armor", "key_red", "key_blue"]
WEAPONS = ["pistol", "shotgun", "chaingun", "rocket_launcher"]


def doom_state(rng):
    px, py = rng.uniform(0, 100), rng.uniform(0, 100); ang = rng.choice(range(0, 360, 15))
    health = rng.choice([100, 90, 75, 60, 45, 30, 20, 12, 8]); armor = rng.choice([0, 0, 25, 50, 100])
    weapon = rng.choice(WEAPONS); ammo = rng.choice([0, 0, 3, 8, 15, 30, 50, 100])
    n_en = rng.choice([0, 0, 1, 1, 1, 2, 2, 3, 4]); enemies = []
    for i in range(n_en):
        t = rng.choice(list(ENEMY_TYPES)); dist = round(rng.uniform(2, 60), 1); rel = rng.choice(range(-180, 180, 5))
        enemies.append({"id": f"E{i + 1}", "type": t, "distance": dist, "relative_angle": rel, "health": rng.choice([ENEMY_TYPES[t][0], ENEMY_TYPES[t][0] // 2, 10]),
                        "visible": bool(abs(rel) <= 45 and rng.random() < 0.9), "attacking": bool(dist < 25 and rng.random() < 0.6)})
    n_it = rng.choice([0, 1, 1, 2]); items = []
    for i in range(n_it):
        items.append({"type": rng.choice(ITEMS), "distance": round(rng.uniform(1, 40), 1), "relative_angle": rng.choice(range(-180, 180, 5))})
    door = rng.random() < 0.15
    state = {"player": {"x": round(px, 1), "y": round(py, 1), "facing_deg": ang, "health": health, "armor": armor, "weapon": weapon, "ammo": ammo, "keys": rng.sample(["red", "blue"], rng.choice([0, 0, 1]))},
             "enemies": enemies, "items": items, "environment": {"door_ahead": door, "door_locked": bool(door and rng.random() < 0.5), "wall_ahead_distance": round(rng.uniform(1, 30), 1), "level": rng.choice(["E1M1", "E1M2", "E2M3", "MAP07"])},
             "last_action": rng.choice(DOOM_ACTIONS + [None]), "tick": rng.randint(0, 5000)}
    return state


def doom_teacher(s):
    p = s["player"]; en = s["enemies"]; items = s["items"]; env = s["environment"]
    visible = [e for e in en if e["visible"]]
    near = [e for e in en if e["distance"] < 15]
    threat = 0
    for e in en:
        w = ENEMY_TYPES[e["type"]][1]
        threat += w * (2 if e["distance"] < 15 else 1) * (1.5 if e["attacking"] else 1)
    if p["health"] < 30:
        threat *= 1.5
    threat_level = 0 if threat == 0 else (1 if threat < 4 else (2 if threat < 10 else 3))
    danger = bool(near and (p["health"] < 40 or any(e["attacking"] for e in near)))
    low_hp = p["health"] < 35
    med = [i for i in items if i["type"] in ("medkit", "stimpack")]
    ammo_items = [i for i in items if i["type"] in ("ammo_clip", "shells")]
    # target: nearest visible enemy with lowest (distance / weight) -> prefer nearest attacking
    target = None
    if visible:
        target = sorted(visible, key=lambda e: (not e["attacking"], e["distance"]))[0]["id"]
    # action
    if p["ammo"] == 0 and ammo_items:
        a = "pick_up" if min(i["distance"] for i in ammo_items) < 3 else ("move_forward" if abs(min(ammo_items, key=lambda i: i["distance"])["relative_angle"]) <= 20 else ("turn_left" if min(ammo_items, key=lambda i: i["distance"])["relative_angle"] < 0 else "turn_right"))
    elif low_hp and med and (not near or p["ammo"] == 0):
        m = min(med, key=lambda i: i["distance"])
        a = "pick_up" if m["distance"] < 3 else ("move_forward" if abs(m["relative_angle"]) <= 20 else ("turn_left" if m["relative_angle"] < 0 else "turn_right"))
    elif low_hp and near and p["ammo"] == 0:
        a = "move_backward"
    elif visible and p["ammo"] > 0:
        t = [e for e in visible if e["id"] == target][0]
        a = "attack" if abs(t["relative_angle"]) <= 10 else ("turn_left" if t["relative_angle"] < 0 else "turn_right")
    elif en and not visible:
        e = min(en, key=lambda e: e["distance"])
        a = "turn_left" if e["relative_angle"] < 0 else "turn_right"
    elif env["door_ahead"] and env["wall_ahead_distance"] < 3:
        a = "use" if not env["door_locked"] or (("red" in p["keys"]) or ("blue" in p["keys"])) else "turn_left"
    elif items:
        i = min(items, key=lambda i: i["distance"])
        a = "pick_up" if i["distance"] < 3 else ("move_forward" if abs(i["relative_angle"]) <= 20 else ("turn_left" if i["relative_angle"] < 0 else "turn_right"))
    elif env["wall_ahead_distance"] < 2:
        a = "turn_left"
    else:
        a = "move_forward"
    retreat = bool(near and (p["health"] < 30 or p["ammo"] == 0))
    return dict(action=a, danger=danger, threat=threat_level, target=target, retreat=retreat, low_hp=low_hp)


ACTION_DESC = {"move_forward": "walk in the facing direction", "move_backward": "back away from the facing direction", "strafe_left": "side-step left", "strafe_right": "side-step right",
               "turn_left": "rotate counter-clockwise", "turn_right": "rotate clockwise", "attack": "fire the current weapon at whatever is in front", "use": "interact with the door or switch ahead", "pick_up": "grab the item within reach"}


def gen_doom(rng, n):
    out = []
    for _ in range(n):
        s = doom_state(rng); lab = doom_teacher(s)
        qs = [Q("What should the player do this tick?", DOOM_ACTIONS, DOOM_ACTIONS.index(lab["action"]), descs=[ACTION_DESC[a] for a in DOOM_ACTIONS]),
              Q("Is the player in immediate danger?", Y, int(lab["danger"]), kind="noul", descs=["no enemy close enough to hurt the player right now, or the player can absorb it", "an enemy within striking range while the player is hurt or under attack"]),
              Q("How threatening is the current situation?", ["0", "1", "2", "3"], lab["threat"], kind="score", descs=["no enemies around", "a weak enemy or one far away", "several enemies or a strong one nearby", "overwhelming: strong enemies close and attacking"]),
              Q("Should the player retreat rather than engage?", Y, int(lab["retreat"]), kind="noul")]
        if s["enemies"]:
            ids = [e["id"] for e in s["enemies"]] + ["none (no enemy is visible)"]
            g = ids.index(lab["target"]) if lab["target"] else len(ids) - 1
            qs.append(Q("Which enemy should the player target first?", ids, g, descs=[f"{e['type']} at distance {e['distance']}" for e in s["enemies"]] + ["no visible enemy to target"]))
        qs.append(Q("Is the player's health critically low?", Y, int(lab["low_hp"]), kind="noul"))
        rng.shuffle(qs)
        out.append(Example(_j(s), qs, "doom"))
    return out


# ------------------------------------------------------------------------------------------ smart home
ROOMS = ["living room", "kitchen", "bedroom", "office", "hallway", "garage", "bathroom"]
DEVICE_ACTIONS = ["turn_on", "turn_off", "set_brightness", "set_temperature", "lock", "unlock", "open", "close", "play", "pause", "no_action"]
REQ_TEMPLATES = {
    ("light", "turn_on"): ["turn on the {room} light", "lights on in the {room}", "it's dark in the {room}", "can you switch the {room} lamp on", "I need light in the {room}"],
    ("light", "turn_off"): ["turn off the {room} light", "kill the lights in the {room}", "{room} lights off please", "I'm leaving the {room}, lights off"],
    ("light", "set_brightness"): ["dim the {room} light to {val}%", "set {room} brightness to {val}", "make the {room} light {val} percent", "brighten the {room} to {val}%"],
    ("thermostat", "set_temperature"): ["set the thermostat to {val} degrees", "make it {val} degrees", "it's too cold, set {val}", "I'm hot, drop the temperature to {val}"],
    ("lock", "lock"): ["lock the {room} door", "make sure the {room} door is locked", "lock up the {room}"],
    ("lock", "unlock"): ["unlock the {room} door", "open the {room} door lock", "let me in through the {room} door"],
    ("blinds", "open"): ["open the {room} blinds", "let some light into the {room}", "raise the {room} blinds"],
    ("blinds", "close"): ["close the {room} blinds", "shut the {room} blinds", "privacy in the {room} please"],
    ("speaker", "play"): ["play some music in the {room}", "start the {room} speaker", "put on jazz in the {room}"],
    ("speaker", "pause"): ["pause the music in the {room}", "stop the {room} speaker", "quiet in the {room}"],
    ("none", "no_action"): ["what's the weather like tomorrow", "remind me to call mom", "tell me a joke", "how many devices are online", "thanks, that's all"],
}
AMBIG = ["turn it off", "make it warmer in here", "lock the door", "lights", "open it", "turn that on"]


def gen_smart_home(rng, n):
    out = []
    for _ in range(n):
        devices = []
        rooms = rng.sample(ROOMS, rng.randint(3, 6))
        for r in rooms:
            devices.append({"id": f"light.{r.replace(' ', '_')}", "type": "light", "room": r, "on": rng.random() < 0.5, "brightness": rng.choice([0, 20, 50, 80, 100])})
            if rng.random() < 0.4:
                devices.append({"id": f"blinds.{r.replace(' ', '_')}", "type": "blinds", "room": r, "position": rng.choice(["open", "closed"])})
            if rng.random() < 0.3:
                devices.append({"id": f"speaker.{r.replace(' ', '_')}", "type": "speaker", "room": r, "playing": rng.random() < 0.3})
        devices.append({"id": "thermostat.main", "type": "thermostat", "room": "hallway", "temperature": rng.choice([17, 19, 21, 23, 25]), "target": rng.choice([19, 20, 21, 22]), "mode": rng.choice(["heat", "cool", "auto"])})
        for r in rng.sample(["front", "garage", "back"], rng.randint(1, 2)):
            devices.append({"id": f"lock.{r}_door", "type": "lock", "room": r + " door", "locked": rng.random() < 0.6})
        ambiguous = rng.random() < 0.12
        if ambiguous:
            req = rng.choice(AMBIG); dev = None; action = "no_action"; needs_conf = True
        else:
            key = rng.choice(list(REQ_TEMPLATES)); dtype, action = key
            cands = [d for d in devices if d["type"] == dtype]
            if dtype == "none":
                dev = None; room = rng.choice(rooms); val = 0
            elif not cands:
                continue
            else:
                dev = rng.choice(cands); room = dev["room"].replace(" door", ""); val = rng.choice([10, 25, 40, 60, 75, 90]) if action == "set_brightness" else rng.choice([18, 19, 20, 21, 22, 24])
            req = rng.choice(REQ_TEMPLATES[key]).format(room=room, val=val)
            needs_conf = action in ("unlock",) or (dtype == "lock" and action == "unlock")
        # occupancy / time context
        ctx = {"time": f"{rng.randint(0, 23):02d}:{rng.choice([0, 15, 30, 45]):02d}", "people_home": rng.randint(0, 3), "user_location": rng.choice(rooms + ["away"])}
        state = {"devices": devices, "context": ctx, "user_request": req}
        dev_opts = [d["id"] for d in devices] + ["none (no device involved)"]
        g_dev = dev_opts.index(dev["id"]) if dev else len(dev_opts) - 1
        qs = [Q("Which device does the request refer to?", dev_opts, g_dev),
              Q("What action should the home controller take?", DEVICE_ACTIONS, DEVICE_ACTIONS.index(action)),
              Q("Is the request ambiguous about which device or room is meant?", Y, int(ambiguous), kind="noul", descs=["the target device and action are clear from the request and the device list", "the request could refer to more than one device, or gives no room or device"]),
              Q("Should the controller ask for confirmation before acting?", Y, int(needs_conf), kind="noul", descs=["safe, reversible action that can be executed right away", "security-sensitive (unlocking) or ambiguous, so confirm first"]),
              Q("Is the request a home-control command at all?", Y, int(action != "no_action" or ambiguous), kind="noul")]
        rng.shuffle(qs)
        out.append(Example(_j(state), qs, "smart_home"))
    return out


# ------------------------------------------------------------------------------------------ support
ISSUES = {
    "unauthorized_charge": ("The customer reports a charge, withdrawal, or login they say they did not make or authorize.", ["I see a charge of ${amt} on my card that I never made.", "Someone charged ${amt} to my account, this wasn't me.", "There is a ${amt} withdrawal I did not authorize."]),
    "card_declined": ("The customer's card or payment is being declined or not working.", ["My card keeps getting declined at checkout.", "Payment failed three times today, the card is fine.", "Why is my card being rejected everywhere?"]),
    "refund_request": ("The customer wants money back for a purchase or service they did receive or agreed to.", ["I want a refund for the ${amt} order, the product is useless.", "Please refund my ${amt} purchase from last week.", "Can I get my ${amt} back? I no longer need the subscription."]),
    "billing_dispute": ("The customer questions a fee, a double charge, or a wrong amount they recognize but consider incorrect.", ["I was charged ${amt} twice for the same order.", "My bill says ${amt} but the plan is supposed to be cheaper.", "There is a fee of ${amt} I don't understand on my statement."]),
    "account_access": ("The customer cannot log in, is locked out, or needs to reset credentials.", ["I can't log in, the reset link never arrives.", "Locked out of my account after too many attempts.", "My password doesn't work anymore and 2FA codes fail."]),
    "cancel_account": ("The customer wants to cancel, close, or downgrade their account or subscription.", ["Please cancel my subscription effective today.", "I want to close my account.", "Downgrade me to the free plan."]),
    "delivery_issue": ("The customer's order has not arrived, is late, or arrived damaged.", ["My order was supposed to arrive Monday and it's still not here.", "The package arrived crushed and the item is broken.", "Tracking says delivered but I got nothing."]),
    "product_question": ("The customer asks how a product or feature works.", ["Does the premium plan include API access?", "How do I export my data?", "Is the device compatible with Android?"]),
    "technical_problem": ("Something in the product is broken or erroring.", ["The app crashes every time I open the reports tab.", "Sync has been failing since the update.", "I get error 502 when uploading files."]),
    "other": ("None of the listed issues.", ["Just wanted to say thanks for the quick help last time.", "Do you have a job opening in support?", "What are your office hours?"]),
}
FRUST = [("Calm and neutral; no sign of irritation.", ["", " Thanks in advance.", " Let me know."]),
         ("Mildly irritated; a hint of impatience but polite.", [" This is the second time I'm writing.", " I'd appreciate a quick answer.", " Kind of annoying honestly."]),
         ("Clearly frustrated; complains directly.", [" This is really frustrating.", " I've been waiting for days and nobody helps.", " Your service has been terrible about this."]),
         ("Angry; strong language, demands, distrust.", [" This is unacceptable, fix it NOW.", " I'm done being patient, I want this resolved today or I'm filing a complaint.", " You people keep lying to me."]),
         ("Hostile or abusive; insults or threats.", [" You are all useless idiots.", " I will make sure everyone knows what a scam this is, you crooks.", " Absolute garbage company, pathetic."])]
NEXT = ["issue_refund", "escalate_to_fraud_team", "verify_identity", "send_password_reset", "check_order_tracking", "explain_charge", "cancel_subscription", "apologize_and_troubleshoot", "answer_question", "close_conversation"]
NEXT_DESC = {"issue_refund": "process a refund for an eligible purchase", "escalate_to_fraud_team": "hand the case to the fraud/unauthorized-activity team", "verify_identity": "confirm the customer's identity before any account change", "send_password_reset": "send a credential reset flow", "check_order_tracking": "look up the shipment and give status", "explain_charge": "walk through the charge or fee on the statement", "cancel_subscription": "process the cancellation or downgrade", "apologize_and_troubleshoot": "acknowledge the bug and start troubleshooting steps", "answer_question": "answer the product question directly", "close_conversation": "nothing to do; wrap up politely"}


def gen_support(rng, n):
    out = []
    for _ in range(n):
        issue = rng.choice(list(ISSUES)); amt = rng.choice([9.99, 14.5, 29.99, 49, 120, 250, 899])
        fl = rng.choice([0, 0, 1, 1, 2, 2, 3, 4])
        msgs = []
        first = rng.choice(ISSUES[issue][1]).replace("{amt}", str(amt)) + rng.choice(FRUST[fl][1])
        msgs.append({"speaker": "customer", "text": first})
        if rng.random() < 0.6:
            msgs.append({"speaker": "assistant", "text": rng.choice(["I'm sorry to hear that. Let me look into it.", "Thanks for reaching out, can you share the order number?", "I understand. Let me check your account."])})
            follow = {"unauthorized_charge": "I never bought anything from that merchant.", "card_declined": "I have plenty of balance, it's not that.", "refund_request": "I just want my money back.", "billing_dispute": "One of the two charges needs to be reversed.", "account_access": "I tried the reset already.", "cancel_account": "I don't need a retention offer, just cancel.", "delivery_issue": "Order 48213, placed nine days ago.", "product_question": "Mostly I need the API for automation.", "technical_problem": "Happens on both my phone and laptop.", "other": "No worries if not."}[issue]
            msgs.append({"speaker": "customer", "text": follow + rng.choice(FRUST[fl][1])})
        tenure = rng.randint(0, 80); prior = rng.randint(0, 6); plan = rng.choice(["free", "basic", "premium"])
        refund_asked = issue in ("refund_request", "billing_dispute") or (issue == "unauthorized_charge" and rng.random() < 0.5 and "money back" in first)
        unauthorized = issue == "unauthorized_charge"
        nxt = {"unauthorized_charge": "escalate_to_fraud_team", "card_declined": "verify_identity", "refund_request": "issue_refund" if tenure >= 1 and plan != "free" else "explain_charge", "billing_dispute": "issue_refund" if "twice" in first else "explain_charge", "account_access": "send_password_reset", "cancel_account": "cancel_subscription", "delivery_issue": "check_order_tracking", "product_question": "answer_question", "technical_problem": "apologize_and_troubleshoot", "other": "close_conversation"}[issue]
        churn = 1 if issue == "cancel_account" or fl >= 3 else 0
        state = {"conversation": msgs, "customer": {"tenure_months": tenure, "plan": plan, "prior_tickets_90d": prior, "recent_charges": [{"amount": amt, "merchant": rng.choice(["ACME Store", "Streamly", "CloudBox", "unknown"]), "days_ago": rng.randint(0, 20)}] if issue in ("unauthorized_charge", "refund_request", "billing_dispute") else []}}
        names = list(ISSUES)
        qs = [Q("What is the customer's primary issue in this conversation, judged by the outcome they want above all else?", names, names.index(issue), descs=[ISSUES[k][0] for k in names]),
              Q("How frustrated is the customer in their latest message, in the context of the conversation?", ["0", "1", "2", "3", "4"], fl, kind="score", descs=[f[0] for f in FRUST]),
              Q("Does the customer ask for money back?", Y, int(refund_asked), kind="noul", descs=["no refund, credit, or reversal is requested", "the customer explicitly wants a refund, credit, or reversal"]),
              Q("Does the customer report a charge, withdrawal, or login that they say they did not make or authorize?", Y, int(unauthorized), kind="noul"),
              Q("What is the best next action for the support agent?", NEXT, NEXT.index(nxt), descs=[NEXT_DESC[k] for k in NEXT]),
              Q("Is there a material risk the customer churns after this conversation?", Y, churn, kind="noul")]
        rng.shuffle(qs)
        out.append(Example(_j(state), qs, "support"))
    return out


# ------------------------------------------------------------------------------------------ security
DETECTORS = [("T1003.001", "OS Credential Dumping: LSASS Memory", "credential_theft"), ("T1021.006", "Remote Services: WinRM", "lateral_movement"), ("T1040", "Network Sniffing", "reconnaissance"),
             ("T1546.018", "Event Triggered Execution: WMI Subscription", "persistence"), ("T1566.001", "Phishing: Spearphishing Attachment", "phishing"), ("T1078", "Valid Accounts", "account_compromise"),
             ("T1486", "Data Encrypted for Impact", "ransomware"), ("T1048", "Exfiltration Over Alternative Protocol", "exfiltration"), ("T1059.001", "PowerShell", "malware_execution")]
ATTACK_TYPES = ["credential_theft", "lateral_movement", "reconnaissance", "persistence", "phishing", "account_compromise", "ransomware", "exfiltration", "malware_execution", "benign_activity"]
SCOPE = ["single_entity", "workgroup", "organization_wide"]


def gen_security(rng, n):
    out = []
    for _ in range(n):
        tid, name, atype = rng.choice(DETECTORS)
        host = f"{rng.choice(['WKS', 'SRV', 'LAP', 'DC'])}-{rng.randint(1000, 9999)}"; user = f"{rng.choice('abcdefghjkmnprstvz')}.{rng.choice(['rasmussen', 'okafor', 'lindqvist', 'tanaka', 'moreau', 'petrov', 'silva', 'nguyen'])}"
        benign = rng.random() < 0.45
        env = rng.choice(["prod", "prod", "dev", "test"]); tier = rng.choice([0, 1, 2, 3]); is_admin = rng.random() < 0.3
        change = {"ticket": f"CHG-{rng.randint(10000, 99999)}", "window_covers_event": benign and rng.random() < 0.7, "requested_by": user if benign and rng.random() < 0.6 else f"x.{rng.choice(['admin', 'ops'])}", "description": rng.choice(["scheduled EDR agent upgrade", "packet capture for latency investigation", "WMI monitoring rollout", "password rotation", "backup job reconfiguration"])} if rng.random() < 0.6 else None
        hr = {"employment_status": "terminated" if (not benign and rng.random() < 0.25) else "active", "department": rng.choice(["IT", "Finance", "Engineering", "Sales", "HR"]), "role": "system administrator" if is_admin else rng.choice(["analyst", "engineer", "account manager", "recruiter"])}
        vpn = {"logged_in_from": rng.choice(["office", "home", "unknown ASN in a country with no employees"]) if not benign else rng.choice(["office", "home"]), "mfa": True if benign else rng.random() < 0.5}
        edr = {"process": rng.choice(["procdump.exe", "mimikatz.exe", "powershell.exe", "wmic.exe", "rundll32.exe", "psexec.exe", "tcpdump", "crowdstrike_updater.exe"]), "parent": rng.choice(["explorer.exe", "services.exe", "cmd.exe", "winword.exe", "outlook.exe"]), "signed": benign or rng.random() < 0.3, "cmdline_contains": rng.choice(["-ma lsass.exe", "sekurlsa::logonpasswords", "-enc JAB", "Invoke-Mimikatz", "/S /D", "-i eth0 -w /tmp/trace.pcap", "update --silent"]) if not benign else rng.choice(["update --silent", "-i eth0 -w /tmp/trace.pcap", "/S /D", "Get-Process"]), "quarantined": rng.random() < 0.3}
        n_hosts = 1 if rng.random() < 0.7 else rng.choice([3, 6, 40])
        spread = (not benign) and n_hosts > 1
        outbound = (not benign) and rng.random() < 0.5
        persistence = (not benign) and (tid in ("T1546.018", "T1078") or rng.random() < 0.3)
        ongoing = (not benign) and rng.random() < 0.6
        contained = bool(edr["quarantined"] and not ongoing and not spread)
        indicators = 0 if benign else (1 + int(not edr["signed"]) + int(vpn["logged_in_from"].startswith("unknown")) + int(hr["employment_status"] == "terminated") + int(edr["cmdline_contains"] in ("sekurlsa::logonpasswords", "Invoke-Mimikatz", "-ma lsass.exe")))
        evidence = 0 if benign else min(3, indicators - 1 if indicators > 1 else 0) if indicators else 0
        if not benign and evidence == 0:
            evidence = 1
        scope = "organization_wide" if n_hosts >= 40 or (host.startswith("DC") and not benign) else ("workgroup" if n_hosts > 1 else "single_entity")
        state = {"alert": f"{tid} | {name}\nasset: {host}    principal: {user}    observed: 2026-0{rng.randint(1, 9)}-{rng.randint(10, 28)} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d} UTC\nevents correlated: {rng.randint(1, 5000)}\nhosts involved: {n_hosts}",
                 "context": {"asset": {"environment": env, "tier": tier, "owner_team": hr["department"]}, "identity": {"user": user, "is_admin": is_admin, "hr": hr}, "change_management": change, "vpn_session": vpn, "edr": edr,
                             "network": {"outbound_connection_active": outbound, "destination": rng.choice(["185.220.101.4:443", "files.dropbox.com", "internal", "pastebin.com"]) if outbound else None},
                             "mailbox": {"suspicious_rules_created": (not benign) and tid == "T1566.001" and rng.random() < 0.6}}}
        qs = [Q("Given the alert and its context records, does this describe unauthorized activity, as opposed to authorized activity that a detector flagged?", Y, int(not benign), kind="noul", descs=["The activity was authorized, expected, or did not happen -- administrative work, a sanctioned tool, a test, or a detector firing on nothing", "Someone is doing something they were not authorized to do"]),
              Q("How strong is the evidence that the activity is unauthorized?", ["0", "1", "2", "3"], evidence, kind="score", descs=["Speculative: hedged language, or a single weak indicator", "Suggestive: one concrete indicator, uncorroborated", "Corroborated: independent indicators agree", "Confirmed: direct proof of unauthorized activity"]),
              Q("How far does this reach?", SCOPE, SCOPE.index(scope), descs=["One host, account or mailbox", "A team, a distribution list, or a handful of related assets", "Everyone, or an asset the whole organization depends on"]),
              Q("What type of attack does the evidence most closely match?", ATTACK_TYPES, ATTACK_TYPES.index("benign_activity" if benign else atype)),
              Q("Is the activity still ongoing?", Y, int(ongoing), kind="noul"),
              Q("Has the activity spread beyond the initial entity?", Y, int(spread), kind="noul"),
              Q("Is there an active outbound channel to an attacker-controlled destination?", Y, int(outbound), kind="noul"),
              Q("Has the attacker established persistence?", Y, int(persistence), kind="noul"),
              Q("Is the threat already contained?", Y, int(contained), kind="noul", descs=["the process is still able to run or the activity continues or has spread", "the malicious process was quarantined and nothing is ongoing or spread"])]
        rng.shuffle(qs)
        out.append(Example(_j(state), qs, "security"))
    return out


# ------------------------------------------------------------------------------------------ invoice
VENDORS = ["Northwind Facilities", "Kestrel Analytics LLC", "Blue Harbor Logistics", "Orion Staffing", "Meridian Legal Services", "Summit Office Supply", "Halcyon Cloud Inc"]
PRICE_BASIS = ["total_fee", "period_fee", "unit_rate", "milestone", "no_stated_basis"]
PB_DESC = {"total_fee": "the contract states ONE TOTAL FEE for the whole engagement", "period_fee": "the contract states a RECURRING FEE PER PERIOD (monthly, quarterly, annual)", "unit_rate": "the contract states a rate per unit, hour, or item", "milestone": "the contract states amounts tied to named milestones or deliverables", "no_stated_basis": "no figure is written in the contract text for these items"}
ACTIONS_AP = ["approve_for_payment", "hold_for_po_mismatch", "hold_for_missing_receipt", "reject_duplicate", "request_vendor_correction", "escalate_to_procurement"]


def gen_invoice(rng, n, long=False):
    out = []
    for _ in range(n):
        vendor = rng.choice(VENDORS); po_no = f"PO-{rng.randint(100000, 999999)}"; inv_no = f"INV-{rng.randint(1000, 99999)}"
        basis = rng.choice(PRICE_BASIS); unit = rng.choice([85, 120, 150, 40]); qty = rng.randint(2, 60)
        contract_total = rng.choice([25000, 48000, 120000, 300000]); period_fee = rng.choice([2500, 4800, 9900])
        amount = {"total_fee": contract_total if rng.random() < 0.5 else round(contract_total / rng.choice([2, 4, 6]), 2), "period_fee": period_fee, "unit_rate": unit * qty, "milestone": rng.choice([12000, 30000]), "no_stated_basis": rng.choice([3400, 7850, 15200])}[basis]
        po_mismatch = rng.random() < 0.2; wrong_po = f"PO-{rng.randint(100000, 999999)}"
        po_remaining = amount + rng.choice([0, 500, 5000]) if not po_mismatch or rng.random() < 0.5 else amount - rng.choice([100, 2000])
        over_po = amount > po_remaining
        receipt_missing = rng.random() < 0.2 and basis != "period_fee"
        duplicate = rng.random() < 0.15
        not_invoice = rng.random() < 0.08
        doc_type = rng.choice(["statement of account", "quotation", "pro-forma invoice", "payment reminder"]) if not_invoice else rng.choice(["tax invoice", "invoice", "credit note"])
        prior = [{"invoice_no": f"INV-{rng.randint(1000, 99999)}", "amount": amount if duplicate else rng.choice([1200, 4800, 9900, amount / 2]), "date": f"2026-0{rng.randint(1, 8)}-{rng.randint(10, 28)}", "status": "paid", "po": po_no} for _ in range(rng.randint(0, 3))]
        if duplicate:
            prior.append({"invoice_no": inv_no, "amount": amount, "date": f"2026-0{rng.randint(1, 8)}-{rng.randint(10, 28)}", "status": "paid", "po": po_no})
        contract_text = {"total_fee": f"Contract Sum. The total fixed fee for the Services described in this Statement of Work shall be USD {contract_total:,.2f}, invoiced in installments.",
                         "period_fee": f"Fees. Client shall pay a monthly service fee of USD {period_fee:,.2f}, invoiced in advance on the first business day of each month.",
                         "unit_rate": f"Rates. Services are billed at USD {unit:.2f} per hour as recorded in approved timesheets.",
                         "milestone": "Payment Schedule. Milestone 1 (design sign-off): USD 12,000.00. Milestone 2 (delivery): USD 30,000.00.",
                         "no_stated_basis": "Fees. Fees are payable in accordance with the Supplier's then-current rate card, as provided separately."}[basis]
        if long:
            contract_text += " " + " ".join(rng.choice(["The Supplier shall maintain insurance coverage of not less than USD 1,000,000 per occurrence.", "Either party may terminate this Agreement upon thirty (30) days written notice.", "All notices shall be in writing and delivered to the addresses set forth above.", "The Supplier warrants that the Services will be performed in a professional and workmanlike manner.", "Confidential Information shall not be disclosed to any third party without prior written consent.", "This Agreement shall be governed by the laws of the State of Delaware.", "Invoices are payable net thirty (30) days from receipt.", "Any purchase order ceiling is a budgetary limit and not a fee."]) for _ in range(rng.randint(8, 30)))
        emails = [{"from": f"ap@{vendor.split()[0].lower()}.com", "subject": rng.choice(["Invoice attached", "Reminder: outstanding balance", "Corrected invoice"]), "body": rng.choice(["Please find our invoice attached for services rendered.", "This is a friendly reminder that the balance below is outstanding; the invoice was sent previously.", "Attached is the corrected invoice replacing the earlier version."])}] if rng.random() < 0.7 else []
        packet = {"task": "Accounts-payable review packet. Evaluate each question about this invoice against the purchase order, contract, delivery evidence, prior invoices, vendor master record and communications.",
                  "document": {"type": doc_type, "number": inv_no, "vendor": vendor, "amount": amount, "currency": "USD", "po_reference": wrong_po if po_mismatch else po_no, "line_items": [{"description": rng.choice(["Consulting services", "Facilities maintenance", "Cloud hosting", "Contract staffing"]), "qty": qty if basis == "unit_rate" else 1, "unit_price": unit if basis == "unit_rate" else amount}], "date": f"2026-0{rng.randint(1, 9)}-{rng.randint(10, 28)}"},
                  "purchase_order": {"number": po_no, "vendor": vendor, "total": po_remaining + sum(p["amount"] for p in prior if p["po"] == po_no), "remaining": po_remaining, "status": "open"},
                  "contract": {"title": f"Master Services Agreement - {vendor}", "text": contract_text},
                  "delivery_evidence": None if receipt_missing else {"goods_receipt": f"GR-{rng.randint(10000, 99999)}", "received_by": "warehouse" if basis == "unit_rate" else "service owner", "quantity_ok": True},
                  "prior_invoices": prior, "vendor_master": {"vendor": vendor, "status": "active", "bank_account_last4": rng.randint(1000, 9999), "payment_terms": "net 30"}, "communications": emails,
                  "exact_facts": {"po_reference_matches_open_po": not po_mismatch, "amount_within_po_remaining": not over_po, "invoice_number_seen_before": duplicate}}
        if not_invoice:
            action = "request_vendor_correction"
        elif duplicate:
            action = "reject_duplicate"
        elif po_mismatch:
            action = "hold_for_po_mismatch"
        elif over_po:
            action = "escalate_to_procurement"
        elif receipt_missing:
            action = "hold_for_missing_receipt"
        else:
            action = "approve_for_payment"
        qs = [Q("The document in the packet is NOT an invoice: it is a statement of account, a quote, a pro-forma, an estimate, a reminder or a remittance summary. A tax invoice or a credit note IS an invoice-type document.", Y, int(not_invoice), kind="noul"),
              Q("Does the invoice reference a purchase order that exists and is open for this vendor?", Y, int(not po_mismatch), kind="noul"),
              Q("Is the invoice amount within the remaining balance of the referenced purchase order?", Y, int(not over_po), kind="noul"),
              Q("Which ONE of these describes the price the contract or SOW text states for the items this invoice bills? A basis needs a figure written in the text.", PRICE_BASIS, PRICE_BASIS.index(basis), descs=[PB_DESC[k] for k in PRICE_BASIS]),
              Q("Is this invoice a duplicate of an invoice already paid?", Y, int(duplicate), kind="noul"),
              Q("Is there delivery or receipt evidence for what the invoice bills?", Y, int(not receipt_missing), kind="noul"),
              Q("What should accounts payable do with this document?", ACTIONS_AP, ACTIONS_AP.index(action)),
              Q("Does the vendor's communication claim the invoice was sent before?", Y, int(bool(emails) and "previously" in emails[0]["body"]), kind="noul")]
        rng.shuffle(qs)
        out.append(Example(_j(packet), qs, "invoice"))
    return out


# ------------------------------------------------------------------------------------------ agent traces
AGENTS = [("Ridgeline Patient Navigator", "help patients schedule, reschedule or cancel appointments and check referral status", ["find_appointments", "book_appointment", "cancel_appointment", "check_referral"]),
          ("Streamly Billing Assistant", "handle refunds, plan changes and billing questions for subscribers", ["lookup_account", "issue_refund", "change_plan", "list_invoices"]),
          ("Kestrel Travel Desk", "book and modify flights and hotels for employees", ["search_flights", "book_flight", "cancel_booking", "hotel_search"]),
          ("Harbor Bank Card Support", "lock cards, dispute charges and update contact info", ["lock_card", "dispute_charge", "update_address", "get_transactions"])]
FAILS = ["none", "tool_error", "wrong_arguments", "wrong_tool", "misread_result", "claimed_without_tool"]


def gen_agent_trace(rng, n):
    out = []
    for _ in range(n):
        name, purpose, tools = rng.choice(AGENTS)
        goal_tool = rng.choice(tools[1:]); fail = rng.choice(FAILS + ["none", "none"])
        user_req = {"find_appointments": "what appointments do I have", "book_appointment": "book me a follow-up next Tuesday morning", "cancel_appointment": "cancel my appointment on the 14th", "check_referral": "has my cardiology referral gone through",
                    "issue_refund": "refund last month's charge, I was double billed", "change_plan": "move me to the annual plan", "list_invoices": "send me my last three invoices", "search_flights": "find flights to Denver Thursday", "book_flight": "book the 9am Denver flight", "cancel_booking": "cancel my hotel in Austin", "hotel_search": "find a hotel near the Austin office",
                    "lock_card": "lock my card, I lost it", "dispute_charge": "dispute the $84 charge from yesterday", "update_address": "update my address to 12 Elm St", "get_transactions": "show my recent transactions", "lookup_account": "look up my account"}.get(goal_tool, "help me")
        steps = []; msgs = [{"role": "user", "content": user_req}]
        pre = tools[0]
        steps.append({"step": 1, "tool": pre, "arguments": {"customer_id": f"C{rng.randint(1000, 9999)}"}, "result": {"ok": True, "record": "found"}})
        t2 = goal_tool; args = {"target": user_req.split()[-1], "confirm": True}; res = {"ok": True, "status": "done", "id": f"X{rng.randint(100, 999)}"}
        if fail == "tool_error":
            res = {"ok": False, "error": rng.choice(["timeout", "permission_denied", "slot_unavailable"])}
        elif fail == "wrong_arguments":
            args = {"target": rng.choice(["", "null", "next Tuesday morning".replace("Tuesday", "Thursday")]), "confirm": False}; res = {"ok": False, "error": "invalid_arguments"}
        elif fail == "wrong_tool":
            t2 = rng.choice([t for t in tools if t != goal_tool]); res = {"ok": True, "status": "done", "note": "unrelated operation completed"}
        elif fail == "misread_result":
            res = {"ok": True, "status": "pending_approval", "id": f"X{rng.randint(100, 999)}"}
        if fail == "claimed_without_tool":
            steps_final = steps
        else:
            steps.append({"step": 2, "tool": t2, "arguments": args, "result": res}); steps_final = steps
        achieved = fail == "none"
        final = {"none": "Done. I've completed that for you and you'll get a confirmation shortly.", "tool_error": "I ran into an error processing that. Would you like me to try again or connect you with a person?",
                 "wrong_arguments": "I've taken care of that for you.", "wrong_tool": "All set, I've handled your request.", "misread_result": "Done, it's been processed.", "claimed_without_tool": "Done, I've completed that for you."}[fail]
        msgs += [{"role": "assistant", "content": final}]
        sat = 4 if achieved and rng.random() < 0.6 else (3 if achieved else rng.choice([0, 0, 1, 1, 2]))
        user_last = {4: "Perfect, thank you so much!", 3: "Great, thanks.", 2: "Ok.", 1: "Hmm, I don't see any confirmation and it still shows the old state.", 0: "This is not what I asked for. Nothing changed. Get me a person."}[sat]
        if rng.random() < 0.8:
            msgs.append({"role": "user", "content": user_last})
        else:
            sat = 2
        unbacked = fail in ("wrong_arguments", "wrong_tool", "misread_result", "claimed_without_tool")
        first_bad = "none" if fail == "none" else (f"step 2: {t2}" if fail != "claimed_without_tool" else "none")
        step_opts = [f"step {s['step']}: {s['tool']}" for s in steps_final] + ["none"]
        g_step = step_opts.index(first_bad) if first_bad in step_opts else len(step_opts) - 1
        state = {"agent": {"name": name, "instructions": f"You are {name}. Your role is to {purpose}. Only report outcomes shown by tool results.", "tools": tools}, "conversation": msgs, "tool_calls": steps_final}
        qs = [Q("According to the tool results and the conversation record -- not the assistant's own words -- did the user get what they asked for?", Y, int(achieved), kind="noul", descs=["no tool result shows the requested outcome; an assistant saying it was done does not count", "the specific outcome the user requested is shown by a tool result"]),
              Q("Which step is the first where the run went wrong: the first tool call that failed, was made with wrong arguments, was the wrong tool for the task, or whose result the assistant then misread?", step_opts, g_step, descs=[f"the run first went wrong at {o}" for o in step_opts[:-1]] + ["no tool step went wrong; any failure lies in the conversation itself, or nothing went wrong"]),
              Q("How satisfied is the user with the outcome, judging only by what they expressed after the assistant's final message?", ["0", "1", "2", "3", "4"], sat, kind="score", descs=["angry or escalating: complains, says the problem is unresolved, demands a person", "dissatisfied: says something was wrong, missing, or unhelpful", "neutral or mixed: no clear signal either way", "satisfied: indicates the issue was handled", "very satisfied: explicitly praises the help or confirms full resolution"]),
              Q("Did the assistant claim an outcome that is not backed by a tool result?", Y, int(unbacked), kind="noul"),
              Q("What kind of failure, if any, occurred in this run?", FAILS, FAILS.index(fail), descs=["nothing went wrong", "a tool call returned an error", "a tool was called with wrong or empty arguments", "the wrong tool was used for the request", "the assistant misread a tool result (e.g. treated pending as done)", "the assistant claimed completion without calling the needed tool"]),
              Q("Should this conversation be escalated to a human agent?", Y, int((not achieved) and sat <= 1), kind="noul")]
        rng.shuffle(qs)
        out.append(Example(_j(state), qs, "agent_trace"))
    return out


# ------------------------------------------------------------------------------------------ catalog alignment
BRANDS = ["Acme", "Nordic", "Zephyr", "Bolt", "Lumen", "Terra"]
PRODS = [("wireless mouse", ["black", "white", "grey"], ["", "ergonomic", "compact"]), ("running shoes", ["blue", "black", "red"], ["men's", "women's"]), ("coffee beans", ["1kg", "500g", "250g"], ["medium roast", "dark roast"]), ("usb-c cable", ["1m", "2m", "0.5m"], ["braided", ""]), ("desk lamp", ["white", "black"], ["LED", "halogen"])]


def gen_catalog(rng, n):
    out = []
    for _ in range(n):
        brand = rng.choice(BRANDS); prod, variants, attrs = rng.choice(PRODS)
        v1 = rng.choice(variants); a1 = rng.choice(attrs); same = rng.random() < 0.5
        v2 = v1 if same or rng.random() < 0.5 else rng.choice([v for v in variants if v != v1]); a2 = a1 if same else (rng.choice([a for a in attrs if a != a1]) if len(attrs) > 1 and v2 == v1 else a1)
        if same:
            v2, a2 = v1, a1
        b2 = brand if same or rng.random() < 0.8 else rng.choice([b for b in BRANDS if b != brand])
        name1 = f"{brand} {a1} {prod} {v1}".replace("  ", " ").strip(); name2 = f"{b2} {prod.title()} - {v2} {a2}".replace("  ", " ").strip()
        if same and rng.random() < 0.4:
            name2 = name2.replace(prod.title(), prod.upper()[:4] + prod[4:])
        p1 = round(rng.uniform(5, 120), 2); p2 = round(p1 * rng.uniform(0.9, 1.1), 2) if same else round(rng.uniform(5, 120), 2)
        r1 = {"sku": f"{brand[:2].upper()}-{rng.randint(1000, 9999)}", "title": name1, "price": p1, "attributes": {"color_or_size": v1, "variant": a1 or None}}
        r2 = {"id": rng.randint(10000, 99999), "name": name2, "price_usd": p2, "spec": f"{v2} {a2}".strip()}
        truly_same = same and b2 == brand
        state = {"record_a": r1, "record_b": r2}
        qs = [Q("Do these two records describe the same product (same brand, model and variant)?", Y, int(truly_same), kind="noul", descs=["different brand, model, size/colour, or variant", "identical product even if the titles are formatted differently"]),
              Q("Are the two records from the same brand?", Y, int(b2 == brand), kind="noul")]
        out.append(Example(_j(state), qs, "catalog"))
    return out


SYNTH = {"doom": gen_doom, "smart_home": gen_smart_home, "support": gen_support, "security": gen_security, "invoice": gen_invoice, "agent_trace": gen_agent_trace, "catalog": gen_catalog}
SYNTH_TRAIN = {"doom": 16000, "smart_home": 9000, "support": 10000, "security": 8000, "invoice": 5000, "agent_trace": 7000, "catalog": 4000}
SYNTH_EVAL = 500


def load_synth(name):
    gen = SYNTH[name]
    tr = gen(random.Random(1000 + zlib.crc32(name.encode()) % 1000), SYNTH_TRAIN[name])
    if name == "invoice":
        tr += gen_invoice(random.Random(77), 1500, long=True)
    ev = gen(random.Random(20000 + zlib.crc32(name.encode()) % 1000), SYNTH_EVAL)
    return tr, ev
