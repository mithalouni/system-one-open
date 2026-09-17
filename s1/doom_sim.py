"""Tiny closed-loop Doom-style arena: the decision model plays as the policy (state JSON -> action) each tick.
Dynamics are simple but consistent with data_synth.doom_state / doom_teacher, so the teacher is a strong baseline.
"""
from __future__ import annotations
import json, math, random
from .data_synth import ENEMY_TYPES, DOOM_ACTIONS, ACTION_DESC, doom_teacher
from .schema import Example, Q

WEAPON_DMG = {"pistol": 15, "shotgun": 45, "chaingun": 25, "rocket_launcher": 120}
ENEMY_SPEED = {"imp": 2.0, "zombieman": 1.5, "demon": 3.0, "cacodemon": 2.5, "shotgun_guy": 1.5, "baron": 2.0}
ENEMY_DMG = {"imp": 6, "zombieman": 4, "demon": 12, "cacodemon": 10, "shotgun_guy": 8, "baron": 20}


class Arena:
    def __init__(self, seed=0, n_enemies=None):
        self.rng = random.Random(seed)
        r = self.rng
        self.p = {"x": 50.0, "y": 50.0, "facing_deg": r.choice(range(0, 360, 15)), "health": 100, "armor": r.choice([0, 25, 50]), "weapon": r.choice(list(WEAPON_DMG)), "ammo": r.choice([10, 20, 40]), "keys": []}
        self.enemies = []
        for i in range(n_enemies or r.choice([2, 3, 4])):
            t = r.choice(list(ENEMY_TYPES)); ang = r.uniform(0, 360); d = r.uniform(20, 55)
            self.enemies.append({"id": f"E{i + 1}", "type": t, "x": 50 + d * math.cos(math.radians(ang)), "y": 50 + d * math.sin(math.radians(ang)), "health": ENEMY_TYPES[t][0], "alive": True, "cool": 0})
        self.items = [{"type": r.choice(["medkit", "medkit", "ammo_clip", "shells", "armor"]), "x": r.uniform(5, 95), "y": r.uniform(5, 95), "taken": False} for _ in range(r.choice([2, 3, 4]))]
        self.tick = 0; self.kills = 0; self.last = None; self.dead = False

    def _rel(self, ox, oy):
        dx, dy = ox - self.p["x"], oy - self.p["y"]
        dist = math.hypot(dx, dy); ang = (math.degrees(math.atan2(dy, dx)) - self.p["facing_deg"] + 180) % 360 - 180
        return round(dist, 1), int(round(ang / 5.0) * 5)

    def state(self):
        en = []
        for e in self.enemies:
            if not e["alive"]:
                continue
            dist, rel = self._rel(e["x"], e["y"])
            en.append({"id": e["id"], "type": e["type"], "distance": dist, "relative_angle": rel, "health": e["health"], "visible": abs(rel) <= 45 and dist <= 60, "attacking": dist < 25})
        items = []
        for it in self.items:
            if it["taken"]:
                continue
            dist, rel = self._rel(it["x"], it["y"])
            if dist <= 45:
                items.append({"type": it["type"], "distance": dist, "relative_angle": rel})
        wall = min(self.p["x"], 100 - self.p["x"], self.p["y"], 100 - self.p["y"])
        return {"player": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in self.p.items()}, "enemies": en, "items": items,
                "environment": {"door_ahead": False, "door_locked": False, "wall_ahead_distance": round(max(0.5, wall), 1), "level": "ARENA"}, "last_action": self.last, "tick": self.tick}

    def step(self, action):
        p = self.p; r = self.rng; self.last = action; self.tick += 1
        rad = math.radians(p["facing_deg"])
        if action == "turn_left":
            p["facing_deg"] = (p["facing_deg"] - 15) % 360
        elif action == "turn_right":
            p["facing_deg"] = (p["facing_deg"] + 15) % 360
        elif action in ("move_forward", "move_backward", "strafe_left", "strafe_right"):
            a = rad + {"move_forward": 0, "move_backward": math.pi, "strafe_left": -math.pi / 2, "strafe_right": math.pi / 2}[action]
            p["x"] = min(99, max(1, p["x"] + 3 * math.cos(a))); p["y"] = min(99, max(1, p["y"] + 3 * math.sin(a)))
        elif action == "attack" and p["ammo"] > 0:
            p["ammo"] -= 1
            cands = []
            for e in self.enemies:
                if e["alive"]:
                    dist, rel = self._rel(e["x"], e["y"])
                    if abs(rel) <= 10 and dist <= 40:
                        cands.append((dist, e))
            if cands:
                e = min(cands, key=lambda c: c[0])[1]; e["health"] -= WEAPON_DMG[p["weapon"]]
                if e["health"] <= 0:
                    e["alive"] = False; self.kills += 1
        elif action == "pick_up":
            for it in self.items:
                if not it["taken"] and math.hypot(it["x"] - p["x"], it["y"] - p["y"]) < 3:
                    it["taken"] = True
                    if it["type"] == "medkit":
                        p["health"] = min(100, p["health"] + 25)
                    elif it["type"] in ("ammo_clip", "shells"):
                        p["ammo"] += 20
                    elif it["type"] == "armor":
                        p["armor"] = min(100, p["armor"] + 50)
        # enemies move toward player and attack
        for e in self.enemies:
            if not e["alive"]:
                continue
            dx, dy = p["x"] - e["x"], p["y"] - e["y"]; d = math.hypot(dx, dy)
            if d > 6:
                s = ENEMY_SPEED[e["type"]]; e["x"] += s * dx / d; e["y"] += s * dy / d
            if d < 25 and e["cool"] <= 0 and r.random() < 0.35:
                dmg = ENEMY_DMG[e["type"]]
                absorb = min(p["armor"], dmg // 2); p["armor"] -= absorb; p["health"] -= (dmg - absorb); e["cool"] = 2
            e["cool"] = max(0, e["cool"] - 1)
        if p["health"] <= 0:
            self.dead = True
        done = self.dead or self.tick >= 200 or not any(e["alive"] for e in self.enemies)
        return done


def questions_for(state):
    """The same question set as data_synth.gen_doom, as API dicts (action first)."""
    qs = [{"id": "action", "type": "choice", "instructions": "What should the player do this tick?", "options": {a: ACTION_DESC[a] for a in DOOM_ACTIONS}},
          {"id": "danger", "type": "noul", "instructions": "Is the player in immediate danger?"},
          {"id": "threat", "type": "score", "instructions": "How threatening is the current situation?", "levels": ["no enemies around", "a weak enemy or one far away", "several enemies or a strong one nearby", "overwhelming: strong enemies close and attacking"]}]
    return qs


def run_episodes(policy, n=10, seed=0, log=None):
    """policy(state_dict) -> action string. Returns aggregate stats."""
    res = []
    for ep in range(n):
        a = Arena(seed=seed + ep); done = False
        while not done:
            s = a.state(); act = policy(s); done = a.step(act)
        res.append(dict(ticks=a.tick, kills=a.kills, survived=not a.dead, health=max(0, a.p["health"]), enemies=len(a.enemies)))
        if log:
            log(f"  ep{ep}: ticks={a.tick} kills={a.kills}/{len(a.enemies)} survived={not a.dead} health={max(0, a.p['health'])}")
    n = len(res)
    return dict(episodes=n, survival_rate=sum(r["survived"] for r in res) / n, mean_kills=sum(r["kills"] for r in res) / n,
                kill_rate=sum(r["kills"] for r in res) / max(1, sum(r["enemies"] for r in res)), mean_ticks=sum(r["ticks"] for r in res) / n, mean_final_health=sum(r["health"] for r in res) / n)


def teacher_policy(s):
    return doom_teacher(s)["action"]


def random_policy(rng):
    return lambda s: rng.choice(DOOM_ACTIONS)


def model_policy(model, decide_fn):
    qs = questions_for(None)
    def pol(s):
        r = decide_fn(model, s, qs, max_state_tokens=1024)
        return r["answers"][0]["choice"]
    return pol
