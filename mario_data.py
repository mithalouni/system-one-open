"""Generate teacher-labelled Super Mario Bros states (real emulator) for a short fine-tune. CPU only.
Writes /vol/data/v1/mario.train.jsonl.gz + mario.eval.jsonl.gz and registers 'mario' in the manifest."""
import modal, os, json
from s1.modal_common import vol
app = modal.App("jev-mario-data")
image = (modal.Image.debian_slim(python_version="3.11").apt_install("build-essential")
         .uv_pip_install("gym==0.23.1", "gym-super-mario-bros==7.4.0", "nes-py==8.2.1", "numpy<2")
         .add_local_python_source("s1"))
NAMES = ["noop", "right", "right jump", "right run", "right run jump", "jump", "left"]
DESC = {"noop": "do nothing", "right": "walk right", "right jump": "jump while moving right", "right run": "run right (hold B)", "right run jump": "long running jump to the right", "jump": "jump in place", "left": "walk left"}
QTEXT = "Super Mario Bros level 1-1. Mario must move right to reach the flag, jump over pits and enemies (goombas, koopas) and onto blocks; running jumps clear wide gaps. Which controller input now?"


def tiles_ahead(ram, x, yscreen):
    def tile(px, py):
        if py < 32 or py >= 240: return 0
        page = (px // 256) % 2; col = (px % 256) // 16; row = (py - 32) // 16
        return int(ram[0x0500 + page * 208 + row * 16 + col])
    ground = [tile(x + dx, 208) != 0 or tile(x + dx, 192) != 0 for dx in (16, 32, 48, 64)]
    block = [tile(x + dx, yscreen) != 0 for dx in (16, 32)]
    return {"ground_ahead_16_32_48_64px": ground, "block_at_mario_height_16_32px": block, "pit_ahead": not all(ground[:3])}


def build_state(ram, info, last, tick):
    x = int(info.get("x_pos", 40)); y = int(info.get("y_pos", 79)); yscreen = 240 - y
    enemies = []
    for i in range(5):
        if ram[0x000F + i]:
            ex = int(ram[0x006E + i]) * 256 + int(ram[0x0087 + i]); ey = int(ram[0x00CF + i])
            enemies.append({"type": "goomba" if ram[0x0016 + i] in (0, 6) else ("koopa" if ram[0x0016 + i] in (1, 2, 3, 4, 5) else "enemy"), "dx": ex - x, "dy": ey - yscreen})
    near = sorted([e for e in enemies if -20 < e["dx"] < 160], key=lambda e: e["dx"])[:3]
    return {"mario": {"x": x, "y_from_ground": max(0, y - 79), "status": info.get("status", "small"), "moving": (int(ram[0x0057]) if ram[0x0057] < 128 else int(ram[0x0057]) - 256)},
            "enemies_ahead": near, "terrain": tiles_ahead(ram, x, max(32, yscreen)), "time_left": info.get("time", 400), "coins": info.get("coins", 0), "last_action": last, "tick": tick}


def teacher(st):
    en = st["enemies_ahead"]; ter = st["terrain"]; airborne = st["mario"]["y_from_ground"] > 2
    close = [e for e in en if 0 < e["dx"] < 60 and abs(e["dy"]) < 28]; very_close = [e for e in en if -8 < e["dx"] < 24 and abs(e["dy"]) < 28]
    danger = bool(very_close) or (ter["pit_ahead"] and not ter["ground_ahead_16_32_48_64px"][0])
    if airborne:
        act = "right run jump" if (close or ter["pit_ahead"]) else "right run"
    elif very_close:
        act = "right run jump"
    elif close or ter["pit_ahead"] or ter["block_at_mario_height_16_32px"][0] or ter["block_at_mario_height_16_32px"][1]:
        act = "right run jump" if (ter["pit_ahead"] or close) else "right jump"
    else:
        act = "right run"
    jump_useful = act in ("right run jump", "right jump", "jump")
    return act, jump_useful, danger


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=4096, timeout=1800)
def gen(episodes: int = 14, decide_every: int = 4, eps: float = 0.15):
    import gym_super_mario_bros, random, gzip
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from s1.schema import Example, Q
    rng = random.Random(0); out = []; progress = []
    for ep in range(episodes):
        env = JoypadSpace(gym_super_mario_bros.make(rng.choice(["SuperMarioBros-1-1-v0", "SuperMarioBros-1-2-v0", "SuperMarioBros-1-1-v0"])), SIMPLE_MOVEMENT)
        env.reset(); info = {}; last = None; act = 3; max_x = 0
        for tick in range(2400):
            if tick % decide_every == 0:
                st = build_state(env.unwrapped.ram, info, last, tick); a, ju, dg = teacher(st)
                qs = [Q(QTEXT, NAMES, NAMES.index(a), kind="choice", descs=[DESC[n] for n in NAMES]),
                      Q("Is a jump useful right now (enemy or pit or block just ahead)?", ["no", "yes"], int(ju), kind="noul"),
                      Q("Is Mario in immediate danger of dying?", ["no", "yes"], int(dg), kind="noul")]
                rng.shuffle(qs); out.append(Example(json.dumps(st, ensure_ascii=False, indent=1), qs, "mario"))
                chosen = a if rng.random() > eps else rng.choice(NAMES); act = NAMES.index(chosen); last = chosen
            _, r, done, info = env.step(act); max_x = max(max_x, int(info.get("x_pos", 0)))
            if done or info.get("flag_get"):
                break
        progress.append(max_x); env.close()
    rng.shuffle(out); n_ev = min(400, len(out) // 8)
    d = "/vol/data/v1"
    with gzip.open(f"{d}/mario.train.jsonl.gz", "wt") as f:
        for e in out[n_ev:]: f.write(e.to_json() + "\n")
    with gzip.open(f"{d}/mario.eval.jsonl.gz", "wt") as f:
        for e in out[:n_ev]: f.write(e.to_json() + "\n")
    m = json.load(open(f"{d}/manifest.json")); m["mario"] = {"name": "mario", "group": "synth", "heldout": False, "train": len(out) - n_ev, "eval": n_ev, "train_questions": 3 * (len(out) - n_ev)}
    json.dump(m, open(f"{d}/manifest.json", "w"), indent=1); vol.commit()
    from collections import Counter
    return {"examples": len(out), "teacher_max_x_per_episode": progress, "action_dist": Counter(q.options[q.gold] for e in out for q in e.qs if q.kind == "choice").most_common()}


@app.local_entrypoint()
def main(episodes: int = 14):
    print(json.dumps(gen.remote(episodes), indent=1))
