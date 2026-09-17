"""Real games played by the open System One model: ViZDoom (deadly corridor) and Super Mario Bros (NES).
Each tick the live game state becomes JSON, the model returns typed decisions (action distribution + situation nouls),
the action is executed, and the real frame + a decision panel are recorded to MP4.

  S1_GPU=H100 modal run games.py --game doom --episodes 1
  S1_GPU=H100 modal run games.py --game mario --episodes 1
"""
import modal, os, json
from s1.modal_common import vol, gpu_kwargs
app = modal.App("jev-games")
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg", "build-essential", "cmake", "libsdl2-dev", "libboost-all-dev", "libopenal-dev", "zlib1g-dev", "libjpeg-dev", "libbz2-dev", "libgtk2.0-dev", "libfluidsynth-dev", "libgme-dev", "libsndfile1-dev", "libwildmidi-dev", "timidity", "nasm", "libpng-dev", "libglib2.0-dev", "libmpg123-dev", "fonts-dejavu-core")
         .uv_pip_install("torch==2.8.0", "transformers==5.17.0", "accelerate==1.15.0", "peft==0.21.0", "numpy<2", "pillow", "vizdoom==1.2.4", "gym==0.23.1", "gym-super-mario-bros==7.4.0", "nes-py==8.2.1", "sentencepiece", "protobuf")
         .env({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
         .add_local_python_source("s1"))
DOOM_MAP = {"Zombieman": "zombieman", "ShotgunGuy": "shotgun_guy", "DoomImp": "imp", "Demon": "demon", "Cacodemon": "cacodemon", "BaronOfHell": "baron", "HellKnight": "baron", "ChaingunGuy": "shotgun_guy"}
ITEM_MAP = {"GreenArmor": "armor", "BlueArmor": "armor", "Medikit": "medkit", "Stimpack": "stimpack", "Clip": "ammo_clip", "ClipBox": "ammo_clip", "Shells": "shells", "ShellBox": "shells", "RedCard": "key_red", "BlueCard": "key_blue"}


def panel_image(title, lines, probs, nouls, latency_ms, w=520, h=480):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (w, h), (15, 21, 25)); d = ImageDraw.Draw(im)
    try:
        F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15); FB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22); FS = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
    except Exception:
        F = FB = FS = ImageFont.load_default()
    d.text((18, 14), title, fill=(230, 236, 241), font=FB)
    d.text((18, 46), f"decision latency {latency_ms:.0f} ms   ·   one forward pass, 0 tokens generated", fill=(147, 161, 174), font=FS)
    y = 78; d.text((18, y), "CONTROLLER PROBABILITIES", fill=(63, 180, 190), font=FS); y += 22
    best = max(probs, key=probs.get)
    for k, v in probs.items():
        col = (76, 194, 122) if k == best else (110, 125, 140)
        d.text((18, y), k, fill=(230, 236, 241) if k == best else (147, 161, 174), font=F); d.rectangle([180, y + 5, 180 + int(300 * v), y + 13], fill=col); d.text((488, y), f"{100*v:.0f}%", fill=(230, 236, 241), font=FS, anchor="ra"); y += 22
    y += 10; d.text((18, y), "SITUATION", fill=(63, 180, 190), font=FS); y += 22
    for k, v in nouls.items():
        d.text((18, y), k, fill=(147, 161, 174), font=F); d.rectangle([180, y + 5, 180 + int(300 * v), y + 13], fill=(224, 161, 58) if v > 0.5 else (110, 125, 140)); d.text((488, y), f"{100*v:.0f}%", fill=(230, 236, 241), font=FS, anchor="ra"); y += 22
    y += 8
    for ln in lines:
        d.text((18, y), ln, fill=(147, 161, 174), font=FS); y += 18
    return im


@app.function(image=image, volumes={"/vol": vol}, timeout=3600, **gpu_kwargs())
def play(game: str, episodes: int, max_ticks: int, decide_every: int, scenario: str = "defend_the_center", run: str = "e2b-full", skill: int = 2):
    RUN = run
    import time, math, numpy as np, subprocess, shutil
    from PIL import Image
    from s1.engine import DecisionModel, decide
    from s1.data_synth import DOOM_ACTIONS, ACTION_DESC
    model = DecisionModel(f"/vol/runs/{RUN}/model"); model.lm.eval()
    outdir = f"/vol/demo/videos/{game}"; shutil.rmtree(outdir, ignore_errors=True); os.makedirs(outdir + "/frames", exist_ok=True)
    stats = []; frame_i = 0; lat = []

    def write(frame_rgb, panel):
        nonlocal frame_i
        fr = Image.fromarray(frame_rgb).resize((640, 480)) if frame_rgb.shape[1] != 640 else Image.fromarray(frame_rgb)
        canvas = Image.new("RGB", (640 + 520, 480), (15, 21, 25)); canvas.paste(fr, (0, 0)); canvas.paste(panel, (640, 0))
        canvas.save(f"{outdir}/frames/f{frame_i:06d}.jpg", quality=88); frame_i += 1

    if game == "doom":
        import vizdoom as vzd
        g = vzd.DoomGame(); g.load_config(os.path.join(vzd.scenarios_path, f"{scenario}.cfg")); g.set_window_visible(False)
        g.set_screen_format(vzd.ScreenFormat.RGB24); g.set_screen_resolution(vzd.ScreenResolution.RES_640X480); g.set_labels_buffer_enabled(True); g.set_depth_buffer_enabled(True)
        g.set_available_buttons([vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK, vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_BACKWARD, vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.USE])
        for v in [vzd.GameVariable.HEALTH, vzd.GameVariable.SELECTED_WEAPON_AMMO, vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y, vzd.GameVariable.ANGLE, vzd.GameVariable.KILLCOUNT, vzd.GameVariable.ARMOR]:
            g.add_available_game_variable(v)
        g.set_doom_skill(skill); g.set_episode_timeout(max_ticks * 4 + 100); g.init()
        BTN = {"strafe_left": 0, "strafe_right": 1, "attack": 2, "move_forward": 3, "move_backward": 4, "turn_left": 5, "turn_right": 6, "use": 7, "pick_up": 3}
        qs = [{"id": "action", "type": "choice", "instructions": "What should the player do this tick?", "options": {a: ACTION_DESC[a] for a in DOOM_ACTIONS}},
              {"id": "danger", "type": "noul", "instructions": "Is the player in immediate danger?"},
              {"id": "retreat", "type": "noul", "instructions": "Should the player retreat rather than engage?"},
              {"id": "threat", "type": "score", "instructions": "How threatening is the current situation?", "levels": ["no enemies around", "a weak enemy or one far away", "several enemies or a strong one nearby", "overwhelming: strong enemies close and attacking"]}]
        for ep in range(episodes):
            g.new_episode(); tick = 0; last = None; kills0 = 0; act = "move_forward"; probs = {a: 1 / len(DOOM_ACTIONS) for a in DOOM_ACTIONS}; nouls = {"immediate danger": 0.0, "should retreat": 0.0}; thr = 0.0; lms = 0.0
            while not g.is_episode_finished() and tick < max_ticks:
                s = g.get_state(); gv = s.game_variables; health, ammo, px, py, ang, kills, armor = gv[:7]
                if tick % decide_every == 0:
                    enemies, items = [], []
                    for L in s.labels:
                        if L.object_name == "DoomPlayer":
                            continue
                        dx, dy = L.object_position_x - px, L.object_position_y - py; dist = math.hypot(dx, dy) / 20.0
                        rel = (math.degrees(math.atan2(dy, dx)) - ang + 540) % 360 - 180
                        if L.object_name in DOOM_MAP:
                            enemies.append({"id": f"E{len(enemies)+1}", "type": DOOM_MAP[L.object_name], "distance": round(dist, 1), "relative_angle": int(round(rel / 5) * 5), "health": 60, "visible": True, "attacking": bool(dist < 25)})
                        elif L.object_name in ITEM_MAP:
                            items.append({"type": ITEM_MAP[L.object_name], "distance": round(dist, 1), "relative_angle": int(round(rel / 5) * 5)})
                    depth = s.depth_buffer; wall = float(depth[depth.shape[0] // 2, depth.shape[1] // 2]) / 4.0 if depth is not None else 30.0
                    state = {"player": {"x": round(px / 10, 1), "y": round(py / 10, 1), "facing_deg": int(ang), "health": int(health), "armor": int(armor), "weapon": "pistol", "ammo": int(ammo), "keys": []}, "enemies": enemies[:4], "items": items[:3],
                             "environment": {"door_ahead": False, "door_locked": False, "wall_ahead_distance": round(min(60.0, wall), 1), "level": scenario.replace("_", " ")}, "last_action": last, "tick": tick}
                    t1 = time.time(); r = decide(model, state, qs, max_state_tokens=1024); lms = (time.time() - t1) * 1000; lat.append(lms)
                    a = r["answers"]; probs = a[0]["probabilities"]; act = a[0]["choice"]; nouls = {"immediate danger": a[1]["noul"], "should retreat": a[2]["noul"]}; thr = a[3]["score"]; last = act
                btn = [0] * 8; btn[BTN[act]] = 1
                rew = g.make_action(btn, 1)
                pan = panel_image(f"System One plays Doom ({scenario.replace(chr(95), chr(32))})", [f"tick {tick}   health {int(health)}   ammo {int(ammo)}   kills {int(kills)}", f"threat level {thr:.2f} / 3   enemies visible {sum(1 for L in s.labels if L.object_name in DOOM_MAP)}", "freedoom assets via ViZDoom · state = labels + depth + game vars"], probs, nouls, lms)
                write(s.screen_buffer, pan); tick += 1
            stats.append({"episode": ep, "ticks": tick, "kills": int(g.get_game_variable(vzd.GameVariable.KILLCOUNT)), "health": int(g.get_game_variable(vzd.GameVariable.HEALTH)), "dead": bool(g.is_player_dead()), "total_reward": float(g.get_total_reward())})
        g.close()
    else:
        import gym_super_mario_bros
        from nes_py.wrappers import JoypadSpace
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        env = JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-1-1-v0"), SIMPLE_MOVEMENT)
        NAMES = ["noop", "right", "right jump", "right run", "right run jump", "jump", "left"]
        DESC = {"noop": "do nothing", "right": "walk right", "right jump": "jump while moving right", "right run": "run right (hold B)", "right run jump": "long running jump to the right", "jump": "jump in place", "left": "walk left"}
        qs = [{"id": "action", "type": "choice", "instructions": "Super Mario Bros level 1-1. Mario must move right to reach the flag, jump over pits and enemies (goombas, koopas) and onto blocks; running jumps clear wide gaps. Which controller input now?", "options": DESC},
              {"id": "jump", "type": "noul", "instructions": "Is a jump useful right now (enemy or pit or block just ahead)?"},
              {"id": "danger", "type": "noul", "instructions": "Is Mario in immediate danger of dying?"}]
        def tiles_ahead(ram, x, y):
            # SMB tile map: pages of 13 rows x 16 cols at 0x0500 (page 0) and 0x05D0 (page 1); x in level pixels
            def tile(px, py):
                if py < 32 or py >= 240: return 0
                page = (px // 256) % 2; col = (px % 256) // 16; row = (py - 32) // 16
                return int(ram[0x0500 + page * 208 + row * 16 + col])
            ground = [tile(x + dx, 208) != 0 or tile(x + dx, 192) != 0 for dx in (16, 32, 48, 64)]
            block = [tile(x + dx, y) != 0 for dx in (16, 32)]
            return {"ground_ahead_16_32_48_64px": ground, "block_at_mario_height_16_32px": block, "pit_ahead": not all(ground[:3])}
        for ep in range(episodes):
            obs = env.reset(); tick = 0; last = None; act = 1; probs = {n: 1 / 7 for n in NAMES}; nouls = {"jump useful now": 0.0, "immediate danger": 0.0}; lms = 0.0; info = {}; max_x = 0
            for tick in range(max_ticks):
                ram = env.unwrapped.ram
                if tick % decide_every == 0:
                    x = int(info.get("x_pos", 40)); y = int(info.get("y_pos", 79)); yscreen = 240 - y
                    enemies = []
                    for i in range(5):
                        if ram[0x000F + i]:
                            ex = int(ram[0x006E + i]) * 256 + int(ram[0x0087 + i]); ey = int(ram[0x00CF + i])
                            enemies.append({"type": "goomba" if ram[0x0016 + i] in (0, 6) else ("koopa" if ram[0x0016 + i] in (1, 2, 3, 4, 5) else "enemy"), "dx": ex - x, "dy": ey - yscreen})
                    state = {"mario": {"x": x, "y_from_ground": max(0, y - 79), "status": info.get("status", "small"), "moving": (int(ram[0x0057]) if ram[0x0057] < 128 else int(ram[0x0057]) - 256)},
                             "enemies_ahead": sorted([e for e in enemies if -20 < e["dx"] < 160], key=lambda e: e["dx"])[:3], "terrain": tiles_ahead(ram, x, max(32, yscreen)), "time_left": info.get("time", 400), "coins": info.get("coins", 0), "last_action": last, "tick": tick}
                    t1 = time.time(); r = decide(model, state, qs, max_state_tokens=1024); lms = (time.time() - t1) * 1000; lat.append(lms)
                    a = r["answers"]; probs = a[0]["probabilities"]; act = NAMES.index(a[0]["choice"]); nouls = {"jump useful now": a[1]["noul"], "immediate danger": a[2]["noul"]}; last = a[0]["choice"]
                obs, rew, done, info = env.step(act); max_x = max(max_x, int(info.get("x_pos", 0)))
                pan = panel_image("System One plays Super Mario Bros", [f"world {info.get('world','1')}-{info.get('stage','1')}   x {info.get('x_pos',0)}   time {info.get('time',0)}   score {info.get('score',0)}", f"decision #{tick//decide_every}   nearby {len([e for e in enemies if -20 < e['dx'] < 160]) if tick % decide_every == 0 else '-'} enemies", "state = RAM (positions, enemies, tiles) · NES emulator via nes-py"], probs, nouls, lms)
                write(obs, pan)
                if done or info.get("flag_get"):
                    break
            stats.append({"episode": ep, "ticks": tick, "max_x": max_x, "flag": bool(info.get("flag_get")), "life": str(info.get("life")), "score": info.get("score")})
        env.close()
    fps = 30 if game == "doom" else 30
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", f"{outdir}/frames/f%06d.jpg", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", f"{outdir}/{game}.mp4"], check=True)
    shutil.rmtree(outdir + "/frames", ignore_errors=True); vol.commit()
    import statistics
    return {"game": game, "stats": stats, "frames": frame_i, "median_decision_ms": round(statistics.median(lat), 1) if lat else None, "decisions": len(lat), "video": f"{outdir}/{game}.mp4", "seconds": round(frame_i / fps, 1)}


@app.local_entrypoint()
def main(game: str = "doom", episodes: int = 1, max_ticks: int = 900, decide_every: int = 3, scenario: str = "defend_the_center", run: str = "e2b-full", skill: int = 2):
    print(json.dumps(play.remote(game, episodes, max_ticks, decide_every, scenario, run, skill), indent=1))
