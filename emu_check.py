import modal
app = modal.App("jev-emu-check")
image = (modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg", "build-essential", "cmake", "libsdl2-dev", "libboost-all-dev", "libopenal-dev", "zlib1g-dev", "libjpeg-dev", "libbz2-dev", "libgtk2.0-dev", "libfluidsynth-dev", "libgme-dev", "libsndfile1-dev", "libwildmidi-dev", "timidity", "nasm", "tar", "libopenal-dev", "libpng-dev", "libglib2.0-dev", "libmpg123-dev")
         .uv_pip_install("vizdoom==1.2.4", "gym==0.23.1", "gym-super-mario-bros==7.4.0", "nes-py==8.2.1", "numpy<2", "pillow", "imageio"))
@app.function(image=image, timeout=900)
def check():
    out = {}
    try:
        import vizdoom as vzd, os
        g = vzd.DoomGame(); g.load_config(os.path.join(vzd.scenarios_path, "defend_the_center.cfg")); g.set_window_visible(False); g.set_screen_format(vzd.ScreenFormat.RGB24); g.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        g.set_labels_buffer_enabled(True); g.add_available_game_variable(vzd.GameVariable.POSITION_X); g.add_available_game_variable(vzd.GameVariable.POSITION_Y); g.add_available_game_variable(vzd.GameVariable.ANGLE)
        g.init(); g.new_episode(); s = g.get_state()
        out["vizdoom"] = dict(ok=True, buttons=[str(b) for b in g.get_available_buttons()], vars=[str(v) for v in g.get_available_game_variables()], labels=[(l.object_name, round(l.object_position_x, 1)) for l in s.labels][:6], screen=list(s.screen_buffer.shape), scenarios=sorted(os.listdir(vzd.scenarios_path))[:20])
        g.close()
    except Exception as e:
        import traceback; out["vizdoom"] = {"error": traceback.format_exc()[-800:]}
    try:
        import gym_super_mario_bros, nes_py
        from nes_py.wrappers import JoypadSpace
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0"); env = JoypadSpace(env, SIMPLE_MOVEMENT)
        obs = env.reset(); r = env.step(1)
        out["mario"] = dict(ok=True, obs=list((r[0] if isinstance(r, tuple) else obs).shape), actions=SIMPLE_MOVEMENT, info={k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in (r[3] if len(r) == 4 else r[4]).items()}, ram=len(env.unwrapped.ram))
        env.close()
    except Exception as e:
        import traceback; out["mario"] = {"error": traceback.format_exc()[-800:]}
    return out
@app.local_entrypoint()
def main():
    import json; print(json.dumps(check.remote(), indent=1, default=str))
