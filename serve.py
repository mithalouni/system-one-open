"""Serve the decision model as a Jev-style HTTP API on Modal.

  S1_GPU=L4 S1_RUN=e2b-full modal deploy serve.py
GET / is the demo page. POST /decide accepts either {"context": str, "schema": {field: {type, description, choices}}}
or the Jev-style form {"state": <str|object>, "questions": [
     {"id":"intent","type":"choice","instructions":"...","options":{"billing":"desc",...} | ["a","b"]},
     {"id":"frustration","type":"score","instructions":"...","levels":["calm",...,"hostile"]},
     {"id":"refund","type":"noul","instructions":"...","criteria":{"true":"...","false":"..."}}]}
-> {"answers":[{"id":"intent","type":"choice","choice":"billing","probabilities":{...},"confidence":0.91}, ...], "latency_ms": 41.2}
"""
import modal, os, json
from s1.modal_common import vol, base_image, gpu_kwargs
RUN = os.environ.get("S1_RUN", "lite-270m")
app = modal.App(f"jev-serve-{RUN}")
serve_image = base_image.env({"S1_RUN": RUN, "S1_GPU_NAME": os.environ.get("S1_GPU", "L4")}).add_local_python_source("s1")   # bake the run name into the container env


@app.cls(image=serve_image, volumes={"/vol": vol}, scaledown_window=120, timeout=600, **gpu_kwargs("L4"))
@modal.concurrent(max_inputs=4)
class Server:
    @modal.enter()
    def load(self):
        import threading
        from s1.engine import DecisionModel, decide
        self.model = DecisionModel(f"/vol/runs/{RUN}/model"); self.model.lm.eval()
        self.lock = threading.Lock()
        decide(self.model, "warm up", [{"type": "noul", "instructions": "is this a warm-up?"}])

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse
        from s1.engine import decide
        from s1.demo_presets import PRESETS
        from s1.demo_page import PAGE
        api = FastAPI(title="System One decisions")

        def schema_to_questions(schema: dict):
            qs = []
            for name, f in schema.items():
                t = (f.get("type") or "").lower(); desc = f.get("description") or name.replace("_", " ")
                if t in ("boolean", "bool", "noul"):
                    qs.append({"id": name, "type": "noul", "instructions": desc})
                elif t in ("scale", "score"):
                    qs.append({"id": name, "type": "score", "instructions": desc, "levels": f.get("levels") or f.get("choices") or []})
                else:
                    qs.append({"id": name, "type": "choice", "instructions": desc, "options": f.get("choices") or f.get("options") or []})
            return qs

        from s1.demo_pages2 import RACE, EMAILS, VIRAL
        from s1.demo_race import RACE_STATE, RACE_QUESTIONS, EMAIL_QUESTIONS, VIRAL_QUESTIONS
        GPU_RATE = {"H100": 3.95, "L4": 0.80, "A10": 1.10, "A100-40GB": 2.10, "A100-80GB": 2.50, "T4": 0.59}.get(os.environ.get("S1_GPU_NAME", "L4"), 0.80)
        self._emails = None

        @api.get("/", response_class=HTMLResponse)
        def demo():
            return PAGE

        from s1.demo_pages3 import DRIVE, HOME

        @api.get("/drive", response_class=HTMLResponse)
        def drive_page():
            return DRIVE

        @api.get("/home", response_class=HTMLResponse)
        def home_page():
            return HOME

        @api.get("/race", response_class=HTMLResponse)
        def race_page():
            return RACE

        @api.post("/race_run")
        def race_run():
            with self.lock:
                r = decide(self.model, RACE_STATE, RACE_QUESTIONS, max_state_tokens=4096, max_q_per_pass=32)
            r["model"] = RUN; r["gpu"] = os.environ.get("S1_GPU_NAME", "L4"); r["cost_usd"] = r["latency_ms"] / 1000 * GPU_RATE / 3600
            return r

        @api.get("/emails", response_class=HTMLResponse)
        def emails_page():
            return EMAILS

        @api.get("/emails_data")
        def emails_data():
            if self._emails is None:
                self._emails = json.load(open("/vol/demo/emails.json"))
            return self._emails

        @api.get("/viral", response_class=HTMLResponse)
        def viral_page():
            return VIRAL

        @api.post("/decide_batch")
        def decide_batch(body: dict):
            """Many states, one question set, one batched forward pass."""
            import time as _t
            from s1.schema import Example
            from s1.engine import score_examples, _api_to_examples
            states = body.get("states") or []; task = body.get("task")
            qs = EMAIL_QUESTIONS if task == "email" else body.get("questions", [])
            exs = []
            for st in states:
                e, _ = _api_to_examples(st, qs, 64); exs.append(e[0])
            t0 = _t.time()
            with self.lock:
                res = score_examples(self.model, exs, max_state_tokens=int(body.get("max_state_tokens", 1024)), max_tokens=int(body.get("max_tokens", 32000)))
            lat = (_t.time() - t0) * 1000
            tok = self.model.tok; tokens_in = sum(min(len(tok.encode(st)), int(body.get("max_state_tokens", 1024))) for st in states)
            out = []
            for per_q in res:
                ans = []
                for q, p in zip(qs, per_q):
                    p = [float(x) for x in p]; kind = q["type"]
                    if kind == "noul":
                        ans.append({"id": q["id"], "type": "noul", "noul": p[1]})
                    elif kind == "score":
                        keys = [str(i) for i in range(len(q["levels"]))]; best = max(range(len(p)), key=lambda i: p[i])
                        ans.append({"id": q["id"], "type": "score", "level": keys[best], "score": sum(i * pi for i, pi in enumerate(p)), "probabilities": dict(zip(keys, p))})
                    else:
                        opts = q["options"]; keys = list(opts.keys()) if isinstance(opts, dict) else list(opts); best = max(range(len(p)), key=lambda i: p[i])
                        ans.append({"id": q["id"], "type": "choice", "choice": keys[best], "probabilities": dict(zip(keys, p))})
                out.append(ans)
            return {"results": out, "latency_ms": round(lat, 1), "tokens_in": tokens_in, "model": RUN, "gpu_rate_per_hour": GPU_RATE}

        @api.get("/info")
        def info():
            return {"model": RUN, "temperature": self.model.temperature, "device": self.model.device, "primitives": ["choice", "score", "noul"], "max_single_pass_options": 52, "presets": PRESETS}

        @api.post("/decide")
        def decide_ep(body: dict):
            if body.get("task") == "viral":
                body = {"state": body.get("state", ""), "questions": VIRAL_QUESTIONS}
            if "schema" in body and "questions" not in body:
                state = body.get("context") or body.get("state") or ""; qs = schema_to_questions(body["schema"])
            else:
                state = body.get("state", ""); qs = body.get("questions", [])
            with self.lock:
                r = decide(self.model, state, qs, max_state_tokens=int(body.get("max_state_tokens", 12000)), max_q_per_pass=int(body.get("max_q_per_pass", 32)))
            r["model"] = RUN
            return r
        return api
