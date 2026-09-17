"""Core data structures + prompt construction.

An Example is one *state* plus one or more typed questions. Question kinds:
  choice : pick one of N options (optionally with descriptions)        -> distribution over options
  noul   : yes/no proposition (optional criteria for true/false)       -> P(yes)
  score  : ordered levels (rubric)                                     -> distribution over levels, expected level

Every question gets one *answer slot* in the prompt ("Answer k: ("). The model is read at that slot only,
restricted to the option-label tokens. No decoding. All slots of an example come from ONE forward pass.
"""
from __future__ import annotations
import json, random
from dataclasses import dataclass, field, asdict

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"   # 52 single-token labels in Gemma's vocab
MAX_OPTIONS = len(LETTERS)
NONE_OPT = "none of the above"


@dataclass
class Q:
    text: str                       # the instruction / question
    options: list                   # canonical option strings (for noul: ["no","yes"]; for score: level labels in order)
    gold: int = -1                  # index into options (-1 unknown)
    kind: str = "choice"            # choice | noul | score
    descs: list | None = None       # optional per-option descriptions (same length as options)
    id: str | None = None           # optional id (api use)


@dataclass
class Example:
    state: str
    qs: list                        # list[Q]
    task: str = ""
    meta: dict = field(default_factory=dict)

    def to_json(self):
        return json.dumps({"state": self.state, "task": self.task, "meta": self.meta,
                           "qs": [asdict(q) for q in self.qs]}, ensure_ascii=False)

    @staticmethod
    def from_json(s):
        d = json.loads(s)
        return Example(d["state"], [Q(**q) for q in d["qs"]], d.get("task", ""), d.get("meta", {}))


KIND_TAG = {"choice": "choice", "noul": "yes/no", "score": "score"}


def render_option(letter, opt, desc):
    opt = str(opt).strip()
    if desc:
        return f"({letter}) {opt} — {str(desc).strip()}"
    return f"({letter}) {opt}"


def truncate_tokens(tok, text, max_tokens):
    """Token-budget truncation keeping head (70%) and tail (30%) of the text."""
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return ids
    head = int(max_tokens * 0.7); tail = max_tokens - head - 8
    mid = tok.encode("\n[... truncated ...]\n", add_special_tokens=False)
    return ids[:head] + mid + ids[-tail:]


def build(example: Example, tok, rng=None, max_state_tokens=1536, shuffle_choice=True, max_options=MAX_OPTIONS,
          train=True):
    """Tokenize one example into a single sequence with one answer slot per question.

    Returns dict(ids, slots, golds, nopts, perms) where perms[k] maps displayed position j -> original option index.
    For 'choice' questions in training, options are shuffled and sub-sampled to <= max_options (gold kept).
    'noul' is always (A) no (B) yes.  'score' levels keep their order.
    """
    rng = rng or random
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    ids = list(bos) + tok.encode("<state>\n", add_special_tokens=False)
    ids += truncate_tokens(tok, example.state, max_state_tokens)
    ids += tok.encode("\n</state>\n", add_special_tokens=False)
    slots, golds, nopts, perms = [], [], [], []
    n = len(example.qs)
    for k, q in enumerate(example.qs):
        opts = list(range(len(q.options)))
        if q.kind == "choice":
            if len(opts) > max_options:
                others = [i for i in opts if i != q.gold]
                keep = rng.sample(others, max_options - 1) + ([q.gold] if q.gold >= 0 else [others[0]])
                opts = keep
                rng.shuffle(opts)          # never leave the kept gold at a fixed position (eval mode included)
            elif shuffle_choice and train:
                rng.shuffle(opts)
        label = f"Question {k + 1}" if n > 1 else "Question"
        alabel = f"Answer {k + 1}" if n > 1 else "Answer"
        lines = [f"\n{label} ({KIND_TAG.get(q.kind, q.kind)}): {q.text.strip()}"]
        if q.kind == "score":
            lines.append("\nLevels:")
        elif q.kind == "choice":
            lines.append("\nOptions:")
        for j, oi in enumerate(opts):
            d = q.descs[oi] if q.descs and oi < len(q.descs) else None
            lines.append("\n" + render_option(LETTERS[j], q.options[oi], d))
        lines.append(f"\n{alabel}: (")
        piece = tok.encode("".join(lines), add_special_tokens=False)
        ids.extend(piece)
        slots.append(len(ids) - 1)
        golds.append(opts.index(q.gold) if q.gold in opts else -1)
        nopts.append(len(opts))
        perms.append(opts)
    return dict(ids=ids, slots=slots, golds=golds, nopts=nopts, perms=perms)


def letter_ids(tok):
    out = []
    for L in LETTERS:
        t = tok.encode(L, add_special_tokens=False)
        assert len(t) == 1, (L, t)
        out.append(t[0])
    return out


def collate(items, pad_id, bucket=64):
    import torch
    T = max(len(it["ids"]) for it in items)
    T = ((T + bucket - 1) // bucket) * bucket
    B = len(items)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    slot_idx, slot_batch, golds, nopts = [], [], [], []
    for b, it in enumerate(items):
        n = len(it["ids"])
        input_ids[b, :n] = torch.tensor(it["ids"])
        attn[b, :n] = 1
        for k, s in enumerate(it["slots"]):
            slot_idx.append(s); slot_batch.append(b); golds.append(it["golds"][k]); nopts.append(it["nopts"][k])
    return dict(input_ids=input_ids, attention_mask=attn, slot_idx=torch.tensor(slot_idx), slot_batch=torch.tensor(slot_batch),
                golds=torch.tensor(golds), nopts=torch.tensor(nopts))


def batches_by_tokens(items, max_tokens, rng, bucket=64, max_rows=384):
    """Group items into micro-batches with a token budget (B*T <= max_tokens)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for i, it in enumerate(items):
        T = ((len(it["ids"]) + bucket - 1) // bucket) * bucket
        groups[T].append(i)
    out = []
    for T, idx in groups.items():
        rng.shuffle(idx)
        B = max(1, min(max_rows, max_tokens // T))
        for c in range(0, len(idx), B):
            out.append(idx[c:c + B])
    rng.shuffle(out)
    return out
