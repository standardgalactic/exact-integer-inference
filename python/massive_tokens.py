"""Which positions and token ids carry the massive residual activation at layer 12 (calibration windows)."""
import json, os, sys, collections
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from tokenizers import Tokenizer

class Stop(Exception): pass
class R:
    def __init__(self): self.x = None
    def chan(self, site, l, t):
        if site == "resid_in" and l == 12:
            self.x = t.clone(); raise Stop
    def flat(self, *a, **k): pass
    def heads(self, *a, **k): pass

setup_determinism()
cfg = load_config(); W = load_weights()
wins, _ = load_token_windows()
tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
pos_counter = collections.Counter(); extras = []
for i in range(64):
    r = R()
    try:
        with torch.no_grad(): forward(W, cfg, torch.from_numpy(wins[i]), rec=r)
    except Stop: pass
    m = r.x.abs().amax(1)
    idx = (m >= 1024).nonzero().flatten().tolist()
    for p in idx:
        pos_counter[p] += 1
        if p != 0:
            extras.append({"window": i, "pos": p, "token_id": int(wins[i][p]), "token": tok.decode([int(wins[i][p])]), "absmax": float(m[p]),
                           "prev": tok.decode([int(wins[i][p-1])])})
    first = float(m[0])
    if i < 3: print("window", i, "pos0 absmax", first, "n massive", len(idx))
out = {"windows": 64, "positions_with_massive_token": dict(pos_counter), "non_first_position_cases": extras,
       "window_first_tokens_massive": pos_counter[0]}
json.dump(out, open(os.path.join(OUT_DIR, "massive_tokens.json"), "w"), indent=1)
print(json.dumps(out, indent=1)[:1500])
