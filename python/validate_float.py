import json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

setup_determinism()
cfg = load_config()
W = load_weights()
wins, info = load_token_windows()
print("windows", wins.shape, "total tokens", info["total_tokens"])

from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32, attn_implementation="eager")
m.eval()

res = []
for i in (0, 1, 70):
    ids = torch.from_numpy(wins[i])
    t0 = time.time()
    with torch.no_grad():
        mine = forward(W, cfg, ids)
        ref = m(ids[None]).logits[0]
    d = (mine - ref).abs()
    n_mine, cnt = nll_sum(mine, ids)
    n_ref, _ = nll_sum(ref, ids)
    res.append({"window": i, "max_abs_logit_diff": d.max().item(), "mean_abs_logit_diff": d.mean().item(),
                "argmax_agree": (mine.argmax(-1) == ref.argmax(-1)).float().mean().item(),
                "nll_mine": n_mine / cnt, "nll_hf": n_ref / cnt, "sec": time.time() - t0})
    print(res[-1])
json.dump(res, open(os.path.join(OUT_DIR, "float_reference_validation.json"), "w"), indent=2)
