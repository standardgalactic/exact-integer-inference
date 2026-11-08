"""Write the pinned model manifest: revision, hashes, software, numerics settings."""
import json, os, platform, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from safetensors import safe_open

setup_determinism()
cfg = load_config()

files = {}
for fn in sorted(os.listdir(MODEL_DIR)):
    p = os.path.join(MODEL_DIR, fn)
    files[fn] = {"bytes": os.path.getsize(p), "sha256": sha256_file(p)}

tensors = {}
with safe_open(os.path.join(MODEL_DIR, "model.safetensors"), "pt") as f:
    for k in sorted(f.keys()):
        t = f.get_tensor(k)
        tensors[k] = {"dtype": str(t.dtype), "shape": list(t.shape),
                      "sha256_of_raw_bytes": hashlib.sha256(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"").hexdigest()}

wins, info = load_token_windows()
data_file = os.path.join(DATA_DIR, "wikitext2_test.parquet")

import transformers, tokenizers, safetensors, pyarrow, numpy
cpu = ""
for line in open("/proc/cpuinfo"):
    if line.startswith("model name"):
        cpu = line.split(":", 1)[1].strip(); break

manifest = {
    "model": {
        "repo": "HuggingFaceTB/SmolLM2-135M",
        "revision": REVISION,
        "license": "apache-2.0",
        "files": files,
        "config_fields_used": {k: cfg[k] for k in [
            "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "vocab_size", "rms_norm_eps", "rope_theta", "rope_interleaved",
            "rope_scaling", "hidden_act", "tie_word_embeddings", "max_position_embeddings",
            "attention_bias", "torch_dtype"]},
        "derived": {"head_dim": cfg["hidden_size"] // cfg["num_attention_heads"],
                    "gqa_group_size": cfg["num_attention_heads"] // cfg["num_key_value_heads"]},
        "stored_weight_dtype": "bfloat16",
        "note_weight_dtype": "Checkpoint tensors are bf16. Widening bf16 to float32 is exact, so the float32 baseline carries exactly the checkpointed weight values.",
        "tensors": tensors,
    },
    "tokenizer": {
        "assets": {k: files[k] for k in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]},
        "encoding": "tokenizers.Tokenizer.from_file(tokenizer.json).encode(text, add_special_tokens=False); no BOS or EOS inserted",
    },
    "heldout_text": {
        "dataset": "Salesforce/wikitext, config wikitext-2-raw-v1, split test",
        "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "file": "wikitext-2-raw-v1/test-00000-of-00001.parquet",
        "file_sha256": sha256_file(data_file),
        "construction": "'\\n\\n'.join(rows of column text), tokenized once as a single stream, cut into non-overlapping windows of 512 tokens, tail dropped",
        "caveat": "Held out from our calibration only. It is not guaranteed absent from SmolLM2 pretraining data.",
        **{k: v for k, v in info.items() if k != "window_start_offsets"},
        "window_start_offsets": "window i starts at token offset 512*i of the single stream",
        "seq_len": SEQ_LEN,
        "calibration_windows": [0, N_CAL - 1],
        "validation_windows": [N_CAL, N_CAL + N_VAL - 1],
        "calibration_ids_sha256": hashlib.sha256(wins[:N_CAL].tobytes()).hexdigest(),
        "validation_ids_sha256": hashlib.sha256(wins[N_CAL:N_CAL + N_VAL].tobytes()).hexdigest(),
        "each_window_independent": "Every window is run as its own sequence with positions 0..511, batch size one, no BOS; the first token of a window acts as the attention sink.",
    },
    "software": {
        "python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__, "safetensors": safetensors.__version__,
        "numpy": numpy.__version__, "pyarrow": pyarrow.__version__,
    },
    "numerics": {
        "dtype": "float32 throughout the reference forward, softmax and normalization in float32, NLL accumulated in float64",
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": True,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "attention": "explicit causal softmax, scale 1/sqrt(64), GQA by repeat_interleave of KV heads (query head h uses KV head h // 3)",
        "rope": "rotate_half convention, non-interleaved, theta=100000, positions 0..T-1",
        "masking": "masked entries are -inf before softmax in the float reference; all range statistics exclude them",
        "platform": platform.platform(), "cpu": cpu,
    },
    "validation_against_huggingface": json.load(open(os.path.join(OUT_DIR, "float_reference_validation.json"))),
}
json.dump(manifest, open(os.path.join(OUT_DIR, "model_manifest.json"), "w"), indent=2)
print("manifest written;", len(tensors), "tensors;", info["n_windows"], "windows")
