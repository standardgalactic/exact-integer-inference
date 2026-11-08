# Exact Integer Inference

An integer-only execution of a pretrained transformer (HuggingFaceTB/SmolLM2-135M, revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2) in which every stored value, rounding point and nonlinear function is fixed by specification, so that independent implementations agree on every intermediate state. A numpy prototype defines the semantics, a scalar C program and a CUDA program reproduce it, and a measured activation profile justifies the number formats.

## Layout

docs/ holds the written results: PROFILE_REPORT.md (pinned manifest, activation profile, scaling proposal, tied-embedding study), INTEGER_SEMANTICS.md (the fixed semantics and the first quality measurements) and CONFORMANCE.md (what the conformance claim covers and what it does not). python/ holds the profiling and reference code. c/ and cuda/ hold the two reimplementations. tools/ holds hash_vectors.sh, which writes and verifies vectors.sha256, and record_conformance.sh, which runs every conformance check and writes a dated record to docs/conformance_runs/. results/ holds the small JSON and CSV outputs. vectors/ holds the test vector manifests; the large expected-state files are regenerated and checked against vectors.sha256.

## Reproducing the conformance check

Install the requirements, fetch the pinned inputs, rebuild the test vectors, and run the checks:

    pip install -r requirements.txt
    sh python/fetch_assets.sh
    python3 python/export_vectors.py vectors/t24 64 24
    python3 python/export_vectors.py vectors/t128 64 128
    python3 python/export_vectors.py vectors/t160 100 160
    mv vectors/t128/model.bin vectors/model.bin
    rm vectors/t24/model.bin vectors/t160/model.bin
    sh tools/hash_vectors.sh --check
    make -C c check
    make -C cuda check

The vector sets t24, t128 and t160 correspond to window 64 with 24 tokens, window 64 with 128 tokens, and window 100 with 160 tokens. The hash check confirms that the regenerated model blob and expected states are byte-identical to the ones the recorded run used. A pass ends with ALL STATES EXACT for each set. To run everything at once, including a negative test and two alternative CUDA builds, use sh tools/record_conformance.sh.

The CUDA check needs an NVIDIA GPU and a CUDA toolkit new enough for the card (12.8 or later for Blackwell cards; 13.3 was used for the recorded run, under WSL2 on an RTX 5060).

## Full pipeline

The profile, scaling proposal and quality measurements in results/ and docs/ come from the longer pipeline below. Profiling runs on one thread with deterministic algorithms; each full profile pass takes about 20 minutes on a two-core CPU.

    python3 python/validate_float.py
    python3 python/manifest.py
    python3 python/profile_acts.py cal 0 64
    python3 python/profile_acts.py val 64 128
    python3 python/summarize.py
    python3 python/extra_analysis.py
    python3 python/propose_scaling.py
    python3 python/run_model.py wide 64 72

## Expected results

Float32 baseline perplexity on the validation windows is 20.89. Over eight validation windows (4,088 tokens) the float baseline scores 3.1523 nats per token, INT8 weights alone 3.1587, the wide integer engine 3.1585, and the all-INT16 variant 3.2731. The C and CUDA programs each reproduce every layer residual and every logit of the numpy prototype exactly; see docs/CONFORMANCE.md and docs/conformance_runs/ for the recorded run.

## License

CC0 1.0 Universal (public domain dedication). See LICENSE.