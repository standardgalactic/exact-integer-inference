# Conformance Record

This document states what the conformance claim of this repository covers, how it is checked, and where the dated evidence lives. It is written to be read without trusting the author: every statement below is either checked by a script in this repository or is marked as an argument that has not been tested exhaustively.

## Claim

The integer reference defined in docs/INTEGER_SEMANTICS.md is implemented three times: in numpy (python/intref.py), in scalar C (c/intref.c) and in CUDA (cuda/intref.cu). For the test sets t24, t128 and t160, all three implementations produce identical int32 values at the embedding output, at the residual stream after each of the 30 decoder blocks, and in the final logits. Identical means equal as integers, with no tolerance. The expected states in vectors/ are produced by the numpy implementation, and the C and CUDA programs compare their own states against them element by element, reporting the first differing index when a difference exists.

## What the claim does not cover

The sets are windows of the wikitext-2 raw test stream, tokenized with the pinned SmolLM2-135M tokenizer, with a window start of the sequence beginning (no BOS token). Agreement on these sets does not establish agreement on every possible input. Accumulator safety is argued from the bounds in docs/INTEGER_SEMANTICS.md, with INT64 accumulation and asserted ranges in the numpy reference, and the sum of squares in the normalization is computed exactly with partial sums that are bounded by construction. Those bounds were not checked by exhaustive or adversarial search. Cached decoding, the Metal backend and ternary or Base-243 packing are not implemented.

The C and CUDA programs assume that right shift of a negative signed integer is arithmetic and that unsigned arithmetic wraps modulo 2^64, as gcc and nvcc both provide. The kernels contain no floating point arithmetic, so compiler flags that change floating point contraction cannot affect results.

Quality, as opposed to conformance, is judged against the float32 baseline and is reported in docs/PROFILE_REPORT.md.

## Inputs

The model is HuggingFaceTB/SmolLM2-135M at revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2. Downloaded assets are pinned by assets.sha256. The model blob, token ids and expected states in vectors/ are pinned by vectors.sha256, which is written and checked by tools/hash_vectors.sh. The exponential, SiLU and RoPE tables are generated at run time with mpmath 1.3.0 and are not separately hashed; an independent cross-check of these tables is open work.

## Procedure

From the repository root on the machine under test, run sh tools/record_conformance.sh. The script verifies the vector hashes, builds and runs the scalar C check, builds and runs the CUDA check on all three sets, and then runs a negative test in which one expected value in t24 is changed by one unit and the CUDA program is required to report a difference. It also rebuilds the CUDA program with -O0 and with a fixed architecture instead of -arch=native, and requires exact agreement on all three sets from each build. The result is written to docs/conformance_runs/ with the date, commit, compiler versions, GPU and driver.

## Recorded runs

The record for the first run is docs/conformance_runs/run-20261108-062226.txt. Add a line for each later run on other hardware.

Date: 2026-11-08. Commit: none, the repository had not been committed at the time of this run. Machine: Ubuntu 24.04 under WSL2 (kernel 6.18.33.1) with an RTX 5060 (8 GB), driver 610.88, gcc 13.3.0. CUDA toolkit: 13.3. Result: t24, t128 and t160 exact at every layer and in the logits; negative test detected; -O0 and fixed-architecture builds exact. The earlier run of the 160-token set reported 0.31 s on device including comparisons.

## Reproducing a failure

If a check fails, the CUDA and C programs print the layer and the first differing index with both values. Compare that layer against the numpy implementation by running python/export_vectors.py for the same window and token count and reading the corresponding checkpoint. A difference at the embedding or at layer zero points at the model blob or the loader; a difference that first appears at a later layer points at the operation named in docs/INTEGER_SEMANTICS.md that is exercised there.

