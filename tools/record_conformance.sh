#!/bin/sh
# Build and run the C and CUDA checks, run a CUDA negative test and two
# build variants, and write a dated record to docs/conformance_runs/.
# Run from the repository root on the machine being recorded.
set -u
export LC_ALL=C
STAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p docs/conformance_runs
REC=docs/conformance_runs/run-$STAMP.txt
TMP=$(mktemp -d)
VEC=vectors
status=0

say() { printf '%s\n' "$*" | tee -a "$REC"; }
section() { say ""; say "== $* =="; }

: > "$REC"
section "environment"
say "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
say "host: $(uname -srm)"
if git rev-parse --git-dir >/dev/null 2>&1; then
    say "commit: $(git rev-parse HEAD)"
    if [ -n "$(git status --porcelain)" ]; then say "tree: dirty"; else say "tree: clean"; fi
fi
say "gcc: $(gcc --version | head -n 1)"
say "nvcc: $(nvcc --version | tail -n 1)"
say "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader)"
say "python: $(python3 --version 2>&1)"

section "vector hashes"
if sh tools/hash_vectors.sh --check >> "$REC" 2>&1; then
    say "vectors match vectors.sha256"
else
    say "VECTOR HASH MISMATCH"; status=1
fi

section "scalar C check"
if make -C c check >> "$REC" 2>&1; then say "c: pass"; else say "c: FAIL"; status=1; fi

section "CUDA check (make -C cuda check)"
if make -C cuda check >> "$REC" 2>&1; then say "cuda: pass"; else say "cuda: FAIL"; status=1; fi
say "exact lines: $(grep -c 'ALL STATES EXACT' "$REC") (three expected, one per set)"

section "CUDA negative test (one unit changed in t24 expected states)"
cp -r "$VEC/t24" "$TMP/t24_bad"
python3 -I - "$TMP/t24_bad/exp_resid_out.i32" <<'PY'
import sys
p = sys.argv[1]
b = bytearray(open(p, "rb").read())
i = len(b) // 2 // 4 * 4            # an aligned int32 in the middle of the file
v = int.from_bytes(b[i:i+4], "little", signed=True) + 1
b[i:i+4] = v.to_bytes(4, "little", signed=True)
open(p, "wb").write(b)
PY
# The second argument is the directory holding model.bin; adjust if the
# loader expects a different layout.
if ./cuda/intref_cuda "$TMP/t24_bad" "$VEC" > "$TMP/neg.txt" 2>&1 \
   && ! grep -Eq 'DIFFERS|MISMATCH' "$TMP/neg.txt"; then
    say "negative test: NOT DETECTED (the check cannot fail, investigate)"; status=1
else
    say "negative test: detected"
    grep -E 'DIFFERS|MISMATCH' "$TMP/neg.txt" | head -n 3 | tee -a "$REC"
fi

section "CUDA build variants"
for flags in "-O0 -arch=native" "-O3 -arch=sm_120"; do
    bin="$TMP/intref_variant"
    if nvcc $flags -o "$bin" cuda/intref.cu >> "$REC" 2>&1; then
        ok=1
        for t in t24 t128 t160; do
            "$bin" "$VEC/$t" "$VEC" 2>&1 | grep -q 'ALL STATES EXACT' || ok=0
        done
        if [ $ok = 1 ]; then say "variant [$flags]: exact on t24 t128 t160"
        else say "variant [$flags]: FAIL"; status=1; fi
    else
        say "variant [$flags]: build failed"; status=1
    fi
done

rm -rf "$TMP"
section "result"
if [ $status = 0 ]; then say "RECORD OK"; else say "RECORD HAS FAILURES"; fi
say "written to $REC"
exit $status
