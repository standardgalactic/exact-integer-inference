#!/bin/sh
# Write or verify vectors.sha256 for the model blob and every expected state.
# Run from the repository root.
#   sh tools/hash_vectors.sh            write vectors.sha256
#   sh tools/hash_vectors.sh --check    verify against vectors.sha256
set -eu
export LC_ALL=C
OUT=vectors.sha256

list() {
    ls vectors/model.bin vectors/*/ids.i64 vectors/*/exp_*.i32 \
       vectors/*/manifest.json 2>/dev/null | sort
}

if [ "${1:-}" = "--check" ]; then
    sha256sum -c "$OUT"
else
    list | xargs sha256sum > "$OUT"
    echo "wrote $OUT ($(wc -l < "$OUT") entries)"
fi

