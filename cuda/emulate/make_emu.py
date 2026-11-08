"""Build a CPU-emulation copy of ../intref.cu (kernel launches rewritten, CUDA headers replaced by shim.h)."""
import re, sys
src = open("../intref.cu").read()
src = src.replace("#include <cuda_runtime.h>", '#include "shim.h"')
src = src.replace("extern __shared__ i64 sm[];", "i64 *sm = g_dyn_shared;")
SYNC = {"k_rmsnorm", "k_attn"}
out, i = [], 0
while True:
    j = src.find("<<<", i)
    if j < 0: out.append(src[i:]); break
    s = j
    while s > 0 and (src[s-1].isalnum() or src[s-1] == "_"): s -= 1
    name = src[s:j]
    k = src.find(">>>", j)
    cfg = src[j+3:k]
    parts, depth, cur = [], 0, ""
    for ch in cfg:
        if ch in "([": depth += 1
        if ch in ")]": depth -= 1
        if ch == "," and depth == 0: parts.append(cur.strip()); cur = ""
        else: cur += ch
    parts.append(cur.strip())
    g, b = parts[0], parts[1]; shm = parts[2] if len(parts) > 2 else "0"
    a0 = src.index("(", k); depth = 0; e = a0
    while True:
        if src[e] == "(": depth += 1
        if src[e] == ")":
            depth -= 1
            if depth == 0: break
        e += 1
    args = src[a0+1:e]
    out.append(src[i:s])
    out.append(f"emu_launch({name}, {'true' if name in SYNC else 'false'}, dim3({g}), dim3({b}), {shm}, {args})")
    i = e + 1
open("intref_emu.cpp", "w").write("".join(out))
print("wrote intref_emu.cpp")
