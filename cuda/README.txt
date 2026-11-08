CUDA check of the integer reference (SmolLM2-135M)

Needs an NVIDIA GPU, a CUDA toolkit that supports it (an RTX 5060 is compute capability 12.0, so CUDA 12.8 or newer),
and a host compiler that toolkit supports. Check with: nvcc --version

From this directory:

  make check

or by hand:

  nvcc -O2 -arch=native -o intref_cuda intref.cu
  ./intref_cuda ../vectors/t24 ../vectors
  ./intref_cuda ../vectors/t128 ../vectors
  ./intref_cuda ../vectors/t160 ../vectors

Each run prints one line per layer, exact or DIFFERS, and ends with ALL STATES EXACT or FAIL. The second argument is the
directory that holds model.bin. The kernels are written for clarity, not speed.

emulate/ contains a CPU emulation of the kernels (threads and barriers) used to test the kernel logic without a GPU:

  cd emulate && python3 make_emu.py && g++ -std=c++20 -O2 -pthread -o intref_emu intref_emu.cpp
  ./intref_emu ../../vectors/t24 ../../vectors
