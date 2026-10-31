# Building bitnet.cpp on macOS arm64 (Apple Silicon)

Verified on Apple M6, macOS 27.0.1, Apple clang 21.0.0 (Xcode CLT), CMake 3.31 (MacPorts), no conda.
Tracking issue: eejd/cpp-bitnet#2 (upstream reports microsoft/BitNet #682, #641, #635, #618, #99).

> **Status of this branch: builds and runs, but I2_S results on ARM are NOT correct yet.**
> The build fixes below make `llama-cli` / `llama-bench` / `llama-server` link and run. The I2_S CPU
> kernel that ships in the pinned `3rdparty/llama.cpp` decodes the weights with the wrong layout on ARM
> (it is the scalar fallback; the NEON kernels in `src/ggml-bitnet-mad.cpp` are not compiled at all), so
> the output is garbage. That is eejd/cpp-bitnet#3, not this document. Use `-ngl 0` and do not trust
> output until #3 is closed.

## Recipe (no conda, out-of-tree build)

```bash
git clone --recurse-submodules git@github.com:eejd/cpp-bitnet.git
cd cpp-bitnet                     # 3rdparty/llama.cpp must be checked out (git submodule update --init)

cmake -S . -B ../bitnet-build \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
      -DBITNET_ARM_TL1=OFF \
      -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_SERVER=ON
cmake --build ../bitnet-build -j 8

../bitnet-build/bin/llama-cli --version
```

These are the flags `setup_env.py` passes, plus `-B` pointing outside the source tree. The following
are applied automatically and need no flags:

| what | why | where |
|---|---|---|
| `BUILD_SHARED_LIBS=OFF` on Apple | `libggml-base` references `quantize_i2_s` / `dequantize_row_i2_s`, which live in `libggml-cpu`; ld64 rejects the undefined symbols of the default shared build (#682, #641, #635) | `CMakeLists.txt` |
| `GGML_METAL=OFF`, `GGML_BLAS=OFF` on Apple | no I2_S/TL1 kernels exist for Metal; with Metal built, creating the context aborts in `ggml_backend_sched_split_graph`; with the Accelerate/BLAS backend built, every prompt batch of >= 32 tokens is claimed by `ggml_backend_blas` and the process dies with SIGSEGV (#682 "SIGSEGV during prompt processing", #641) | `CMakeLists.txt` |
| `patches/llama.cpp/0001-*.patch` | `src1_cont` is undeclared in `ggml-cpu.c` on every CPU with i8mm (M2 and later) (#618, #682, #641) | `cmake/ApplyLlamaPatches.cmake`, run at configure time |

Override any of them explicitly (`-DBUILD_SHARED_LIBS=ON`, `-DGGML_METAL=ON`, ...) if you know why;
`-DBITNET_APPLY_LLAMA_PATCHES=OFF` skips the patch queue.

### Why patches instead of a submodule bump

`3rdparty/llama.cpp` points at `isHuangXin/llama.cpp`, which this repository cannot push to. Fixes that must
live inside it are carried as `patches/llama.cpp/NNNN-*.patch` and applied with `git apply` at configure time
(idempotent: a patch that is already applied is detected with `git apply --reverse --check` and skipped; one that
neither applies nor reverse-applies stops the configure). The submodule working tree therefore shows as
modified; that is expected.

### Optional

* `-DLLAMA_BUILD_UI=OFF` skips the web UI, which otherwise runs `npm install` / `npm run build` during the build
  (network access, Node.js).
* `-DLLAMA_OPENSSL=OFF` avoids linking OpenSSL (it is found under `/opt/local` if MacPorts' openssl3 is installed).
* Conda is not needed for the C++ build. For the Python converters, make sure the **submodule's** `gguf` is the
  one imported, otherwise you get `AttributeError: type object 'MODEL_ARCH' has no attribute 'BITNET_B158'` /
  `... 'GGMLQuantizationType' has no attribute 'TL1'` (#99; a PyPI/MacPorts `gguf` 0.19.0 shadows it):

  ```bash
  PYTHONPATH=$PWD/3rdparty/llama.cpp/gguf-py python3 utils/convert-hf-to-gguf-bitnet.py --help
  ```

## Smoke test

```bash
../bitnet-build/bin/llama-completion -m path/to/ggml-model-i2_s.gguf -p "The capital of France is" -n 16 -ngl 0 -t 4 -no-cnv --temp 0
../bitnet-build/bin/llama-bench      -m path/to/ggml-model-i2_s.gguf -ngl 0 -t 4 -p 64 -n 16
```

`--version` works for `llama-cli`, `llama-server`, `llama-completion` (`llama-bench` / `llama-quantize` do not take it).

## Reproduction notes (what failed before these changes)

```
# default setup_env.py configuration, shared libs
ld: Undefined symbols for architecture arm64:
  "_dequantize_row_i2_s", referenced from: _type_traits in ggml.c.o
  "_quantize_i2_s", referenced from: _ggml_quantize_chunk in ggml.c.o

# after linking (static libs)
ggml/src/ggml-cpu/ggml-cpu.c:1495:40: error: use of undeclared identifier 'src1_cont'

# after building, Metal + BLAS backends on, prompt of >= 31 tokens, -ngl 0
engine: fatal signal 11 ... ggml_backend_blas_graph_compute ... ggml_backend_sched_graph_compute_async

# Metal on, BLAS off
ggml.c:1789: GGML_ASSERT(view_src == NULL || data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src)) failed
  ggml_view_tensor <- ggml_backend_sched_split_graph <- ggml_backend_sched_reserve <- llama_init_from_model
```

Not done here: Windows / Linux builds (the `BUILD_SHARED_LIBS` default is Apple-only), the TL1 path
(`-DBITNET_ARM_TL1=ON` needs model-specific generated kernels), and the upstream PR #635 `tensor->backend`
guard (this ggml has no `tensor->backend` / `GGML_BACKEND_TYPE_CPU`, so it would not compile).
