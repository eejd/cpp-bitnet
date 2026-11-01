# Building bitnet.cpp on macOS arm64 (Apple Silicon)

Verified on Apple M6, macOS 27.0.1, Apple clang 21.0.0 (Xcode CLT), CMake 3.31 (MacPorts), no conda.
Tracking issue: eejd/cpp-bitnet#2 (upstream reports microsoft/BitNet #682, #641, #635, #618, #99).

> **Status of this branch.** The build fixes make `llama-cli` / `llama-bench` / `llama-server` link and run.
> On top of them, `patches/llama.cpp/0002` (I2_S NEON kernel in the canonical 128-weight/32-byte layout,
> both `vec_dot` calling conventions) and `0003` (squared-ReLU FFN) make the **official**
> `ggml-model-i2_s.gguf` agree with mlx_lm's `bitnet` model on Apple M6 (`-ngl 0`): teacher-forced top-1
> agreement 1.000, mean KL 0.0026 nats, perplexity 7.1785 vs 7.1532 on the gate text, greedy continuations
> equal except at reference near-ties (tolerances and method: `utils/i2s_gate/README.md` on
> `test/i2s-correctness-gate`, eejd/cpp-bitnet#3). Without 0002/0003 the output is garbage
> (`The capital of France is@@@@@@@@@@@@@@@@`). `patches/llama.cpp/0004` makes `llama-quantize ... I2_S`
> (what `setup_env.py` uses to make a local GGUF) usable: it was `invalid ftype 'I2_S'`, and
> isHuangXin/llama.cpp#2 wrote a non-canonical packing. With 0004 the 14 I2_S tensors of the synthetic f32 model
> are byte-identical to the canonical packing at 1, 4 and default threads, and the gate passes on that file.
> The official GGUF has no `tokenizer.ggml.pre`; `patches/llama.cpp/0005` makes the loader assume `llama-bpe` for
> exactly that case (arch `bitnet-b1.58`, 128256 tokens, 280147 merges, no pre key; the merge table is identical to
> Llama-3.2's). Without it, prompts containing e.g. `(n):\n` tokenized differently from the Llama-3 tokenizer and the
> gate's tokenizer parity failed on 2 of 5 prompts; with it all 5 pass on the unmodified file. Any other GGUF
> without the key still gets the "missing pre-tokenizer" warning and the `default` regexes. The converter already
> writes `llama-bpe` for this tokenizer (its check-hash matches Meta-Llama-3), so only existing files needed this.

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
| `GGML_METAL=OFF`, `GGML_BLAS=OFF` on Apple | no I2_S/TL1 kernels exist for Metal; with Metal built (BLAS off) creating the context aborts in `ggml_backend_sched_split_graph`; with Metal + Accelerate/BLAS built, prompt batches of >= 32 tokens are claimed by `ggml_backend_blas` and the process dies with SIGSEGV (#682 "SIGSEGV during prompt processing", #641); with BLAS built and Metal off there is no crash but the BLAS backend still claims those batches and returns garbage (synthetic gate: perplexity 4e27 vs 3.7e5, kernels otherwise correct) | `CMakeLists.txt` |
| `patches/llama.cpp/0001-*.patch` | `src1_cont` is undeclared in `ggml-cpu.c` on every CPU with i8mm (M2 and later) (#618, #682, #641) | `cmake/ApplyLlamaPatches.cmake`, run at configure time |
| `patches/llama.cpp/0002-*.patch` ... `0005-*.patch` | I2_S kernel layout/offset/calling-convention fix; squared-ReLU FFN; `llama-quantize ... I2_S` with the canonical layout; `llama-bpe` pre-tokenizer fallback for the official BitNet GGUF (see status above) | same |

Override any of them explicitly (`-DBUILD_SHARED_LIBS=ON`, `-DGGML_METAL=ON`, ...) if you know why;
`-DBITNET_APPLY_LLAMA_PATCHES=OFF` skips the patch queue (and with it the correctness fixes).

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
# BLAS on, Metal off: no crash, wrong numbers for batches >= 32 tokens

# Metal on, BLAS off
ggml.c:1789: GGML_ASSERT(view_src == NULL || data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src)) failed
  ggml_view_tensor <- ggml_backend_sched_split_graph <- ggml_backend_sched_reserve <- llama_init_from_model
```

Not done here: Windows / Linux builds (the `BUILD_SHARED_LIBS` default is Apple-only), the TL1 path
(`-DBITNET_ARM_TL1=ON` needs model-specific generated kernels), and the upstream PR #635 `tensor->backend`
guard (this ggml has no `tensor->backend` / `GGML_BACKEND_TYPE_CPU`, so it would not compile).
