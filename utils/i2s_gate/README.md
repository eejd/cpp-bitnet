# I2_S correctness gate

Purpose: decide, with evidence that cannot pass on garbage, whether this fork's CPU path (`-ngl 0`)
computes the same function as the trusted reference for the **same weights**. "The model loaded and
printed words" is not accepted anywhere here.

| piece | file |
|---|---|
| tolerances + rationale (registered before any model-level run) | `tolerances.json` |
| comparison logic (pure numpy) and its unit tests on synthetic data | `compare.py`, `test_compare.py` |
| fixed prompts, fixed perplexity text | `prompts.json`, `ppl_text.txt` |
| reference runner: mlx_lm `bitnet` model on Metal, float32 | `ref_mlx.py` |
| engine runner: libllama CPU, dumps per-position logits / greedy ids / tokenizer ids | `engine_logits.cpp`, `build_engine_logits.sh` |
| orchestrator, exit codes 0 PASS / 1 FAIL / 2 error / 3 BLOCKED | `run_gate.py` |
| tiny random BitNet in both formats (same ternary weights) for plumbing tests | `make_synthetic_model.py` |
| model-independent kernel tests | `../i2s_kernel_test/` |

## Running (needs both model files)

```bash
# 1. build the repo (see docs/macos-arm64.md) with static libs, then the logits tool
utils/i2s_gate/build_engine_logits.sh build build/bin/engine_logits

# 2. official GGUF (microsoft/BitNet-b1.58-2B-4T-gguf, ggml-model-i2_s.gguf) + the reference weights
#    (mlx-community/bitnet-b1.58-2B-4T) must exist locally; the gate never downloads anything.
python3.14 utils/i2s_gate/run_gate.py \
    --gguf models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    --ref-model mlx-community/bitnet-b1.58-2B-4T \
    --engine-bin build/bin/engine_logits --threads 4 --out gate-out/i2s-2B4T --tag "$(git rev-parse --short HEAD)"
```

If a file is missing the script exits with status 3 and writes `result.json` with `"status": "BLOCKED"`
and what is missing. `--ref-model` may be a directory or a repo id present in the local HF cache
(`HF_HUB_CACHE`, `~/.cache/huggingface/hub`, `/hive-cache/codeagent/models/{hf/hub,hub}`).
Reference compute is float32 by default (`--ref-dtype`), ~10 GB for the 2B model.

## What is compared

The reference produces, per fixed prompt, a 16-token greedy continuation and the logits of every
position of `prompt + continuation[:-1]`. The engine is then asked for (a) the logits of the very same
token sequence (teacher forcing), (b) its own free-running greedy continuation of the prompt, (c) its own
tokenization of the prompt text (must equal the reference ids), and for the fixed text (d) the logits of
every position, from which both sides compute perplexity on the same ids.

| check | pass rule (see `tolerances.json`) |
|---|---|
| teacher-forced, prompts pooled and perplexity text | no NaN/Inf; top-1 agreement >= 0.90 on positions outside the near-tie band (reference top1-top2 gap >= max(0.25, 4·sigma), sigma = measured RMS error of that gap in the same run, capped at 10 % of the logit std - see `tolerances.json` history); mean KL(ref‖engine) <= 0.02 nats; max per-position KL <= 0.25; min centered-logit cosine >= 0.99 |
| free-running greedy, per prompt | >= 12 of 16 tokens identical, or the first divergence is at a reference near tie (same band) |
| perplexity | `|ln(ppl_engine/ppl_ref)| <= 0.02` |
| tokenizer parity | engine ids == reference ids |

Why not bit-equality: mlx_lm multiplies **float** activations with the ternary weights; bitnet.cpp
quantizes each activation row to int8 (absmax) first. The tolerances allow that documented noise and are
set so that every defect seen upstream (wrong weight layout, wrong +1 offset, constant output, SiLU instead
of relu², NaN/garbage) fails by orders of magnitude; `test_compare.py` demonstrates this on synthetic data
(`python3.14 utils/i2s_gate/test_compare.py -v`). The tolerances are **not calibrated on a real
engine/reference pair** (none existed when written). If the first real run is within a few percent of a
bound, change `tolerances.json` only in a separate, explained commit.

## Recording the code path

`engine_logits` writes `meta.json` with `llama_print_system_info()` (SIMD features compiled into libllama:
NEON, DOTPROD, MATMUL_INT8, SME, ...), the thread count and the model path; `run_gate.py` copies it into
`result.json`. Pass `--tag` with the commit/patch set. To compare kernel paths build the same commit with
`-DCMAKE_C_FLAGS=-mcpu=apple-m1+nodotprod` etc. and rerun.

## Synthetic model (plumbing only)

```bash
python3.14 utils/i2s_gate/make_synthetic_model.py --out /tmp/syn
python3.14 utils/i2s_gate/run_gate.py --gguf /tmp/syn/synthetic-i2s.gguf --ref-model /tmp/syn/mlx --no-tokenizer \
    --prompts /tmp/syn/prompts.json --ppl-ids /tmp/syn/ppl_ids.json --engine-bin build/bin/engine_logits --out /tmp/syn/out
```

Random weights, byte tokenizer, 2 layers. It proves the loader, kernels, graph (incl. relu²) and the gate
itself agree on a model whose truth is known; it says **nothing** about the official checkpoint (real scale
conventions, tokenizer, embeddings, chat template, accumulated error over 30 layers).
