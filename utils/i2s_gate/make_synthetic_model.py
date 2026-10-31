#!/usr/bin/env python3
"""Write a tiny random BitNet-b1.58 model in BOTH formats from the same ternary weights.

  python3.14 make_synthetic_model.py --out DIR [--seed 0]

  DIR/mlx/               config.json + model.safetensors  (what mlx_lm's `bitnet` model loads)
  DIR/synthetic-i2s.gguf arch bitnet-b1.58, linear weights I2_S in the canonical 128/32 layout
                         (same packing as utils/convert-bitnet-embedding-to-gguf.py), f16 embedding,
                         f32 norms, tied output
  DIR/prompts.json, DIR/ppl_ids.json   id-based inputs for the gate

THIS IS NOT THE OFFICIAL MODEL. It exercises the loader, the I2_S kernels and the whole forward
pass against an independent implementation, but says nothing about the real checkpoint (weights,
tokenizer, chat template, scale conventions). Random weights give low-contrast logits.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "3rdparty" / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402


def pack_canonical(t: np.ndarray) -> np.ndarray:
    """t: int8 ternary [M, K] (K % 128 == 0) -> uint8 [M*K/4]: 128-weight blocks in 32 bytes,
    byte gp of a block holds weights gp, 32+gp, 64+gp, 96+gp at bit positions 6,4,2,0."""
    code = (t + 1).astype(np.uint8).reshape(-1, 4, 32)
    return ((code[:, 0] << 6) | (code[:, 1] << 4) | (code[:, 2] << 2) | code[:, 3]).reshape(-1)


def pack_mlx(t: np.ndarray) -> np.ndarray:
    """mlx_lm BitLinear layout: uint8 [M/4, K]; byte [r,k] holds rows r + i*(M/4) at bit 2i."""
    M, K = t.shape
    code = (t + 1).astype(np.uint8).reshape(4, M // 4, K)
    return code[0] | (code[1] << 2) | (code[2] << 4) | (code[3] << 6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--ffn", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--vocab", type=int, default=300)
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--rope-theta", type=float, default=10000.0)
    a = ap.parse_args()
    H, F, L, NH, NKV, V = a.hidden, a.ffn, a.layers, a.heads, a.kv_heads, a.vocab
    HD = H // NH
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); (out / "mlx").mkdir(parents=True, exist_ok=True)

    emb = (rng.normal(0, 0.25, (V, H)).astype(np.float16)).astype(np.float32)  # f16-representable
    def norm(n): return (1.0 + 0.1 * rng.normal(size=n)).astype(np.float32)
    def tern(m, k):
        t = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(m, k), p=[0.25, 0.5, 0.25])
        s = np.float32(rng.uniform(0.7, 1.4) / np.sqrt(0.5 * k))
        return t, s

    shapes = {"q": (H, H), "k": (NKV * HD, H), "v": (NKV * HD, H), "o": (H, H),
              "gate": (F, H), "up": (F, H), "down": (H, F)}
    hf_names = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj",
                "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}
    gg_names = {"q": "attn_q", "k": "attn_k", "v": "attn_v", "o": "attn_output",
                "gate": "ffn_gate", "up": "ffn_up", "down": "ffn_down"}

    layers = []
    for _ in range(L):
        d = {"attn_norm": norm(H), "attn_sub_norm": norm(H), "ffn_norm": norm(H), "ffn_sub_norm": norm(F)}
        for k, (m, kk) in shapes.items():
            d[k] = tern(m, kk)
        layers.append(d)
    out_norm = norm(H)

    # ---- mlx checkpoint
    import mlx.core as mx
    w = {"model.embed_tokens.weight": mx.array(emb), "model.norm.weight": mx.array(out_norm)}
    for i, d in enumerate(layers):
        p = f"model.layers.{i}."
        w[p + "input_layernorm.weight"] = mx.array(d["attn_norm"])
        w[p + "self_attn.attn_sub_norm.weight"] = mx.array(d["attn_sub_norm"])
        w[p + "post_attention_layernorm.weight"] = mx.array(d["ffn_norm"])
        w[p + "mlp.ffn_sub_norm.weight"] = mx.array(d["ffn_sub_norm"])
        for k in shapes:
            t, s = d[k]
            w[p + hf_names[k] + ".weight"] = mx.array(pack_mlx(t))
            w[p + hf_names[k] + ".weight_scale"] = mx.array(np.array([1.0 / s], dtype=np.float32))  # inverted by mlx_lm
    mx.save_safetensors(str(out / "mlx" / "model.safetensors"), w)
    (out / "mlx" / "config.json").write_text(json.dumps({
        "model_type": "bitnet", "hidden_size": H, "num_hidden_layers": L, "intermediate_size": F,
        "num_attention_heads": NH, "num_key_value_heads": NKV, "rms_norm_eps": a.eps, "vocab_size": V,
        "rope_theta": a.rope_theta, "max_position_embeddings": 2048, "tie_word_embeddings": True,
        "quantization_config": {"quant_method": "bitnet", "linear_class": "bitlinear"}}))

    # ---- GGUF (I2_S)
    g = gguf.GGUFWriter(str(out / "synthetic-i2s.gguf"), "bitnet-b1.58")
    g.add_name("synthetic-bitnet-b1.58-tiny")
    g.add_context_length(2048); g.add_embedding_length(H); g.add_block_count(L)
    g.add_feed_forward_length(F); g.add_head_count(NH); g.add_head_count_kv(NKV)
    g.add_layer_norm_rms_eps(a.eps); g.add_rope_freq_base(a.rope_theta); g.add_rope_dimension_count(HD)
    g.add_uint32("general.file_type", 41)  # LLAMA_FTYPE_MOSTLY_I2_S
    toks = ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)]
    types = [gguf.TokenType.UNKNOWN, gguf.TokenType.CONTROL, gguf.TokenType.CONTROL] + [gguf.TokenType.BYTE] * 256
    extra = [chr(ord("a") + i) for i in range(26)] + ["\u2581" + c for c in "abcdefghijklmno"]
    toks += extra[: V - len(toks)]; types += [gguf.TokenType.NORMAL] * (len(toks) - len(types))
    assert len(toks) == V, (len(toks), V)
    g.add_tokenizer_model("llama"); g.add_token_list(toks); g.add_token_scores([0.0] * V); g.add_token_types(types)
    g.add_bos_token_id(1); g.add_eos_token_id(2); g.add_unk_token_id(0); g.add_add_bos_token(True)
    g.add_tensor("token_embd.weight", emb.astype(np.float16))
    g.add_tensor("output_norm.weight", out_norm)
    for i, d in enumerate(layers):
        g.add_tensor(f"blk.{i}.attn_norm.weight", d["attn_norm"]); g.add_tensor(f"blk.{i}.attn_sub_norm.weight", d["attn_sub_norm"])
        g.add_tensor(f"blk.{i}.ffn_norm.weight", d["ffn_norm"]); g.add_tensor(f"blk.{i}.ffn_sub_norm.weight", d["ffn_sub_norm"])
        for k, (m, kk) in shapes.items():
            t, s = d[k]
            packed = pack_canonical(t)
            blob = np.zeros(m * kk // 4 + 32, dtype=np.uint8)
            blob[: packed.size] = packed
            blob[packed.size: packed.size + 4] = np.frombuffer(np.float32(s).tobytes(), dtype=np.uint8)
            g.add_tensor(f"blk.{i}.{gg_names[k]}.weight", blob, raw_shape=(m, kk), raw_dtype=gguf.GGMLQuantizationType.I2_S)
    g.write_header_to_file(); g.write_kv_data_to_file(); g.write_tensors_to_file(); g.close()

    # ---- id-based gate inputs (byte tokens 3..258 behave like text bytes)
    ids = lambda s: [1] + [3 + b for b in s.encode()]
    (out / "prompts.json").write_text(json.dumps({"prompts": [
        {"name": n, "ids": ids(s)} for n, s in [("a", "The capital of France is"), ("b", "Water boils at"),
                                                ("c", "def f(n):\n    "), ("d", "Once upon a time,"), ("e", "12 + 30 =")]]}))
    txt = (Path(__file__).with_name("ppl_text.txt")).read_text().strip()[:400]
    (out / "ppl_ids.json").write_text(json.dumps(ids(txt)))
    print(f"synthetic model written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
