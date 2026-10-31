#!/usr/bin/env python3
"""Trusted reference for the I2_S gate: mlx_lm's `bitnet` model (Metal), float32 compute.

  python3.14 ref_mlx.py --model mlx-community/bitnet-b1.58-2B-4T --prompts prompts.json \
        --ppl-text ppl_text.txt --out ref_dir [--n-new 16] [--dtype float32]

For every prompt it produces (a) the greedy continuation, (b) logits of all positions of
prompt+continuation[:-1] (teacher forcing), and for the perplexity text the logits of every
position. mlx_lm's BitLinear multiplies *float* activations with the ternary weights, so this is
the "no activation quantization" reference; the gate tolerances account for that.

Output: ref_dir/ref.json (+ ref_dir/<name>.npy logits float32 [n_pos, n_vocab]).
Prompts with "ids" bypass the tokenizer (synthetic models have none).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local dir or HF repo id of the mlx-format checkpoint")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--ppl-text", help="text file; ignored if --ppl-ids is given")
    ap.add_argument("--ppl-ids", help="json file with a list of token ids (synthetic models)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-new", type=int, default=16)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--no-tokenizer", action="store_true", help="model dir has no tokenizer (ids only)")
    a = ap.parse_args()

    import mlx.core as mx
    from mlx_lm.utils import load_model, load

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.no_tokenizer:
        from mlx_lm.utils import _download  # type: ignore
        model, _ = load_model(Path(a.model))
        tok = None
    else:
        model, tok = load(a.model)
    model.set_dtype(getattr(mx, a.dtype))
    mx.eval(model.parameters())

    def logits_for(ids: list[int]) -> np.ndarray:
        lg = model(mx.array([ids]))[0]
        return np.array(lg.astype(mx.float32))

    def enc(p: dict) -> list[int]:
        if "ids" in p:
            return list(p["ids"])
        if tok is None:
            raise SystemExit(f"prompt {p['name']} has text but no tokenizer is available")
        return list(tok.encode(p["text"]))

    res = {"model": a.model, "dtype": a.dtype, "n_new": a.n_new, "prompts": [], "ppl": None}
    for p in json.loads(Path(a.prompts).read_text())["prompts"]:
        ids = enc(p)
        seq = list(ids)
        gen, margins = [], []
        for _ in range(a.n_new):
            last = logits_for(seq)[-1]
            top2 = np.partition(last, -2)[-2:]
            margins.append(float(top2[1] - top2[0]))
            nxt = int(last.argmax())
            gen.append(nxt); seq.append(nxt)
        tf_seq = ids + gen[:-1]                       # positions len(ids)-1 .. end predict gen[0..n-1]
        lg = logits_for(tf_seq)[len(ids) - 1:]
        np.save(out / f"{p['name']}.npy", lg.astype(np.float32))
        res["prompts"].append({"name": p["name"], "prompt_ids": ids, "text": p.get("text"),
                               "greedy_ids": gen, "greedy_margins": margins, "tf_seq": tf_seq,
                               "eval_from": len(ids) - 1})
    if a.ppl_ids or a.ppl_text:
        ids = json.loads(Path(a.ppl_ids).read_text()) if a.ppl_ids else list(tok.encode(Path(a.ppl_text).read_text().strip()))
        lg = logits_for(ids)[:-1]
        np.save(out / "ppl.npy", lg.astype(np.float32))
        res["ppl"] = {"ids": ids, "n_targets": len(ids) - 1}
    (out / "ref.json").write_text(json.dumps(res))
    print(f"reference written to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
