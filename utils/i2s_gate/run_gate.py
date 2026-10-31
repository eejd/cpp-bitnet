#!/usr/bin/env python3
"""I2_S correctness gate: engine (bitnet.cpp GGUF, CPU) vs mlx_lm `bitnet` reference.

  python3.14 utils/i2s_gate/run_gate.py \
      --gguf models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
      --ref-model mlx-community/bitnet-b1.58-2B-4T \
      --engine-bin build/bin/engine_logits   # built by build_engine_logits.sh

Exit status: 0 PASS, 1 FAIL, 2 error, 3 BLOCKED (a required model file/repo is missing; nothing was run).
Tolerances: tolerances.json (registered before any model-level run). Details: README.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare as C  # noqa: E402


def blocked(out: Path, missing: list[str]) -> int:
    msg = {"status": "BLOCKED", "missing": missing}
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(msg, indent=1))
    print("GATE BLOCKED - missing: " + "; ".join(missing), file=sys.stderr)
    return 3


def ref_available(model: str, no_tokenizer: bool) -> bool:
    if Path(model).exists():
        return True
    # HF repo id: look in the local HF cache only (the gate never downloads)
    roots = [os.environ.get("HF_HUB_CACHE"), os.environ.get("HF_HOME") and os.environ["HF_HOME"] + "/hub",
             str(Path.home() / ".cache/huggingface/hub"), "/hive-cache/codeagent/models/hf/hub", "/hive-cache/codeagent/models/hub"]
    d = "models--" + model.replace("/", "--")
    return any(r and (Path(r) / d / "snapshots").exists() for r in roots)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--ref-model", required=True, help="mlx-format checkpoint (dir or HF repo id in the local cache)")
    ap.add_argument("--engine-bin", required=True)
    ap.add_argument("--prompts", default=str(HERE / "prompts.json"))
    ap.add_argument("--ppl-text", default=str(HERE / "ppl_text.txt"))
    ap.add_argument("--ppl-ids", help="json list of ids (synthetic models; overrides --ppl-text)")
    ap.add_argument("--no-tokenizer", action="store_true", help="reference has no tokenizer: prompts must carry ids; skips tokenizer parity")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-python", default=sys.executable)
    ap.add_argument("--ref-dtype", default="float32")
    ap.add_argument("--reuse-ref", action="store_true", help="reuse OUT/ref from an earlier run (same reference model)")
    ap.add_argument("--tag", default="", help="free text recorded in the result (commit, patch set, ...)")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    missing = []
    if not Path(a.gguf).is_file():
        missing.append(f"GGUF not found: {a.gguf}")
    if not ref_available(a.ref_model, a.no_tokenizer):
        missing.append(f"reference weights not found locally: {a.ref_model}")
    if not Path(a.engine_bin).is_file():
        missing.append(f"engine binary not built: {a.engine_bin} (utils/i2s_gate/build_engine_logits.sh)")
    if missing:
        return blocked(out, missing)

    tol = C.load_tolerances()
    tol_sha = hashlib.sha256((HERE / "tolerances.json").read_bytes()).hexdigest()[:12]
    n_new = tol["n_new_tokens"]

    # ---- reference
    ref_dir = out / "ref"
    if not (a.reuse_ref and (ref_dir / "ref.json").exists()):
        cmd = [a.ref_python, str(HERE / "ref_mlx.py"), "--model", a.ref_model, "--prompts", a.prompts,
               "--out", str(ref_dir), "--n-new", str(n_new), "--dtype", a.ref_dtype]
        cmd += ["--ppl-ids", a.ppl_ids] if a.ppl_ids else ["--ppl-text", a.ppl_text]
        if a.no_tokenizer:
            cmd.append("--no-tokenizer")
        subprocess.run(cmd, check=True)
    ref = json.loads((ref_dir / "ref.json").read_text())

    # ---- engine
    eng_dir = out / "engine"; eng_dir.mkdir(exist_ok=True)
    lines = []
    for i, p in enumerate(ref["prompts"]):
        lines.append("L %d %s" % (i, " ".join(map(str, p["tf_seq"]))))
        lines.append("G %d %d %s" % (100 + i, n_new, " ".join(map(str, p["prompt_ids"]))))
        if p.get("text") is not None and not a.no_tokenizer:
            tf = eng_dir / f"text{i}.txt"; tf.write_text(p["text"])
            lines.append(f"K {200 + i} 1 {tf}")
    if ref["ppl"]:
        lines.append("L 1000 %s" % " ".join(map(str, ref["ppl"]["ids"])))
    (eng_dir / "cmds.txt").write_text("\n".join(lines) + "\n")
    subprocess.run([a.engine_bin, "--model", a.gguf, "--threads", str(a.threads), "--cmds", str(eng_dir / "cmds.txt"), "--out", str(eng_dir)], check=True)
    meta = json.loads((eng_dir / "meta.json").read_text())
    n_vocab = meta["n_vocab"]

    def read_logits(idx: int, n_pos: int) -> np.ndarray:
        return np.fromfile(eng_dir / f"{idx}.f32", dtype=np.float32).reshape(n_pos, n_vocab)

    checks: dict[str, dict] = {}
    ref_pool, eng_pool = [], []
    pending_greedy = []
    for i, p in enumerate(ref["prompts"]):
        rl = np.load(ref_dir / f"{p['name']}.npy")
        if rl.shape[1] != n_vocab:
            return blocked(out, [f"vocab size differs: reference {rl.shape[1]} vs engine {n_vocab}"])
        el = read_logits(i, len(p["tf_seq"]))[p["eval_from"]:]
        ref_pool.append(rl); eng_pool.append(el)
        eng_ids = [int(x) for x in (eng_dir / f"{100 + i}.ids").read_text().split()]
        pending_greedy.append((p, eng_ids))
        if (eng_dir / f"{200 + i}.ids").exists():
            tids = [int(x) for x in (eng_dir / f"{200 + i}.ids").read_text().split()]
            ok = tids == p["prompt_ids"]
            checks[f"tokenizer_parity[{p['name']}]"] = {"pass": ok, "engine_ids": tids, "ref_ids": p["prompt_ids"], "reasons": [] if ok else ["engine tokenizer ids differ from reference tokenizer ids"]}
    rp, ep = np.concatenate(ref_pool), np.concatenate(eng_pool)
    ppl_rl = ppl_el = None
    if ref["ppl"]:
        ids = ref["ppl"]["ids"]; n = len(ids) - 1
        ppl_rl = np.load(ref_dir / "ppl.npy"); ppl_el = read_logits(1000, len(ids))[:-1]
    # near-tie threshold measured on ALL teacher-forced positions of this run (prompts + ppl text)
    all_r = np.concatenate([rp] + ([ppl_rl] if ppl_rl is not None else []))
    all_e = np.concatenate([ep] + ([ppl_el] if ppl_el is not None else []))
    finite = np.isfinite(all_e).all()
    nt = C.near_tie(all_r, all_e, tol) if finite else {"sigma": float("nan"), "logit_std": float(all_r.std()), "threshold": tol["near_tie"]["floor_logits"], "sigma_ok": False}
    for p, eng_ids in pending_greedy:
        checks[f"greedy[{p['name']}]"] = C.evaluate_greedy(p["greedy_ids"], eng_ids, np.array(p["greedy_margins"]), tol, nt["threshold"]) | {"ref_ids": p["greedy_ids"], "engine_ids": eng_ids}
    checks["teacher_forced[prompts,pooled]"] = C.evaluate_teacher_forced(rp, ep, tol, nt)
    if ref["ppl"]:
        checks["teacher_forced[ppl_text]"] = C.evaluate_teacher_forced(ppl_rl, ppl_el, tol, nt)
        checks["perplexity"] = C.evaluate_ppl(C.mean_nll(ppl_rl, ids[1:]), C.mean_nll(ppl_el, ids[1:]), tol) | {"n_targets": n}

    passed = all(c["pass"] for c in checks.values())
    result = {"status": "PASS" if passed else "FAIL", "tag": a.tag, "gguf": a.gguf, "ref_model": a.ref_model,
              "tolerances_sha256_12": tol_sha, "tolerances_version": tol["version"], "near_tie": nt, "threads": a.threads, "engine_system_info": meta["system_info"], "checks": checks}
    (out / "result.json").write_text(json.dumps(result, indent=1, default=float))
    for name, c in checks.items():
        extra = ""
        if "metrics" in c and c["metrics"]:
            m = c["metrics"]; extra = f"top1={m['top1_agreement_non_tie']:.3f} meanKL={m['mean_kl']:.4g} maxKL={m['max_kl']:.4g} minCos={m['min_cosine']:.4f}"
        elif "ppl_ref" in c:
            extra = f"ppl_ref={c['ppl_ref']:.4f} ppl_engine={c['ppl_engine']:.4f}"
        elif "matching_prefix" in c:
            extra = f"prefix={c['matching_prefix']}/{c['n']} ({c['reason']})"
        print(f"{'PASS' if c['pass'] else 'FAIL'}  {name:36s} {extra}  {'; '.join(c.get('reasons', []))}")
    print(f"\nGATE {result['status']}   [{meta['system_info'][:110]}...]  tolerances {tol_sha}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
