"""Comparison logic of the I2_S correctness gate (pure numpy, no model, no engine).

Inputs are logits arrays [n_positions, n_vocab] from the trusted reference and from the engine under
test, computed on the *same* token ids (teacher forcing), plus greedy token lists.
All tolerances come from tolerances.json; nothing here is tuned per run.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

TOL_PATH = Path(__file__).with_name("tolerances.json")


def load_tolerances(path: Path = TOL_PATH) -> dict:
    return json.loads(Path(path).read_text())


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=-1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def top_margin(logits: np.ndarray) -> np.ndarray:
    """top-1 minus top-2 logit per position."""
    part = np.partition(np.asarray(logits, dtype=np.float64), -2, axis=-1)
    return part[:, -1] - part[:, -2]


def position_metrics(ref: np.ndarray, eng: np.ndarray) -> dict:
    ref = np.asarray(ref, dtype=np.float64)
    eng = np.asarray(eng, dtype=np.float64)
    if ref.shape != eng.shape:
        raise ValueError(f"shape mismatch ref {ref.shape} vs engine {eng.shape}")
    lp_r, lp_e = log_softmax(ref), log_softmax(eng)
    kl = (np.exp(lp_r) * (lp_r - lp_e)).sum(axis=-1)
    rc, ec = ref - ref.mean(-1, keepdims=True), eng - eng.mean(-1, keepdims=True)
    denom = np.linalg.norm(rc, axis=-1) * np.linalg.norm(ec, axis=-1)
    cos = np.where(denom > 0, (rc * ec).sum(-1) / np.where(denom > 0, denom, 1.0), 0.0)
    return {
        "kl": kl,
        "cos": cos,
        "top1_equal": ref.argmax(-1) == eng.argmax(-1),
        "ref_margin": top_margin(ref),
    }


def evaluate_teacher_forced(ref: np.ndarray, eng: np.ndarray, tol: dict) -> dict:
    t = tol["teacher_forced"]
    reasons: list[str] = []
    if not np.isfinite(np.asarray(eng)).all():
        n_bad = int((~np.isfinite(np.asarray(eng))).sum())
        return {"pass": False, "reasons": [f"engine logits contain {n_bad} non-finite values"], "metrics": {}}
    if not np.isfinite(np.asarray(ref)).all():
        return {"pass": False, "reasons": ["reference logits contain non-finite values (gate invalid)"], "metrics": {}}
    m = position_metrics(ref, eng)
    non_tie = m["ref_margin"] >= t["near_tie_margin_logits"]
    agree = float(m["top1_equal"][non_tie].mean()) if non_tie.any() else 1.0
    metrics = {
        "n_positions": int(len(m["kl"])),
        "n_non_tie_positions": int(non_tie.sum()),
        "top1_agreement_non_tie": agree,
        "mean_kl": float(m["kl"].mean()),
        "max_kl": float(m["kl"].max()),
        "min_cosine": float(m["cos"].min()),
        "mean_cosine": float(m["cos"].mean()),
    }
    if agree < t["min_top1_agreement_non_tie"]:
        reasons.append(f"top-1 agreement {agree:.3f} < {t['min_top1_agreement_non_tie']}")
    if metrics["mean_kl"] > t["max_mean_kl_nats"]:
        reasons.append(f"mean KL {metrics['mean_kl']:.4g} > {t['max_mean_kl_nats']}")
    if metrics["max_kl"] > t["max_position_kl_nats"]:
        reasons.append(f"max per-position KL {metrics['max_kl']:.4g} > {t['max_position_kl_nats']}")
    if metrics["min_cosine"] < t["min_position_cosine_centered"]:
        reasons.append(f"min centered cosine {metrics['min_cosine']:.4f} < {t['min_position_cosine_centered']}")
    return {"pass": not reasons, "reasons": reasons, "metrics": metrics}


def evaluate_greedy(ref_ids: list[int], eng_ids: list[int], ref_margins: np.ndarray, tol: dict) -> dict:
    """Free-running greedy continuation. ref_margins[i] = reference top1-top2 logit gap when it chose ref_ids[i]."""
    g = tol["greedy_free_running"]
    n = min(len(ref_ids), len(eng_ids))
    d = next((i for i in range(n) if ref_ids[i] != eng_ids[i]), None)
    if len(eng_ids) < len(ref_ids) and d is None:
        d = len(eng_ids)
    out = {"matching_prefix": len(ref_ids) if d is None else d, "n": len(ref_ids)}
    if d is None:
        out.update(pass_=True, reason="identical")
    elif d >= g["min_matching_prefix"]:
        out.update(pass_=True, reason=f"diverged at {d} >= {g['min_matching_prefix']}")
    elif d < len(ref_margins) and ref_margins[d] < g["divergence_excused_if_ref_margin_below"]:
        out.update(pass_=True, reason=f"diverged at {d} but reference margin {ref_margins[d]:.3f} is a near tie")
    else:
        m = float(ref_margins[d]) if d < len(ref_margins) else float("nan")
        out.update(pass_=False, reason=f"diverged at {d} (< {g['min_matching_prefix']}) with reference margin {m:.3f}")
    out["pass"] = out.pop("pass_")
    return out


def mean_nll(logits: np.ndarray, target_ids: list[int]) -> float:
    """logits[i] predicts target_ids[i]."""
    lp = log_softmax(logits)
    return float(-lp[np.arange(len(target_ids)), np.asarray(target_ids)].mean())


def evaluate_ppl(nll_ref: float, nll_eng: float, tol: dict) -> dict:
    if not (math.isfinite(nll_ref) and math.isfinite(nll_eng)):
        return {"pass": False, "reasons": ["non-finite perplexity"], "ppl_ref": float("nan"), "ppl_engine": float("nan")}
    lr = nll_eng - nll_ref  # ln(ppl_eng/ppl_ref)
    ok = abs(lr) <= tol["perplexity"]["max_abs_log_ratio"]
    return {
        "pass": ok,
        "ppl_ref": math.exp(nll_ref),
        "ppl_engine": math.exp(nll_eng),
        "abs_log_ratio": abs(lr),
        "reasons": [] if ok else [f"|ln(ppl_eng/ppl_ref)| = {abs(lr):.4f} > {tol['perplexity']['max_abs_log_ratio']}"],
    }
