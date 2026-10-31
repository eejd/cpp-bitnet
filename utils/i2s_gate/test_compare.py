"""Unit tests of compare.py on synthetic data. Run: python3.14 utils/i2s_gate/test_compare.py -v

They establish that the gate (a) accepts the documented quantization noise and (b) rejects each failure
mode seen upstream: scrambled weights, constant output, NaN/garbage, SiLU-vs-relu^2-sized distortions.
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare as C

TOL = C.load_tolerances()
V, P = 600, 48


def make_ref(seed=0, scale=3.0):
    rng = np.random.default_rng(seed)
    # peaked, language-model-like logits: a few strong candidates per position
    x = rng.normal(0, 1.0, size=(P, V))
    top = rng.integers(0, V, size=(P, 4))
    for i in range(P):
        x[i, top[i]] += np.array([8.0, 6.0, 4.5, 3.5]) * scale / 3.0
    return x


class TeacherForced(unittest.TestCase):
    def setUp(self):
        self.ref = make_ref()

    def test_identical_passes(self):
        self.assertTrue(C.evaluate_teacher_forced(self.ref, self.ref, TOL)["pass"])

    def test_small_activation_quantization_like_noise_passes(self):
        rng = np.random.default_rng(1)
        eng = self.ref + rng.normal(0, 0.03, size=self.ref.shape)
        r = C.evaluate_teacher_forced(self.ref, eng, TOL)
        self.assertTrue(r["pass"], r)
        self.assertLess(r["metrics"]["mean_kl"], 0.005)

    def test_large_noise_fails(self):
        rng = np.random.default_rng(2)
        eng = self.ref + rng.normal(0, 1.0, size=self.ref.shape)
        self.assertFalse(C.evaluate_teacher_forced(self.ref, eng, TOL)["pass"])

    def test_scrambled_vocab_fails(self):
        rng = np.random.default_rng(3)
        eng = self.ref[:, rng.permutation(V)]
        r = C.evaluate_teacher_forced(self.ref, eng, TOL)
        self.assertFalse(r["pass"])
        self.assertLess(r["metrics"]["top1_agreement_non_tie"], 0.2)

    def test_unrelated_logits_fail(self):
        eng = make_ref(seed=99)
        self.assertFalse(C.evaluate_teacher_forced(self.ref, eng, TOL)["pass"])

    def test_constant_output_fails(self):
        eng = np.tile(self.ref[0], (P, 1))  # same logits at every position (upstream #610: constant output)
        self.assertFalse(C.evaluate_teacher_forced(self.ref, eng, TOL)["pass"])

    def test_flat_logits_fail(self):
        self.assertFalse(C.evaluate_teacher_forced(self.ref, np.zeros_like(self.ref), TOL)["pass"])

    def test_nan_fails(self):
        eng = self.ref.copy(); eng[5, 7] = np.nan
        r = C.evaluate_teacher_forced(self.ref, eng, TOL)
        self.assertFalse(r["pass"]); self.assertIn("non-finite", r["reasons"][0])

    def test_inf_and_1e27_garbage_fail(self):
        eng = self.ref.copy(); eng[3, 0] = np.inf
        self.assertFalse(C.evaluate_teacher_forced(self.ref, eng, TOL)["pass"])
        eng = self.ref * 1e27
        self.assertFalse(C.evaluate_teacher_forced(self.ref, eng, TOL)["pass"])

    def test_temperature_distortion_fails(self):
        # a systematically flattened distribution (e.g. wrong activation function) must not pass
        self.assertFalse(C.evaluate_teacher_forced(self.ref, self.ref * 0.6, TOL)["pass"])

    def test_near_tie_flip_is_excused_but_clear_flip_is_not(self):
        ref = self.ref.copy()
        ref[10, :] = 0.0; ref[10, 5] = 10.0; ref[10, 6] = 9.9   # near tie (margin 0.1)
        eng = ref.copy(); eng[10, 5] = 9.85                       # flips top-1 to 6
        r = C.evaluate_teacher_forced(ref, eng, TOL)
        self.assertTrue(r["pass"], r)
        ref2 = self.ref.copy()
        eng2 = ref2.copy()
        for i in range(0, P, 2):                                  # flip many clear top-1s
            j = ref2[i].argmax(); k = (j + 1) % V
            eng2[i, j], eng2[i, k] = eng2[i, k], eng2[i, j]
        self.assertFalse(C.evaluate_teacher_forced(ref2, eng2, TOL)["pass"])

    def test_adaptive_near_tie_follows_measured_noise_but_is_capped(self):
        rng = np.random.default_rng(7)
        # engine noise comparable to the synthetic end-to-end run (logit std ~4, noise ~0.12)
        ref = make_ref(seed=3, scale=3.0) * (4.0 / make_ref(seed=3, scale=3.0).std())
        eng = ref + rng.normal(0, 0.12, ref.shape)
        nt = C.near_tie(ref, eng, TOL)
        self.assertGreater(nt["threshold"], TOL["near_tie"]["floor_logits"])   # wider than the floor
        self.assertTrue(nt["sigma_ok"])
        self.assertTrue(C.evaluate_teacher_forced(ref, eng, TOL)["pass"])
        # heavy noise: sigma exceeds 10 % of the logit std -> cannot excuse itself, fails on the cap
        eng2 = ref + rng.normal(0, 2.0, ref.shape)
        nt2 = C.near_tie(ref, eng2, TOL)
        self.assertFalse(nt2["sigma_ok"])
        r = C.evaluate_teacher_forced(ref, eng2, TOL)
        self.assertFalse(r["pass"])

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            C.evaluate_teacher_forced(self.ref, self.ref[:-1], TOL)


class Greedy(unittest.TestCase):
    margins = np.full(16, 2.0)

    def test_identical(self):
        self.assertTrue(C.evaluate_greedy(list(range(16)), list(range(16)), self.margins, TOL)["pass"])

    def test_late_divergence_ok(self):
        e = list(range(16)); e[13] = 999
        self.assertTrue(C.evaluate_greedy(list(range(16)), e, self.margins, TOL)["pass"])

    def test_early_divergence_fails(self):
        e = list(range(16)); e[2] = 999
        self.assertFalse(C.evaluate_greedy(list(range(16)), e, self.margins, TOL)["pass"])

    def test_early_divergence_at_near_tie_excused(self):
        m = self.margins.copy(); m[2] = 0.05
        e = list(range(16)); e[2] = 999
        self.assertTrue(C.evaluate_greedy(list(range(16)), e, m, TOL)["pass"])

    def test_constant_repeat_fails(self):  # '@@@@@@' style output
        self.assertFalse(C.evaluate_greedy(list(range(16)), [31] * 16, self.margins, TOL)["pass"])

    def test_threshold_argument_is_used(self):
        m = self.margins.copy(); m[2] = 0.4
        e = list(range(16)); e[2] = 999
        self.assertFalse(C.evaluate_greedy(list(range(16)), e, m, TOL)["pass"])                         # floor 0.25
        self.assertTrue(C.evaluate_greedy(list(range(16)), e, m, TOL, near_tie_threshold=0.6)["pass"])  # measured noise

    def test_truncated_fails(self):
        self.assertFalse(C.evaluate_greedy(list(range(16)), list(range(3)), self.margins, TOL)["pass"])


class Perplexity(unittest.TestCase):
    def test_nll_matches_manual(self):
        lg = np.log(np.array([[0.5, 0.5], [0.25, 0.75]]))
        self.assertAlmostEqual(C.mean_nll(lg, [0, 1]), -(math.log(0.5) + math.log(0.75)) / 2)

    def test_within_two_percent_passes(self):
        self.assertTrue(C.evaluate_ppl(2.0, 2.0 + 0.015, TOL)["pass"])

    def test_three_percent_fails(self):
        self.assertFalse(C.evaluate_ppl(2.0, 2.0 + 0.031, TOL)["pass"])
        self.assertFalse(C.evaluate_ppl(2.0, 2.0 - 0.031, TOL)["pass"])

    def test_garbage_ppl_fails(self):
        self.assertFalse(C.evaluate_ppl(2.9, 6.4, TOL)["pass"])  # ~ ln(600): uniform over vocab
        self.assertFalse(C.evaluate_ppl(2.9, float("nan"), TOL)["pass"])


if __name__ == "__main__":
    unittest.main()
