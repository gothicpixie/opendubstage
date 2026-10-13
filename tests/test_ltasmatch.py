"""Synthetic-signal tests for opendubstage.match.ltasmatch.  Run: python -m pytest tests/test_ltasmatch.py"""
import shutil, subprocess, sys, unittest
from pathlib import Path

import numpy as np

from opendubstage.match import ltasmatch as C

SR = 48000


def lfilter(b, a, x):
    from scipy.signal import lfilter as lf
    return lf(b, a, x)


class TestLTAS(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.x = self.rng.standard_normal(SR * 20).astype(np.float32)

    def test_white_noise_rises_3db_per_band_ratio(self):
        acc, n = C.ltas_power(self.x, SR)
        db = C.to_db(acc)
        # white noise: band power ∝ bandwidth ∝ centre → +1 dB per 1/3-octave band (10·log10(2^(1/3)))
        slope = np.polyfit(np.log2(C.THIRD_OCTAVE), db, 1)[0]
        self.assertAlmostEqual(slope, 3.0, delta=0.3)
        self.assertGreater(n, 200)

    def test_fit_undoes_a_known_coloration(self):
        ref = C.to_db(C.ltas_power(self.x, SR)[0])
        y = self.x.copy()
        for fc, g in ((400, 4.0), (3000, -5.0)):
            b, a = C.peq_coeffs(fc, g, 1.0, SR); y = lfilter(b, a, y)
        before = C.to_db(C.ltas_power(y.astype(np.float32), SR)[0])
        self.assertGreater(C.deviation(before, ref)["max_abs_dev_db"], 3.0)
        g = C.fit_eq(before, ref)
        self.assertTrue(np.all(np.abs(g) <= 6.0))
        after = before + C.eq_band_response(g)
        self.assertLess(C.deviation(after, ref)["max_abs_dev_db"], 1.5)

    def test_gain_limit(self):
        ref = np.zeros(len(C.THIRD_OCTAVE)); before = ref.copy(); before[10:14] -= 20
        g = C.fit_eq(before, ref)
        self.assertTrue(np.all(np.abs(g) <= 6.0))

    def test_protect_below_leaves_low_bands(self):
        ref = np.zeros(len(C.THIRD_OCTAVE)); before = ref.copy(); before[:4] += 8   # 80–160 Hz hump
        g = C.fit_eq(before, ref, protect_below=180)
        self.assertLess(abs(g[0]), 1.5)

    def test_vad_ignores_silence(self):
        y = np.concatenate([self.x[:SR * 2] * 0.3, np.zeros(SR * 2, np.float32) + 1e-5 * self.x[:SR * 2]])
        m = C.energy_vad(y, SR)
        half = len(m) // 2
        self.assertGreater(m[:half - 3].mean(), 0.9)
        self.assertLess(m[half + 3:].mean(), 0.1)

    def test_interval_mask(self):
        m = C.interval_mask([(1.0, 2.0)], 200, SR)
        centers = (np.arange(200) * 1024 + 2048) / SR
        self.assertTrue(np.all((centers[m] > 1.0) & (centers[m] < 2.0)))


class TestPitchAndRoom(unittest.TestCase):
    def test_median_f0(self):
        t = np.arange(SR) / SR
        for f0 in (110.0, 200.0):
            x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))
            self.assertAlmostEqual(C.median_f0(x, SR), f0, delta=f0 * 0.02)
        self.assertTrue(np.isnan(C.median_f0(np.random.default_rng(1).standard_normal(SR) * 1e-9 * 0, SR)))

    def test_early_reflections_string(self):
        s = C.early_reflections()
        self.assertTrue(s.startswith("aecho="))
        _, ig, og, d, g = s.split("=")[0], *s.split("=")[1].split(":")
        self.assertEqual(len(d.split("|")), len(g.split("|")))
        gs = [float(v) for v in g.split("|")]
        self.assertTrue(all(a > b for a, b in zip(gs, gs[1:])))


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not on PATH")
class TestFfmpegAgreement(unittest.TestCase):
    def test_peq_model_matches_ffmpeg_equalizer(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(SR * 10).astype(np.float32)
        gains = [0, 3.0, -4.0, 0, 2.5, -3.0, 0, 0]
        p = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
                            "-af", C.eq_string(gains), "-f", "f32le", "-"], input=x.tobytes(), capture_output=True, check=True)
        y = np.frombuffer(p.stdout, np.float32)
        meas = 10 * np.log10(C.ltas_power(y, SR)[0] / C.ltas_power(x, SR)[0])
        pred = C.eq_band_response(gains)
        c = np.array(C.THIRD_OCTAVE); m = (c >= 100) & (c <= 10000)
        self.assertLess(np.abs(meas - pred)[m].max(), 0.5)


if __name__ == "__main__":
    unittest.main()
