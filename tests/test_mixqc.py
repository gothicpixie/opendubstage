"""Synthetic fixtures for opendubstage.qc.mixqc: a clean programme passes every check; each variant trips exactly one.

Run:  python -m pytest tests/test_mixqc.py   (needs numpy, scipy, soundfile; ffmpeg optional for the meter cross-check)
"""
import json, re, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

from opendubstage.qc import mixqc as sd_qc

SR = 48000
DUR = 40.0
T = sd_qc.Thresholds(lufs_range=(-40.0, -5.0), lra_programme=(0.0, 15.0), lra_segment_max={"*": None})


def speech(seconds, seed, lowpass=None):
    """Speech-like signal: band-limited noise with a 4 Hz syllable envelope and word gaps."""
    rng = np.random.default_rng(seed); n = int(seconds * SR); t = np.arange(n) / SR
    x = sosfilt(butter(4, [200, 5000], "bandpass", fs=SR, output="sos"), rng.standard_normal(n))
    if lowpass: x = sosfilt(butter(4, lowpass, "lowpass", fs=SR, output="sos"), x)
    env = np.sin(np.pi * 4 * t) ** 2 * (np.sin(2 * np.pi * t / 1.6) > -0.6)
    x = x * env; x *= 0.02 / np.sqrt(np.mean(x ** 2))
    return fade(x, 0.1, 0.1)


def fade(x, fi, fo):
    x = x.copy(); a, b = int(fi * SR), int(fo * SR)
    if a: x[:a] *= np.linspace(0, 1, a)
    if b: x[-b:] *= np.linspace(1, 0, b)
    return x


def music(seconds, amp=0.03, band_noise=0.0, fo=3.0, step=None, seed=9):
    t = np.arange(int(seconds * SR)) / SR
    x = amp * (np.sin(2 * np.pi * 110 * t) + 0.5 * np.sin(2 * np.pi * 220 * t))
    if band_noise:
        y = sosfilt(butter(4, [1000, 4000], "bandpass", fs=SR, output="sos"), np.random.default_rng(seed).standard_normal(len(t)))
        x = x + band_noise * y / np.sqrt(np.mean(y ** 2))
    if step:                                   # (time, dB): smooth 1 s level step
        at, db = step; x = x * 10 ** (np.clip((t - at), 0, 1) * db / 20)
    return fade(x, 0.0, fo)


class Fixture:
    """Builds cue files, the mix, and the cue sheet in a temp dir."""
    def __init__(self, d):
        self.d = Path(d); self.cues = []; self.mix = np.zeros(int(DUR * SR))

    def cue(self, cid, role, start, x, cls=None, fade_in=0.1, fade_out=0.1, **kw):
        p = self.d / f"{cid}.wav"; sf.write(p, np.stack([x, x], 1), SR, subtype="FLOAT")
        i = int(round(start * SR)); self.mix[i:i + len(x)] += x[:len(self.mix) - i]
        c = {"id": cid, "role": role, "start": start, "end": start + len(x) / SR, "fade_in": fade_in,
             "fade_out": fade_out, "file": p.name, **kw}
        if cls: c["class"] = cls
        self.cues.append(c)

    def write(self, gain=1.0, edit=None):
        m = self.mix * gain
        if edit: edit(m)
        sf.write(self.d / "mix.wav", np.stack([m, m], 1), SR, subtype="FLOAT")
        (self.d / "cues.json").write_text(json.dumps({"duration": DUR, "cues": self.cues}))
        return self.d / "mix.wav", self.d / "cues.json"


def base(fx, b_gain=1.0, b_start=12.5, b_lowpass=None, b_class="narration", mus=None, b_fade=0.1):
    fx.cue("m1", "music", 0.0, mus if mus is not None else music(DUR), fade_in=0.0, fade_out=3.0)
    fx.cue("n1", "dialogue", 2.0, speech(10, 1), cls="narration")
    fx.cue("n2", "dialogue", b_start, b_gain * speech(9.5, 2, b_lowpass), cls=b_class, fade_in=b_fade)


def run(build, **write_kw):
    with tempfile.TemporaryDirectory() as d:
        fx = Fixture(d); build(fx); mix, cues = fx.write(**write_kw)
        met, probs = sd_qc.qc(mix, cues, thresholds=T, log=None)
        return met, probs, {e["check"] for e in met["events"]}


class TestSdQc(unittest.TestCase):
    def test_clean_passes(self):
        met, probs, checks = run(base)
        self.assertEqual(probs, [], probs)
        self.assertEqual(met["jump_edits_checked"], 1)
        self.assertEqual(met["masking_mode"], "stems")

    def test_loudness_true_peak(self):
        _, probs, checks = run(base, gain=None or 1.0, edit=lambda m: m.__imul__(0.999 / np.abs(m).max()))
        self.assertEqual(checks, {"loudness"}, probs)

    def test_lra(self):
        _, probs, checks = run(lambda fx: base(fx, mus=music(DUR, amp=0.003, step=(24.0, 44.0))))
        self.assertEqual(checks, {"lra"}, probs)

    def test_jump_between_voices(self):
        _, probs, checks = run(lambda fx: base(fx, b_gain=2.5))     # +8 dB on the second voice
        self.assertEqual(checks, {"jump"}, probs)

    def test_jump_exempt_at_scene_change(self):
        def b(fx):
            base(fx, b_gain=2.5)
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(d); b(fx); mix, cues = fx.write()
            sheet = json.loads(cues.read_text()); sheet["scenes"] = [12.5]; cues.write_text(json.dumps(sheet))
            _, probs = sd_qc.qc(mix, cues, thresholds=T, log=None)
        self.assertEqual(probs, [])

    def test_masking(self):
        _, probs, checks = run(lambda fx: base(fx, mus=music(DUR, band_noise=0.03)))
        self.assertEqual(checks, {"masking"}, probs)

    def test_click_at_edit(self):
        def edit(m): m[int(12.5 * SR)] += 0.3
        _, probs, checks = run(lambda fx: base(fx, b_fade=0.0), edit=edit)     # a HARD edit (no fade) that clicks
        self.assertEqual(checks, {"click"}, probs)

    def test_faded_edge_cannot_click(self):
        def edit(m): m[int(12.5 * SR)] += 0.3                                  # a transient AT a faded edge: not the edit's fault
        _, probs, checks = run(base, edit=edit)
        self.assertNotIn("click", checks, probs)

    def test_hard_stop(self):
        def b(fx):
            fx.cue("m1", "music", 0.0, music(20.0, fo=0.2), fade_in=0.0, fade_out=0.2)
            fx.cue("a1", "ambience", 19.0, music(DUR - 19.0, fo=3.0),
                   fade_in=0.0, fade_out=3.0)
            fx.cue("n1", "dialogue", 2.0, speech(10, 1), cls="narration")
            fx.cue("n2", "dialogue", 12.5, speech(9.5, 2), cls="narration")
        _, probs, checks = run(b)
        self.assertEqual(checks, {"hard_stop"}, probs)

    def test_hard_stop_exempt_when_material_ended(self):
        def b(fx):
            x = music(20.0, fo=0.0); x[-int(2 * SR):] = 0.0          # the music ended 2 s before the cue's edge
            fx.cue("m1", "music", 0.0, x, fade_in=0.0, fade_out=0.2)
            fx.cue("a1", "ambience", 19.0, music(DUR - 19.0, fo=3.0), fade_in=0.0, fade_out=3.0)
            fx.cue("n1", "dialogue", 2.0, speech(10, 1), cls="narration")
            fx.cue("n2", "dialogue", 12.5, speech(9.5, 2), cls="narration")
        _, probs, checks = run(b)
        self.assertNotIn("hard_stop", checks, probs)

    def test_dead_air(self):
        def b(fx):
            fx.cue("m1", "music", 0.0, music(23.0), fade_in=0.0, fade_out=3.0)
            fx.cue("n1", "dialogue", 2.0, speech(10, 1), cls="narration")
            fx.cue("n2", "dialogue", 12.5, speech(9.5, 2), cls="narration")
        _, probs, checks = run(b)
        self.assertEqual(checks, {"dead_air"}, probs)

    def test_tonal_balance(self):
        _, probs, checks = run(lambda fx: base(fx, b_start=15.5, b_lowpass=500, b_class="actuality"))
        self.assertEqual(checks, {"tonal"}, probs)

    def test_programme_only_without_cues(self):
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(d); base(fx); mix, _ = fx.write()
            met, probs = sd_qc.qc(mix, None, thresholds=T, log=None)
        self.assertEqual(probs, []); self.assertTrue(met["masking_mode"].startswith("unavailable"))

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_meter_agrees_with_ffmpeg(self):
        with tempfile.TemporaryDirectory() as d:
            fx = Fixture(d); base(fx); mix, cues = fx.write()
            met, _ = sd_qc.qc(mix, None, thresholds=T, log=None)
            e = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(mix), "-af", "ebur128", "-f", "null", "-"],
                               capture_output=True, text=True).stderr
            I = float(re.findall(r"I:\s+(-?[\d.]+) LUFS", e)[-1]); lra = float(re.findall(r"LRA:\s+(-?[\d.]+) LU", e)[-1])
        self.assertAlmostEqual(met["integrated_lufs"], I, delta=0.3)
        self.assertAlmostEqual(met["lra_programme"], lra, delta=1.0)


if __name__ == "__main__":
    unittest.main()
