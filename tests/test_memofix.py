"""Synthetic tests for opendubstage.restore.memofix: harmonic "words" over a car-like rumble. No real recordings."""
import json

import numpy as np
import pytest
from scipy import signal

from opendubstage.restore import memofix as M

SR = 48000
WORDS = [(1.0, 1.8), (2.4, 3.3), (5.0, 5.7), (6.4, 7.6), (9.0, 9.8), (10.5, 11.4)]
DUR = 13.0


def rumble(n, seed=0, level_db=-30.0):
    """Engine/road rumble: low-passed noise below ~120 Hz plus a 38 Hz hum."""
    rng = np.random.default_rng(seed)
    b, a = signal.butter(4, 120 / (SR / 2))
    x = signal.lfilter(b, a, rng.standard_normal(n))
    x += 0.5 * np.sin(2 * np.pi * 38.0 * np.arange(n) / SR) * np.std(x)
    return x / np.sqrt(np.mean(x ** 2)) * 10 ** (level_db / 20)


def speech(n, words=WORDS, seed=1):
    """Harmonic bursts (f0 140-200 Hz, harmonics to ~4 kHz) at the word timings, plus a faint broadband floor."""
    rng = np.random.default_rng(seed)
    x = 10 ** (-70 / 20) * rng.standard_normal(n)
    for i, (a, b) in enumerate(words):
        t = np.arange(int((b - a) * SR)) / SR
        f0 = 140 + 12 * i
        w = sum(np.sin(2 * np.pi * f0 * k * t + k) / k for k in range(1, int(4000 / f0)))
        w *= np.sin(np.pi * t / t[-1]) ** 0.5
        w *= 10 ** (-22 / 20) / np.sqrt(np.mean(w ** 2))
        n0 = int(a * SR); x[n0:n0 + len(w)] += w
    return x


def band_db(x, lo, hi, spans):
    sos = signal.butter(4, [lo, hi], "band", fs=SR, output="sos")
    y = signal.sosfilt(sos, x)
    e = sum(float((y[int(a * SR):int(b * SR)] ** 2).sum()) for a, b in spans)
    n = sum(int(b * SR) - int(a * SR) for a, b in spans)
    return 10 * np.log10(e / n + 1e-30)


GAPS = [(4.0, 4.7), (8.0, 8.7), (12.0, 12.8)]


@pytest.fixture(scope="module")
def memo():
    n = int(DUR * SR)
    clean = speech(n)
    return clean, clean + rumble(n)


def test_rumble_in_gaps_drops_relative_to_speech(memo):
    clean, x = memo
    y, rep = M.memofix(x, SR, WORDS, eq=False)
    assert len(y) == len(x) and y.dtype == np.float32
    before = band_db(x, 30, 150, GAPS) - band_db(x, 1000, 4000, WORDS)
    after = band_db(y, 30, 150, GAPS) - band_db(y, 1000, 4000, WORDS)
    assert before - after >= 6.0, (before, after)
    assert rep["gap_frames"] > 10 and rep["speech_frames"] > 10


def test_presence_band_untouched_without_eq(memo):
    """Above lf_max nothing is processed: the 1-4 kHz band keeps its shape (only the global RMS match changes it)."""
    _, x = memo
    y, _ = M.memofix(x, SR, WORDS, eq=False)
    sos = signal.butter(4, [1000, 4000], "band", fs=SR, output="sos")
    a, b = signal.sosfilt(sos, x), signal.sosfilt(sos, y.astype(float))
    c = np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))
    assert c > 0.999


def test_energy_fallback_without_word_timings(memo):
    _, x = memo
    y, rep = M.memofix(x, SR, None, eq=False)
    before = band_db(x, 30, 150, GAPS) - band_db(x, 1000, 4000, WORDS)
    after = band_db(y, 30, 150, GAPS) - band_db(y, 1000, 4000, WORDS)
    assert before - after >= 3.0
    assert rep["gap_frames"] > 0


def test_eq_is_clamped_and_only_cuts_low(memo):
    _, x = memo
    p = M.MemoParams()
    _, rep = M.memofix(x, SR, WORDS, p)
    eq = {int(k): v for k, v in rep["eq_db"].items()}
    assert all(-p.max_cut_db - 1e-9 <= v <= p.max_boost_db + 1e-9 for v in eq.values())
    assert all(v <= 0 for f, v in eq.items() if f < p.cut_only_below)


def test_output_rms_matches_input(memo):
    _, x = memo
    y, _ = M.memofix(x, SR, WORDS)
    assert abs(10 * np.log10(np.mean(y.astype(float) ** 2) / np.mean(x ** 2))) < 0.1


def test_silence_is_safe():
    y, _ = M.memofix(np.zeros(SR * 2), SR, None)
    assert np.all(np.isfinite(y)) and np.max(np.abs(y)) < 1e-6


def test_load_words_formats(tmp_path):
    p = tmp_path / "w.json"
    p.write_text(json.dumps([{"t": 1.0, "e": 1.5, "type": "word"}, {"t": 2.0, "e": 2.2, "type": "spacing"},
                             {"start": 3.0, "end": 3.5}]))
    assert M.load_words(p, offset=1.0) == [(0.0, 0.5), (2.0, 2.5)]
    p.write_text(json.dumps({"words": [[0.5, 1.0]]}))
    assert M.load_words(p) == [(0.5, 1.0)]


def test_ltass_table_is_consistent():
    assert len(M.LTASS_BANDS) == len(M.LTASS_DB)
    assert np.all(np.diff(M.LTASS_BANDS) > 0)
