"""Synthetic tests for specrepair (no real recordings). Run: python -m pytest tests/test_specrepair.py"""
import os
import sys
import tempfile

import numpy as np
import soundfile as sf
from scipy import signal

from opendubstage.restore.specrepair import Params, level_db, repair_array, repair_file  # noqa: E402
from opendubstage.restore.specrepair import band_db  # noqa: E402

SR = 48000


def room(n, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n)
    b, a = signal.butter(2, 400 / (SR / 2))                  # rumbly car-like noise
    x = signal.lfilter(b, a, w) + 0.15 * w
    return x / np.sqrt(np.mean(x ** 2)) * 10 ** (-50 / 20)


def word(dur, f0, seed):
    t = np.arange(int(dur * SR)) / SR
    env = np.sin(np.pi * t / dur) ** 0.6
    x = sum(np.sin(2 * np.pi * f0 * k * t + k) / k for k in range(1, 12))
    return env * x / np.sqrt(np.mean(x ** 2)) * 10 ** (-20 / 20)


def scene(seed=1):
    """12 s: words at 2.0-2.6, 3.4-4.0 (gap 2.6-3.4 holds a 'cough'), 8.0-8.6, 9.0-9.6; the rest is room tone."""
    x = room(12 * SR, seed)
    words = [(2.0, 2.6), (3.4, 4.0), (8.0, 8.6), (9.0, 9.6)]
    for i, (a, b) in enumerate(words):
        n0 = int(a * SR); w = word(b - a, 200 + 15 * i, i); x[n0:n0 + len(w)] += w
    rng = np.random.default_rng(seed + 10)
    burst = rng.standard_normal(int(0.25 * SR))
    bb, ba = signal.butter(2, [300 / (SR / 2), 4000 / (SR / 2)], "band")
    burst = signal.lfilter(bb, ba, burst) * np.hanning(len(burst))
    burst *= 10 ** (-18 / 20) / np.sqrt(np.mean(burst ** 2))
    n0 = int(2.85 * SR); x[n0:n0 + len(burst)] += burst
    return x, words, (2.85, 3.10)


def test_room_tone_replace_reduces_event_and_keeps_words():
    x, words, (ea, eb) = scene()
    y, rep = repair_array(x, SR, events=[(2.7, 3.3)], protected=words)
    assert len(y) == len(x)
    ev = [r for r in rep if r.get("method") == "room_tone"]
    assert ev, rep
    seg = slice(int(ea * SR), int(eb * SR))
    red = level_db(x[seg]) - level_db(y[seg])
    assert red >= 15, red
    for a, b in words:                       # protected samples are bit-identical
        s = slice(int(a * SR), int(b * SR))
        assert np.array_equal(x[s], y[s])
    # changed samples stay inside the reported region
    ch = np.flatnonzero(x != y) / SR
    assert ch.min() >= ev[0]["start"] - 1e-3 and ch.max() <= ev[0]["end"] + 1e-3
    # the fill sits at room-tone level (within 3 dB of the floor)
    assert abs(level_db(y[seg]) - level_db(x[int(5 * SR):int(7 * SR)])) < 3


def test_spectral_attenuate_when_event_overlaps_word():
    x, words, _ = scene()
    rng = np.random.default_rng(3)
    n0 = int(2.3 * SR); burst = rng.standard_normal(int(0.1 * SR)) * np.hanning(int(0.1 * SR)) * 0.05
    x[n0:n0 + len(burst)] += burst
    y, rep = repair_array(x, SR, events=[(2.3, 2.4)], protected=words)
    r = [r for r in rep if r.get("method") == "spectral_attenuate"]
    assert r and r[0]["reduction_db"] > 3, rep
    far = slice(int(8 * SR), int(9.6 * SR))
    assert np.array_equal(x[far], y[far])


def test_thump_attenuated_hf_untouched():
    x = room(6 * SR, 5)
    t = np.arange(int(0.12 * SR)) / SR
    th = np.sin(2 * np.pi * 55 * t) * np.exp(-t / 0.03) * 10 ** (-12 / 20)
    n0 = int(3.0 * SR); x[n0:n0 + len(th)] += th
    y, rep = repair_array(x, SR, thumps=True)
    th_r = [r for r in rep if r["kind"] == "thump"]
    assert len(th_r) == 1, rep
    s = slice(n0, n0 + len(th))
    assert band_db(x[s], SR, 20, 150) - band_db(y[s], SR, 20, 150) > 12
    assert abs(band_db(x[s], SR, 400, 20000) - band_db(y[s], SR, 400, 20000)) < 0.5


def test_low_voice_is_not_a_thump():
    x = room(6 * SR, 9)
    w = word(0.5, 95, 4)                      # low-pitched voiced sound with energy under 150 Hz
    n0 = int(3.0 * SR); x[n0:n0 + len(w)] += w
    _, rep = repair_array(x, SR, thumps=True)
    assert not [r for r in rep if r["kind"] == "thump"], rep


def test_cluster_outside_nominal_span_left_alone():
    x, words, _ = scene(11)
    rng = np.random.default_rng(2)
    n0 = int(2.62 * SR); tail = rng.standard_normal(int(0.08 * SR)) * 0.02   # loud blip right after a word
    x[n0:n0 + len(tail)] += tail
    # blip and event are separate clusters (merge_gap below their 0.15 s spacing)
    y, rep = repair_array(x, SR, events=[(2.85, 3.1)], protected=words, p=Params(merge_gap=0.1))
    s = slice(int(2.61 * SR), int(2.70 * SR))
    assert any(r.get("method") == "room_tone" for r in rep)
    assert np.array_equal(x[s], y[s]), rep


def test_file_roundtrip_bit_exact_outside_regions():
    x, words, _ = scene(7)
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in.flac"), os.path.join(d, "out.flac")
        sf.write(src, x, SR, subtype="PCM_24")
        reps = repair_file(src, dst, events=[(2.7, 3.3)], protected=words, log=lambda *a: None)
        a = sf.read(src, dtype="int32")[0]; b = sf.read(dst, dtype="int32")[0]
        assert len(a) == len(b)
        diff = np.flatnonzero(a != b) / SR
        live = [r for r in reps if "start" in r]
        assert len(diff) and all(any(r["start"] - 1e-3 <= t <= r["end"] + 1e-3 for r in live) for t in diff)
        c = signal.correlate(b.astype(float), a.astype(float), mode="full", method="fft")
        assert np.argmax(c) - (len(a) - 1) == 0


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
