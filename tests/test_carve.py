"""Synthetic tests for opendubstage.mix.carve: pink-noise "music" and gated speech-band-noise "dialogue". There is no real audio."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from opendubstage.mix import carve as sc

SR = 48000


def pink(n, ch=2, seed=0):
    rng = np.random.default_rng(seed)
    X = np.fft.rfft(rng.standard_normal((ch, n)), axis=1)
    f = np.fft.rfftfreq(n, 1 / SR); f[0] = f[1]
    x = np.fft.irfft(X / np.sqrt(f), n=n, axis=1).T
    return (0.1 * x / np.sqrt((x ** 2).mean())).astype(np.float32)


def speech_like(n, on, seed=1):
    """Band-limited (300 Hz–3.4 kHz) noise, gated on the (a, b) second intervals in `on`, about -20 dBFS RMS."""
    rng = np.random.default_rng(seed)
    s = sosfilt(butter(4, (300, 3400), "band", fs=SR, output="sos"), rng.standard_normal(n))
    s = 0.1 * s / np.sqrt((s ** 2).mean())
    g = np.zeros(n)
    for a, b in on: g[int(a * SR):int(b * SR)] = 1.0
    return np.stack([s * g, s * g], 1).astype(np.float32)


def band_db(x, a, b, band=(1000, 4000)):
    m = x.mean(1) if x.ndim == 2 else x
    seg = m[int(a * SR):int(b * SR)]
    X = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    f = np.fft.rfftfreq(len(seg), 1 / SR)
    return 10 * np.log10(X[(f >= band[0]) & (f <= band[1])].sum())


DUR = 20.0
ON = [(4.0, 10.0), (14.0, 16.0)]


@pytest.fixture(scope="module")
def sig():
    n = int(DUR * SR)
    mx, dx = pink(n), speech_like(n, ON)
    return mx, dx, sc.carve(mx, dx, SR)


@pytest.mark.parametrize("cut", [-3.0, -4.5, -6.0])
def test_band_drops_by_configured_amount_under_speech(cut):
    n = int(DUR * SR)
    mx, dx = pink(n), speech_like(n, ON)
    y = sc.carve(mx, dx, SR, cut_db=cut)
    d = band_db(y, 5.0, 9.5) - band_db(mx, 5.0, 9.5)           # steady state inside a line
    assert abs(d - cut) <= 1.0, d


def test_unchanged_outside_speech(sig):
    mx, dx, y = sig
    for a, b in [(0.5, 3.5), (11.0, 13.5), (17.0, 19.5)]:      # after release (hold 0.2 + release 0.3)
        assert abs(band_db(y, a, b) - band_db(mx, a, b)) < 0.5
        assert abs(band_db(y, a, b, (20, 20000)) - band_db(mx, a, b, (20, 20000))) < 0.5


def test_outside_band_unchanged_under_speech(sig):
    mx, dx, y = sig
    for band in [(50, 600), (6500, 16000)]:                     # beyond the half-octave tapers
        assert abs(band_db(y, 5, 9.5, band) - band_db(mx, 5, 9.5, band)) < 0.5


@pytest.mark.parametrize("block_s", [1.0, 3.7, 30.0])
def test_blocks_match_single_pass(sig, block_s):
    mx, dx, y = sig
    yb = sc.carve_blocks(mx, dx, SR, int(block_s * SR))
    assert yb.shape == y.shape
    assert np.max(np.abs(yb - y)) < 1e-5


def test_mono_and_pad_check():
    n = SR * 3
    y = sc.carve(pink(n, 1)[:, 0], speech_like(n, [(1, 2)])[:, 0], SR)
    assert y.shape == (n,)
    with pytest.raises(ValueError):
        sc.carve(np.zeros((SR, 2)), np.zeros((SR, 2)), SR, pad=100)


def test_silent_dialogue_is_bit_exact_passthrough():
    n = SR * 5
    mx = pink(n)
    assert np.array_equal(sc.carve(mx, np.zeros_like(mx), SR), mx)


# --- posts -----------------------------------------------------------------------------------------------------------
GAPS = [1.0, 2.0, 2.4, 2.6, 3.0, 5.0, 7.9, 12.0]


def _lines():
    t, act = 2.0, []
    for g in GAPS + [None]:
        act.append((t, t + 4.0)); t += 4.0 + (g or 0)
    return act, t + 6.0


def test_posts_only_in_gaps_of_2_5_s_or_more():
    act, total = _lines()
    db = sc.posts(int(total * sc.FR) + 1, act, depth=-13.0, post_db=-6.0, return_db=True)
    for (a0, b0), (a1, _), g in zip(act, act[1:], GAPS):
        seg = db[int(b0 * sc.FR):int(a1 * sc.FR)]
        rises = seg.max() > -13.0 + 0.5
        assert rises == (g >= 2.5), (g, seg.max())
        if rises:
            assert (seg > -13.0 + 0.01).sum() / sc.FR >= 1.5   # never a post shorter than 1.5 s
            target = 0.0 if g >= 8.0 else -6.0
            assert abs(seg.max() - target) < 0.05
    for a, b in act:                                            # fully down under every word
        assert np.all(np.abs(db[int(a * sc.FR):int(b * sc.FR)] + 13.0) < 1e-6)


def test_posts_pre_dip_lands_before_the_word():
    act = [(2.0, 5.0), (8.0, 11.0)]                             # 3 s gap: post
    db = sc.posts(1500, act, depth=-13.0, post_db=-6.0, lead=0.25, dip=0.5, return_db=True)
    i = int((8.0 - 0.25) * sc.FR)
    assert abs(db[i] + 13.0) < 1e-6                             # down by lead-in time
    assert db[int((8.0 - 0.25 - 0.5) * sc.FR) - 1] > -6.5       # still at post just before the dip


def test_posts_linear_units_and_shape():
    env = np.ones(1000, np.float32)
    g = sc.posts(env, [(1.0, 3.0)])
    assert g.shape == env.shape and g.dtype == np.float32 and g.max() <= 1.0 + 1e-6


def test_posts_no_rise_when_post_below_depth():
    act = [(2.0, 4.0), (7.0, 9.0)]
    db = sc.posts(1500, act, depth=-5.0, post_db=-6.0, return_db=True)
    assert np.allclose(db[int(4.2 * sc.FR):int(6.7 * sc.FR)], -5.0)


# --- phrase_safe_end -------------------------------------------------------------------------------------------------
def _map():
    bpm = 120.0; bar = 2.0
    db = [0.5 + k * bar for k in range(60)]
    return {"bpm": bpm, "dur": 125.0, "downbeats": db,
            "sections": [{"start": 0.0, "end": 32.5}, {"start": 32.5, "end": 64.5}, {"start": 64.5, "end": 120.5}]}


def test_phrase_end_prefers_section():
    r = sc.phrase_safe_end(_map(), 30.0)
    assert r["kind"] == "section" and r["end"] == 32.5 and r["cue_end"] == pytest.approx(32.5 + r["fade"])
    assert 1.5 <= r["fade"] <= 4.0


def test_phrase_end_falls_back_to_phrase_then_downbeat():
    r = sc.phrase_safe_end(_map(), 80.0, search=6.0)
    assert r["kind"] == "phrase" and (r["end"] - 64.5) % 8.0 == pytest.approx(0.0)
    r = sc.phrase_safe_end(_map(), 83.0, search=0.6)       # 82.5 = bar 9 of the section: not a phrase
    assert r["kind"] == "downbeat" and (r["end"] - 0.5) % 2.0 == pytest.approx(0.0)
