"""ltasmatch: offline dialogue "EQ match" that writes plain ffmpeg filter strings.

Dialogue editors make one voice sit next to another with an EQ match (FabFilter Pro-Q "EQ Match", iZotope Ozone /
Neutron "Match EQ", iZotope RX "Dialogue Match"). They compare the long-term average spectrum (LTAS) of a reference
voice with the LTAS of the line that sounds different. Then they apply a *smoothed, gain-limited* difference curve.
Smoothing matters because matching at fine resolution copies the other voice's formants and sounds phasey. Limiting
the gain matters because large boosts lift the noise floor. RX Dialogue Match also matches ambience and reverb.

This module does the same thing offline:

* ``ltas_power``: per-band power over speech-active frames (1/3-octave bands by default).
* ``energy_vad`` / ``interval_mask``: choose the speech frames, from signal energy or from known word timings.
* ``fit_eq``: least-squares fit of a few octave-spaced peaking bands (RBJ biquads, which is ffmpeg's ``equalizer``),
  each limited to ±gmax dB, to the smoothed difference between target and measured.
* ``eq_string``: turn the fitted gains into an ffmpeg ``equalizer`` chain.
* ``early_reflections``: an ``aecho`` string of short, quickly decaying taps that gives a dry voice a small room.
* ``median_f0``: an autocorrelation pitch estimate, used to split speakers by voice pitch.

The only dependency is numpy; ``decode`` also needs an ``ffmpeg`` binary on PATH.
"""
from __future__ import annotations

import subprocess
import numpy as np

THIRD_OCTAVE = (80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
                5000, 6300, 8000, 10000, 12500)
OCTAVE_EQ = (125, 250, 500, 1000, 2000, 4000, 8000, 12000)
HALF_OCTAVE_EQ = (125, 180, 250, 355, 500, 710, 1000, 1400, 2000, 2800, 4000, 5600, 8000, 11300)


# ------------------------------------------------------------------------------------------------ audio I/O
def decode(path, sr=48000, ss=0.0, dur=None, af=None, ffmpeg="ffmpeg"):
    """Decode a stretch of any ffmpeg-readable file, optionally through a filter chain, to mono float32 at ``sr``."""
    cmd = [ffmpeg, "-v", "error", "-ss", f"{ss:.3f}"] + (["-t", f"{dur:.3f}"] if dur else []) + ["-i", str(path)]
    chain = f"aresample={sr},aformat=sample_fmts=flt:channel_layouts=mono" + (f",{af}" if af else "")
    p = subprocess.run(cmd + ["-af", chain, "-f", "f32le", "-ac", "1", "-"], capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


# ------------------------------------------------------------------------------------------------ measurement
def band_edges(centers=THIRD_OCTAVE, fraction=3):
    c = np.asarray(centers, float)
    return c * 2 ** (-1 / (2 * fraction)), c * 2 ** (1 / (2 * fraction))


def _frames(x, nfft, hop):
    n = 1 + (len(x) - nfft) // hop if len(x) >= nfft else 0
    return n


def energy_vad(x, sr, nfft=4096, hop=1024, top_db=20.0, floor_db=12.0):
    """Boolean mask of speech-active frames, for material without word timings. A frame counts as speech when its
    RMS is within ``top_db`` of the 95th-percentile frame and at least ``floor_db`` above the 10th-percentile floor."""
    n = _frames(x, nfft, hop)
    if not n: return np.zeros(0, bool)
    idx = np.arange(n)[:, None] * hop + np.arange(0, nfft, 64)[None, :]          # strided RMS estimate (fast)
    d = 10 * np.log10(np.mean(x[idx] ** 2, axis=1) + 1e-18)
    hi, lo = np.percentile(d, 95), np.percentile(d, 10)
    return (d > hi - top_db) & (d > lo + floor_db)


def interval_mask(intervals, n_frames, sr, nfft=4096, hop=1024):
    """Boolean frame mask from (start_s, end_s) intervals, such as word timings. The middle half of the frame must lie
    inside an interval."""
    centers = (np.arange(n_frames) * hop + nfft // 2) / sr
    half = nfft / 4 / sr
    m = np.zeros(n_frames, bool)
    for a, b in intervals:
        m |= (centers - half >= a) & (centers + half <= b)
    return m


def ltas_power(x, sr, mask=None, centers=THIRD_OCTAVE, nfft=4096, hop=1024):
    """Summed per-band power over the selected frames → (acc[n_bands], n_frames_used).
    ``mask`` is a boolean array (one entry per frame), a callable ``(x, sr, nfft, hop) -> mask``, or None for all
    frames. Accumulators from several files can be added together before ``to_db``."""
    n = _frames(x, nfft, hop)
    if not n: return np.zeros(len(centers)), 0
    if callable(mask): mask = mask(x, sr, nfft, hop)
    idx = np.arange(n) if mask is None else np.nonzero(np.asarray(mask)[:n])[0]
    if not len(idx): return np.zeros(len(centers)), 0
    win = np.hanning(nfft).astype(np.float32)
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    lo, hi = band_edges(centers)
    sel = [(freqs >= l) & (freqs < h) for l, h in zip(lo, hi)]
    P = np.zeros(len(freqs))
    for k in range(0, len(idx), 1024):
        fr = np.stack([x[i * hop:i * hop + nfft] for i in idx[k:k + 1024]]) * win
        P += (np.abs(np.fft.rfft(fr, axis=1)) ** 2).sum(0)
    return np.array([P[m].sum() for m in sel]), len(idx)


def to_db(acc, centers=THIRD_OCTAVE, ref=(200, 6300)):
    """Band powers → dB *shape*: 0 dB is the mean of the bands whose centres lie in ``ref``."""
    db = 10 * np.log10(np.maximum(np.asarray(acc, float), 1e-30))
    c = np.asarray(centers)
    return db - db[(c >= ref[0]) & (c <= ref[1])].mean()


def smooth(db, passes=1):
    """[¼ ½ ¼] smoothing across adjacent bands, applied ``passes`` times (one pass ≈ 2/3 octave on 1/3-octave data)."""
    y = np.asarray(db, float)
    for _ in range(passes):
        y = np.convolve(np.pad(y, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    return y


def deviation(after, target, centers=THIRD_OCTAVE, rng=(200, 6300)):
    """Level-free deviation (after − target, mean over ``rng`` removed). Returns max |dev|, rms and per-band dev."""
    c = np.asarray(centers); m = (c >= rng[0]) & (c <= rng[1])
    d = np.asarray(after, float) - np.asarray(target, float)
    d = d - d[m].mean()
    return {"max_abs_dev_db": round(float(np.abs(d[m]).max()), 2), "rms_dev_db": round(float(np.sqrt((d[m] ** 2).mean())), 2),
            "dev_db": [round(float(v), 2) for v in d]}


# ------------------------------------------------------------------------------------------------ EQ model + fit
def peq_db(f, fc, gain_db, q, sr=48000):
    """Magnitude (dB) at frequencies ``f`` of an RBJ peaking biquad, which is what ffmpeg ``equalizer=t=q`` uses."""
    A = 10 ** (gain_db / 40); w0 = 2 * np.pi * fc / sr; al = np.sin(w0) / (2 * q)
    b = np.array([1 + al * A, -2 * np.cos(w0), 1 - al * A]); a = np.array([1 + al / A, -2 * np.cos(w0), 1 - al / A])
    z = np.exp(-1j * 2 * np.pi * np.asarray(f, float) / sr)
    return 20 * np.log10(np.abs((b[0] + b[1] * z + b[2] * z ** 2) / (a[0] + a[1] * z + a[2] * z ** 2)))


def peq_coeffs(fc, gain_db, q, sr=48000):
    """(b, a) normalised coefficients of the same biquad (for scipy.signal.lfilter in tests / offline use)."""
    A = 10 ** (gain_db / 40); w0 = 2 * np.pi * fc / sr; al = np.sin(w0) / (2 * q)
    b = np.array([1 + al * A, -2 * np.cos(w0), 1 - al * A]); a = np.array([1 + al / A, -2 * np.cos(w0), 1 - al / A])
    return b / a[0], a / a[0]


def eq_band_response(gains, eq_freqs=OCTAVE_EQ, q=1.4, centers=THIRD_OCTAVE, sr=48000):
    """Response of the whole EQ chain, power-averaged over each analysis band."""
    lo, hi = band_edges(centers)
    out = np.empty(len(centers))
    for b in range(len(centers)):
        f = np.geomspace(lo[b], min(hi[b], sr / 2 * 0.999), 9)
        r = sum(peq_db(f, fc, g, q, sr) for fc, g in zip(eq_freqs, gains))
        out[b] = 10 * np.log10(np.mean(10 ** (np.asarray(r) / 10)))
    return out


def fit_eq(before, target, centers=THIRD_OCTAVE, eq_freqs=OCTAVE_EQ, q=1.4, gmax=6.0, sr=48000, rng=(200, 6300),
           protect_below=None, cap_above=(9000, 1.5), smooth_passes=1, iters=25, ridge=0.05):
    """Fit peaking gains so that ``before + EQ`` approaches ``target`` (both dB shapes on ``centers``).

    * The wanted correction is ``smooth(target - before)``. Bands inside ``rng`` get weight 1; the rest get 0.25.
    * ``protect_below=f`` leaves bands below f uncorrected. Use it when the source is a different voice type: a low
      voice's fundamental region is not a mic colouration to remove.
    * ``cap_above=(f, db)`` limits boosts above f, so hiss and air in band-limited sources are not lifted.
    * The fit ignores level, because loudness normalisation sets the level later. Gains are clipped to ±gmax dB.
    * ``ridge`` pulls gains towards 0 dB. This removes the free "all bands up" direction that a level-free fit has
      and keeps neighbouring bands from fighting each other.
    """
    c = np.asarray(centers, float)
    want = smooth(np.asarray(target, float) - np.asarray(before, float), smooth_passes)
    w = np.where((c >= rng[0]) & (c <= rng[1]), 1.0, 0.25)
    if protect_below:
        want = np.where(c < protect_below, 0.0, want); w = np.where(c < protect_below, 0.5, w)
    if cap_above:
        want = np.where(c > cap_above[0], np.minimum(want, cap_above[1]), want)
    want = want - (want * w).sum() / w.sum()
    g = np.zeros(len(eq_freqs))
    resp = lambda gg: eq_band_response(gg, eq_freqs, q, centers, sr)
    for _ in range(iters):
        r0 = resp(g)
        r = r0 - (r0 * w).sum() / w.sum()
        J = np.stack([(resp(g + 0.5 * np.eye(len(g))[i]) - r0) / 0.5 for i in range(len(g))], 1)
        J = J - (J * w[:, None]).sum(0) / w.sum()
        A = np.vstack([J * np.sqrt(w)[:, None], np.sqrt(ridge) * np.eye(len(g))])
        y = np.concatenate([(want - r) * np.sqrt(w), -np.sqrt(ridge) * g])
        step, *_ = np.linalg.lstsq(A, y, rcond=None)
        g = g + 0.7 * step
        g = np.clip(g - np.median(g), -gmax, gmax)      # level is free: centre the gains before limiting them
    return np.round(g, 1)


def eq_string(gains, eq_freqs=OCTAVE_EQ, q=1.4, min_abs=0.3):
    """ffmpeg filter string for fitted gains. Bands smaller than ``min_abs`` dB are dropped. Returns 'anull' if empty."""
    parts = [f"equalizer=f={f}:t=q:w={q}:g={g:+.1f}" for f, g in zip(eq_freqs, gains) if abs(g) >= min_abs]
    return ",".join(parts) or "anull"


# ------------------------------------------------------------------------------------------------ room / reverb
def early_reflections(delays_ms=(3.1, 5.3, 7.9, 11.3, 17.2, 26.0, 39.0), first=0.30, rt60_ms=110.0, in_gain=0.9,
                      out_gain=0.9):
    """ffmpeg ``aecho`` string for a small, damped space, such as a car cabin or a small furnished room.
    Tap gains fall exponentially so that the level drops 60 dB in ``rt60_ms``. Use delays that are not related by
    simple ratios, so the comb ripple averages out instead of ringing at one pitch."""
    d = np.asarray(delays_ms, float)
    gains = first * 10 ** (-3 * (d - d[0]) / rt60_ms)
    return (f"aecho={in_gain}:{out_gain}:" + "|".join(f"{x:g}" for x in d) + ":" +
            "|".join(f"{g:.3f}" for g in gains))


def decay_ratio_db(x, sr, intervals, early=(0.0, 0.03), late=(0.03, 0.12)):
    """Crude reverberance probe: across word offsets followed by ≥ 0.25 s of gap, the energy in [end+late] minus the
    energy in the final ``early`` window of the word, in dB. A more negative value means a drier signal."""
    vals = []
    iv = sorted(intervals)
    for k, (a, b) in enumerate(iv):
        nxt = iv[k + 1][0] if k + 1 < len(iv) else b + 1.0
        if nxt - b < 0.25 or b - a < 0.15: continue
        e0 = x[int((b - 0.03) * sr):int(b * sr)]
        e1 = x[int((b + late[0]) * sr):int((b + late[1]) * sr)]
        if len(e0) and len(e1):
            vals.append(10 * np.log10((np.mean(e1 ** 2) + 1e-18) / (np.mean(e0 ** 2) + 1e-18)))
    return float(np.median(vals)) if vals else float("nan")


# ------------------------------------------------------------------------------------------------ pitch
def median_f0(x, sr, fmin=70.0, fmax=400.0, frame=2048, hop=512, voicing=0.45):
    """Median fundamental frequency (Hz) of the voiced frames in ``x``, by normalised autocorrelation; NaN if
    unvoiced. This is meant for sorting words or phrases by speaker pitch, not for pitch tracking."""
    x = np.asarray(x, float)
    if len(x) < frame: return float("nan")
    lmin, lmax = int(sr / fmax), min(int(sr / fmin), frame - 1)
    f0s = []
    for s in range(0, len(x) - frame + 1, hop):
        fr = x[s:s + frame] - x[s:s + frame].mean()
        e = (fr ** 2).sum()
        if e <= 1e-12: continue
        ac = np.fft.irfft(np.abs(np.fft.rfft(fr, 2 * frame)) ** 2)[:frame]
        ac /= ac[0]
        lag = lmin + int(np.argmax(ac[lmin:lmax]))
        if ac[lag] < voicing: continue
        if 1 <= lag < frame - 1:                                         # parabolic refinement
            y0, y1, y2 = ac[lag - 1], ac[lag], ac[lag + 1]
            den = y0 - 2 * y1 + y2
            lag = lag + (0.5 * (y0 - y2) / den if den else 0.0)
        f0s.append(sr / lag)
    return float(np.median(f0s)) if f0s else float("nan")


# ------------------------------------------------------------------------------------------------ command line
def _mono(path, ss=0.0, dur=None):
    """Read a file as mono float32 with soundfile; falls back to ffmpeg (``decode``) for formats it cannot read."""
    try:
        import soundfile as sf
        info = sf.info(str(path))
        a = int(ss * info.samplerate)
        b = None if dur is None else a + int(dur * info.samplerate)
        x, sr = sf.read(str(path), start=a, stop=b, dtype="float32", always_2d=True)
        return x.mean(1), sr
    except Exception:  # noqa: BLE001 - any soundfile failure: let ffmpeg try
        return decode(path, 48000, ss, dur), 48000


def main(argv=None) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(prog="opendubstage ltasmatch",
                                 description="Fit a smoothed, gain-limited EQ that makes SOURCE's long-term average "
                                             "speech spectrum match REFERENCE's; prints an ffmpeg filter string.")
    ap.add_argument("reference"); ap.add_argument("source")
    ap.add_argument("--gmax", type=float, default=6.0, help="per-band gain limit (dB)")
    ap.add_argument("--protect-below", type=float, help="leave bands below this (Hz) uncorrected")
    ap.add_argument("--half-octave", action="store_true", help="use half-octave EQ bands instead of octave bands")
    ap.add_argument("--all-frames", action="store_true", help="measure every frame (skip the energy speech detector)")
    ap.add_argument("--json", action="store_true", help="print gains and deviation as JSON instead")
    a = ap.parse_args(argv)
    eqf = HALF_OCTAVE_EQ if a.half_octave else OCTAVE_EQ
    shapes = []
    for path in (a.reference, a.source):
        x, sr = _mono(path)
        acc, n = ltas_power(x, sr, mask=None if a.all_frames else energy_vad)
        if not n:   # no level contrast (e.g. a steady bed): measure every frame instead
            import sys
            print(f"{path}: no speech-active frames found; using all frames", file=sys.stderr)
            acc, n = ltas_power(x, sr)
        if not n:
            raise SystemExit(f"{path}: too short to measure")
        shapes.append(to_db(acc))
    target, before = shapes
    g = fit_eq(before, target, eq_freqs=eqf, gmax=a.gmax, protect_below=a.protect_below)
    after = before + eq_band_response(g, eqf)
    if a.json:
        print(json.dumps({"eq": eq_string(g, eqf), "gains_db": dict(zip(eqf, g.tolist())),
                          "before": deviation(before, target), "after": deviation(after, target)}, indent=1))
    else:
        print(eq_string(g, eqf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
