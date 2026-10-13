"""memofix: remove low-frequency rumble from a voice memo recorded in a moving car (or any boomy room), then nudge
its tonal balance toward an average speech spectrum. Pure DSP (numpy / scipy); the voice is never resynthesised.

Two stages:

1. **Gap-learned LF subtraction.** The rumble's power spectrum is learned from the *gaps between words* (frames
   at least ``gap_guard`` s away from any speech interval), as a rolling median over a ``profile_window`` s window
   so the profile follows changing road or engine noise. Below ``lf_max`` Hz each STFT bin gets a Wiener-like gain
   ``1 - over * N / P``, floored at ``floor_db`` and smoothed across frames, so there is no "musical noise". Bins at
   and above ``lf_max`` pass untouched, so consonants and the voice's presence region are not processed at all.
2. **Matching EQ toward the long-term average speech spectrum (LTASS).** The reference is the
   international LTASS of Byrne et al. (1994), "An international comparison of long-term average speech spectra",
   JASA 96(4). It is measured on speech frames only and both curves are normalised at 500 Hz to 2 kHz.
   The difference is smoothed, applied at ``strength`` (partial), limited to ``-max_cut_db`` / ``+max_boost_db``
   and only allowed to *cut* below ``cut_only_below`` Hz (car boom), so it removes boxiness and lifts a dull top
   without inventing low end. The result is RMS-matched to the input.

Speech intervals come from word timings when you have them (any ASR's word list); otherwise a simple energy
detector in the 300 Hz to 3 kHz band picks the quietest frames as gaps.

API: ``memofix(x, sr, speech=None, ...) -> (y, report)``. CLI: ``opendubstage memofix IN.wav OUT.wav
[--words WORDS.json] [--offset S] [--strength 0.6]``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve, firwin2, istft, stft

# 1/3-octave centres (Hz) and the Byrne et al. (1994) international LTASS (dB SPL at 1 m, combined male/female,
# "normal" vocal effort) at those centres, as transcribed for this tool; only the *shape* is used.
# TODO: re-verify each value against Table II of the paper before a 1.0 release.
LTASS_BANDS = np.array([63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
                        4000, 5000, 6300, 8000, 10000, 12500, 16000], float)
LTASS_DB = np.array([38.6, 43.5, 54.4, 57.7, 56.8, 60.2, 60.3, 59.0, 62.1, 62.1, 60.5, 56.8, 53.7, 53.0, 52.0, 48.7,
                     48.1, 46.8, 45.6, 44.5, 44.3, 43.7, 43.4, 41.3, 40.7], float)


@dataclass(frozen=True)
class MemoParams:
    nfft: int = 4096
    hop: int = 1024
    lf_max: float = 400.0          # Hz; subtraction only below this
    over: float = 1.5              # over-subtraction factor
    floor_db: float = -15.0        # most a bin is ever reduced
    profile_window: float = 10.0   # s; rolling noise profile (+/- half on each side)
    gap_guard: float = 0.2         # s kept clear around each speech interval when choosing gap frames
    strength: float = 0.6          # fraction of the LTASS correction applied
    max_cut_db: float = 6.0
    max_boost_db: float = 3.0
    cut_only_below: float = 300.0  # Hz; the EQ may only cut below this
    eq_taps: int = 2047


def _frame_times(n_frames, sr, hop):
    return np.arange(n_frames) * hop / sr


def speech_masks(times, speech, gap_guard):
    """(speech, gap) boolean masks over frame centre times from (start, end) speech intervals."""
    sp = np.zeros(len(times), bool)
    gap = np.ones(len(times), bool)
    for a, b in speech:
        sp |= (times >= a) & (times <= b)
        gap &= ~((times >= a - gap_guard) & (times <= b + gap_guard))
    return sp, gap


def energy_gaps(P, f, quantile=15.0):
    """Fallback gap mask without word timings: the quietest frames in the 300 Hz to 3 kHz band."""
    e = P[(f >= 300) & (f < 3000)].sum(0)
    return e < np.percentile(e, quantile)


def lf_gains(P, f, times, gap, p: MemoParams = MemoParams()):
    """Per-bin amplitude gains (same shape as P) for the LF subtraction; 1.0 at and above ``p.lf_max``."""
    G = np.ones_like(P)
    lf = f < p.lf_max
    if not lf.any() or not gap.any():
        return G
    PL = P[lf]
    floor = 10 ** (p.floor_db / 10)                       # power-domain floor
    half = p.profile_window / 2
    for i in range(P.shape[1]):
        m = gap & (np.abs(times - times[i]) < half)
        if m.sum() < 4:
            m = gap
        noise = np.median(PL[:, m], axis=1)
        G[lf, i] = np.clip(1 - p.over * noise / (PL[:, i] + 1e-20), floor, 1.0)
    G[lf] = np.sqrt(G[lf])                                # power gain -> amplitude gain
    for i in range(1, P.shape[1]):                        # one-pole smoothing across frames (no musical noise)
        G[lf, i] = 0.5 * G[lf, i] + 0.5 * G[lf, i - 1]
    return G


def ltass_correction(Z, f, speech, p: MemoParams = MemoParams()):
    """Clamped, smoothed per-band EQ (dB on LTASS_BANDS) that moves the speech frames of ``Z`` toward the LTASS."""
    Ps = np.abs(Z[:, speech]) ** 2 if speech.sum() > 20 else np.abs(Z) ** 2
    meas = np.array([10 * np.log10(Ps[(f >= b / 2 ** (1 / 6)) & (f < b * 2 ** (1 / 6))].mean() + 1e-20)
                     for b in LTASS_BANDS])
    ref = (LTASS_BANDS >= 500) & (LTASS_BANDS <= 2000)
    corr = (LTASS_DB - LTASS_DB[ref].mean()) - (meas - meas[ref].mean())
    corr = np.convolve(np.pad(corr, 1, mode="edge"), [0.25, 0.5, 0.25], "valid")
    corr = np.clip(corr * p.strength, -p.max_cut_db, p.max_boost_db)
    low = LTASS_BANDS < p.cut_only_below
    corr[low] = np.minimum(corr[low], 0.0)
    return corr


def eq_fir(corr_db, sr, taps=2047):
    """Linear-phase FIR realising a per-band dB curve on LTASS_BANDS (bands above Nyquist are dropped)."""
    keep = LTASS_BANDS < sr / 2
    b, c = LTASS_BANDS[keep], np.asarray(corr_db)[keep]
    fx = np.concatenate([[0.0], b, [sr / 2]])
    gx = 10 ** (np.concatenate([[c[0]], c, [c[-1]]]) / 20)
    return firwin2(taps, fx / (sr / 2), gx)


def memofix(x, sr, speech=None, p: MemoParams = MemoParams(), eq=True):
    """Process a mono (or multichannel, averaged to mono) signal. ``speech``: [(start_s, end_s)] relative to x[0],
    or None to detect gaps from energy. Returns (y float32 mono, report dict)."""
    x = np.asarray(x, np.float64)
    if x.ndim == 2:
        x = x.mean(1)
    f, tt, Z = stft(x, sr, nperseg=p.nfft, noverlap=p.nfft - p.hop)
    P = np.abs(Z) ** 2
    if speech:
        sp, gap = speech_masks(tt, speech, p.gap_guard)
        if gap.sum() < 10:
            gap = energy_gaps(P, f)
    else:
        gap = energy_gaps(P, f)
        sp = ~gap
    Z2 = Z * lf_gains(P, f, tt, gap, p)
    _, y = istft(Z2, sr, nperseg=p.nfft, noverlap=p.nfft - p.hop)
    y = y[:len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    corr = np.zeros(len(LTASS_BANDS))
    if eq:
        corr = ltass_correction(Z2, f, sp, p)
        y = fftconvolve(y, eq_fir(corr, sr, p.eq_taps), mode="same")
    rx, ry = np.mean(x ** 2), np.mean(y ** 2)
    if ry > 0 and rx > 0:
        y = y * np.sqrt(rx / ry)                          # RMS-match to the input
    report = {"gap_frames": int(gap.sum()), "speech_frames": int(sp.sum()),
              "eq_db": dict(zip(LTASS_BANDS.astype(int).tolist(), np.round(corr, 1).tolist()))}
    return y.astype(np.float32), report


def load_words(path, offset=0.0, kinds=("word", "audio_event")):
    """Speech intervals from a word-timing JSON: a list of {"t"/"start", "e"/"end", "type"?} or [start, end] pairs.
    ``offset`` (s) is the position of the clip's first sample on the word list's timeline."""
    data = json.loads(open(path).read())
    if isinstance(data, dict):
        data = data.get("words", data.get("segments", []))
    out = []
    for w in data:
        if isinstance(w, (list, tuple)):
            a, b = w[0], w[1]
        else:
            if w.get("type", "word") not in kinds:
                continue
            a, b = w.get("t", w.get("start")), w.get("e", w.get("end"))
        out.append((float(a) - offset, float(b) - offset))
    return out


def main(argv=None) -> int:
    import argparse
    import soundfile as sf
    ap = argparse.ArgumentParser(prog="opendubstage memofix", description=__doc__.split("\n\n")[0])
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--words", help="word timings JSON (list of {t,e} / {start,end} or [a, b] pairs)")
    ap.add_argument("--offset", type=float, default=0.0, help="clip start on the word list's timeline (s)")
    ap.add_argument("--strength", type=float, default=MemoParams.strength)
    ap.add_argument("--no-eq", action="store_true", help="LF subtraction only")
    a = ap.parse_args(argv)
    x, sr = sf.read(a.inp, dtype="float64", always_2d=False)
    speech = load_words(a.words, a.offset) if a.words else None
    y, rep = memofix(x, sr, speech, MemoParams(strength=a.strength), eq=not a.no_eq)
    sf.write(a.out, y, sr)
    print(json.dumps({"out": a.out, **rep}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
