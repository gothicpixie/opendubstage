"""Room tone from explicitly clean gaps, using random-phase overlap-add synthesis.

Input timing JSON accepts [[start, end], ...] or {"words": [{"start": ..., "end": ...}]}.
Words/exclusions are protected, NOT automatically detected. Complete Hann analysis
frames are collected inside their complement; no frame crosses a protected boundary.
A normalized average power spectrum learns colour independently of level. Slow level
variation is approximated by randomly resampled, smoothed frame RMS statistics, not
by replaying a recording. Independent random phases avoid a copied loop period.

Engineering defaults: 128 ms FFT for low-frequency resolution; 75% overlap to smooth
random frames; 20 ms word guard; 0.5 s minimum total clean material (at least four
frames); 0.5 s envelope knots, +/-6 dB envelope clamp for conservative modulation.
These are not a perceptual guarantee. Mono only; no stereo-coherence claims.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d


@dataclass(frozen=True)
class Params:
    frame_s: float = 0.128
    guard_s: float = 0.020
    min_clean_s: float = 0.5
    modulation_s: float = 0.5
    modulation_limit_db: float = 6.0


def _audio(x, sr):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("input must be nonempty finite mono audio")
    if not isinstance(sr, (int, np.integer)) or isinstance(sr, bool) or sr < 1000:
        raise ValueError("sample rate must be an integer >= 1000")
    return x


def _params(p):
    vals = (p.frame_s, p.guard_s, p.min_clean_s, p.modulation_s, p.modulation_limit_db)
    if not all(np.isfinite(v) for v in vals) or p.frame_s <= 0 or p.guard_s < 0 or p.min_clean_s <= 0 or p.modulation_s <= 0 or p.modulation_limit_db < 0:
        raise ValueError("invalid room-tone parameters")


def clean_intervals(duration, protected=(), excluded=(), guard_s=0.02):
    """Complement of validated, padded protected/excluded intervals in seconds."""
    busy = []
    for a, b in list(protected) + list(excluded):
        if not np.isfinite([a, b]).all() or a < 0 or b <= a or b > duration + 1e-9:
            raise ValueError("intervals must satisfy 0 <= start < end <= duration")
        busy.append((max(0., a - guard_s), min(duration, b + guard_s)))
    cursor, out = 0., []
    for a, b in sorted(busy):
        if a > cursor:
            out.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < duration:
        out.append((cursor, duration))
    return out


def learn(x, sr, protected=(), excluded=(), p=Params()):
    """Learn model from gap samples only. Raises rather than inventing missing tone."""
    x = _audio(x, sr); _params(p)
    nfft = max(32, int(round(p.frame_s * sr / 2)) * 2)
    hop = nfft // 4
    win = signal.windows.hann(nfft, sym=False)
    spectra, levels, slow = [], [], []
    used, intervals = 0, []
    for a, b in clean_intervals(len(x) / sr, protected, excluded, p.guard_s):
        i, j = int(np.ceil(a * sr)), int(np.floor(b * sr))
        if j - i < nfft:
            continue
        seg = x[i:j]
        frames = np.lib.stride_tricks.sliding_window_view(seg, nfft)[::hop]
        powers = np.mean(frames ** 2, axis=1)
        # Silent frames cannot teach spectral colour; silence-only input is handled separately.
        keep = powers > 1e-30
        if keep.any():
            z = np.fft.rfft(frames[keep] * win, axis=1)
            spectra.append(np.abs(z) ** 2 / powers[keep, None] / np.sum(win ** 2) * nfft)
            lv = 10 * np.log10(powers[keep])
            slow.extend(uniform_filter1d(lv, size=max(1, int(p.modulation_s * sr / hop)), mode="nearest"))
        levels.extend(powers)
        used += len(seg); intervals.append([i / sr, j / sr])
    if used < p.min_clean_s * sr or len(levels) < 4:
        raise ValueError("insufficient clean gaps: need minimum clean duration and four complete analysis frames")
    power = float(np.mean(levels))
    spec = np.mean(np.concatenate(spectra), axis=0) if spectra else np.zeros(nfft // 2 + 1)
    log_level = np.asarray(slow) if slow else np.array([0.])
    variation = np.clip(log_level - np.mean(log_level), -p.modulation_limit_db, p.modulation_limit_db)
    return dict(sr=int(sr), nfft=nfft, hop=hop, spectrum=spec, rms=np.sqrt(power), modulation_db=variation,
                modulation_s=p.modulation_s, clean_seconds=used / sr, frames=len(levels), intervals=intervals)


def synthesize(model, duration, seed=0):
    """Return (mono audio, JSON-safe report). Seed makes synthesis repeatable."""
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite")
    sr, nfft, hop = model["sr"], model["nfft"], model["hop"]
    count = int(round(duration * sr))
    if count < 1:
        raise ValueError("duration is shorter than one sample")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    # Pad both ends, then crop into the stationary overlap region.
    total = count + 2 * nfft
    y, weight = np.zeros(total + nfft), np.zeros(total + nfft)
    win = signal.windows.hann(nfft, sym=False)
    mag = np.sqrt(model["spectrum"])
    for start in range(0, total, hop):
        phase = rng.uniform(-np.pi, np.pi, len(mag))
        z = mag * np.exp(1j * phase)
        z[0] = mag[0] * rng.choice([-1., 1.])
        z[-1] = mag[-1] * rng.choice([-1., 1.])
        y[start:start+nfft] += np.fft.irfft(z, n=nfft) * win
        weight[start:start+nfft] += win ** 2
    y = y[nfft:nfft+count] / np.sqrt(weight[nfft:nfft+count])
    step = max(1, int(round(model["modulation_s"] * sr)))
    knots = np.arange(0, count + step, step)
    env = rng.choice(model["modulation_db"], len(knots))
    # Cosine interpolation gives zero slope at every knot, with no repeated audio blocks.
    pos = np.arange(count) / step
    index = pos.astype(int); blend = (1 - np.cos(np.pi * (pos-index))) / 2
    y *= 10 ** ((env[index] * (1-blend) + env[index+1] * blend) / 20)
    measured = np.sqrt(np.mean(y ** 2))
    if measured > 0:
        y *= model["rms"] / measured
    peak = float(np.max(np.abs(y)))
    # Never silently clip or lower learned level.
    if peak > 1:
        raise ValueError("synthesized tone exceeds full scale; supply quieter clean material")
    report = dict(method="random_phase_overlap_add", seed=seed, sample_rate=sr, samples=count,
                  duration=count/sr, clean_seconds=model["clean_seconds"], analysis_frames=model["frames"],
                  clean_intervals=model["intervals"], frame_samples=nfft,
                  source_rms=float(model["rms"]), output_rms=float(np.sqrt(np.mean(y*y))), peak=peak,
                  modulation_std_db=float(np.std(model["modulation_db"])),
                  limitations=["mono", "explicit clean-gap selection", "statistical approximation; listening validation required"])
    return y, report


def generate(x, sr, duration, protected=(), excluded=(), seed=0, p=Params()):
    return synthesize(learn(x, sr, protected, excluded, p), duration, seed)



def read_intervals(path):
    """Strict timing schema: a supplied but unrecognized file must never mean no protection."""
    if not path:
        return []
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        keys = [key for key in ("words", "events", "intervals") if key in data]
        if len(keys) != 1:
            raise ValueError("timing object must contain exactly one of words/events/intervals")
        data = data[keys[0]]
    if not isinstance(data, list):
        raise ValueError("timings must be a list")
    result = []
    for item in data:
        if isinstance(item, list) and len(item) == 2:
            a, b = item
        elif isinstance(item, dict) and "start" in item and "end" in item:
            a, b = item["start"], item["end"]
        elif isinstance(item, dict) and "t" in item and "e" in item:
            a, b = item["t"], item["e"]
        else:
            raise ValueError("each timing needs [start,end] or start/end (t/e) fields")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (a,b)):
            raise ValueError("timing endpoints must be numbers")
        if not np.isfinite([a,b]).all() or a < 0 or b <= a:
            raise ValueError("timings must have finite 0 <= start < end")
        result.append((float(a),float(b)))
    return result


def main(argv=None):
    import soundfile as sf
    ap = argparse.ArgumentParser(prog="opendubstage roomtone", description=__doc__)
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--protected", help="word/kept-sound timing JSON; all other samples must be clean tone")
    ap.add_argument("--excluded", help="other unwanted event timing JSON")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report")
    a = ap.parse_args(argv)
    inputs = {Path(v).resolve() for v in (a.src, a.protected, a.excluded) if v}
    outputs = [Path(v).resolve() for v in (a.dst, a.report) if v]
    if inputs.intersection(outputs) or len(set(outputs)) != len(outputs):
        ap.error("output/report must not overwrite audio or timing inputs, or each other")
    try:
        x, sr = sf.read(a.src, dtype="float64")
        y, report = generate(x, sr, a.duration, read_intervals(a.protected), read_intervals(a.excluded), a.seed)
    except ValueError as exc:
        ap.error(str(exc))
    sf.write(a.dst, y, sr, subtype="FLOAT")
    if a.report:
        Path(a.report).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
