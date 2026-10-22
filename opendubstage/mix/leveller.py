"""Conservative offline phrase levelling from explicit word timings (no VAD or compressor).

Timing JSON: {"words": [{"start": 0.2, "end": 1.0}, ...],
              "breaths": [{"start": 1.2, "end": 1.4}, ...]}.
Times are seconds, ordered, non-overlapping and inside the recording. Breaths must not
intersect words. Unknown fields (e.g. transcript text) are ignored, never copied to reports.

Measurement is ungated K-weighted mean-square loudness over each phrase, using the
BS.1770 filter and mono/stereo channel sum. It is NOT standardized 3-second short-term
or gated programme loudness. The target -24 is a starting point inspired by A/85,
not a delivery compliance claim. Other defaults are conservative engineering choices:
300 ms phrase gap, 150 ms raised-cosine ramps in pauses, -12/+6 dB gain limits,
0.5 LU no-op tolerance, -60 loudness floor (do not boost near-silence), -1 dBFS
sample-peak ceiling, and breaths kept unchanged unless explicitly attenuated by 6 dB.
Ramps require the configured duration at internal boundaries, using at most half each
available gap. If a full ramp does not fit, the entire event is left unchanged and
reported as insufficient_pause. Phrase gains never change mid-word. Clipping prevention caps
positive gain using the entire affected support. This is not
an inter-sample true-peak limiter; already-hot untouched input is reported, not repaired.

API: level(audio, sr, timings, **options) -> (audio_copy, report). Mono/stereo float
arrays only. Report schema opendubstage.leveller/0.1 contains parameters, event sample
bounds, gains, before/after phrase measurements, headroom constraints, actual changed
and untouched sample ranges, and sample peaks. Null loudness means digital silence.
CLI preserves the input WAV subtype; bit-exact means decoded samples in untouched
regions, not byte-identical WAV containers. Report measurements precede output encoding.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import sosfilt

from opendubstage.qc.mixqc import kweight_sos


def phrase_loudness(audio, sr):
    """Ungated K-weighted phrase loudness; null for digital silence."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    z = sosfilt(kweight_sos(sr), x, axis=0)
    power = float(np.mean(np.sum(z * z, axis=1)))
    return -0.691 + 10 * math.log10(power) if power > 0 else None


def _intervals(items, sr, n, name):
    if not isinstance(items, list):
        raise ValueError(f'{name} must be a list')
    result = []
    previous = 0.0
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f'{name} entries must have start/end seconds')
        try:
            a, b = float(item['start']), float(item['end'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'{name} entries must have numeric start/end') from exc
        if not (math.isfinite(a) and math.isfinite(b) and previous <= a < b <= n / sr):
            raise ValueError(f'{name} must be ordered, non-overlapping and within audio')
        # Protect every sample touched by a word, including fractional-sample edges.
        lo, hi = int(math.floor(a * sr)), min(n, int(math.ceil(b * sr)))
        if result and lo < result[-1][1]:
            raise ValueError(f'{name} overlap at sample resolution')
        result.append((lo, hi))
        previous = b
    return result


def _ranges(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [[int(a), int(b)] for a, b in edges.reshape(-1, 2)]


def _peak(x):
    p = float(np.max(np.abs(x))) if x.size else 0.0
    return 20 * math.log10(p) if p else None


def level(audio, sr, timings, *, target=-24.0, phrase_gap=0.3, ramp=0.15,
          min_gain=-12.0, max_gain=6.0, tolerance=0.5, floor=-60.0,
          peak_ceiling=-1.0, breath_policy='keep', breath_db=-6.0):
    """Return a new float array and JSON-safe report; never modify the input."""
    x = np.asarray(audio)
    if x.dtype.kind != 'f' or x.ndim not in (1, 2) or (x.ndim == 2 and x.shape[1] not in (1, 2)):
        raise ValueError('audio must be a mono/stereo floating-point array')
    if not np.all(np.isfinite(x)) or not isinstance(sr, (int, np.integer)) or sr < 8000:
        raise ValueError('audio must be finite and sample rate an integer >= 8000')
    params = dict(target=target, phrase_gap=phrase_gap, ramp=ramp, min_gain=min_gain,
                  max_gain=max_gain, tolerance=tolerance, floor=floor,
                  peak_ceiling=peak_ceiling, breath_policy=breath_policy, breath_db=breath_db)
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for k, v in params.items() if k != 'breath_policy'):
        raise ValueError('parameters must be finite numbers')
    if not (0 <= phrase_gap and 0 < ramp and min_gain <= 0 <= max_gain <= 60 and
            min_gain >= -120 and tolerance >= 0 and floor < target <= 0 and
            peak_ceiling <= 0 and -120 <= breath_db <= 0 and breath_policy in ('keep', 'attenuate')):
        raise ValueError('invalid leveller parameters')
    if not isinstance(timings, dict) or 'words' not in timings:
        raise ValueError('timings must contain a words list')
    n = len(x)
    words = _intervals(timings['words'], sr, n, 'words')
    breaths = _intervals(timings.get('breaths', []), sr, n, 'breaths')
    if any(a < d and c < b for a, b in words for c, d in breaths):
        raise ValueError('breaths must not overlap words')
    phrases = []
    for a, b in words:
        if (phrases and a - phrases[-1][1] <= phrase_gap * sr and
                not any(c < a and d > phrases[-1][1] for c, d in breaths)):
            phrases[-1][1] = b
        else:
            phrases.append([a, b])
    ramp_samples = max(2, int(math.ceil(ramp * sr)))
    events = sorted([(a, b, 'phrase') for a, b in phrases] + [(a, b, 'breath') for a, b in breaths])
    envelope = np.ones(n, dtype=np.float64)
    report_events = []
    for i, (a, b, kind) in enumerate(events):
        left = events[i - 1][1] if i else 0
        right = events[i + 1][0] if i + 1 < len(events) else n
        pre = min(ramp_samples, (a - left) // 2)
        post = min(ramp_samples, (right - b) // 2)
        lo, hi = a - pre, b + post
        before = phrase_loudness(x[a:b], sr)
        requested = 0.0
        reason = 'breath_kept' if kind == 'breath' else 'within_tolerance'
        if kind == 'breath' and breath_policy == 'attenuate':
            requested, reason = breath_db, 'breath_attenuated'
        elif kind == 'phrase':
            if before is None or before < floor:
                reason = 'below_floor'
            elif abs(target - before) > tolerance:
                requested, reason = target - before, 'levelled'
        gain = float(np.clip(requested, min_gain, max_gain)) if kind == 'phrase' else requested
        peak = _peak(x[lo:hi])
        headroom_limited = gain > 0 and peak is not None and gain > max(0.0, peak_ceiling - peak)
        if headroom_limited:
            gain = max(0.0, peak_ceiling - peak)
        insufficient_pause = gain != 0 and ((a > 0 and pre < ramp_samples) or (b < n and post < ramp_samples))
        if insufficient_pause:
            gain, reason = 0.0, 'insufficient_pause'
        if gain != 0:
            envelope[a:b] = 10 ** (gain / 20)
            if pre:
                t = np.arange(pre) / pre
                envelope[lo:a] = 10 ** (gain * (0.5 - 0.5 * np.cos(np.pi * t)) / 20)
            if post:
                t = np.arange(1, post + 1) / post
                envelope[b:hi] = 10 ** (gain * (0.5 + 0.5 * np.cos(np.pi * t)) / 20)
        report_events.append(dict(kind=kind, start_sample=a, end_sample=b,
                                  support=[lo, hi], before_loudness=before,
                                  requested_gain_db=requested, gain_db=gain, reason=reason,
                                  gain_limited=not math.isclose(gain, requested),
                                  headroom_limited=bool(headroom_limited),
                                  insufficient_pause=bool(insufficient_pause)))
    y = x.copy()
    changed = envelope != 1.0
    y[changed] = x[changed] * (envelope[changed, None] if x.ndim == 2 else envelope[changed])
    for event in report_events:
        event['after_loudness'] = phrase_loudness(y[event['start_sample']:event['end_sample']], sr)
    report = dict(schema='opendubstage.leveller/0.1', sample_rate=int(sr), samples=n,
                  measurement='ungated K-weighted phrase mean-square (not EBU short-term)',
                  parameters=params, events=report_events, changed_ranges=_ranges(changed),
                  untouched_ranges=_ranges(~changed), input_sample_peak_dbfs=_peak(x),
                  output_sample_peak_dbfs=_peak(y), peak_ceiling_is_true_peak=False)
    return y, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--timings', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    for name, default in [('target', -24.), ('phrase-gap', .3), ('ramp', .15),
                          ('min-gain', -12.), ('max-gain', 6.), ('tolerance', .5),
                          ('floor', -60.), ('peak-ceiling', -1.), ('breath-db', -6.)]:
        parser.add_argument('--' + name, type=float, default=default)
    parser.add_argument('--breath-policy', choices=['keep', 'attenuate'], default='keep')
    args = vars(parser.parse_args(argv))
    src, dst, timing_path, report_path = [args.pop(k) for k in ('input', 'output', 'timings', 'report')]
    if len({p.resolve() for p in (src, dst, timing_path, report_path)}) != 4:
        parser.error('input, output, timings and report must be distinct paths')
    try:
        info = sf.info(src)
        if info.format not in ('WAV', 'WAVEX') or dst.suffix.lower() != '.wav':
            raise ValueError('input and output must be WAV files')
        if info.subtype not in ('PCM_U8', 'PCM_16', 'PCM_24', 'PCM_32', 'FLOAT', 'DOUBLE'):
            raise ValueError('WAV must use lossless PCM or floating-point samples')
        x, sr = sf.read(src, dtype='float64')
        y, report = level(x, sr, json.loads(timing_path.read_text()), **args)
        encoded_report = json.dumps(report, indent=2, allow_nan=False) + '\n'
        sf.write(dst, y, sr, subtype=info.subtype)
        report_path.write_text(encoded_report)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
