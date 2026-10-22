# Experimental dialogue leveller

`opendubstage leveller input.wav output.wav --timings words.json --report report.json`

This offline gain rider takes explicit word timings. It does not detect speech, transcribe,
compress, denoise, or send audio anywhere. Input and output are mono/stereo PCM or floating-point
WAV files, with the input sample rate and subtype preserved. The full recording is held in RAM.

## Timing schema

```json
{
  "words": [{"start": 0.5, "end": 1.0}, {"start": 1.1, "end": 1.8}],
  "breaths": [{"start": 2.0, "end": 2.2}]
}
```

Times are seconds relative to sample zero. Each list must be ordered and non-overlapping,
within the recording, and have positive durations. Breaths must not intersect words.
Fractional sample edges round outward to protect words. Extra fields such as transcript
text are ignored and not copied into the report. Empty word lists are valid.

## Method and defaults

Words separated by at most `--phrase-gap 0.3` seconds form one phrase. An explicit breath
splits a phrase even when the gap is short. All channels receive the same gain, held constant
from the first word's start to the last word's end, including internal gaps. The meter resets
at each phrase and measures ungated K-weighted mean-square energy with mono/stereo channel
summing. This is a useful phrase descriptor, **not** EBU standardized 3-second short-term,
gated integrated loudness, or a claim of delivery compliance.

- `--target -24`: phrase target, inspired by A/85 dialogue level as a starting point.
- `--min-gain -12`, `--max-gain 6`: conservative gain bounds, in dB.
- `--tolerance 0.5`: leave phrases already this close to target untouched.
- `--floor -60`: do not boost near-silent phrases. Digital silence remains unchanged.
- `--ramp 0.15`: raised-cosine ramps in dB before/after each event, entirely outside words
  and breaths. Each ramp requires the configured duration (rounded up, at least two samples),
  using at most half the available gap. If a full ramp cannot fit at either internal
  boundary, the entire event stays unchanged, with `insufficient_pause` recorded.
  Thus a 150 ms ramp requires at least 300 ms of gap on that side. Farther gaps stay
  untouched; recording start/end need no outside ramp. Choose shorter ramps explicitly
  only after listening; the algorithm never silently substitutes a sharp transition.
- `--peak-ceiling -1`: cap positive gain against the maximum sample peak across the whole
  affected phrase and ramp support. Existing over-ceiling material is not globally repaired.
  This is a sample-peak guard, **not** a true-peak limiter; use `opendubstage qc` on the result.
- `--breath-policy keep`: keep explicitly marked breath samples bit-exact (the default).
  `--breath-policy attenuate --breath-db -6` instead applies fixed attenuation, with the same
  pause-only ramp protection. Unmarked breaths inside phrases receive that phrase's gain.

Other than the target's broadcast inspiration, these defaults are engineering starting
points chosen for restraint, not perceptually validated universal settings. In particular,
phrase boundaries and breath timings need review. When gain/headroom limits bind, the target
is deliberately not reached. The synthetic tests establish signal properties, not perceived
improvement; blind listening review is still needed before use on productions.

## Report and exactness

The JSON report uses schema `opendubstage.leveller/0.1`. It includes parameters, per-event
sample bounds and ramp supports, requested/applied gains, gain/headroom-limit flags,
before/after phrase loudness, and input/output sample peaks. Digital silence is represented
by `null`, never non-standard JSON infinity. All sample ranges are half-open `[start, end)`.
`changed_ranges` describes non-unity gain; `untouched_ranges` its complement. A changed range
can contain zero samples that numerically remain zero.

Measurements are made before WAV encoding. Unchanged decoded PCM/float samples are preserved,
not the entire WAV container or metadata. Changed PCM samples are quantized to the original
subtype. No-op does not mean "normalize the whole file." The API returns a new array and never
modifies the source. The CLI requires distinct input, output, timings and report paths; like
other tools it can overwrite existing output destinations, so choose fresh filenames.

```python
from opendubstage.mix.leveller import level
processed, report = level(audio_float_array, sample_rate, timings_dict)
```
