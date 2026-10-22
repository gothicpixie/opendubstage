# Room-tone synthesis (experimental)

`opendubstage roomtone` generates a mono statistical approximation of the recording's
own ambience. It does not loop a captured waveform, download a model, detect words,
or promise that matching spectra sounds indistinguishable. Listening validation is pending.

```sh
opendubstage roomtone input.wav tone.wav --duration 30 --protected words.json \
  --excluded events.json --seed 0 --report tone.json
```

Timing files use seconds: `[[start, end], ...]` or `{"words": [{"start": 0.5, "end": 1.2}]}`.
Protected words/kept sounds and excluded events are never learning material. Their complement
must genuinely be clean room tone. Without timings the entire source is explicitly assumed
clean. Mark breaths, music and noises as protected/excluded too; this is not a voice detector.
Invalid/out-of-bounds intervals are rejected, overlaps are merged. Output is floating-point WAV
with the source sample rate. Output and report paths must not overwrite either audio or timing inputs.
A supplied but unrecognized/malformed timing schema fails closed, never silently
removes protection.

## Method and defaults

- Pad protected/excluded intervals by **20 ms** as a conservative word-edge guard.
- Analyze only complete **128 ms** Hann frames within clean gaps, with **75% overlap**.
  No analysis frame spans a word, event or separate gap. This resolves useful low bands
  without requiring multi-second clean stretches. Gaps shorter than one frame are discarded.
- Require **0.5 seconds total** usable clean material and at least **four frames**. More,
  representative clean material is better. Insufficient evidence raises an error, never
  substitutes unrelated noise. A genuinely silent reference produces silence.
- Normalize each analysis spectrum by its frame power, average it, and independently retain
  smoothed frame-level statistics. Synthesis randomizes every frame's phase, Hann-overlap-adds
  frames, and compensates the sum of squared windows. Padding/cropping removes boundary
  startup fades. DC and Nyquist bins have random real signs.
- Approximate slow amplitude variation by drawing from smoothed log-RMS statistics at
  **0.5 second** knots with cosine interpolation. Deviations are clamped to **±6 dB**.
  This retains approximate modulation depth, not the original trajectory or modulation spectrum.
- Match overall RMS to the mean source analysis-frame power. This is an RMS measure,
  not LUFS. Output duration is rounded to the nearest sample. Peak above full scale is an
  error rather than silent clipping/attenuation. There is no limiter.
- A seed (default **0**) makes output reproducible. Different seeds produce different waveforms;
  duration is not constrained by the clean material's length. Processing is in-memory and
  long requested output uses proportionate memory/time.

These are engineering defaults, not a perceptual standard. Mono only: stereo width,
interchannel correlation, narrow tonal hum, and strongly nonstationary room events are not
claimed to be reproduced faithfully. Random phase is particularly inappropriate when a
specific tonal phase trajectory must be preserved.

The JSON report records method, seed, sample count/rate, used clean intervals, analysis-frame
count, source/output RMS, sample peak, learned modulation deviation, and limitations. It does
not claim a quality score or hide insufficient-evidence failures.

## Opt-in spectral-repair fallback

```sh
opendubstage specrepair input.wav repaired.wav --events events.json --protected words.json \
  --synthesize-tone --tone-seed 0 --report repairs.json
```

Existing contiguous-tone selection remains first choice and all defaults remain unchanged.
Only if no candidate meets `tone_min_len` does the opt-in fallback gather shorter quiet islands
inside `context`. It retains the repairer's protected/excluded/event-gap exclusions and quiet
energy limits, then requires the minimum evidence above. It reports `room_tone_synthesis` and
embeds the synthesis report (clean intervals use absolute source time, with the analyzed
window offset also recorded). If unavailable it reports a skipped repair and reason. The same
patch splicing and integer-PCM writer preserve every sample outside reported patches exactly.
The synthesizer learns the short islands' level directly, so the legacy post-synthesis EQ match
is not applied again. Users must still review the repaired transition by ear.

## Verification

Synthetic tests verify, for stationary colored noise, third-octave spectral power within
**±1.5 dB** over resolved bands **100–6300 Hz at 16 kHz**, RMS within **0.2 dB**, and maximum
absolute normalized autocorrelation below **0.05** at **0.1–5 s** lags. Short-lag correlation is
expected from the spectral colour; the bound is not universal for arbitrary tonal recordings.
Tests also cover modulation, timing exclusion, repeatable/fresh seeds, exact length, silence,
malformed input, opt-in fallback, insufficient/context-limited evidence, CLI, and bit-exact
integer-PCM samples outside patches. Numerical checks do not substitute for blind listening.
