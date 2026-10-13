# LTAS matching: method, algebra, pitfalls

## Pipeline (as implemented in `opendubstage.match.ltasmatch`)

1. `decode(path, sr=48000, ss, dur, af)` → mono float32 through an optional ffmpeg filter chain (so you can measure a voice *after* its existing chain).
2. Speech mask: `energy_vad(x, sr, nfft=4096, hop=1024, top_db=20, floor_db=12)` or `interval_mask(word_intervals, n_frames, sr)`.
3. `ltas_power(x, sr, mask, centers=THIRD_OCTAVE)` → per-band power accumulator; `to_db(acc, ref=(200, 6300))` normalises so the 200 Hz–6.3 kHz mean is 0 dB (loudness-independent).
4. `smooth(db, passes=1)` → 1/6-octave-ish smoothing across bands.
5. `fit_eq(before, target, eq_freqs=OCTAVE_EQ | HALF_OCTAVE_EQ, q=1.4 | 2.1, gmax=6.0, rng=(200, 6300), protect_below=None)` → per-band gains by least squares against the smoothed difference, each clamped to ±gmax.
6. `eq_string(gains, eq_freqs, q, min_abs=0.3)` → `equalizer=f=…:t=q:w=…:g=…` chain; bands below 0.3 dB are dropped.
7. `deviation(after, target, rng=(200, 6300))` → max absolute band deviation; pass ≤ 3 dB.

## Why smoothing and gain limits

- A fine-resolution match copies the *reference's formants* onto the source: two people saying different vowels then share the same resonances and the result sounds phasey or like a vocoder. 1/3-octave measurement plus octave or half-octave correction bands keep the match to spectral *tilt and room*, which is what the ear hears as "same place".
- Large boosts lift the noise floor by the same amount; a +10 dB shelf at 8 kHz on an 11 kHz-bandwidth recording is 10 dB more hiss and no more voice. Hence ±6 dB (±5 above 8 kHz) and `--protect-below` for a voice whose fundamentals you must not touch (a 90 Hz second-order high-pass keeps 100–120 Hz fundamentals).

## Amount

`G(f) = amount × (P_ref − P_src)`. RX documents 10–40 % as "often enough" for lav vs boom. For a studio TTS against a phone recording made in a vehicle or small room the sources differ far more, so 0.4–0.45 is needed; for camera speech that should remain recognisably "elsewhere", 0.25.

Zero-mean the curve over 300 Hz–3 kHz before applying so the loudness normaliser that follows does not fight the EQ.

## Room

A dry voice gets a room by early reflections, not by a long tail. `early_reflections(delays_ms=(3.1, 5.3, 7.9, 11.3, 17.2, 26.0, 39.0), first=0.30, rt60_ms=110, in_gain=0.9)` returns an `aecho` string whose taps decay to the RT60. Measure the reference's `decay_ratio_db(x, sr, intervals, early=(0, 0.03), late=(0.03, 0.12))` on word endings and tune RT60 until the bridge's ratio is within 3 dB. A car: RT60 < 0.3 s. A hall: ~1.1 s (then use a real IR via `afir`, not `aecho`).

Also give it the reference's **floor**: a learned room-tone loop at −46 LUFS under the bridge. Without the floor, the bridge sits in a different room however good the EQ.

## Splitting speakers by pitch

`median_f0(x, sr, fmin=70, fmax=400, frame=2048, hop=512, voicing=0.45)` gives a robust median fundamental per span. When two speakers differ by more than ~40 Hz in median f0, label spans by pitch and confirm the diarisation; where they overlap in pitch, fall back to the level rule (spans ≤ 12 words ≥ 6 dB below the cue median).

## Pitfalls seen in practice

- Measuring the LTAS on frames that include music or the other speaker: use word timings when you have them.
- Fitting against a reference measured *before* its own chain while applying the EQ *after* the source's chain: measure both at the point in the graph where the match will be applied.
- Side files that mutate chains at import (a fitted-EQ JSON read by the mixer): record them in the mix report or inline the curve in the cue sheet, or your cache keys will silently change when you re-measure.
- A match that passes the deviation test and still sounds wrong usually fails on **room** or **level**, not spectrum: check the decay ratio and the ±2 LU short-term rule before touching the EQ again.
