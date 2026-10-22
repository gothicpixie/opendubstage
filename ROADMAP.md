# Roadmap

OpenDubStage grows one small, tested, documented tool at a time. A tool lands when it has a written method,
defaults with their reasoning, synthetic tests that pin its behaviour, and a CLI. "Better" is decided by blind listening
plus objective checks, never by a metric alone.

## 0.1: foundations (this release)

- [x] `restore.specrepair`: room-tone Replace, spectral Attenuate and LF thump removal, bit-exact outside the patched regions.
- [x] `restore.memofix`: gap-learned LF rumble subtraction plus a clamped LTASS matching EQ.
- [x] `match.ltasmatch`: an LTAS dialogue EQ match that writes `ffmpeg` filter strings, an early-reflections room, and a pitch probe.
- [x] `mix.carve`: speech-keyed 1–4 kHz carve, music posts and phrase-safe outs.
- [x] `mix.transitions`: J/L-cuts, tails and crossfades on a documented cue-sheet schema.
- [x] `qc.mixqc`: public-radio mix QC with blocking and advisory checks.
- [x] `foley`: placement rules, an independent checker, property tests, analysers, a post chain and backends (see
  [docs/foley-roadmap.md](docs/foley-roadmap.md) for its own plan).
- [x] A single `opendubstage <tool>` CLI, CI on Linux and macOS for Python 3.10–3.13, and no model downloads in tests.

## 0.2: planned tools

### Dialogue leveller (experimental word-timing implementation)

Implemented as `opendubstage leveller`; see [method, defaults and limitations](docs/leveller.md).
Synthetic invariants cover convergence when limits allow, word-safe gains and exact untouched samples.
VAD input, real-world listening validation and a release remain pending.
The current phrase meter is ungated K-weighted mean square, not standardized three-second short-term loudness.

Original goal:
A sentence-level leveller in the spirit of a dialogue-leveller plug-in, as opposed to a compressor:
- Measure short-term loudness per phrase (from word timings or a VAD) and ride each phrase toward a target with slow,
  phrase-aligned gain moves. Gain changes happen in pauses, never mid-word.
- Separate handling for breaths: either de-breath (attenuate by a fixed amount) or keep, as a policy.
- Tests: synthetic phrases at known levels converge to the target within ±1 LU, gain never moves inside a word, and
  the output is bit-exact where no ride is needed.

### Room-tone synthesis (experimental mono implementation)

Implemented as `opendubstage roomtone`, with opt-in `specrepair --synthesize-tone` fallback;
see [method, defaults and limitations](docs/roomtone.md). Synthetic tests cover spectral/level
match, long-lag nonperiodicity and protected samples. Real-world listening validation remains pending.

Original goal:
Generate arbitrary lengths of a recording's own room tone, for fills longer than any clean stretch available
(an "ambience match"):
- Learn the noise floor's spectral envelope and its slow modulation from gaps between words; synthesise by
  overlap-add of randomised-phase STFT frames with matching envelope statistics, so the tone has no loop seam.
- Feeds `specrepair` when no tone stretch of `tone_min_len` exists within `context`.
- Tests: spectral match within ±1.5 dB per 1/3 octave, no periodicity (autocorrelation peak below a bound), level match.

### De-reverb
A light, classical (non-neural) de-reverb for interview audio recorded in live rooms:
- Late-reverb spectral subtraction with a decay model fitted from word offsets (re-using `ltasmatch.decay_ratio_db`),
  limited in depth so the room is reduced, not removed.
- An optional neural backend behind an extra, never in the default install or in CI.
- Tests: on synthetic speech convolved with an exponential-decay impulse response, the decay ratio improves by at least
  a set amount while direct-path energy stays within ±1 dB.

### True-peak limiter helper
A delivery helper, not a mastering limiter:
- Measure true peak (4× oversampled, BS.1770) and compute the minimal linear gain or the look-ahead limiter settings that
  meet a target after a lossy codec, with headroom per codec.
- Emits an `ffmpeg` chain (`alimiter` / `loudnorm` in measured two-pass mode) and re-checks the result with `qc`.
- Tests: synthetic inter-sample-peak signals land at or under the ceiling and integrated loudness stays within ±0.5 LU.

### MLX VoxCPM LoRA training port
An Apple-silicon (MLX) port of VoxCPM LoRA fine-tuning that produces LoRA weights the official PyTorch inference loads
unchanged, with numerical parity tests against the pinned upstream trainer. The full build specification, acceptance
tests and licence rules are in [docs/voxcpm-mlx-lora.md](docs/voxcpm-mlx-lora.md). It will live in its own repository
(it has heavy dependencies and model downloads) and be linked from here.

## Later

- An exporter for each cue-sheet tool to EDL/AAF markers, a Reaper project and an FCPXML audio lane.
- `qc` checks proposed by practice but not yet implemented: a "texture floor" (cleaned dialogue's noise floor must not
  fall more than a set amount below the raw floor), a proximity-band check for headphone deliverables, and an
  anti-fatigue window (something changes in the soundtrack at least every few minutes).
- Vocal-aware masking in `qc` (a sung track under speech, not only energy in the presence band).
- Stable API and a 1.0 release once the interfaces have been used on a few productions.
