# Restoration tools: what each does, what it costs the voice

Surveyed 2026-10-12. "AS" = runs natively on Apple silicon; "RT" = real-time factor.

## RX-style processes and open equivalents

| Process | Open / local equivalent | Quality vs the commercial tool | Notes |
|---|---|---|---|
| Spectral de-noise | DeepFilterNet3 (`deep-filter`, 27 MB binary, 48 kHz WAV only, `-a` attenuation cap) | ≈ equal on hiss and road noise; can gate laughter and distant speakers at full strength | MIT/Apache; > 10× RT on CPU. Cap at 12 dB on the main voice, 18 dB on a separated vocal stem; blend 70/30 |
| | `afftdn=nr=8:nf=-45:tn=1` (ffmpeg) | below on stationary noise; smears consonants above nr 10 | built in |
| | `arnndn=m=<rnnoise model>:mix=0.7` | fine for hiss | check per-model licence |
| De-rumble | `asubcut=cutoff=70,highpass=f=90:p=2` | same thing | |
| De-reverb | `nara_wpe` (late reverb only, mild, no artefacts) | below RX on strong rooms, equal on mild | MIT; apply before denoise |
| | VoiceRestore (flow-matching, 301 M) / Sidon (CoreML, 246 M) / Resemble Enhance | generative; can beat RX on bad rooms, can resynthesise timbre | blend ≤ 50 %; narration only; blind test required |
| De-click / mouth de-click | `adeclick=w=55:o=75:a=2:t=2` (AR interpolation) + your own detector | good for < 10 ms clicks | RX's mouth mode is a detector tuned to lip smacks |
| Spectral Repair (Attenuate / Replace) | `opendubstage specrepair` | reproducible for Attenuate and Replace; Partials+Noise not implemented | numpy / scipy / soundfile only |
| EQ Match / Dialogue Match | `opendubstage ltasmatch` + `early_reflections` + a learned room-tone bed | equal (it is LTAS arithmetic) | see `voice-matching` |
| Dialogue Isolate / Music Rebalance | `audio-separator` BS-RoFormer (SDR 12.9; MPS + CoreML) | ≈ RX | MIT; 639 MB checkpoint; separate whole spans, not 10 s chunks |
| Leveler | sentence-level gain ride (your own) | ≈ RX Leveler | `speechnorm` / `dynaudnorm` are not substitutes |
| De-breath | detector (RMS −48…−26 dBFS, centroid 1.2–4 kHz, flatness > 0.25, runs ≥ 120 ms in gaps ≥ 250 ms) + −8 dB | | never remove |
| Ambience Match | room-tone synthesis from the recording's own gaps | | see SKILL.md §3 |
| Bandwidth extension | AudioSR (6 GB), ClearerVoice MossFormer2_SR_48K | generative; breathy / metallic | rarely worth it on a mix |
| Metrics | `speechmos` (DNSMOS), `torchmetrics` DNSMOS, `pystoi` (needs a clean reference), Whisper WER, speaker embeddings | | DNSMOS alone is not a decision |

## Tool table

| Tool | Local | AS | Licence | Batch | Verdict |
|---|---|---|---|---|---|
| DeepFilterNet3 | yes (binary or pip) | yes, prebuilt | MIT / Apache-2 | CLI, directory | first choice for hiss and broadband noise on close voices; test every distant source separately |
| Demucs v4 (`htdemucs_ft`) | yes | PyTorch, CPU ≈ 1.5× RT | MIT (archived) | CLI | second opinion for separation |
| audio-separator (BS/Mel-RoFormer) | yes | MPS / CoreML | MIT | CLI | first choice for speech-over-band separation |
| ClearerVoice-Studio (MossFormer2_SE_48K) | yes | CPU; MPS unstated | Apache-2 | Python | ≈ DFN3, sometimes smoother |
| Resemble Enhance | yes / hosted | CPU fallback | MIT | CLI (`--denoise_only` is the safe mode) | enhancer stage is generative |
| VoiceFixer | yes | CPU | MIT | CLI | dated vocoder; not recommended |
| nara_wpe | yes | numpy | MIT | Python | safe mild de-reverb |
| Adobe Podcast Enhance v2 | hosted | n/a | proprietary | web only | strongest one-click de-reverb; removes laughter and overlapping replies at full strength; use 30–50 % |
| ElevenLabs Voice Isolator | hosted API | n/a | proprietary | API | no de-reverb; adds a little reverb around the voice |
| iZotope RX 12, dxRevive Pro, Hush Pro, Clarity Vx | GUI / plug-in | native | proprietary | no scripting | dxRevive Pro and Hush Pro won the 2025 Production Expert blind test |
| NVIDIA Maxine | no macOS | | | | excluded |

## Symptoms of over-processing (and the fix)

- Underwater / watery / musical noise → lower reduction, blend dry.
- Gated or pumping background → lighter setting, add constant room tone.
- Thin, bass-less voice → many isolators strip LF; check the 100–250 Hz proximity band.
- Removed emotion (laughter, gasps, overlapping replies) → the model treated it as noise; measure WER on the secondary speaker.
- Hallucinated phonemes / changed timbre → generative stage; ASR can get worse on fabricated detail.
- Processed and unprocessed shots jump at cuts → process a continuous span with one setting.

## A/B protocol (short clips, before any batch run)

Six 20–30 s clips covering: a clean passage (control; cleanup should be inaudible), a quiet
low-level passage, the roomiest passage with laughter, an outdoor gust, a distant line, a
speech over a band. For each render: original, ffmpeg-only chain, model at 100 %, model at
70/30, and the escalation candidate. Blind-listen with randomised names on headphones and on
a TV; delta-check; objective: Whisper WER vs the verified transcript, speech-to-floor,
loudness range. Present pairs to any model judge in both orders (see `ai-review-of-audio`).
