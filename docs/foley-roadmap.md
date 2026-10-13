# Foley roadmap (`opendubstage.foley`)

Inherited from the standalone stillfoley project; the repo-wide plan is in [../ROADMAP.md](../ROADMAP.md).

## 0.1: foundations (this release)

- [x] Pure-Python placement engine: levels, density, timing, dialogue protection, no-sound classes, and designed silences.
- [x] An independent checker, plus property-based tests that the planner always satisfies it.
- [x] A keyword analyser, and an Apple Vision analyser (labels and face size, with no model download).
- [x] JSON sequence in, JSON cue sheet out, with an audit trail of every rejected candidate.
- [x] An ffmpeg post chain (era EQ, worldizing, fades, loudness) and per-layer stems.
- [x] Backends: a library of files, and the local Woosh-DVFlow model (still+text, 48 kHz).

## 0.2: make the generated sound usable

- **Event windowing.** Generators return 8 s clips, but a spot is 1–3 s. Pick the window around the strongest onset, and use the quietest region for texture loops.
- **Seamless loops for beds and textures**, using a crossfaded loop point instead of a hard `-stream_loop`.
- **Variant generation.** Make 3 variants per class from different seeds, rejecting any whose CLAP similarity to the prompt falls below a threshold.
- **A VLM analyser.** A small local VLM that emits `{bed, sources:[{class, distance, confidence}]}` directly. It must beat the keyword analyser on the evaluation set before it becomes the default.
- **Text-only backends.** Stable Audio Open (Stability Community Licence, the only route with a commercial option) and Woosh-DFlow.
- **An API backend** for hosted text-to-sound-effects services, using the same `Backend` protocol.
- **Exporters:** EDL/AAF markers, a Reaper project and an FCPXML audio lane.

## Evaluation protocol: what "documentary-appropriate" means, measurably

The target is sound that viewers do not notice as sound design, but that makes them prefer the cut. Neither objective metrics nor a single listening test captures that alone, so we use both.

### Test set

- 60 public-domain stills: LoC FSA/OWI, the NYPL Photochrom collection, and NARA. They are stratified into:
  - street,
  - rural,
  - water,
  - interior,
  - crowd,
  - portrait or detail (the no-sound control),
  - weak picture (the no-sound control).
- Each still carries its catalogue caption and a hand-written reference "what is audible here" annotation from two annotators, with agreement measured.
- 6 short sequences (2–3 min each) with recorded public-domain or CC0 narration and marked key lines.

### Blind listening

- **Paired comparison (2AFC).** Each sequence is played *with* the planned sound and *with the bed only*, in random order. The listener answers two questions:
  - "Which do you prefer?"
  - "In which one did you notice added sound effects?"
- **The target is high preference with low detection.** The sound should be preferred more than 60% of the time and detected as an effect at or below 55%, close to chance.
- **MUSHRA-style rating** of individual spots on three 0–100 scales:
  - plausibility for the picture;
  - distance and era fit;
  - intrusiveness (lower is better).

  An anchor (a loud, dry, modern library effect) and a hidden reference (a human sound editor's choice) are included.
- **Panels:**
  - at least 15 general viewers;
  - at least 3 working sound editors or documentary editors.

  Results are reported separately for each panel.
- **The witness test for personal films.** People who were there judge whether "that is how it sounded". This is qualitative and logged per still.

### Objective metrics (automated, per release)

- **Rule compliance:** `check()` must return no violations. This is binding and runs in CI.
- **Level compliance:** each rendered cue's integrated loudness is within ±1.5 LU of target, and its short-term maximum stays at least 6 LU below dialogue short-term.
- **Dialogue masking proxy:** the change in STOI or ESTOI on the narration with and without the effects stems (target: less than 0.01 drop). The speech-to-effects ratio is reported in the 1–4 kHz band.
- **Semantic fit:** CLAP text-audio similarity between the generated clip and (a) the prompt and (b) the human "what is audible" annotation, plus image-audio similarity using an audio-visual embedding (e.g. ImageBind-style) with the still. The latter is reported, not gated, because these embeddings reward literal, foreground sound.
- **Restraint statistics:** spots per minute, the fraction of stills that get any sound, and the fraction of no-sound-class stills given sound (this must be 0).
- **Artefacts:** clipping, DC offset, and presence of speech (a VAD on the effects stem should find none) or music (a music/speech classifier on the effects stem).
- **Distribution distance:** FAD (VGGish and PANNs embeddings) against a reference set of human-edited documentary effects stems, if one can be licensed.

### Ablations to run

1. Still+text conditioning against text only, for the same prompt and seed. The prototype already produces both (`*_img.wav` / `*_txt.wav`).
2. Keyword analyser, against the Vision analyser, against a VLM analyser, scored against the human "what is audible" annotations (precision matters more than recall).
3. With and without the post chain (era EQ and worldizing), in a blind A/B.

## Fine-tuning plan (only if the evaluation says the base models fall short)

**Hypothesis:** base models produce the right *source* but the wrong *perspective*. They sound too close, too clean and too modern, and prompting plus the post chain do not close the gap.

**Approach:** a LoRA or adapter on a text-to-audio model, conditioned with perspective tokens: `distance ∈ {close, medium, far}`, `era ∈ {modern, 1990s camcorder, 1970s 16mm, 1930s acetate}`, `room`. Images condition through the analyser, not the generator, so the model never has to learn still-image grounding.

**Base model candidates:**
- Stable Audio Open 1.0 or Small: Stability Community Licence, clean CC training data, and the best route if commercial use matters.
- Woosh-Flow: CC-BY-NC.

**Data,** with each source's licence and how we would use it:

| Source | Licence | Use |
|---|---|---|
| Freesound, CC0 subset | CC0 | Primary: field recordings and ambience. Filter by tags (`field-recording`, `ambience`, `distant`) and duration. |
| Freesound, CC-BY subset | CC-BY 3.0/4.0 | Secondary; keep attribution manifests. |
| BBC Sound Effects archive | RemArc licence: personal, educational and research only | Evaluation and listening references only; not for training a released model. |
| AudioSet (ontology plus YouTube clips) | Labels CC-BY 4.0; the audio is YouTube content with no redistribution rights | Label taxonomy and sampling strategy only. Avoid training on scraped audio for a released model. |
| VGGSound | CC-BY 4.0 annotations; the audio and video are YouTube content | As for AudioSet: research only. It also skews toward close, literal foley, the bias we are trying to remove. |
| Clotho, AudioCaps | Clotho: Freesound-derived (mixed CC); AudioCaps: AudioSet-derived | Captions for prompt style; check Clotho's per-clip licences. |
| Self-recorded "perspective pairs" | Ours (CC0 release) | The key new dataset: the same source recorded close, medium and far, plus degraded copies through period-mic simulations. Small (hours), but exactly on target. |

**Synthetic perspective augmentation:** clean CC0 sources degraded by distance (air absorption, early reflections, a lower direct-to-reverberant ratio) and by era (band-limit, wow and flutter, hiss, compression). The adapter learns the mapping, and the evaluation checks it on real archival recordings.

**Budget:** a LoRA on SAO-Small-class models fits on a single consumer GPU. Do not try it on a busy laptop.

**Stopping rule:** if the adapter does not beat prompt-plus-post-chain on the blind "noticed as effect" rate at equal preference, it does not ship.

## Later

- Spatial placement for wide stills: pan a spot toward its source in the frame, using Vision's object bounding boxes, kept subtle.
- Archival footage without sound: per-shot analysis on keyframes, with the same rules. Sync is deliberately loose, because these are textures and not lip-sync foley.
- An interactive review UI: a timeline with stills and cues, where you accept or reject each cue and the decision log shows the reason.
