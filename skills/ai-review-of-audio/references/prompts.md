# Prompts and templates for model review of audio

Replace the bracketed parts with the piece's own facts. Keep the structure.

## Review brief: picture + sound

```
You are a senior documentary editor and re-recording mixer reviewing part of a [feature-length / short]
[kind of film]. It should feel [timeless, unhurried and cinematic: a feature documentary, not a slideshow].
[The narration is X; there is live sound from Y; the score is Z.] Watch AND listen closely to the whole clip.
Report concrete, timestamped problems (timestamps are seconds from the start of THIS clip):
- pacing: anything rushed (stills changing too fast, no pause after a line, scenes not allowed to play), or dragging;
- music: songs starting mid-phrase, cut or ducked before they establish, ending abruptly or not on a phrase end,
  fighting the narration, pumping level;
- dialogue/narration: hard to understand, harsh, muddy, noisy, hissy, roomy, level jumps between voices;
- ambience: dead silence under pictures, abrupt sound starts/stops, missing room tone;
- picture: awkward dissolves, soft/dark/blown frames, bad framing, repeated-looking shots, sync errors;
- moments that work well and should be protected.
Be specific and practical. Do not transcribe lyrics.
```

## Review brief: radio cut (sound only)

```
You are a senior documentary editor and re-recording mixer listening to the RADIO CUT (sound only, no picture yet)
of a [kind of piece]. It must feel [raw, emotional and timeless]. [The narration is X (a second voice sometimes
chimes in); there is live sound from Y; the score is Z.] The picture will be cut to this audio afterwards.
Listen closely to the whole clip. Report concrete, timestamped problems (seconds from the start of THIS clip):
- narration edits: a cut that starts or ends mid-sentence or mid-thought, a clipped first/last word, a fade that eats
  a syllable, a line that lands without the context needed to understand it, a jump that sounds like a splice;
- emotion: moments that are moving and must be protected; places where an emotional beat is rushed, stepped on by
  music, or cut short; places that feel flat, logistical or rambling;
- music: wrong mood for what is being said, starting mid-phrase, ducked before it establishes, ending abruptly, pumping;
- pacing: no breath after a big line, gaps that feel dead, sections that drag;
- dialogue/mix: hard to understand, harsh, noisy, level jumps between voices or between narration and live sound.
Be specific and practical. Do not transcribe lyrics.
```

## Structured-output schema

```json
{"type": "OBJECT", "properties": {
  "summary": {"type": "STRING"},
  "pacing_score": {"type": "INTEGER", "description": "1 = rushed, 5 = ideal, 9 = drags"},
  "mix_score": {"type": "INTEGER", "description": "1-10, 10 = professional feature mix"},
  "notes": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
    "t": {"type": "NUMBER"}, "t_end": {"type": "NUMBER"},
    "category": {"type": "STRING", "enum": ["pacing", "music", "dialogue", "ambience", "picture", "praise"]},
    "severity": {"type": "INTEGER", "description": "1 minor - 3 must fix"},
    "note": {"type": "STRING"}, "fix": {"type": "STRING"}},
    "required": ["t", "category", "severity", "note"]}}},
 "required": ["summary", "pacing_score", "mix_score", "notes"]}
```

## Blind A/B (two variants, run in BOTH orders)

```
You will hear two versions of the same [20–30 s] clip of [spoken word / live recording], labelled A and B,
in that order. They differ only in audio processing. For each version give integer scores 1–10 for
intelligibility, naturalness (does it still sound like a real person in a real place; penalise gating,
"underwater" or robotic artefacts, lost breaths or laughter, thinned bass) and noise (10 = no distracting
noise), plus one sentence of evidence. Then state which version you would use in a documentary mix and why,
in one sentence. If they are indistinguishable, say so.
```

Then repeat with the clips swapped (B first). Result table per clip:

| variant | intell. (AB) | natural. (AB) | noise (AB) | intell. (BA) | natural. (BA) | noise (BA) | pick AB | pick BA | order-consistent? |
|---|---|---|---|---|---|---|---|---|---|

A variant wins only if `pick AB == pick BA`. Report wins / ties / flips.

## Multi-variant set (4–5 chains, randomised order, ≥ 2 orderings)

```
You will hear N versions of the same clip, labelled [letters]. Rate each (intelligibility, naturalness, noise, 1–10)
with a one-line note, then rank them for use in a documentary mix. Prefer the version that keeps the voice and
the room natural over the one that is merely quietest.
```

Keep only variants whose rank is stable within ±1 across orderings. Include the original as one of the variants;
if it ranks first, the correct answer is "no processing".

## Hidden vocals in a generated underscore

```
This is an instrumental music bed. Does it contain any human voice, singing, humming, chanting, choir, vocoded
voice, or spoken words, however faint or buried? Answer exactly `none`, or list each occurrence with a timestamp
(seconds from the start of this clip), a description, and a confidence 1–5. Do not transcribe words.
```

Run on 30–60 s windows covering the whole file, twice with the windows in different orders. Confidence ≥ 3 in
both runs → separate vocals and inspect, or drop the track.

## Cross-check against `opendubstage qc`

For each model note of kind ∈ {dead_air, masking, level, edge, tone}, find the nearest `qc` problem of the matching
check within ±15 s. Print:

| complaint kind | caught | missed |
|---|---|---|

and list the misses with the note text, so the human and model listens that must stay in the gate are explicit.

## Control runs

- Two identical clips under different labels: the model must call them indistinguishable or split evenly across orders.
- A deliberately damaged variant (e.g. 12 dB of denoise with a hard gate): the model must rank it last in every order.
If either control fails, discard the run and lower the chunk length or raise the proxy bitrate before retrying.
