# Foley for stills (`opendubstage.foley`)

This module began as the standalone *stillfoley* project and is now part of OpenDubStage.

**Documentary-grade sound for still photographs and archival footage.**

A documentary built from photographs (the Ken Burns way) needs a little sound under its pictures:
- a bed of room or air;
- now and then a distant car, water at a dock or a train far off;
- and, much more often, nothing at all.

Generative audio models can now make those sounds. Left to themselves, though, they make too many, too loud and too literal, they land on the cut and they follow the camera move. The result is a cartoon.

This module is the editorial layer in between. It analyses each picture, proposes sounds, and then applies documentary placement rules. The output is a cue sheet you can audit, and optionally rendered stems. The generator is pluggable: a library of real recordings, a local open model, or an API.

```
photo (+ caption, + sequence timing, + narration timing)
   │
   ▼
scene analysis ── caption / catalogue record, Apple Vision labels, face size
   │               → bed, candidate sounds, no-sound classes
   ▼
placement rules ── levels in LU relative to dialogue, density per minute,
   │                never on the cut, onset and key-line protection,
   │                music-up / live / designed-silence blocks
   ▼
cue sheet (JSON) ── every accepted cue and every rejected candidate, with the reason
   │
   ├─► backend: library files │ local model (Woosh-DVFlow) │ API
   ▼
post chain ── era EQ, worldizing room, fades, loudness → per-layer stems
```

**Status: alpha.** The rules engine, analysers, cue sheets and post chain are implemented and tested; the local-model backend is a prototype.

## Quick start

```bash
# plan sound for five public-domain photographs (examples/foley/public_domain/SOURCES.md)
opendubstage foley plan examples/foley/public_domain/sequence.json -o cues.json
opendubstage foley plan examples/foley/public_domain/sequence.json -o cues.json --vision   # + Apple Vision (macOS)
opendubstage foley check examples/foley/public_domain/sequence.json cues.json            # re-verify any hand edits

# fill sources from a folder of your own recordings named <class>_<variant>.wav, then render stems
opendubstage foley generate examples/foley/public_domain/sequence.json cues.json --backend library --library ~/fx
opendubstage foley render cues.json -o stems/
```

The foley core has **no dependencies** beyond the Python standard library, plus `ffmpeg` for rendering.

## Input: a sequence

```json
{
  "dialogue_lufs": -24,
  "stills": [{"id": "street", "path": "street.jpg", "start": 8, "dur": 8, "chapter": "town",
              "caption": "Saturday afternoon on the main street, cars parked", "time_of_day": "day",
              "hero": false, "kind": "photo", "has_sync_audio": false, "tags": [], "quality": {}}],
  "narration": [{"start": 17.0, "end": 23.0, "key": true}],
  "music": [{"start": 40, "end": 70, "level": "up"}],
  "live": [], "silences": []
}
```

## Output: a cue sheet

```json
{"schema": "opendubstage.foley.cuesheet/0.1", "dialogue_lufs": -24.0, "duration": 46.0,
 "cues": [{"id": "s0003", "layer": "spot", "sound_class": "car_far", "at": 9.081, "dur": 3.0,
           "target_lufs": -32.0, "fade_in": 0.03, "fade_out": 0.3, "still_id": "street",
           "prompt": "a car passing slowly on a quiet road, distant, quiet, recorded on location, ...",
           "variant": 2, "room": "street", "src": null}],
 "decisions": [{"still_id": "mud", "sound_class": "car_far", "accepted": false,
                "reason": "same sound class too recently"}]}
```

## The rules

The rules, with their reasoning and sources, are in [docs/rules.md](foley-rules.md). In short:

| | Default |
|---|---|
| Bed / texture / spot / hero spot / transition | −16 / −12 / −8 / −4 / −2 LU relative to dialogue |
| Spots per still | ≤ 1, and only for a visible source |
| Spots per minute | ≤ 4 under narration, ≤ 6 in a narration-free montage, 0 under music-up or live sound |
| Placement | 0.8 s ± 0.3 s after the cut; never within 0.4 s of it; ends before the next cut |
| Dialogue | nothing from 0.5 s before to 0.3 s after an onset; bed only from 4 s before to 3 s after a key line |
| No-sound classes | portraits and close-ups, documents and details, technically weak pictures, silent objects |
| Repeats | same sample never within 10 min; same class never within 30 s; ≥ 5 s between spots |
| Prompts | one action, material, place and distance; lint rejects whooshes, risers, music, voices and "epic" |

`check()` re-verifies every rule on any cue sheet, and property-based tests assert that `plan()` always passes it.

## Generative backend (local)

The survey in [docs/survey.md](foley-survey.md) explains the choice: **wrap open models; do not train from scratch.**

The prototype backend is Sony AI's **Woosh-DVFlow-8s**: video+text to 48 kHz audio in 4 steps, with MPS auto-selected. A still is presented as a static 8 s clip, so the picture's content conditions the model but no motion does. Total weights are about 4.6 GB.

```bash
scripts/install_woosh.sh        # clones Woosh, installs it, fetches 3 weight archives; refuses if disk would drop below 40 GB free
.venv/bin/python scripts/prototype_woosh.py examples/foley/public_domain/sequence.json examples/foley/public_domain/cuesheet.json samples/woosh
```

On a shared machine, wrap heavy steps with `OPENDUBSTAGE_LOCK=/path/to/.heavy.lock scripts/with_heavy_lock.sh ...`.

**Licences matter here.** OpenDubStage is Apache-2.0 and ships no weights. Woosh's weights, like those of nearly every capable open audio model, are **CC-BY-NC 4.0**, so the audio you make with them is for non-commercial use. For commercial work, use the library backend with licensed or self-recorded effects, Stable Audio Open (Stability Community Licence; backend on the roadmap), or a paid API. The rules apply equally to all of them.

## Module layout

```
opendubstage/foley/
  model.py            data types (Still, Sequence, Cue, Layer, ...)
  levels.py           level table, LU relative to dialogue
  taxonomy.py         sound classes, bed selectors, no-sound classes, rooms
  prompts.py          prompt template + anti-cartoon lint
  analysis.py         keyword analyser (caption/tags)
  analysis_vision.py  Apple Vision analyser (labels + face size), macOS
  rules.py            the planner and the independent checker
  io.py               sequence / cue sheet JSON (also reads legacy stillfoley.cuesheet/0.1)
  post.py             ffmpeg post chain (era EQ, worldizing, fades, loudness)
  render.py           per-layer stems
  backends/           library, woosh (local model); API backends plug in here
  cli.py              analyze | plan | check | generate | render
```

The foley roadmap, evaluation protocol and fine-tuning plan are in [foley-roadmap.md](foley-roadmap.md).
