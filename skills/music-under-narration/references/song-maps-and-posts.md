# Song maps, posts and phrase-safe exits

## Song maps (once per song, cached)

Every music cue in the edit list should be expressed as `(song, section_or_bar_in, bar_out)`; the planner converts to seconds. Cache per song a `song_map.json`:

```
{"bpm": 92.0, "beats": [...], "downbeats": [...], "bars": [[t0, t1], ...],
 "phrases": [...4- and 8-bar groups from the first downbeat...],
 "sections": [{"start": 0.0, "end": 14.2, "label": "intro", "vocals": false}, ...]}
```

Tools:
- **allin1** (MIT): `allin1.analyze(path)` → `bpm`, `beats`, `downbeats`, `segments` with labels from {start, end, intro, outro, break, bridge, inst, solo, verse, chorus}. Best single tool: both grids at once. https://github.com/mir-aidj/all-in-one
- **beat_this** (ISMIR 2024, MIT): `File2Beats(checkpoint_path="final0", device="cpu", dbn=False)(path)` → `(beats, downbeats)`; state of the art for bars; weaker on continuity. Verify allin1 where they disagree by > 60 ms. https://github.com/CPJKU/beat_this
- **msaf**: boundaries only, fallback. librosa `beat_track`: tempo only, no downbeats.
- Vocals per section: a separation pass (BS-RoFormer vocals stem) plus an energy threshold per section, or a transcript of the song. Mark `vocals: true/false` per section; the "nothing sung under speech" rule reads this field.

The verify step rejects a cue whose in- or out-point is more than 60 ms off a downbeat (or the song's start/end).

## `phrase_safe_end(song_map, t_end, search=6.0, bars_per_phrase=4)`

Returns the nearest section end, 4-bar phrase end or downbeat to the wanted out-point within ±6 s, plus a fade length equal to one bar clipped to 1.5–4 s. Prefer section ends, then phrase ends, then downbeats. Apply it to every `under` cue end, not only to budget exits.

## `posts(env, activity, depth=-14, post_db=-6, fr=100, min_gap=2.5, full_gap=8.0, min_post=1.5, ...)`

Input: a frame-rate (100 fps) linear-gain envelope and a speech-activity mask. Output: the same envelope with posts.

1. Merge speech lines closer than `min_gap` (2.5 s): no post between them.
2. For each gap ≥ 2.5 s: hang 0.15 s after the last word, rise 1.0 s to `post_db`, hold, dip 0.5 s ending 0.25 s before the next word. A post window shorter than `min_post` (1.5 s) is not a post; stay down.
3. For gaps ≥ `full_gap` (8 s): rise fully (2.0 s), dip 0.8 s before the next line.
4. Gaps > 12 s with music and no speech are reported as interludes: a plan problem.

Keep-spans (laughs, sighs, sentence-final breaths) are unioned into the activity mask before `posts` runs, so music does not rise into a laugh.

## Cue-length histogram (what good looks like)

- Cues 0:40–2:30, median ~1:15 (Menjivar).
- Posts: median 4–7 s; tails of 2–3 and 10–12 s acceptable.
- Music-free stretches between songs: 2–6 min in a feature (so each entry is an event).
- Per 10-min window: music 30–70 % of the time.

## Crossfades between songs

- Same tempo family: `acrossfade=d=<one bar>:c1=qsin:c2=qsin`, bar-aligned.
- Different keys: 2–4 bars with the outgoing cue first dropped 6 dB, or (better) end the outgoing song on a downbeat, 0.5–1 s of ambience, start the incoming at a section start (see `scene-transitions`).
