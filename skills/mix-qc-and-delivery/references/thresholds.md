# `opendubstage qc` thresholds and cue-sheet schema

## Cue-sheet format (times in seconds on the programme timeline)

```json
{"duration": 3600.0,
 "cues": [{"id": "n1", "role": "dialogue", "class": "narration",
           "start": 12.0, "end": 40.5, "fade_in": 0.1, "fade_out": 0.3,
           "file": "cue.wav", "file_offset": 0.0, "gain_env": "duck",
           "speech_active": [[12.4, 18.0]], "natural_end": false}],
 "envelopes": {"duck": {"rate": 100, "values": [], "db": false}},
 "stems": {"dialogue": "dx.wav", "background": "bg.wav"},
 "segments": [{"start": 0, "end": 600, "type": "narrated"}],
 "scenes": [0.0, 600.0]}
```

`role` ∈ dialogue | music | ambience | effects. `class` is the dialogue voice class (narration,
host, actuality, bridge, live…). `speech_active` is derived from `file` when absent. `stems`
are full-length files aligned with the mix (used for masking and tonal checks). `scenes` are
boundaries where level jumps are allowed.

## `Thresholds` defaults (with sources)

| Field | Default | Source |
|---|---|---|
| `lra_programme` | (4.0, 18.0) LU | [NFLX] near-field range; [3343] LRA is descriptive |
| `lra_segment_max` | {"*": 15.0}; set narrated ≤ 10 (strict ≤ 7), live ≤ 15 | [NFLX] dialogue ≤ 7; [AUPH] talk ≈ 5 |
| `jump_lu` | 5.0 | [JND] ~1 LU; [A85] ±2 LU tolerance → > 2× is audible |
| `jump_window_s` / `jump_speech_s` / `jump_max_gap_s` / `jump_scene_s` / `jump_floor_lufs` | 0.4 / 1.5 / 1.5 / 2.0 / −45 | [3341] momentary; practice |
| `mask_band` | (1000, 4000) Hz | [SHIR] consonant cues |
| `mask_ratio_db` | −10.0 (strict −14 in 1.2–4.8 kHz) | [TORC] |
| `mask_min_s` / `mask_smooth_s` | 3.0 / 1.0 | sustained stretches only |
| `click_ratio` / `click_win_ms` / `click_ref_ms` / `click_min_dbfs` | 8.0 / 5 / 50 / −60 | practice; only edges with fades < 50 ms can click |
| `hard_stop_fade_s` / `hard_stop_end_s` / `hard_stop_tail_db` | 1.5 / 0.5 / −30 | [BYERS]; practice |
| `dead_air_dbfs` / `dead_air_s` / `dead_air_edge_s` | −60 / 3.0 / 5.0 (strict −50 / 2.0) | `silencedetect` convention; practice |
| `tonal_band` / `tonal_dev_db` | (1000, 4000) / 4.0 | a 4 dB presence difference reads as a different voice |
| `lufs_range` | (−26, −22) for a −24 programme; set wider (−27…−16) when anchoring on dialogue and reporting the programme | [PRSS] [A85] |
| `true_peak_max` | −2.0 dBTP | [PRSS] [NFLX]; R128/TD1004 allow −1 but lossy codecs overshoot |
| `expected_lufs` / `expected_tol` | None / 0.5 | cross-check against the mixer's own report |
| `blocking` | ("loudness", "dead_air", "click", "masking") | tune per programme |

Source keys: [R128] EBU R 128 · [3341]/[3342]/[3343] EBU Tech · [1770] ITU-R BS.1770-4 ·
[A85] ATSC A/85 · [NFLX] Netflix v1.5 · [TD1004] AES TD1004.1.15-10 · [TORC] Torcoli et al.
AES 146 (2019) · [SHIR] Shirley & Kendrick (BBC R&D / Salford) · [JND] loudness JND · [PRSS]
PRSS spec · [AUPH] Auphonic 2019 · [BYERS] NPR Training checklist 2018.

## Checks (in the order they run)

1. `loudness`: integrated (BS.1770-4 gating) and true peak (4× oversampling); optional cross-check with another meter.
2. `lra`: programme and per segment type (Tech 3342: short-term, −70 absolute / −20 relative gates, 95th − 10th percentile).
3. `dead_air`: sample peak below the floor for longer than allowed; head and tail exempt.
4. `jump`: momentary max within 1.5 s each side of a voice-to-voice edit; scenes exempt.
5. `masking`: background (music + ambience) vs dialogue energy in the band while speech is active, median-smoothed over 1 s, speech pauses ≤ 1 s bridged; needs stems or per-cue renders.
6. `tonal`: presence-band share per dialogue class vs the mean of the classes.
7. `click`: max |x[n] − x[n−1]| within ±5 ms of a cue edge vs the RMS of the difference over ±50 ms.
8. `hard_stop`: non-dialogue cues whose rendered tail is still within 30 dB of the cue's median level at a fade shorter than 1.5 s.

Each problem is `"[check] H:MM:SS message"`; `metrics["events"]` holds the same findings as
structured records. The mix is read in blocks, so programme length is not limited by RAM
(a 190-minute mix ran in ~2 minutes in practice).

## Mastering chain that satisfies the gate

```
measure dialogue-gated loudness on DX + LIVE-speech  (ebur128=peak=true:dualmono=true)
→ one linear gain to −24 LUFS
→ alimiter=limit=0.74:attack=5:release=50:level=false:latency=true   (−2.6 dBFS sample ≈ ≤ −2.3 dBTP)
→ 24-bit WAV; encode; re-measure true peak on the encoded file (≤ −2 dBTP)
```

No `loudnorm`, no `dynaudnorm` on the bus. LRA is reported, not clamped.

## Cross-checking a model reviewer against the gate

Match each of the model's timestamped notes to the nearest problem of the same kind within
±15 s; print a caught / missed table per complaint kind (dead_air, masking, level, edge,
tone). The misses tell you which human or model listens must stay in the gate.
