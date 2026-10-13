# Prompt template and keyword → cue classes

## Spot (one-shot) prompt template

```
{one action} {one material/surface} {one place}, {distance}, quiet, recorded on location, slightly muffled, no music, no voices, one-shot
```

Generate 3 variants each, 1.0–3.0 s, prompt influence ≈ 0.6, WAV 48 kHz. Examples:

- "Single wine glass clink against another glass on a wooden table, medium distance, quiet outdoor dinner, slightly muffled, no music, no voices, one-shot"
- "Three footsteps on dry pine needles, close, forest, no music, no voices, one-shot"
- "Old DSLR camera shutter with mirror slap, medium distance, outdoors, no music, no voices, one-shot"
- "Wooden folding chair creak and settle, close, quiet porch at night, no music, no voices, one-shot"
- "A car door closing on a dirt road, distant, no music, no voices, one-shot"
- "Soft laughter from a small group of adults, distant, outdoors under trees, muffled, no music, no clear words, one-shot" (the one permitted "voice" sound; unintelligible and far)
- "Wooden ladder being set against a tree and a strip of tape being pulled, medium distance, outdoors, no music, no voices, one-shot"
- "Shoes shuffling on a wooden dance floor with distant murmur, medium distance, muffled, no music, no clear voices, one-shot"

## Texture and bed prompts

`loop=true`, 22–30 s, prompt influence ≈ 0.3 for natural variation:

- "Light wind through tall pine trees, afternoon, a few distant songbirds, no insects, no music, no voices, ambience loop"
- "Dusk in a pine forest, crickets beginning, very light breeze, no music, no voices, ambience loop"
- "Distant party murmur through trees at night, occasional faint clink, no music, no clear words, ambience loop"
- "High mountain ridge wind, thin air, no birds, no insects, no music, no voices, ambience loop"
- "Quiet cabin interior at morning, faint refrigerator hum, wood settling, no music, no voices, ambience loop"
- "Inside a parked car with the engine off, faint road noise outside, no music, no voices, ambience loop" (a narrator's room, for a floor under bridge lines or one deliberate reveal)

## Post-process every file the same way

```
highpass=f=80, lowpass=f=9000, equalizer=f=3500:t=q:w=1.2:g=-3, afade in/out 30 ms
→ loudness-normalise to the layer table (bed −16 LU, texture −12, spot −8 / hero −4, transition −2 relative to dialogue)
→ the scene's room (trees / street / small room / hall)
```

Name files `fx/{class}_{variant}.flac` and keep `fx/manifest.json` with prompt, parameters, backend and date.

## Keyword → class map (first version; extend from your catalogue)

| Keywords in the description (lower-case substrings) | Class | Cue |
|---|---|---|
| laugh, laughing | spot (texture if a crowd) | `laugh_group_far` |
| glass, toast, champagne, drink, wine | spot | `glass_clink` |
| walk, walking, path, aisle, procession | spot | `steps_pine` (outdoor) / `steps_wood` (porch, hall) |
| car, truck, road, driving, parking | spot | `car_door_far` or `car_pass_far` |
| camera, photographer, photographing | spot | `dslr_shutter` |
| danc, dance floor | texture | `dance_shuffle_murmur` |
| ladder, tent, string lights, setting up, tape | spot | `setup_ladder_tape` |
| chair, porch, swing | spot | `chair_creak` |
| fire, bonfire, campfire | texture | `fire_crackle_far` (check peaks; crackle files often fail the level table) |
| lantern, candle, cake, ring, dress, bouquet, padlock | none | bed only |
| portrait, close-up, smile, smiling, looking at camera | none | bed only |
| blur, overexposed, dark, underexposed | none | bed only |
| pine, tree, forest, woods + time of day | bed selector | `pines_day` / `pines_dusk` / `pines_night` |
| mountain, ridge, snow, cloud, overlook | bed selector | `ridge_wind` |
| lake, river, creek, water | texture | `water_lap` |
| cabin, bedroom, kitchen, interior | bed selector | `cabin_interior` |
| reception, tables, dinner, band, dance (night) | bed selector | `reception_far` / `reception_near` |

Time-of-day bands for bed selection: before 17:30 day, 17:30–19:30 dusk, after night (adjust per season and latitude).

## Sources

Walter Murch, "Dense Clarity, Clear Density" (Transom Review 2005); Ken Burns / Florentine Films sound team on *The War* (Mix); Peter Albrechtsen (A Sound Effect; FilmDaft); *They Shall Not Grow Old* team (Mix); *Apollo 11* (Electronic Sound); *Fire of Love* (A Sound Effect); Torcoli et al. (AES 2019).
