# The placement rules, and why

Every number below is a default in `opendubstage.foley.rules.Rules` or `opendubstage.foley.levels`. The planner (`plan`) enforces them and an independent checker (`check`) re-verifies them on any cue sheet. Property-based tests in `tests/foley/test_properties.py` assert that the planner's output always passes the checker.

These are rules of thumb drawn from how acclaimed documentaries treat sound under photographs and silent archive. They are not laws of nature: change them in `Rules(...)`, and the checker follows.

## Layers and levels (relative to dialogue)

| Layer | What | Default level vs dialogue | At −24 LUFS dialogue |
|---|---|---|---|
| Bed | continuous scene ambience, crossfaded between chapters | −16 LU | −40 |
| Texture | semi-continuous: distant voices, water, birds, a train far off; 10–40 s | −12 LU | −36 |
| Spot | one-shot matched to a visible source | −8 LU (hero still: −4) | −32 (−28) |
| Transition | a hand-requested designed sound that *is* the cut | −2 LU | −26 |

**Distance is the dial.** The person telling the story is present; the photograph is a memory. A sound that comes across nearer than the narrator is wrong.

The anchors are:
- the published −10 to −15 LU range for music under speech (Torcoli et al.);
- dialogue-anchored delivery specifications.

Walter Murch's observation that listeners follow about two and a half simultaneous layers ("Dense Clarity, Clear Density") is why spots get only "half a layer" once dialogue and a bed are present.

## Density

- **One spot per still at most.** Only visible sources get a sound.
- **At most 4 spots per minute under narration, and at most 6 per minute in a narration-free montage.**
- **At least 5 s between spots, and 30 s before the same class of sound repeats.**
- **The same sample never repeats within 10 minutes.** Each class has 3 variants, rotated.
- **Only one texture at a time.** Textures change with the scene, not with the still.
- **At most one designed transition per chapter**, and only when it is requested by hand.

## Timing

- **Never on the cut.** A spot lands 0.8 s ± 0.3 s after the picture appears, and never within 0.4 s of the cut. Real sounds do not coincide with edits; a sound exactly on the cut reads as a stinger. If that slot is blocked, the planner tries the still's midpoint once.
- **A spot belongs to its picture.** It ends 0.1 s before the next cut.
- **Spots never follow the Ken Burns move.** The taxonomy has no whooshes or risers, and the prompt lint rejects them.

## Protecting dialogue

- **Narration onsets:** nothing sounds in the 0.5 s before a narration onset or in the first 0.3 s of it, because a spot there masks the first word.
- **Key lines:** from 4 s before a key line to 3 s after it, the sound is the bed alone: no spots, no textures. Textures end before the window, so their 3 s fade completes in time. The 3 s afterwards is the breath.
- **Music up:** when music is the foreground there are no spots and no textures. The bed stays.
- **Live or sync sound:** archive footage that already carries its real sound, or any stretch marked `live`, gets no added effects.
- **Designed silences:** inside a designed silence nothing plays, not even the bed. Use one or two per film at most.

## When not to add sound (no-sound classes)

- **Portraits, close-ups and posed pictures:** the face is the subject. With Apple Vision, a face covering at least 3% of the frame counts as a portrait.
- **Details and documents:** letters, maps, certificates, clippings.
- **Technically weak pictures:** blurry, dark or overexposed. The picture is already asking for forgiveness, so sound should not call attention to it.
- **Objects that make no sound:** a lantern does not tick and a cake does not rustle. The taxonomy simply has no class for them.

## Avoiding the cartoon feel (prompts and post chain)

1. **One action, one material, one place, one distance** per prompt. The tail is always "no music, no voices".
2. **The lint** (`prompts.lint`) rejects:
   - whooshes, risers and stingers;
   - comedic sounds;
   - music;
   - intelligible speech;
   - "epic" or "cinematic" scale.
3. **Match the era and the mic** (`post.ERA_EQ`): an 80 Hz high-pass, a 9 kHz low-pass and a −3 dB dip at 3.5 kHz.
4. **Worldize** with a short room matched to the bed: trees, open land, street, small room or hall.
5. **Prefer textures to hits.** Laughter is a distant group, not a sitcom laugh. Footsteps are a few steps, not a march.
6. **Witness test:** play a viewer a handful of stills with and without sound. If they can point at the effect, lower it 3 dB or remove it.

## Sources (public)

- Walter Murch, "Dense Clarity, Clear Density", *Transom Review* 5(1), 2005.
- Ken Burns and the Florentine Films sound team on sound for still photographs (*The Civil War*, *The War*).
- Peter Albrechtsen (sound designer, *The Cave*) on recording foley "badly" on purpose, and on one or two total silences per film.
- *They Shall Not Grow Old* (2018) sound team on period-matching and distance.
- *Apollo 11* (2019) on leaving archival sound alone and moving ambience treble with the picture on a few hero shots.
- M. Torcoli et al. on preferred music-to-speech loudness differences in broadcast.
