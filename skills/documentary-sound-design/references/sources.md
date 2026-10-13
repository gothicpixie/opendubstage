# Sources shared by the OpenDubStage skills

Consulted 2026-10-12. Each skill cites these by short name.

## Standards and controlled studies

- **Torcoli, Freke-Morin, Paulus, Simon, Shirley**, "Preferred levels for background ducking to produce esthetically pleasing audio for TV with clear speech", AES 146th Convention (2019); JAES 2019. ≥ 10 LU commentary over music, ≥ 15 LU over ambience; non-experts prefer ~4 LU more. A companion voice-over-voice study found experts at 11.5–17 LU and recommends ≥ 20 LU. https://cris.fau.de/publications/221136558 · https://aes2.org/publications/elibrary-page/?id=20308
- **Shirley & Kendrick** (BBC R&D / Salford), "clear speech" work: background music and effects are the main cause of intelligibility complaints; the 1–4 kHz band matters most. BBC rule of thumb: when the mix is done, try the music 4 dB lower. https://salford-repository.worktribe.com/OutputFile/1488623
- **ANSI S3.5** Speech Intelligibility Index band-importance weights (peak ≈ 1.5–3.5 kHz).
- **ITU-R BS.1770-4**: K-weighted gated loudness; true peak at ≥ 4× oversampling.
- **EBU R 128** (−23 LUFS ± 1, −1 dBTP), **Tech 3341** (momentary 400 ms, short-term 3 s), **Tech 3342** (LRA), **Tech 3343** (LRA is descriptive). https://tech.ebu.ch/publications/tech3343
- **ATSC A/85** (2013): dialogue-anchored −24 LKFS ± 2.
- **PRSS** (public radio distribution): −24 LUFS ± 2 LU, −2 dBTP (sample peak −3). Current (2014); Radio World, "PRSS tackles audio program consistency". https://current.org/?p=60099
- **Netflix Sound Mix Specifications & Best Practices v1.5**: dialogue −27 LKFS ± 2 dialog-gated, −2 dBTP, LRA guidance programme 4–18 LU, dialogue ≤ 7 LU. https://backlothelp.netflix.com/hc/en-us/articles/360001794307
- **AES TD1004.1.15-10** streaming: −16…−20 LUFS, ≤ −1 dBTP; lossy codecs overshoot.
- **Apple Podcasts**: −16 LKFS ± 1, −1 dBTP. **Auphonic** (2019): talk ≈ 5 LU LRA, features/drama 5–25 LU.
- **WCAG 1.4.7** (AAA): background ≥ 20 dB below speech.

## Public radio practice

- **Jonathan Menjivar** (This American Life), Transom 2015, "Using Music": entrances and exits signal shifts; enter mid-action; "don't slowly fade up music, choose a spot and start it"; posts 4–7 s (2–3 or 10–12); cues 1:00–1:30, sometimes 0:40–2:30; "go as long without music as you did with it". https://transom.org/2015/using-music-jonathan-menjivar-for-this-american-life/
- **Rob Rosenthal**, Transom 2019, "Scoring Stories pt 1": build the story without music; music marks scene breaks. https://transom.org/2019/scoring-stories-part-1/
- **Jonathan Mitchell**, Transom 2014: take music away and the next line feels more important. https://transom.org/2014/using-music-jonathan-mitchell/
- **Rob Byers**, NPR Training mixing checklist (2018): every voice roughly the same volume; "turn the listening level down halfway, can you understand every word?"; check all edits; mix in sentences and phrases; balance ambience and music. https://npr.brightspotcdn.com/dd/e4/4c73d7c74227a94967e00362b73a/npr-training-audio-mixing-checklist.pdf
- **NPR Training 2025**, score best practices and mixing music stories: "the spoken word should drive the music"; use an equalizer to keep music from clashing with narration. https://www.npr.org/sections/npr-training/2025/05/31/g-s1-67187/
- **Podcast Engineering School**, mixing phone audio: match the better source down to the worse; cut 250–450 Hz, add 2–3 kHz; thin source slightly louder. https://podcastengineeringschool.com/properly-mixing-phone-call-audio
- **Joe Richman / Radio Diaries** (NPR Training 2025; Nieman Storyboard 2012): intimacy; subject-narrators locate themselves; a finale on "hard exhalations".
- **Erica Heilman, *Finn and the Bell*** (Peabody 2022; Third Coast 2023): stammers kept in; one real bell as bookend; "listen to the very last second".
- **Sean Cole, *Time Bandit*** (Third Coast 2021): ending music commissioned "for editorial reasons as much as aesthetic ones".
- **Walter Murch**, "Dense Clarity, Clear Density", Transom Review 5(1), 2005: two and a half layers; the "shell game" of layers; silence as metaphor. https://transom.org/2005/walter-murch/

## Dialogue editing

- **Post Audio Tips**: DX fill and the golden rule of dialogue editing (fill every hole; room tone never stacks under speech; fills ≤ 20 ms equal-power). https://postaudiotips.beehiiv.com/p/dx-fill
- **Adobe Audition community**: 5 ms automatic crossfades at every edit boundary remove clicks from non-zero crossings; 60 ms (two video frames) is too long for dialogue editing.
- **IRPR Sound**: breaths are reduced, never deleted; "if the music drops 15–20 dB on every line the audience hears the automation"; room tone glossary. https://sounddesign.irpr.agency/
- **iZotope RX** docs: Spectral Repair (Attenuate / Replace / Pattern / Partials+Noise); EQ Match ("10–40 % is often enough"); Dialogue Match (EQ + reverb + ambience).
- **Mokrý & Rajmic 2024**: autoregressive (Janssen) inpainting holds to ~45–80 ms gaps. https://arxiv.org/pdf/2403.04433
- Breath-locator features: arXiv 2404.15143.

## Restoration tools and their artefacts

- DeepFilterNet3 (MIT/Apache; PESQ 3.17 VB-DMD) https://github.com/Rikorose/DeepFilterNet
- Resemble Enhance (MIT; generative enhancer) https://github.com/resemble-ai/resemble-enhance
- VoiceFixer (MIT; 2021) https://github.com/haoheliu/voicefixer
- nara_wpe (MIT; late-reverb suppression) https://github.com/fgnt/nara_wpe
- VoiceRestore (MIT; flow-matching) https://github.com/skirdey/voicerestore
- ClearerVoice-Studio (Apache-2; MossFormer2 48 kHz) https://github.com/modelscope/ClearerVoice-Studio
- audio-separator / BS-RoFormer (MIT; SDR 12.9) https://github.com/nomadkaraoke/python-audio-separator
- Production Expert 2025 dialogue-restoration blind listening test (dxRevive Pro, Hush Pro winners). https://www.production-expert.com/production-expert-1/results-dialogue-restoration-listening-test-noise-reduction
- "To dereverb or not to dereverb" (perceptual study): artefacts can outweigh the gain.
- Smule 2025 DNSMOS comparison: https://arxiv.org/pdf/2510.21659

## Feature-documentary interviews

- **Fire of Love** (Patrice LeBlanc, A Sound Effect): archive not cleaned; "respect the diverse qualities of the different audio sources"; foley as the first link. https://www.asoundeffect.com/fire-of-love-documentary-sound/
- **They Shall Not Grow Old** (Burge, Kwok, de la Rey, Heywood; Mix): only visible sources get a sound; per-line EQ; distance is the dial; libraries sound too modern. https://www.mixonline.com/?p=105012
- **Apollo 11** (Eric Milano; Electronic Sound): breathing room before the peak; leave the sacred thing alone; perspective by EQ; witnesses test the mix.
- **Amy** (Kapadia; Televisual/BFI): "the sound is what elevates it"; songs as text.
- **Senna** (Pandey; ayrton-senna.net): remove the cue before the moment so the return lands.
- **Free Solo** (Schnupp; CAS Quarterly 2019): close, dry, human foley makes a wide shot intimate.
- **20 Days in Mariupol** (Jordan Dykstra, Film Score Monthly): a chord for three seconds, space for six.
- **Ken Burns, *The War*** (Ewers, Spiegel, Tavella; Mix): "when you see a German 88 you hear a German 88"; density is the failure mode; "nothing makes sound more effective than the absence of sound"; checked on six to eight systems. https://www.mixonline.com/?p=78678
- **Peter Albrechtsen** (A Sound Effect; FilmDaft): foley recorded "badly" on purpose; total silence once or twice per film; music underscores an emotion already established.
- **Cameraperson / Navalny** (Pete Horner): the maker's breath and handling are texture; perspective shifts reserved.
- **Becoming Led Zeppelin** (Nick Bergh; Sound on Sound, Mix): "no broadband denoise anywhere"; "the best restoration is no restoration"; one EQ/limiter across recordings erases what makes each one itself. https://www.soundonsound.com/techniques/restoring-led-zeppelin
- **32 Sounds** (Mark Mangini; Post Magazine, A Sound Effect): hero mix on headphones; "no loudspeaker can reproduce proximity".
- **The Beatles: Get Back** (Kwok, Burge, de la Rey; Awards Daily): separate speech from room but keep the audience "in the room"; watch audio fatigue over ten-minute stretches.
- **Deaf President Now!** (Cinema Eye 2026): silence as point of view; actors who voiced interviewees are credited.
- Award lists: CAS documentary mixing (2017–2026), MPSE Golden Reel feature documentary, Primetime Emmy nonfiction sound, Cinema Eye Honors sound design, BAFTA TV Craft Sound: Factual, Third Coast / Driehaus, Peabody, Ambies, Prix Europa / Prix Italia.

## Beat and structure tools

- **beat_this** (Foscarin, Schlüter, Widmer, ISMIR 2024; MIT) https://github.com/CPJKU/beat_this
- **allin1** (Kim; MIT): bpm, beats, downbeats, sections. https://github.com/mir-aidj/all-in-one
- **msaf** (fallback boundaries). librosa `beat_track` for tempo only.

## Multimodal review

- Position bias in pairwise LLM judging is well documented (e.g. Zheng et al. 2023, "Judging LLM-as-a-judge", and later audio-judge work); measured in practice here: the same two mixes swapped order and the model's preference followed the slot. Count only order-consistent wins.
