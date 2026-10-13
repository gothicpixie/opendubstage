# Model survey and decision (October 2026)

This survey asks one question: should we **train** a model, **fine-tune** one, or **wrap** existing open models?
The task is subtle, documentary-appropriate ambience and spot sounds for a still photograph, running locally on Apple silicon.

The facts below come from the GitHub API, Hugging Face API file sizes, model cards and issue trackers, all checked on 2026-10-12. Star counts move, so treat them as a snapshot. Anything marked "unverified" could not be confirmed from a primary source.

## Decision

**Wrap now; fine-tune later only if the evaluation says so; do not train from scratch.**

1. **The hard part is not generation.** Current open models can already make a distant car, water lapping on a dock, or wind in trees. What goes wrong in documentary photo sound is editorial:
   - too many sounds;
   - sounds on the cut;
   - sounds that follow the camera move;
   - sounds louder than the memory they illustrate;
   - sounds on portraits.

   Those are placement, level and restraint problems. A rules engine solves them deterministically and testably, and that engine is this repository's core.
2. **Still-image conditioning is weak everywhere, and that is acceptable.** Only MMAudio has a documented image-only path, and its own README calls it experimental ("not something MMAudio has been trained for"). Every video-to-audio model was trained on VGGSound/AudioSet-style video, which skews toward close, literal, foreground foley. That is the opposite of what a still needs. The robust path is:

   picture -> scene analysis (caption, catalogue record, on-device vision labels, face size) -> a restrained text prompt -> a strong text(+image) generator -> a post chain that adds era, distance and room.
3. **A from-scratch model is not justified.** It would need:
   - a paired dataset of archival stills and documentary-grade sound, which does not exist openly;
   - weeks of GPU time;
   - and it would still need the same placement rules.

   Nothing in the survey suggests open models *cannot* make subtle ambience. They need to be *asked* for it and *placed* with restraint.
4. **When fine-tuning would pay off.** A small LoRA/adapter could teach a text-to-audio model the "distant, slightly muffled, period-appropriate" timbre that prompts alone only approximate. That is a measurable hypothesis; ROADMAP.md describes the protocol. The base models with the cleanest data provenance for that are Stable Audio Open (CC0/CC-BY Freesound and FMA training data, Stability Community Licence) and Woosh (CC-BY-NC weights).

### Prototype choice

- **MMAudio was the first choice** because of its image mode, but its smallest variant needs about **6.7 GB** of weights: 629 MB model, 687 MB VAE, 449 MB vocoder, 950 MB Synchformer and 3,947 MB DFN5B CLIP. That exceeds this prototype's 6 GB download budget.
- **The prototype uses Sony AI's Woosh-DVFlow-8s** (2026) instead:
  - It is distilled video+text-to-audio: 4 sampling steps, 48 kHz.
  - The reference scripts select MPS automatically.
  - It needs about **4.6 GB** in total: the AE, DVFlow and TextConditionerV releases plus the 950 MB Synchformer. The roberta-large text encoder loads lazily and only its config and tokenizer are downloaded.
  - A still is presented as a static 8 s clip.
- **AudioLDM2 was dropped.** It is text only, 16 kHz, and its README warns it needs about 20 GB of RAM on MPS, which is unsafe on a 24 GB machine that is rendering.

## Main table

| Model | Stars / last push | Code licence | Weights licence (commercial?) | Input | Output | Download (approx.) | Apple silicon / MPS |
|---|---|---|---|---|---|---|---|
| **MMAudio** (hkchengrex) | 2,280 / 2026-02 | MIT | CC-BY-NC 4.0 (no) | video+text, text, **image (experimental, Gradio/Python only)** | 16 or 44.1 kHz, trained on 8 s | small_16k **6.7 GB**; large_44k_v2 ~10.7 GB | Runs (M1 Max, M4 Max reports); MPS RNG crackle (generate noise on CPU); Synchformer conv3d needs `PYTORCH_ENABLE_MPS_FALLBACK=1`; use fp32 |
| **HunyuanVideo-Foley** (Tencent) | 1,056 / 2025-09 | Tencent Hunyuan Community | same; commercial < 100M MAU but **excludes EU, UK, South Korea** | video+text | 48 kHz stereo, 1-15 s | XL ~10.5 GB, XXL ~15 GB | CLI accepts `--device mps`; no reports; 8-20 GB VRAM |
| **ThinkSound** (FunAudioLLM) | 1,380 / 2026-04 | Apache-2.0, README says research only | no commercial | video+text (chain-of-thought) | 44.1 kHz, ~9 s | light ~9.2 GB + encoders | One report: M1 Pro, 10 min per 5 s, poor quality |
| **PrismAudio** (ThinkSound successor) | shared repo | Apache-2.0, research only | no commercial | video+CoT text | unverified | ~6.2 GB + encoders | none |
| **FoleyCrafter** (open-mmlab) | 666 / 2026-06 | Apache-2.0 | base Auffusion CC-BY-NC-SA | video (+text) | 16 kHz, ~10 s | ~9 GB + CLIP | No: hard-coded `.to("cuda")` |
| **Diff-Foley** | 207 / 2024-05 | Apache-2.0 | MIT tag, VGGSound-trained | video | 16 kHz, 8 s | ~6.1 GB | no evidence; superseded |
| **V-AURA** | 37 / 2026-02 | MIT | unverified | video | 2.56 s default | ~9 GB | No: CUDA only |
| **Stable Audio Open 1.0** | 3,882 (tools) / 2026-10 | MIT | Stability Community Licence (**commercial under revenue threshold**), gated | text | 44.1 kHz stereo, up to 47 s | ~4.9 GB | Open MPS bugs (#261, #181); conflicting reports |
| **Stable Audio Open Small** | same | MIT | same, gated | text | 44.1 kHz stereo, up to 11 s | ~1.7 GB + T5 | same stack; Arm-CPU optimised per card |
| **AudioLDM2** | 2,642 / 2024-09 | CC-BY-NC-SA | CC-BY-NC-SA (no) | text | 16 kHz | base ~4.5 GB (safetensors) | `--device mps` documented, ~20 GB RAM |
| **Tango 2** | 1,241 / 2025-07 | CC-BY-NC-ND | CC-BY-NC-SA (no) | text | 16 kHz, 10 s | ~5.3 GB | no; torch 1.13 + cu117 pin |
| **Make-An-Audio 2** | 198 / 2024-05 | MIT | MIT tag (data caveats) | text | 16 kHz | ~9.6 GB | no evidence |
| **Woosh** (Sony AI, 2026) | 390 / 2026-09 | MIT + Apache-2.0 (V2A parts) | **CC-BY-NC** | text (Flow/DFlow); video+text (VFlow/DVFlow) | 48 kHz mono, 5 s T2A / 8 s V2A | DFlow ~3.4 GB; DVFlow ~4.6 GB incl. Synchformer | Scripts auto-select MPS; **this repo's prototype ran it on M5 (see README)** |
| **TangoFlux** | 885 / 2026-01 | Stability CL + WavCaps academic | non-commercial | text | 44.1 kHz stereo, 30 s | ~4.1 GB + T5 | none (issue #29 open) |
| **AudioX** (HKUST) | 1,552 / 2026-03 | CC-BY-NC | CC-BY-NC | text, video, audio | 44.1 kHz, 11 s | ~6-7 GB | CUDA recommended |
| **AudioGen** (AudioCraft) | 23,691 / 2026-03 | MIT | CC-BY-NC | text | 16 kHz | ~3.9 GB + T5 | poor (xFormers issues) |
| **im2wav** | 125 / 2023-01 | MIT | unverified (Google Drive) | **image** | unverified | unverified | none; stale |

Excluded:
- **Kling-Foley:** benchmark only, no weights.
- **SpecMaskFoley and TARO:** no official repository with weights.

## Licence summary for users

Almost every capable open weight is **non-commercial**. OpenDubStage is Apache-2.0 and ships no weights. Whoever runs a generative backend inherits that backend's weight licence for the audio they make.

For a commercial documentary, the realistic options are:
- the **library backend**, with licensed or self-recorded effects;
- **Stable Audio Open**, under the Stability Community Licence;
- or a paid API.

The placement rules apply equally to all three.

## Sources

- MMAudio: https://github.com/hkchengrex/MMAudio , https://huggingface.co/hkchengrex/MMAudio , arXiv 2412.15322. MPS: PRs and issues #19, #26, #97, #99.
- HunyuanVideo-Foley: https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley
- ThinkSound / PrismAudio: https://github.com/FunAudioLLM/ThinkSound (issue #57 on M1 Pro)
- FoleyCrafter: https://github.com/open-mmlab/FoleyCrafter
- Diff-Foley: https://github.com/luosiallen/Diff-Foley
- V-AURA: https://github.com/ilpoviertola/V-AURA
- Stable Audio Open: https://huggingface.co/stabilityai/stable-audio-open-1.0 , https://github.com/Stability-AI/stable-audio-tools (issues #181, #261)
- AudioLDM2: https://github.com/haoheliu/AudioLDM2
- Tango / TangoFlux: https://github.com/declare-lab/tango , https://github.com/declare-lab/TangoFlux
- Make-An-Audio 2: https://github.com/bytedance/Make-An-Audio-2
- Woosh: https://github.com/SonyResearch/Woosh , arXiv 2604.01929
- AudioX: https://github.com/ZeyueT/AudioX
- AudioGen: https://github.com/facebookresearch/audiocraft
- im2wav: https://github.com/RoySheffer/im2wav
- Kling-Foley: https://github.com/klingfoley/Kling-Foley
