# Spec: open-source LoRA fine-tuning for VoxCPM on Apple silicon (MLX)

> Status: planned (see [ROADMAP](../ROADMAP.md)). This is the build specification for a future OpenDubStage
> companion tool. It is written as a brief for whoever (human or agent) implements it; nothing here is built yet.

## Goal

Build an open-source tool that fine-tunes **VoxCPM 1.5** with **LoRA** on Apple silicon using **MLX** (Apple's Metal-native
array framework). The tool must:

- produce LoRA weights that the official PyTorch VoxCPM inference code loads without changes;
- match the official PyTorch training numerically (see the parity tests below);
- run materially faster than PyTorch on the MPS backend on the same Mac.

The motivation: today the only way to fine-tune VoxCPM is the official PyTorch script (CUDA-first; MPS works but is slow
and needs `PYTORCH_ENABLE_MPS_FALLBACK=1`). Community MLX ports of VoxCPM exist for *inference* only. No MLX training path
exists.

## Ground truth and starting points (verify each; don't trust this summary over the code)

- **Upstream model and training code:** https://github.com/OpenBMB/VoxCPM. Pin commit
  `cce58a5b59303c9bd63d12f23afe6a49a4a80c59` for all parity work.
  - The reference trainer is `scripts/train_voxcpm_finetune.py`.
  - Model code is under `src/voxcpm/`.
  - Read the fine-tuning docs: https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html.
- **Base weights:** `openbmb/VoxCPM1.5` on Hugging Face. Do not redistribute weights; download at runtime.
- **Inference port to start from:** the VoxCPM backend in **mlx-audio** (https://github.com/Blaizzy/mlx-audio, documented at
  https://voxcpm.readthedocs.io/en/latest/deployment/mlx_audio.html). It supports VoxCPM 1.0/1.5 inference.
  - Reuse its module ports where they are correct.
  - Where they differ from upstream, upstream wins.
- **Reference training config.** This is a real config that has been used with the upstream trainer; keep its semantics:
  ```yaml
  sample_rate: 44100
  batch_size: 4
  grad_accum_steps: 4
  num_iters: 600
  learning_rate: 0.0001
  weight_decay: 0.01
  warmup_steps: 50
  max_batch_tokens: 4096
  lambdas: {loss/diff: 1.0, loss/stop: 1.0}
  lora: {enable_lm: true, enable_dit: true, enable_proj: false, r: 32, alpha: 32, dropout: 0.0}
  ```
- **Manifest format** (keep it identical so existing datasets work): JSONL, one `{"audio": "<path>", "text": "<transcript>"}`
  per line, with separate train and val manifests.

## Scope

**In scope:**
1. An MLX implementation of every module the upstream trainer runs in the forward and backward pass for VoxCPM 1.5
   training: the text and audio embedding paths, the base LM, the residual LM, the local diffusion transformer and its
   flow-matching or diffusion objective, the stop head, and the audio VAE encoder used to make training latents. Confirm
   the exact module list from the upstream code; name each module and its upstream file in your design doc.
2. LoRA adapters on exactly the layers that upstream's `enable_lm` / `enable_dit` / `enable_proj` select, with the same
   rank, alpha scaling, dropout and initialisation.
3. The training loop:
   - the same losses and weights (`loss/diff`, `loss/stop`) and the same noise and timestep sampling;
   - AdamW with the same warmup/decay schedule;
   - gradient accumulation, and `max_batch_tokens` bucketing or packing as upstream does it;
   - bf16 training with fp32 master weights where upstream does that (check);
   - gradient clipping if upstream clips;
   - a validation loop at `valid_interval`, checkpoints at `save_interval`, and resume from a checkpoint;
   - deterministic seeding.
4. **LoRA export** in upstream's format (key names, dtype, file layout). The official PyTorch loader must load it with no
   code changes. Optionally add an exporter for the mlx-audio inference path too.
5. A CLI that accepts the upstream YAML config. Any extra MLX-only keys are optional and documented.
6. A precompute step that encodes audio to VAE latents once and caches them, if that speeds training without changing
   the results. Prove it doesn't change the results (test 3c).

**Phase 2 (required): VoxCPM 2.** VoxCPM 2 is the newest release: about 2B parameters, 48 kHz output, and upstream ships
LoRA training for it in `conf/voxcpm_v2/voxcpm_finetune_lora.yaml`, with a 16 kHz encoder input, 48 kHz output,
gradient clipping, and about 20 GB of VRAM per the upstream docs.

Once every Phase 1 (VoxCPM 1.5) acceptance item passes:
- Extend the port to VoxCPM 2 with the same test and validation suite (parity, equivalence run, cross-loading,
  performance). Keep one shared code base with per-model configs.
- No Python MLX port of VoxCPM 2 exists to start from. A Swift MLX inference port exists (`speech-swift`, an int8 model at
  https://huggingface.co/aufklarer/voxcpm2-mlx-int8). Use it only as a reference, check its licence, and keep the upstream
  PyTorch code as the oracle.
- Report the memory each Mac configuration needs for VoxCPM 2 LoRA training (for example 24 GB versus 32 GB or more), and
  which settings fit.

**Out of scope:** full fine-tuning, multi-GPU, new model features, and redistributing weights.

## Hard rules

- **Public data only.** Use public datasets for every test, benchmark, example and CI fixture, for example LJSpeech
  (single speaker, public domain) or a LibriTTS-R speaker subset (check its licence). Never commit any private or personal
  voice data. Generated test fixtures must be small (under 5 MB total in the repo).
- **Licences.** Read upstream VoxCPM's licence and mlx-audio's licence before copying any code. Keep the required
  attribution and notices, and choose a compatible licence for the new repo (state why in the README).
- **Upstream is the oracle.** Every numerical claim is checked against the pinned upstream PyTorch code running on the
  same inputs. If you find an upstream bug, don't silently "fix" it in MLX. Match upstream, document the bug, and offer a
  flag.
- **Validation log.** Record the commands you ran and their results in `docs/VALIDATION.md` as you go.

## Deliverables

1. **A new public GitHub repo**, for example `voxcpm-mlx-train`, or a PR series to mlx-audio if its maintainers' scope
   fits. Decide, state why, and ask before opening an upstream PR. It contains the package, CLI, tests, and these docs:
   - `README.md`: install, quickstart, the config, the numbers from the validation report, and known limits.
   - `docs/DESIGN.md`: a module map from upstream to MLX, and the dtype policy.
   - `docs/VALIDATION.md`: the full results tables below.
2. **A reproducible example**: fine-tune on an LJSpeech subset, then synthesise held-out sentences with (a) the official
   PyTorch inference loading the MLX-trained LoRA and (b) the MLX inference path.
3. **A benchmark script**, plus results for at least one Apple-silicon machine. Record the chip, memory, macOS, MLX version
   and PyTorch version.

## Testing and validation (all must pass; record every number)

### 1. Unit tests (pytest, fast, run in CI)
- Shapes and dtypes for every module on random inputs.
- LoRA math: merged and unmerged outputs are equal, and a zero-initialised B gives an output identical to the base model.
- The config parser accepts the upstream YAML; unknown keys raise clear errors.
- The manifest loader handles missing files, empty text and unicode.
- Checkpoint save, load and resume yield bit-identical optimizer state.
- The LR schedule matches upstream's formula at steps 0, warmup, mid and end.

### 2. Module parity against upstream PyTorch (local, on the pinned commit, with the same weights)
For each module, feed identical inputs, using fixed seeds and identical injected noise and timesteps.
- **fp32:** max abs error ≤ 1e-4 and relative error ≤ 1e-4 on the outputs.
- **bf16:** within a documented tolerance, justified by a PyTorch-bf16-vs-PyTorch-fp32 baseline. MLX may not be worse than
  PyTorch is against itself.
- **The full model** on one real training batch:
  - each loss term within 1e-4 relative (fp32);
  - **LoRA gradient** cosine similarity ≥ 0.999 per parameter tensor, and norm ratio within 1 ± 0.01;
  - **one AdamW step** leaves LoRA weights within 1e-5 relative (fp32).
- **20-step micro-run** with identical data order and noise: the MLX loss curve stays within 2% of PyTorch at every step.

### 3. Training validation
- **a. Overfit test:** 8 clips, about 300 steps. Loss drops to near its floor and the model reproduces the training clips
  intelligibly.
- **b. Equivalence run:** the same config on the same public single-speaker subset (≥1 h of audio, held-out val set) in
  both PyTorch (CUDA if available, else MPS) and MLX.
  - **Loss:** final train and val losses within 3%, with similar curve shapes. Plot both in VALIDATION.md.
  - **Objective metrics** on 30 held-out sentences, comparing the two LoRAs:
    - **speaker similarity:** cosine of a public speaker-verification embedding (for example ECAPA-TDNN or WavLM-SV)
      against held-out real clips of the target speaker;
    - **WER** with Whisper large-v3 (or turbo);
    - **a naturalness predictor:** UTMOS or DNSMOS.

    Acceptance: MLX is not worse than PyTorch by more than 0.02 speaker similarity, 1.0 point of absolute WER, or 0.1 on
    the naturalness score. Report the base model without the LoRA too, to show that fine-tuning helps.
- **c. Latent-cache test** (if a cache is implemented): training from cached latents equals training from on-the-fly
  encoding (same losses at step 1 and step 20).
- **d. Cross-loading:**
  - an MLX-trained LoRA loaded by the official PyTorch inference makes the same audio as the MLX inference, within a
    documented tolerance (or matching metrics if sampling is stochastic: fix the seeds);
  - a PyTorch-trained LoRA loads into the MLX inference.
- **e. Resume:** kill training at step N and resume. The loss curve continues exactly, matching an uninterrupted run with
  the same seed.

### 4. Performance
- Measure steps/s, clips/s and peak unified memory for the reference config on the same Mac:
  - MLX bf16;
  - MLX fp32;
  - PyTorch MPS (with `PYTORCH_ENABLE_MPS_FALLBACK=1`, and record which ops fall back).
- **Target:** MLX at least 2× faster than PyTorch MPS on the same machine. If you miss it, profile, report the bottleneck
  ops, and document what would close the gap. Don't game the benchmark: identical configs, warm-up excluded, and the
  median of ≥3 runs.
- Report the wall-clock time for the full 600-step reference config.

### 5. CI
- GitHub Actions on Apple-silicon macOS runners runs the unit tests and a tiny smoke train (a few steps on the fixtures).
- Parity tests, the equivalence run and benchmarks need large downloads or long runs, so they're marked and run locally.
- Their recorded results (with commit SHA, date and machine) live in VALIDATION.md, and the README links to them.

## Acceptance checklist (report each as PASS or FAIL with numbers)
- [ ] All unit tests are green in CI.
- [ ] Module parity: fp32 within tolerance for every module, and bf16 within the justified tolerance.
- [ ] Full-model loss, gradient and one-step parity.
- [ ] The 20-step loss curve is within 2%.
- [ ] The overfit test passes.
- [ ] The equivalence run is within the loss and metric acceptance limits.
- [ ] Cross-loading works both ways.
- [ ] Resume is exact.
- [ ] The performance table is filled in, with ≥2× over MPS or a documented gap.
- [ ] Licences and attribution are checked, and no private data is in the repo or its history.
- [ ] The README quickstart works from a clean clone on a fresh venv (paste the transcript into VALIDATION.md).

## Reporting

The implementation report lists:
- the repo URL;
- the acceptance checklist with numbers;
- the performance table;
- known limitations;
- anything you'd want reviewed before announcing it.

If you are blocked (licence, missing upstream detail, a parity gap you can't close), stop and report the evidence instead
of loosening a tolerance.
