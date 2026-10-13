"""Local generative backend: Sony AI's Woosh-DVFlow (video+text -> 48 kHz audio, 4 steps).

A still is presented to the model as a static 8 s clip, so the visual features
carry the picture's content but no motion (which is what we want: sound never
follows a Ken Burns move). Text conditioning carries the restrained prompt.

Woosh code: MIT / Apache-2.0. Woosh weights: CC-BY-NC 4.0 (non-commercial).
Install: see README "Generative backend". Requires `woosh` importable and the
checkpoints directory (Woosh-DVFlow-8s, Woosh-AE, TextConditionerV) present.
"""
from __future__ import annotations

import os
from pathlib import Path

from . import SoundRequest

CLIP_S = 8.0          # the model's training window
FPS = 24
LATENT_FRAMES = 801   # 8 s at the AE's 100 latent frames/s (+1), from the reference script


class WooshBackend:  # pragma: no cover - exercised by scripts/prototype_woosh.py, not unit tests
    name = "woosh"

    def __init__(self, woosh_root: str | Path, device: str | None = None, steps: int = 4, cfg: float = 3.0,
                 use_image: bool = True):
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Synchformer conv3d has no MPS kernel
        import torch

        self.torch = torch
        self.root = Path(woosh_root)
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if torch.backends.mps.is_available() else "cpu")
        self.steps, self.cfg, self.use_image = steps, cfg, use_image
        cwd = os.getcwd()
        os.chdir(self.root)  # Woosh resolves "checkpoints/..." relative to its repo
        try:
            from woosh.components.base import LoadConfig
            from woosh.model.flowmap_from_pretrained import FlowMapFromPretrained
            from woosh.utils.video import SynchformerProcessor

            self.ldm = FlowMapFromPretrained(LoadConfig(path="checkpoints/Woosh-DVFlow-8s")).eval().to(self.device)
            self.sync = SynchformerProcessor(frame_rate=FPS).eval().to(self.device)
        finally:
            os.chdir(cwd)

    def _frames(self, image: str | None):
        torch = self.torch
        from PIL import Image

        n = int(CLIP_S * FPS)
        if image and self.use_image:
            im = Image.open(image).convert("RGB")
            im.thumbnail((448, 448))  # Synchformer centre-crops to 224 anyway
            frame = torch.from_numpy(__import__("numpy").asarray(im).copy())
        else:  # text-only control: a flat mid-grey frame
            frame = torch.full((224, 224, 3), 128, dtype=torch.uint8)
        return frame.unsqueeze(0).expand(n, *frame.shape).contiguous()

    def generate(self, req: SoundRequest, out_dir: Path) -> Path:
        torch = self.torch
        import torchaudio
        from woosh.inference.flowmap_sampler import sample_euler

        out_dir.mkdir(parents=True, exist_ok=True)
        tag = "img" if (req.image and self.use_image) else "txt"
        dst = out_dir / f"{req.sound_class}_{req.variant}_{tag}.wav"
        g = torch.Generator().manual_seed(req.seed * 1000 + req.variant)
        noise = torch.randn(1, 128, LATENT_FRAMES, generator=g).to(self.device)  # CPU RNG: MPS RNG crackles
        with torch.inference_mode():
            feats = self.sync(self._frames(req.image).to(self.device), FPS)
            cond = self.ldm.get_cond({"audio": None, "description": [req.prompt], "synch_out": feats["synch_out"]},
                                     no_dropout=True, device=self.device)
            x = sample_euler(model=self.ldm, noise=noise, cond=cond, num_steps=self.steps,
                             renoise=[0, 0.5, 0.5, 0.3][: self.steps], cfg=self.cfg)
            audio = self.ldm.autoencoder.inverse(x).cpu()[0]
        peak = float(audio.abs().max())
        if peak > 1.0:
            audio = audio / peak
        torchaudio.save(str(dst), audio, sample_rate=48000)
        return dst
