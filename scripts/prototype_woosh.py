#!/usr/bin/env python3
"""Prototype: generate sound for the public-domain demo stills with Woosh-DVFlow.

For every still it generates two 8 s clips from the same prompt and seed:
  *_img.wav  still + text conditioning (the still is presented as a static clip)
  *_txt.wav  text only (flat grey frames), as the control for "does the picture matter?"
It also generates one bed per chapter and fills the cue sheet's sources, then
writes samples/manifest.json with prompts, parameters, device and timings.

    .venv/bin/python scripts/prototype_woosh.py examples/foley/public_domain/sequence.json \
        examples/foley/public_domain/cuesheet.json samples/woosh
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

from opendubstage.foley.analysis_vision import VisionAnalyzer
from opendubstage.foley.backends import SoundRequest
from opendubstage.foley.backends.woosh_backend import WooshBackend
from opendubstage.foley.io import cuesheet_from_dict, cuesheet_to_dict, load_sequence
from opendubstage.foley.model import Layer
from opendubstage.foley.taxonomy import bed_prompt

SEED = 7


def main(seq_path: str, sheet_path: str, out: str, woosh_root: str = "vendor/woosh") -> None:
    seq, base = load_sequence(seq_path)
    for s in seq.stills:
        s.path = str((base / s.path).resolve()) if s.path else None
    sheet = cuesheet_from_dict(json.loads(Path(sheet_path).read_text()))
    out_dir = Path(out)
    an = VisionAnalyzer()

    t0 = time.perf_counter()
    img_be = WooshBackend(woosh_root, use_image=True)
    txt_be = WooshBackend.__new__(WooshBackend)  # share the loaded model, flip the image switch
    txt_be.__dict__.update(img_be.__dict__, use_image=False)
    load_s = time.perf_counter() - t0
    manifest = {"model": "Woosh-DVFlow-8s (Sony AI, weights CC-BY-NC 4.0)", "device": img_be.device,
                "steps": img_be.steps, "cfg": img_be.cfg, "seed": SEED, "machine": platform.machine(),
                "load_s": round(load_s, 1), "items": []}

    for s in seq.stills:
        a = an.analyze(s)
        best = max(a.candidates, key=lambda c: c.confidence, default=None)
        key = best.sound_class if best else f"bed_{a.bed or 'room_tone'}"
        prompt = best.prompt if best else bed_prompt(a.bed or "room_tone", s.time_of_day)
        for be in (img_be, txt_be):
            t = time.perf_counter()
            p = be.generate(SoundRequest(f"{s.id}__{key}", prompt, 8.0, 0, s.path, seed=SEED), out_dir)
            manifest["items"].append({"still": s.id, "file": p.name, "prompt": prompt,
                                      "conditioning": "still+text" if be.use_image else "text only",
                                      "no_sound_reason": a.no_sound_reason, "secs": round(time.perf_counter() - t, 1)})
            print(manifest["items"][-1], flush=True)

    stills = {s.id: s for s in seq.stills}
    for c in sheet.cues:
        if c.layer is Layer.BED:
            first = next(s for s in seq.stills if s.end > c.at + 0.01)
            p = img_be.generate(SoundRequest(f"bed__{c.sound_class}", c.prompt, 8.0, 0, first.path, seed=SEED), out_dir)
        else:
            p = out_dir / f"{c.still_id}__{c.sound_class}_0_img.wav"
            if not p.exists():
                p = img_be.generate(SoundRequest(f"{c.still_id}__{c.sound_class}", c.prompt, 8.0, 0,
                                                 stills[c.still_id].path, seed=SEED), out_dir)
        c.src = str(p.resolve())
    Path(sheet_path).write_text(json.dumps(cuesheet_to_dict(sheet), indent=2) + "\n")
    manifest["total_s"] = round(time.perf_counter() - t0, 1)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("total", manifest["total_s"], "s on", img_be.device)


if __name__ == "__main__":
    main(*sys.argv[1:])
