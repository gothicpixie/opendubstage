"""Render a cue sheet with sources into per-layer stems (bed, texture, spot, transition) with ffmpeg."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .model import Layer
from .post import render_file
from .rules import CueSheet


def stem_command(inputs: list[tuple[Path, float]], dst: Path, duration: float, sample_rate: int = 48000) -> list[str]:
    """ffmpeg command that places each (file, at_seconds) on a silent timeline of `duration`."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-t", f"{duration:.3f}", "-i", f"anullsrc=r={sample_rate}:cl=stereo"]
    for path, _ in inputs:
        cmd += ["-i", str(path)]
    chains, labels = [], ["[0:a]"]
    for i, (_, at) in enumerate(inputs, start=1):
        ms = max(0, round(at * 1000))
        chains.append(f"[{i}:a]adelay={ms}|{ms}[d{i}]")
        labels.append(f"[d{i}]")
    graph = ";".join(chains + [f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0:duration=first[out]"])
    return cmd + ["-filter_complex", graph, "-map", "[out]", "-t", f"{duration:.3f}", str(dst)]


def render_stems(sheet: CueSheet, out_dir: Path, work: Path | None = None) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    work = work or out_dir / "_cues"
    stems: dict[str, Path] = {}
    for layer in Layer:
        cues = [c for c in sheet.by_layer(layer) if c.src]
        if not cues:
            continue
        processed = []
        for c in cues:
            p = render_file(c.src, work / f"{c.id}.wav", c.target_lufs, c.dur,
                            room=c.room if layer is not Layer.BED else None,
                            fade_in=c.fade_in, fade_out=c.fade_out, loop=layer in (Layer.BED, Layer.TEXTURE))
            processed.append((p, c.at))
        dst = out_dir / f"{layer.value}.wav"
        subprocess.run(stem_command(processed, dst, sheet.duration), check=True)
        stems[layer.value] = dst
    return stems
