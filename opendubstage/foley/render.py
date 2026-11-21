"""Render a cue sheet with sources into per-layer stems (bed, texture, spot, transition) with ffmpeg."""
from __future__ import annotations

import math
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
        processed = []
        for c in sheet.by_layer(layer):
            if not c.src:
                continue
            at, dur = c.at, c.dur
            if layer is Layer.BED:
                # The FFmpeg trim and delay use milliseconds. Round both edges
                # inward so the rendered bed stays inside the checked cue.
                start_ms = max(0, math.ceil(c.at * 1000))
                end_ms = math.floor(c.end * 1000)
                if end_ms <= start_ms:
                    continue
                at, dur = start_ms / 1000, (end_ms - start_ms) / 1000
            p = render_file(c.src, work / f"{c.id}.wav", c.target_lufs, dur,
                            room=c.room if layer is not Layer.BED else None,
                            fade_in=c.fade_in, fade_out=c.fade_out, loop=layer in (Layer.BED, Layer.TEXTURE))
            processed.append((p, at))
        if not processed:
            continue
        dst = out_dir / f"{layer.value}.wav"
        subprocess.run(stem_command(processed, dst, sheet.duration), check=True)
        stems[layer.value] = dst
    return stems
