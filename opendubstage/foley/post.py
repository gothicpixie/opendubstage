"""Post chain: make a generated effect sit in an old photograph instead of on top of it.

1. Match the era and the mic: band-limit to ~80 Hz-9 kHz and dip the 2-5 kHz
   presence region a little, so effects do not sound more modern than the archive.
2. Worldize: a short room/space matched to the scene. Distance and room are what
   tell the ear that a sound belongs to the picture.
3. Short fades so a one-shot never clicks; loudness-normalise to the layer target.

Everything here builds ffmpeg filter strings (pure, testable); `render_file`
is the only function that runs a subprocess.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

# aecho in_gain:out_gain:delays(ms):decays. Early reflections only outdoors; longer indoors.
ROOMS: dict[str, str] = {
    "outdoor_trees": "aecho=0.85:0.5:37|61|89:0.25|0.18|0.12",
    "outdoor_open": "aecho=0.9:0.4:53|97:0.12|0.07",
    "street": "aecho=0.85:0.5:29|71|113:0.22|0.15|0.1",
    "small_room": "aecho=0.8:0.6:13|29|47|71:0.3|0.22|0.15|0.1",
    "hall": "aecho=0.8:0.7:41|97|163|251|353:0.35|0.28|0.22|0.16|0.11",
}

ERA_EQ = "highpass=f=80,lowpass=f=9000,equalizer=f=3500:t=q:w=1.2:g=-3"


def filter_chain(target_lufs: float, dur: float, room: str | None = None,
                 fade_in: float = 0.03, fade_out: float = 0.3, era_eq: bool = True) -> str:
    if dur <= 0:
        raise ValueError("duration must be positive")
    parts: list[str] = []
    if era_eq:
        parts.append(ERA_EQ)
    if room:
        if room not in ROOMS:
            raise ValueError(f"unknown room {room!r}; known: {sorted(ROOMS)}")
        parts.append(ROOMS[room])
    fo = min(fade_out, dur / 2)
    parts.append(f"afade=t=in:st=0:d={min(fade_in, dur / 2):.3f}")
    parts.append(f"afade=t=out:st={max(0.0, dur - fo):.3f}:d={fo:.3f}")
    # loudnorm needs >= 3 s of programme to measure integrated loudness reliably; for
    # short one-shots it still lands within ~1-2 LU, which is inside the tolerance we use.
    parts.append(f"loudnorm=I={target_lufs:.1f}:TP=-3:LRA=11")
    return ",".join(parts)


def render_file(src: str | Path, dst: str | Path, target_lufs: float, dur: float, room: str | None = None,
                fade_in: float = 0.03, fade_out: float = 0.3, sample_rate: int = 48000, loop: bool = False) -> Path:
    """Process one source into a cue-ready file. `loop` repeats a short source to fill `dur` (beds, textures)."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    chain = filter_chain(target_lufs, dur, room, fade_in, fade_out)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           *(["-stream_loop", "-1"] if loop else []), "-i", str(src), "-t", f"{dur:.3f}",
           "-af", chain, "-ar", str(sample_rate), "-ac", "2", str(dst)]
    subprocess.run(cmd, check=True)
    return dst
