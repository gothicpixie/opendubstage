"""Pluggable sound sources.

A backend turns a request (prompt, optional image, duration, variant) into an
audio file. Backends are imported lazily so the core library has no heavy
dependencies.

- `library`: pick from a folder of pre-made files (`<class>_<variant>.<ext>`). No ML.
- `woosh`: local open model (Sony AI Woosh-DVFlow), still+text conditioned, 48 kHz. See README.
- API backends (e.g. a hosted text-to-sound-effects service) can implement the same protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class SoundRequest:
    sound_class: str
    prompt: str
    dur: float
    variant: int = 0
    image: str | None = None       # path to the still, for image-conditioned models
    negative_prompt: str = "music, speech, voices, singing, whoosh"
    seed: int = 0


class Backend(Protocol):
    name: str

    def generate(self, req: SoundRequest, out_dir: Path) -> Path: ...


def get_backend(name: str, **kwargs) -> Backend:
    if name == "library":
        from .library import LibraryBackend
        return LibraryBackend(**kwargs)
    if name == "woosh":
        from .woosh_backend import WooshBackend
        return WooshBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r}")
