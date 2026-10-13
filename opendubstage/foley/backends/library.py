"""Library backend: choose a pre-made file for a sound class and variant."""
from __future__ import annotations

from pathlib import Path

from . import SoundRequest

EXTS = (".wav", ".flac", ".aif", ".aiff", ".mp3", ".ogg")


class LibraryBackend:
    name = "library"

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)

    def files_for(self, sound_class: str) -> list[Path]:
        return sorted(p for p in self.root.glob(f"{sound_class}_*") if p.suffix.lower() in EXTS)

    def generate(self, req: SoundRequest, out_dir: Path) -> Path:
        files = self.files_for(req.sound_class)
        if not files:
            raise FileNotFoundError(f"no library files for class {req.sound_class!r} in {self.root}")
        return files[req.variant % len(files)]
