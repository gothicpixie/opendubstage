"""Plain data types shared by every stage of the pipeline.

Times are seconds on the programme timeline. Loudness targets are integrated
LUFS per cue, before any mix envelope (ducking) is applied.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Layer(str, Enum):
    """The four sound layers, in priority order (see docs/rules.md)."""

    BED = "bed"                # continuous scene ambience
    TEXTURE = "texture"        # semi-continuous, placeable (birds, distant voices, water)
    SPOT = "spot"              # one-shot matched to a visible object
    TRANSITION = "transition"  # a designed sound that *is* the cut; hand-requested only


@dataclass(frozen=True)
class Interval:
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"interval ends before it starts: {self.start}..{self.end}")

    @property
    def dur(self) -> float:
        return self.end - self.start

    def overlaps(self, start: float, end: float) -> bool:
        """Open-interval overlap test: touching edges do not overlap."""
        return start < self.end and end > self.start

    def contains(self, t: float) -> bool:
        return self.start <= t < self.end


@dataclass(frozen=True)
class NarrationSegment(Interval):
    """A stretch of dialogue/narration. `key` marks a line the mix must clear the way for."""

    key: bool = False


@dataclass(frozen=True)
class MusicSegment(Interval):
    """`up` = music is the foreground (no spots); `under` = music sits beneath dialogue."""

    level: str = "under"


@dataclass
class Still:
    """One picture on the timeline: a still photo (usually with a move) or an archival clip."""

    id: str
    start: float
    dur: float
    path: str | None = None
    caption: str = ""
    tags: list[str] = field(default_factory=list)
    chapter: str = ""
    time_of_day: str | None = None      # "day" | "dusk" | "night" | None
    kind: str = "photo"                 # "photo" | "footage"
    has_sync_audio: bool = False        # footage that already carries its real sound
    hero: bool = False                  # a hand-picked still that may take a louder spot
    transition: str | None = None       # hand-requested designed transition class
    quality: dict[str, bool] = field(default_factory=dict)  # {"blurry": True, "dark": True, ...}

    @property
    def end(self) -> float:
        return self.start + self.dur


@dataclass
class Sequence:
    stills: list[Still]
    narration: list[NarrationSegment] = field(default_factory=list)
    music: list[MusicSegment] = field(default_factory=list)
    live: list[Interval] = field(default_factory=list)       # sync sound already present
    silences: list[Interval] = field(default_factory=list)   # designed total silences
    dialogue_lufs: float = -24.0
    seed: str = "stillfoley"

    def __post_init__(self) -> None:
        self.stills = sorted(self.stills, key=lambda s: s.start)
        self.narration = sorted(self.narration, key=lambda s: s.start)

    @property
    def end(self) -> float:
        return max((s.end for s in self.stills), default=0.0)


@dataclass
class SoundCandidate:
    """A sound an analyser thinks the picture could carry. Placement decides if it gets one."""

    sound_class: str
    layer: Layer
    prompt: str
    confidence: float = 1.0
    source: str = "caption"   # "caption" | "tags" | "model" | "manual"
    dur: float = 2.0


@dataclass
class SceneAnalysis:
    still_id: str
    candidates: list[SoundCandidate] = field(default_factory=list)
    bed: str | None = None
    room: str | None = None
    no_sound_reason: str | None = None   # set => bed only, never a spot or texture


@dataclass
class Cue:
    id: str
    layer: Layer
    sound_class: str
    at: float
    dur: float
    target_lufs: float
    fade_in: float = 0.03
    fade_out: float = 0.03
    still_id: str | None = None
    prompt: str = ""
    variant: int = 0
    room: str | None = None
    src: str | None = None

    @property
    def end(self) -> float:
        return self.at + self.dur

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["layer"] = self.layer.value
        return d


@dataclass
class Decision:
    """Why a candidate was accepted or rejected: the planner's audit trail."""

    still_id: str | None
    sound_class: str | None
    accepted: bool
    reason: str
