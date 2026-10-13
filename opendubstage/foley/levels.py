"""Level table: every layer is set in LU *relative to dialogue*, never as an absolute.

Distance is the dial. A sound under a photograph is a memory, so it must sit
further away than the voice telling the story. The offsets below are rules of
thumb anchored on the published range for music under speech (roughly -10 to
-15 LU below dialogue) and dialogue-anchored delivery specs; see docs/rules.md.
"""
from __future__ import annotations

from .model import Layer

# LU relative to the dialogue's integrated loudness
LEVEL_OFFSETS_LU: dict[str, float] = {
    "bed": -16.0,
    "texture": -12.0,
    "spot": -8.0,
    "spot_hero": -4.0,
    "transition": -2.0,
}

DEFAULT_DIALOGUE_LUFS = -24.0

# Nothing in this tool is ever allowed to reach dialogue level.
MAX_OFFSET_LU = -2.0


def offset_for(layer: Layer, hero: bool = False) -> float:
    if layer is Layer.SPOT and hero:
        return LEVEL_OFFSETS_LU["spot_hero"]
    return LEVEL_OFFSETS_LU[layer.value]


def target_lufs(layer: Layer, dialogue_lufs: float = DEFAULT_DIALOGUE_LUFS, hero: bool = False) -> float:
    """Per-cue integrated loudness target for `layer` under dialogue at `dialogue_lufs`."""
    off = offset_for(layer, hero)
    if off > MAX_OFFSET_LU:  # defensive: a table edit must never put effects at dialogue level
        raise ValueError(f"offset {off} LU is louder than the {MAX_OFFSET_LU} LU ceiling")
    return dialogue_lufs + off
