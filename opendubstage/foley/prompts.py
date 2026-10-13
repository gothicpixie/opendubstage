"""Prompt construction and the anti-cartoon lint.

Every generated effect names one action, one material, one place and a
distance, then a fixed tail that rules out music and voices. The lint rejects
prompts that ask for the things that make photo sound design feel cartoonish:
whooshes on camera moves, stingers, sitcom laughter, music, intelligible speech.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

SPOT_TAIL = "quiet, recorded on location, slightly muffled, no music, no voices, one-shot"
TEXTURE_TAIL = "no music, no voices, ambience"

# (pattern, reason). Patterns are matched against the lower-cased prompt.
BANNED: list[tuple[str, str]] = [
    (r"\b(whoosh|swoosh|swish|riser|rise|swell|braam|impact hit|stinger|sting)\b",
     "move-linked or trailer sound: spots never follow the camera move"),
    (r"\b(laugh track|sitcom|cartoon|comedic|funny|boing|slide whistle)\b", "comedic sound"),
    (r"\b(music|melody|song|singing|choir|chord|beat|drum loop)\b", "music belongs to the music stem"),
    (r"\b(saying|says|speech|talking clearly|dialogue|narrat\w*|announcer)\b", "voices compete with dialogue"),
    (r"\b(epic|cinematic|dramatic|huge|massive|explosion)\b", "scale is wrong for a memory"),
]

_NEGATED = re.compile(r"\bno (music|voices|clear words|speech)\b")


@dataclass(frozen=True)
class PromptParts:
    action: str
    material: str = ""
    place: str = ""
    distance: str = "medium distance"


def build_prompt(parts: PromptParts, layer: str = "spot") -> str:
    head = " ".join(p for p in (parts.action, parts.material, parts.place) if p).strip()
    tail = SPOT_TAIL if layer in ("spot", "transition") else TEXTURE_TAIL
    return f"{head}, {parts.distance}, {tail}"


def lint(prompt: str) -> list[str]:
    """Return a list of violations (empty = clean). Negations like 'no music' are allowed."""
    text = _NEGATED.sub(" ", prompt.lower())
    out = []
    for pat, reason in BANNED:
        m = re.search(pat, text)
        if m:
            out.append(f"{m.group(0)!r}: {reason}")
    return out
