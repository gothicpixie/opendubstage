"""Scene analysis: picture (+ caption, tags) -> SceneAnalysis.

The default analyser reads the caption and tags. It is intentionally a
recall-limited, precision-first matcher: a missed sound is invisible, a wrong
sound breaks the spell. A vision-language analyser can implement the same
`Analyzer` protocol and emit class keys from `taxonomy.SOUND_CLASSES`.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Protocol

from .model import SceneAnalysis, SoundCandidate, Still
from .taxonomy import (
    BED_SELECTORS, DEFAULT_BED, NO_SOUND_PATTERNS, NO_SOUND_TAGS, QUALITY_FLAGS, ROOM_FOR_BED, SOUND_CLASSES,
)


class Analyzer(Protocol):
    def analyze(self, still: Still) -> SceneAnalysis: ...


def _text(still: Still) -> str:
    return " ".join([still.caption, *still.tags]).lower()


def no_sound_reason(still: Still) -> str | None:
    if still.has_sync_audio:
        return "footage carries its own sync sound"
    tags = {t.lower() for t in still.tags}
    hit = tags & NO_SOUND_TAGS
    if hit:
        return f"tagged {sorted(hit)[0]}"
    bad = [k for k, v in still.quality.items() if v and k in QUALITY_FLAGS]
    if bad:
        return f"technically weak picture ({bad[0]})"
    text = _text(still)
    for pat, reason in NO_SOUND_PATTERNS:
        if re.search(pat, text):
            return reason
    return None


def select_bed(text: str) -> str | None:
    for key, pats in BED_SELECTORS:
        if any(re.search(p, text) for p in pats):
            return key
    return None


class KeywordAnalyzer:
    """Caption/tag matcher. Precision first; at most a handful of candidates per still."""

    def analyze(self, still: Still) -> SceneAnalysis:
        text = _text(still)
        bed = select_bed(text)
        out = SceneAnalysis(still_id=still.id, bed=bed, room=ROOM_FOR_BED.get(bed) if bed else None)
        reason = no_sound_reason(still)
        if reason:
            out.no_sound_reason = reason
            return out
        for cls in SOUND_CLASSES.values():
            if any(re.search(p, text) for p in cls.keywords) and not any(re.search(p, text) for p in cls.exclude):
                out.candidates.append(SoundCandidate(cls.key, cls.layer, cls.prompt, 0.6, "caption", cls.dur))
        return out


def chapter_beds(stills: list[Still], analyses: dict[str, SceneAnalysis]) -> dict[str, tuple[str, str | None]]:
    """Majority bed per chapter, plus the chapter's most common time of day."""
    by_ch: dict[str, list[Still]] = {}
    for s in stills:
        by_ch.setdefault(s.chapter, []).append(s)
    out = {}
    for ch, items in by_ch.items():
        beds = Counter(analyses[s.id].bed for s in items if analyses.get(s.id) and analyses[s.id].bed)
        tods = Counter(s.time_of_day for s in items if s.time_of_day)
        out[ch] = (beds.most_common(1)[0][0] if beds else DEFAULT_BED, tods.most_common(1)[0][0] if tods else None)
    return out
