"""Plain helpers shared by the foley tests (kept out of conftest.py so they can be imported normally)."""
from __future__ import annotations

from opendubstage.foley.model import Sequence, Still


def mk_still(i: int, caption: str, start: float, dur: float = 6.0, chapter: str = "a", **kw) -> Still:
    return Still(id=f"s{i}", start=start, dur=dur, caption=caption, chapter=chapter, **kw)


def analyze_all(seq: Sequence, analyzer) -> dict:
    return {s.id: analyzer.analyze(s) for s in seq.stills}
