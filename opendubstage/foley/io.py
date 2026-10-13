"""JSON in (sequence) and JSON out (cue sheet)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model import Cue, Decision, Interval, Layer, MusicSegment, NarrationSegment, Sequence, Still
from .rules import CueSheet

CUESHEET_SCHEMA = "opendubstage.foley.cuesheet/0.1"
SEQUENCE_SCHEMA = "opendubstage.foley.sequence/0.1"
# cue sheets written by the standalone stillfoley releases are read unchanged
_LEGACY_CUESHEET_SCHEMAS = {"stillfoley.cuesheet/0.1"}

_STILL_FIELDS = {f for f in Still.__dataclass_fields__}


def sequence_from_dict(d: dict[str, Any]) -> Sequence:
    stills = []
    for i, s in enumerate(d["stills"]):
        unknown = set(s) - _STILL_FIELDS
        if unknown:
            raise ValueError(f"still {i}: unknown fields {sorted(unknown)}")
        stills.append(Still(**s))
    ids = [s.id for s in stills]
    if len(set(ids)) != len(ids):
        raise ValueError("still ids must be unique")
    return Sequence(
        stills=stills,
        narration=[NarrationSegment(**n) for n in d.get("narration", [])],
        music=[MusicSegment(**m) for m in d.get("music", [])],
        live=[Interval(**x) for x in d.get("live", [])],
        silences=[Interval(**x) for x in d.get("silences", [])],
        dialogue_lufs=float(d.get("dialogue_lufs", -24.0)),
        seed=str(d.get("seed", "stillfoley")),
    )


def load_sequence(path: str | Path) -> tuple[Sequence, Path]:
    p = Path(path)
    return sequence_from_dict(json.loads(p.read_text())), p.parent


def cuesheet_to_dict(sheet: CueSheet) -> dict[str, Any]:
    return {
        "schema": CUESHEET_SCHEMA,
        "dialogue_lufs": sheet.dialogue_lufs,
        "duration": round(sheet.duration, 3),
        "cues": [c.to_dict() for c in sheet.cues],
        "decisions": [d.__dict__ for d in sheet.decisions],
    }


def cuesheet_from_dict(d: dict[str, Any]) -> CueSheet:
    if d.get("schema") != CUESHEET_SCHEMA and d.get("schema") not in _LEGACY_CUESHEET_SCHEMAS:
        raise ValueError(f"unsupported cue sheet schema {d.get('schema')!r}")
    cues = []
    for c in d["cues"]:
        c = dict(c)
        c["layer"] = Layer(c["layer"])
        cues.append(Cue(**c))
    return CueSheet(cues=cues, decisions=[Decision(**x) for x in d.get("decisions", [])],
                    dialogue_lufs=d["dialogue_lufs"], duration=d.get("duration", 0.0))


def save_cuesheet(sheet: CueSheet, path: str | Path) -> None:
    Path(path).write_text(json.dumps(cuesheet_to_dict(sheet), indent=2) + "\n")
