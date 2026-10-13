"""Command line: analyze -> plan -> check -> generate -> render."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .analysis import KeywordAnalyzer
from .io import cuesheet_from_dict, cuesheet_to_dict, load_sequence, save_cuesheet
from .model import Layer, Sequence
from .rules import check, plan


def _resolve(seq: Sequence, base: Path) -> None:
    for s in seq.stills:
        if s.path and not Path(s.path).is_absolute():
            s.path = str((base / s.path).resolve())


def _analyzer(vision: bool):
    if vision:
        from .analysis_vision import VisionAnalyzer
        return VisionAnalyzer()
    return KeywordAnalyzer()


def cmd_analyze(a) -> int:
    seq, base = load_sequence(a.sequence)
    _resolve(seq, base)
    an = _analyzer(a.vision)
    for s in seq.stills:
        r = an.analyze(s)
        print(json.dumps({**asdict(r), "candidates": [{**asdict(c), "layer": c.layer.value} for c in r.candidates]}))
    return 0


def cmd_plan(a) -> int:
    seq, base = load_sequence(a.sequence)
    _resolve(seq, base)
    an = _analyzer(a.vision)
    sheet = plan(seq, {s.id: an.analyze(s) for s in seq.stills})
    errs = check(sheet, seq)
    if errs:  # should be impossible; the property tests guard it
        print("\n".join(errs), file=sys.stderr)
        return 2
    save_cuesheet(sheet, a.out)
    n = {l.value: len(sheet.by_layer(l)) for l in Layer}
    print(f"{a.out}: {n}; {sum(not d.accepted for d in sheet.decisions)} candidates rejected (see 'decisions')")
    return 0


def cmd_check(a) -> int:
    seq, _ = load_sequence(a.sequence)
    sheet = cuesheet_from_dict(json.loads(Path(a.cuesheet).read_text()))
    errs = check(sheet, seq)
    print("\n".join(errs) if errs else "ok")
    return 1 if errs else 0


def cmd_generate(a) -> int:
    from .backends import SoundRequest, get_backend

    seq, base = load_sequence(a.sequence)
    _resolve(seq, base)
    stills = {s.id: s for s in seq.stills}
    sheet = cuesheet_from_dict(json.loads(Path(a.cuesheet).read_text()))
    kwargs = {"root": a.library} if a.backend == "library" else {"woosh_root": a.woosh_root}
    be = get_backend(a.backend, **kwargs)
    out = Path(a.out)
    for c in sheet.cues:
        img = stills[c.still_id].path if c.still_id in stills else None
        c.src = str(be.generate(SoundRequest(c.sound_class, c.prompt, c.dur, c.variant, img), out))
        print(c.id, c.sound_class, "->", c.src)
    Path(a.cuesheet).write_text(json.dumps(cuesheet_to_dict(sheet), indent=2) + "\n")
    return 0


def cmd_render(a) -> int:
    from .render import render_stems

    sheet = cuesheet_from_dict(json.loads(Path(a.cuesheet).read_text()))
    for k, v in render_stems(sheet, Path(a.out)).items():
        print(k, v)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="opendubstage foley", description=__doc__)
    sub = p.add_subparsers(required=True)
    s = sub.add_parser("analyze", help="print the scene analysis of every still")
    s.add_argument("sequence"); s.add_argument("--vision", action="store_true", help="use Apple Vision labels")
    s.set_defaults(fn=cmd_analyze)
    s = sub.add_parser("plan", help="sequence.json -> cuesheet.json")
    s.add_argument("sequence"); s.add_argument("-o", "--out", default="cuesheet.json")
    s.add_argument("--vision", action="store_true")
    s.set_defaults(fn=cmd_plan)
    s = sub.add_parser("check", help="verify a (possibly hand-edited) cue sheet against the rules")
    s.add_argument("sequence"); s.add_argument("cuesheet"); s.set_defaults(fn=cmd_check)
    s = sub.add_parser("generate", help="fill each cue's source with a backend")
    s.add_argument("sequence"); s.add_argument("cuesheet")
    s.add_argument("--backend", choices=["library", "woosh"], default="library")
    s.add_argument("--library"); s.add_argument("--woosh-root", default="vendor/woosh")
    s.add_argument("-o", "--out", default="fx"); s.set_defaults(fn=cmd_generate)
    s = sub.add_parser("render", help="render per-layer stems with ffmpeg")
    s.add_argument("cuesheet"); s.add_argument("-o", "--out", default="stems"); s.set_defaults(fn=cmd_render)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
