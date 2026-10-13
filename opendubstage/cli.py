"""``opendubstage <tool> ...``: one entry point that forwards to each tool's own command line.

Every tool is also runnable on its own, e.g. ``python -m opendubstage.qc.mixqc --help``.
"""
from __future__ import annotations

import importlib
import sys

from . import __version__

# tool -> (module, one-line help). Modules are imported lazily so `--help` stays fast and light.
TOOLS = {
    "specrepair": ("opendubstage.restore.specrepair", "spectral repair: room-tone replace, attenuate, LF thumps"),
    "memofix": ("opendubstage.restore.memofix", "remove car/room rumble learned from word gaps, LTASS matching EQ"),
    "ltasmatch": ("opendubstage.match.ltasmatch", "dialogue EQ match: fit an ffmpeg equalizer chain from LTAS"),
    "carve": ("opendubstage.mix.carve", "speech-keyed 1-4 kHz dynamic carve of a music stem"),
    "transitions": ("opendubstage.mix.transitions", "J/L-cut pre-laps, tails, crossfades on a cue sheet"),
    "qc": ("opendubstage.qc.mixqc", "public-radio mix QC (loudness, LRA, true peak, dead air, jumps, tonal)"),
    "foley": ("opendubstage.foley.cli", "sound for stills: analyze / plan / check / generate / render"),
}


def _usage() -> str:
    w = max(map(len, TOOLS))
    lines = [f"opendubstage {__version__}", "", "usage: opendubstage <tool> [args...]", "", "tools:"]
    lines += [f"  {k:<{w}}  {h}" for k, (_, h) in TOOLS.items()]
    lines += ["", "Run 'opendubstage <tool> --help' for a tool's options."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0
    if argv[0] in ("-V", "--version"):
        print(__version__)
        return 0
    tool, rest = argv[0], argv[1:]
    if tool not in TOOLS:
        print(f"opendubstage: unknown tool {tool!r}\n\n{_usage()}", file=sys.stderr)
        return 2
    mod = importlib.import_module(TOOLS[tool][0])
    rc = mod.main(rest)
    return int(rc or 0)


if __name__ == "__main__":
    raise SystemExit(main())
