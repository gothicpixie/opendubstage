"""Scene-transition pass for a film cue sheet: J-cuts, L-cuts and music crossfades at chapter boundaries.

Documentary practice, applied mechanically to a cue sheet before it is mixed:

1. Ambience pre-laps (J-cuts): the incoming scene's ambience or live room sound starts `prelap` seconds before the
   cut, under the outgoing audio.
2. Tails (L-cuts): outgoing music, ambience and live sound run `tail` seconds past the cut with a fade, under the
   incoming scene. Nothing stops dead on a boundary.
3. Music-to-music seams (one music cue ending where another begins, at a boundary or inside a chapter): an
   equal-power crossfade of `xfade` seconds. When both cues are foreground ("up") songs from different sources,
   a straight crossfade clashes (tempo, key), so the outgoing song fades out, `gap` seconds of ambience play, and
   the incoming song fades in.
4. Edges: live-sound chapter edges get fades of at least `live_edge_fade`; no cue ends with a fade shorter than
   `min_fade` unless another cue in the same stem continues straight on from it.
5. Constraints: dialogue cues are never touched (so two dialogue cues never come to overlap); `total` is unchanged;
   no cue is extended past its source's length, and none is extended outside [0, total].

The pass never moves a cue's sync point (`at` of live sound and of "up" music): live sound is extended with
`pre`/`post` (extra source audio before/after, as the renderer supports), and music/ambience are extended by moving
`ss`/`at` together, which keeps every source sample at the same film time.

Cue-sheet schema (times in seconds; "film time" is time on the output timeline)
------------------------------------------------------------------------------
  {"total": float,                       # programme length
   "dx":   [cue, ...],                   # dialogue / narration (never modified by this pass)
   "live": [cue, ...],                   # sync sound from picture sources
   "mx":   [cue, ...],                   # music
   "amb":  [cue, ...]}                   # ambience beds and effects
  cue = {"src": str,                     # source file
         "ss": float,                    # source in-point
         "dur": float,                   # length on the timeline (from `at`)
         "at": float,                    # film time where source time `ss` plays
         "pre"?: float, "post"?: float,  # extra source audio before `ss` / after `ss+dur` (pre is clipped to ss)
         "fade"?: float,                 # live/amb: edge fade, both ends (default `default_fade[stem]`)
         "fade_in"?: float, "fade_out"?: float,   # overrides; mx defaults are mx_fade_in / mx_fade_out
         "mode"?: "up" | "under",        # mx only: foreground song vs ducked score
         ...}                            # any other keys are passed through untouched
  The audible span of a cue is [at - min(pre, ss), at + dur + post]; fades apply at the ends of that span and are
  equal-power when the renderer uses a quarter-sine curve (ffmpeg afade curve=qsin).

Chapters: a list of {"t0": float, ...} (other keys ignored) in film time; every t0 after the first is a boundary.

API
---
  transitions(cues, chapters, settings=None, src_dur=None, log=None) -> new cue sheet (the input is not modified)
  check_transitions(cues, chapters, settings=None, src_dur=None) -> list of problem strings
  bridge_gaps(cues, chapters) -> boundaries where no non-dialogue sound spans the cut
  ProbeCache(path=None) -> callable src -> duration (ffprobe, cached in memory and optionally in a JSON file)

CLI:  opendubstage transitions CUES.json CHAPTERS.json OUT.json [--probe-cache FILE]
      (CHAPTERS.json is a list of chapters or {"chapters": [...]})
"""
from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

STEMS = ("live", "mx", "amb")


@dataclass
class Settings:
    prelap: float = 2.0            # J-cut: incoming ambience / live room sound starts this early (1.5-3 s)
    tail: float = 3.0              # L-cut: outgoing music / ambience / live sound runs this far past the cut (1.5-4 s)
    min_tail: float = 1.5          # a tail shorter than this is not worth adding
    tail_fade: float = 4.5         # fade-out length for a tailed music cue (starts before the cut, ends in the next scene)
    xfade: float = 3.0             # music-to-music crossfade (2-4 s)
    min_xfade: float = 2.0
    gap: float = 0.75              # up-song to different up-song: ambience gap between fade-out and fade-in (0.5-1 s)
    gap_fade: float = 2.5          # fade-out before such a gap
    gap_fade_in: float = 1.5       # fade-in after such a gap
    live_edge_fade: float = 2.0    # minimum fade at a live chapter edge
    min_fade: float = 1.5          # nothing ends with a shorter fade (unless a same-stem cue continues from it)
    edge_window: float = 0.5       # an edge this close to a boundary is "at" the boundary
    reach: float = 1.5             # edges this far on the other side of a boundary still count as at it
    seam_before: float = 2.0       # mx seam: outgoing ends from this much before the incoming start ...
    seam_after: float = 5.0        # ... to this much after it
    dx_clear: float = 0.25         # added live pre-laps/tails stop this far from dialogue in the neighbouring scene
    src_margin: float = 0.05       # stay this far inside the source's end
    mx_fade_in: float = 0.0        # renderer defaults (what an absent key means)
    mx_fade_out: float = 6.0
    default_fade: dict = field(default_factory=lambda: {"live": 2.0, "amb": 3.0})


# ---------------------------------------------------------------------------------------------------------------- probe
class ProbeCache:
    """src -> duration in seconds (None if unknown), via ffprobe, memoised; optionally persisted as JSON."""

    def __init__(self, path: Optional[str | Path] = None, ffprobe: str = "ffprobe"):
        self.path, self.ffprobe = (Path(path) if path else None), ffprobe
        self.d: dict = {}
        if self.path and self.path.exists():
            try: self.d = json.loads(self.path.read_text())
            except ValueError: self.d = {}

    def __call__(self, src: str) -> Optional[float]:
        if src not in self.d:
            v = None
            try:
                p = subprocess.run([self.ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", src],
                                   capture_output=True, text=True, timeout=60)
                v = float(p.stdout.strip().splitlines()[0])
            except (ValueError, IndexError, OSError, subprocess.SubprocessError):
                v = None
            self.d[src] = v
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True); self.path.write_text(json.dumps(self.d, indent=0))
        return self.d[src]


def _src_dur_fn(src_dur) -> Callable[[str], Optional[float]]:
    if src_dur is None: return lambda s: None
    if isinstance(src_dur, dict): return lambda s: src_dur.get(s)
    return src_dur


# ------------------------------------------------------------------------------------------------------- cue geometry
def span(stem, c, src_dur=None):
    """Audible (start, end) in film time, as the renderer plays it."""
    ss = c.get("ss", 0.0); pre = min(c.get("pre", 0.0), ss)
    end = c["at"] + c["dur"] + c.get("post", 0.0)
    sd = src_dur(c["src"]) if src_dur else None
    if sd: end = min(end, c["at"] + (sd - ss))
    return c["at"] - pre, end


def fades(stem, c, S: Settings):
    if stem == "mx": return c.get("fade_in", S.mx_fade_in), c.get("fade_out", S.mx_fade_out)
    f = c.get("fade") or S.default_fade[stem]
    return c.get("fade_in", f), c.get("fade_out", f)


def _set_fades(stem, c, S, fi=None, fo=None):
    """Set one end's fade explicitly without changing the other end (live/amb share `fade`)."""
    cur_in, cur_out = fades(stem, c, S)
    if stem != "mx":
        c.setdefault("fade_in", cur_in); c.setdefault("fade_out", cur_out)
    if fi is not None: c["fade_in"] = round(fi, 3)
    if fo is not None: c["fade_out"] = round(fo, 3)


def _src_room_after(c, src_dur, S):
    """Seconds of source available after the cue's current audible end (inf if unknown)."""
    sd = src_dur(c["src"])
    if not sd: return math.inf
    return max(0.0, sd - S.src_margin - (c.get("ss", 0.0) + c["dur"] + c.get("post", 0.0)))


def _extend_end(stem, c, new_end, src_dur, S, total):
    """Move the audible end to new_end (later), within the source and the programme. Returns the end reached."""
    _, end = span(stem, c)
    want = min(new_end, total) - end
    if want <= 0: return end
    add = min(want, _src_room_after(c, src_dur, S))
    if add <= 0: return end
    if stem == "live": c["post"] = round(c.get("post", 0.0) + add, 3)
    else: c["dur"] = round(c["dur"] + add, 3)
    return end + add


def _set_end(stem, c, new_end, src_dur, S, total):
    """Move the audible end to new_end (earlier or later). Returns the end reached."""
    s, end = span(stem, c)
    if new_end >= end: return _extend_end(stem, c, new_end, src_dur, S, total)
    cut = end - max(new_end, s + 0.5)
    if stem == "live":
        p = c.get("post", 0.0); dp = min(p, cut); c["post"] = round(p - dp, 3); cut -= dp
    c["dur"] = round(c["dur"] - cut, 3)
    return span(stem, c)[1]


def _extend_start(stem, c, new_start, S):
    """Move the audible start earlier to new_start (≥ 0), using the source before `ss`. Returns the start reached."""
    s, _ = span(stem, c)
    want = s - max(0.0, new_start)
    if want <= 0: return s
    if stem == "live":
        have = c.get("ss", 0.0) - min(c.get("pre", 0.0), c.get("ss", 0.0))   # unused source before the current pre
        add = min(want, have)
        if add > 0: c["pre"] = round(min(c.get("pre", 0.0), c.get("ss", 0.0)) + add, 3)
        return s - max(0.0, add)
    add = min(want, c.get("ss", 0.0))
    if add > 0:
        c["ss"] = round(c["ss"] - add, 3); c["at"] = round(c["at"] - add, 3); c["dur"] = round(c["dur"] + add, 3)
    return s - max(0.0, add)


def continuous(a, b, tol=0.5):
    """b continues a's recording (same source, source time continuous at b's start, b starts as a ends)."""
    if a["src"] != b["src"] or abs((a["at"] + a["dur"] + a.get("post", 0.0)) - b["at"]) > 1.0: return False
    return abs((a.get("ss", 0.0) + (b["at"] - a["at"])) - b.get("ss", 0.0)) < tol


def boundaries(chapters):
    return [float(ch["t0"]) for ch in chapters[1:]]


def _all(cues):
    return [(k, c) for k in STEMS for c in cues.get(k, [])]


# --------------------------------------------------------------------------------------------------------------- pass
def transitions(cues: dict, chapters: list, settings: Settings | None = None, src_dur=None, log: list | None = None) -> dict:
    """Return a new cue sheet with documentary transitions at every chapter boundary (see module docstring)."""
    S = settings or Settings()
    sd = _src_dur_fn(src_dur)
    out = copy.deepcopy(cues)
    total = float(out["total"])
    B = boundaries(chapters)
    dx = sorted((c["at"], c["at"] + c["dur"]) for c in out.get("dx", []))
    note = (lambda *a: log.append(" ".join(str(x) for x in a))) if log is not None else (lambda *a: None)
    done_end, done_start = set(), set()

    def near_b(t, lo, hi):
        return next((b for b in B if b - lo <= t <= b + hi), None)

    # 0. a cue that asks for more than its source holds would get its fade-out placed after the audio ends (a hard
    #    stop): pull the end back inside the source so the fade lands on real audio
    for k, c in _all(out):
        if sd(c["src"]) and c.get("ss", 0.0) + c["dur"] + c.get("post", 0.0) > sd(c["src"]) - S.src_margin + 1e-6:
            over = c.get("ss", 0.0) + c["dur"] + c.get("post", 0.0) - (sd(c["src"]) - S.src_margin)
            dp = min(c.get("post", 0.0), over)
            if dp: c["post"] = round(c["post"] - dp, 3)
            c["dur"] = round(c["dur"] - (over - dp), 3)
            note(k, "clamped to source", Path(c["src"]).name, f"-{over:.2f}s")

    # 1. music seams: one mx cue ends around where another starts
    mx = sorted(out.get("mx", []), key=lambda c: c["at"])
    for bc in mx:
        bs, _ = span("mx", bc, sd)
        cands = [ac for ac in mx if ac is not bc and id(ac) not in done_end and span("mx", ac, sd)[0] < bs - S.seam_after
                 and bs - S.seam_before <= span("mx", ac, sd)[1] <= bs + S.seam_after]
        if not cands or id(bc) in done_start: continue
        ac = min(cands, key=lambda a: abs(span("mx", a, sd)[1] - bs))
        a_pos = ac.get("ss", 0.0) + (bs - ac["at"])                 # outgoing source time at the incoming start
        if ac["src"] == bc["src"] and abs(a_pos - (bc.get("ss", 0.0) - (bc["at"] - bs))) < 0.5:
            # the same recording continuing (split across a cut): butt-splice, since overlapping two near-aligned
            # copies of the same audio comb-filters
            _set_end("mx", ac, bs + 0.05, sd, S, total)
            _set_fades("mx", ac, S, fo=0.1); _set_fades("mx", bc, S, fi=0.1)
            note("mx splice", f"{bs:.2f}", Path(ac["src"]).name)
            done_end.add(id(ac)); done_start.add(id(bc)); continue
        up_up = ac.get("mode") == "up" and bc.get("mode") == "up" and ac["src"] != bc["src"]
        if up_up:          # fade-out -> ambience gap -> fade-in (a straight crossfade of two songs clashes)
            e = _set_end("mx", ac, bs - S.gap, sd, S, total)
            _set_fades("mx", ac, S, fo=max(S.gap_fade, S.min_fade))
            _set_fades("mx", bc, S, fi=max(fades("mx", bc, S)[0], S.gap_fade_in))
            note("mx gap", f"{bs:.2f}", Path(ac["src"]).name, "->", Path(bc["src"]).name, f"out end {e:.2f}")
        else:              # equal-power crossfade; mid-song "under" entries pre-lap by half the crossfade
            if bc.get("mode") != "up" and bc.get("ss", 0.0) >= 0.5:
                w0 = _extend_start("mx", bc, bs - S.xfade / 2, S)
            else:
                w0 = bs
            e = _set_end("mx", ac, w0 + S.xfade, sd, S, total)
            x = e - w0
            if x < S.min_xfade:            # outgoing source ran out: fade it fully before the incoming starts
                _set_fades("mx", ac, S, fo=max(S.min_fade, fades("mx", ac, S)[1]))
                _set_fades("mx", bc, S, fi=max(fades("mx", bc, S)[0], S.min_xfade))
            else:
                _set_fades("mx", ac, S, fo=max(x, min(fades("mx", ac, S)[1], x + 3.0)))
                _set_fades("mx", bc, S, fi=max(fades("mx", bc, S)[0], x))
            note("mx xfade", f"{bs:.2f}", Path(ac["src"]).name, "->", Path(bc["src"]).name, f"window {w0:.2f}-{e:.2f}")
        done_end.add(id(ac)); done_start.add(id(bc))

    # 2. boundary edges: tails (L-cuts) and pre-laps (J-cuts)
    for b in B:
        prev_dx_end = max((e for s, e in dx if s < b), default=-math.inf)
        next_dx_start = min((s for s, e in dx if s >= b - 0.01), default=math.inf)
        for k, c in _all(out):
            s, e = span(k, c, sd)
            # outgoing edge
            if id(c) not in done_end and b - S.edge_window <= e <= b + S.reach:
                target = b + S.tail
                if k == "live":
                    target = min(target, max(e, next_dx_start - S.dx_clear))
                e2 = _extend_end(k, c, target, sd, S, total) if target - e >= 0.05 else e
                fo = fades(k, c, S)[1]
                need = {"mx": S.tail_fade if e2 - b >= S.min_tail else S.min_fade * 2,
                        "amb": max(S.min_fade, 3.0), "live": max(S.live_edge_fade, 3.0)}[k]
                _set_fades(k, c, S, fo=max(fo, min(need, e2 - s)))
                done_end.add(id(c))
                note(k, "tail", f"b={b:.2f}", Path(c["src"]).name, f"end {e:.2f}->{e2:.2f}")
            # incoming edge
            if id(c) not in done_start and b - S.reach <= s <= b + S.edge_window:
                if k in ("amb", "live"):
                    target = b - S.prelap
                    if k == "live": target = max(target, min(s, prev_dx_end + S.dx_clear))
                    s2 = _extend_start(k, c, target, S) if s - target >= 0.05 else s
                    fi = fades(k, c, S)[0]
                    _set_fades(k, c, S, fi=max(fi, S.live_edge_fade if k == "live" else S.min_fade, min(s - s2 + 1.0, 4.0)))
                    note(k, "prelap", f"b={b:.2f}", Path(c["src"]).name, f"start {s:.2f}->{s2:.2f}")
                elif c.get("ss", 0.0) >= 0.5 and fades(k, c, S)[0] < S.min_fade:   # mid-song entry on a cut
                    _set_fades(k, c, S, fi=S.min_fade)
                    note("mx", "soft entry", f"b={b:.2f}", Path(c["src"]).name)
                done_start.add(id(c))

    # 3. no short fade on an end that nothing in the same stem continues from
    for k in STEMS:
        lst = out.get(k, [])
        for c in lst:
            s, e = span(k, c, sd)
            fo = fades(k, c, S)[1]
            if fo >= S.min_fade or e >= total - 0.01: continue
            if any(o is not c and e - 1.0 <= span(k, o, sd)[0] <= e + 0.1 for o in lst): continue   # chained clip
            if k == "live":
                lim = min((s0 for s0, _ in dx if s0 >= e - 0.01), default=math.inf) - S.dx_clear
                _extend_end(k, c, min(e + 1.0, max(e, lim)), sd, S, total)
            s, e = span(k, c, sd)
            _set_fades(k, c, S, fo=min(S.min_fade, max(fo, e - s)))
            note(k, "min fade", Path(c["src"]).name, f"end {e:.2f}")

    assert out["total"] == cues["total"]
    assert out.get("dx") == cues.get("dx")
    return out


# -------------------------------------------------------------------------------------------------------------- checks
def check_transitions(cues: dict, chapters: list, settings: Settings | None = None, src_dur=None) -> list[str]:
    """Problems: an mx/live/amb cue that starts or ends within `edge_window` of a boundary with a fade under
    `min_fade` and no other non-dialogue cue overlapping that edge (continuing past an end / already playing at a
    start); plus constraint violations (overlapping dialogue, a cue past its source)."""
    S = settings or Settings()
    sd = _src_dur_fn(src_dur)
    probs = []
    items = [(k, c, *span(k, c, sd)) for k, c in _all(cues)]
    for b in boundaries(chapters):
        for k, c, s, e in items:
            fi, fo = fades(k, c, S)
            name = Path(c["src"]).name
            if abs(e - b) <= S.edge_window and fo < S.min_fade:
                if not any(continuous(c, o) for k2, o, *_ in items if k2 == k) and not any(o is not c and s2 < e - 0.25 and e2 > e + 1.0 for _, o, s2, e2 in items):
                    probs.append(f"{k} {name} ends {e:.2f} at boundary {b:.2f} with a {fo:.2f}s fade and nothing continuing")
            if abs(s - b) <= S.edge_window and fi < S.min_fade:
                if not any(continuous(o, c) for k2, o, *_ in items if k2 == k) and not any(o is not c and s2 < s - 1.0 and e2 > s + 0.25 for _, o, s2, e2 in items):
                    probs.append(f"{k} {name} starts {s:.2f} at boundary {b:.2f} with a {fi:.2f}s fade and nothing under it")
    d = sorted(cues.get("dx", []), key=lambda c: c["at"])
    for a, c in zip(d, d[1:]):
        if c["at"] < a["at"] + a["dur"] - 1e-6:
            probs.append(f"dx overlap at {c['at']:.2f}")
    for k, c, s, e in items:
        dur = sd(c["src"])
        if dur and c.get("ss", 0.0) + c["dur"] + c.get("post", 0.0) > dur + 0.02:
            probs.append(f"{k} {Path(c['src']).name} runs past its source ({dur:.2f}s)")
    return probs


def bridge_gaps(cues: dict, chapters: list, src_dur=None, overlap: float = 1.5) -> list[float]:
    """Boundaries where no non-dialogue cue plays from `overlap` s before the cut to `overlap` s after it, and no pair
    of cues overlaps across it (outgoing still sounding when the incoming has started) by at least `overlap` s."""
    sd = _src_dur_fn(src_dur)
    spans = [span(k, c, sd) for k, c in _all(cues)]
    gaps = []
    for b in boundaries(chapters):
        through = any(s <= b - overlap and e >= b + overlap for s, e in spans)
        outs = [e for s, e in spans if s < b and b - 0.5 <= e]
        ins = [s for s, e in spans if e > b and s <= b + 0.5]
        xf = any(e - s2 >= overlap for e in outs for s2 in ins)
        if not (through or xf): gaps.append(b)
    return gaps


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="opendubstage transitions",
                                 description="J/L-cut pre-laps, tails and music crossfades at chapter boundaries.")
    ap.add_argument("cues", help="cue sheet JSON")
    ap.add_argument("chapters", help='chapters JSON: a list of {"t0": ...} or {"chapters": [...]}')
    ap.add_argument("out", help="output cue sheet JSON")
    ap.add_argument("--probe-cache", help="JSON file caching source durations (ffprobe)")
    a = ap.parse_args(argv)
    cues = json.loads(Path(a.cues).read_text())
    ch = json.loads(Path(a.chapters).read_text()); ch = ch["chapters"] if isinstance(ch, dict) else ch
    pc = ProbeCache(a.probe_cache)
    lg: list = []
    before = check_transitions(cues, ch, src_dur=pc)
    new = transitions(cues, ch, src_dur=pc, log=lg)
    after = check_transitions(new, ch, src_dur=pc)
    Path(a.out).write_text(json.dumps(new, indent=1))
    print("\n".join(lg))
    print(f"check_transitions: {len(before)} before, {len(after)} after; "
          f"bridge gaps: {len(bridge_gaps(cues, ch, pc))} before, {len(bridge_gaps(new, ch, pc))} after")
    for p in after: print("  ", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
