"""Documentary placement rules: candidates -> cue sheet.

Pure Python, deterministic for a given seed, no audio dependencies. The planner
is conservative by construction: every rule can only *remove* a candidate, and
`check()` re-verifies every invariant independently on any cue sheet,
including hand-edited ones. See docs/rules.md for the reasoning and sources
behind each number.
"""
from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from dataclasses import dataclass, field

from .analysis import chapter_beds
from .levels import target_lufs
from .model import Cue, Decision, Interval, Layer, SceneAnalysis, Sequence, Still
from .taxonomy import SOUND_CLASSES, TRANSITION_CLASSES, bed_prompt, ROOM_FOR_BED


@dataclass(frozen=True)
class Rules:
    # placement relative to the picture
    spot_offset_s: float = 0.8          # real sounds do not coincide with edits
    spot_jitter_s: float = 0.3
    min_from_cut_s: float = 0.4         # never on (or right next to) the cut frame
    end_before_cut_s: float = 0.1       # a spot belongs to its picture and ends before the next cut
    min_spot_dur_s: float = 0.5
    # density
    max_spots_per_still: int = 1
    max_spots_per_min_narration: int = 4
    max_spots_per_min_free: int = 6
    min_spot_gap_s: float = 5.0
    same_class_gap_s: float = 30.0
    repeat_window_s: float = 600.0      # never the same variant twice in 10 minutes
    variants_per_class: int = 3
    # dialogue protection
    pre_onset_guard_s: float = 0.5      # nothing in the 0.5 s before a narration onset
    first_word_guard_s: float = 0.3     # ...nor on the first syllable
    key_pre_s: float = 4.0              # bed only from 4 s before a key line...
    key_post_s: float = 3.0             # ...to 3 s after it (the breath)
    # textures
    texture_min_s: float = 10.0
    texture_max_s: float = 40.0
    texture_fade_s: float = 3.0
    # beds
    bed_crossfade_s: float = 5.0
    # transitions
    max_transitions_per_chapter: int = 1
    # fades for one-shots
    spot_fade_in_s: float = 0.03
    spot_fade_out_s: float = 0.3


@dataclass
class CueSheet:
    cues: list[Cue] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    dialogue_lufs: float = -24.0
    duration: float = 0.0

    def by_layer(self, layer: Layer) -> list[Cue]:
        return [c for c in self.cues if c.layer is layer]


# ---------------------------------------------------------------- helpers

def _rng(seed: str, key: str) -> random.Random:
    h = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def _chapter_runs(stills: list[Still]) -> list[list[Still]]:
    """Contiguous runs of stills that share a chapter label."""
    runs: list[list[Still]] = []
    for s in stills:
        if runs and runs[-1][0].chapter == s.chapter:
            runs[-1].append(s)
        else:
            runs.append([s])
    return runs


def blocked_for_spots(seq: Sequence, r: Rules, layer: Layer = Layer.SPOT) -> list[tuple[Interval, str]]:
    """Every interval in which no spot (or, with `layer=TEXTURE`, no texture) may sound, with the reason.

    Textures are low and continuous, so the per-onset guard does not apply to them;
    live sound, music up, designed silences and key-line windows do."""
    out: list[tuple[Interval, str]] = []
    out += [(iv, "live sync sound") for iv in seq.live]
    out += [(iv, "music up") for iv in seq.music if iv.level == "up"]
    out += [(iv, "designed silence") for iv in seq.silences]
    for s in seq.stills:
        if s.has_sync_audio:
            out.append((Interval(s.start, s.end), "live sync sound"))
    for n in seq.narration:
        if layer is Layer.SPOT:
            out.append((Interval(max(0.0, n.start - r.pre_onset_guard_s), n.start + r.first_word_guard_s),
                        "narration onset"))
        if n.key:
            out.append((Interval(max(0.0, n.start - r.key_pre_s), n.end + r.key_post_s), "key line"))
    return out


def _first_block(blocks, start: float, end: float) -> str | None:
    for iv, why in blocks:
        if iv.overlaps(start, end):
            return why
    return None


def _under_narration(seq: Sequence, t0: float, t1: float) -> bool:
    return any(n.overlaps(t0, t1) for n in seq.narration)


def _subtract(span: Interval, holes: list[Interval]) -> list[Interval]:
    parts = [span]
    for h in sorted(holes, key=lambda i: i.start):
        nxt = []
        for p in parts:
            if not h.overlaps(p.start, p.end):
                nxt.append(p)
                continue
            if h.start > p.start:
                nxt.append(Interval(p.start, h.start))
            if h.end < p.end:
                nxt.append(Interval(h.end, p.end))
        parts = nxt
    return [p for p in parts if p.dur > 1e-6]


class _VariantPool:
    """Rotate variants so the same sample never repeats inside the window."""

    def __init__(self, n: int, window: float):
        self.n, self.window = n, window
        self.last: dict[tuple[str, int], float] = {}

    def pick(self, cls: str, t: float, rng: random.Random) -> int | None:
        free = [v for v in range(self.n) if t - self.last.get((cls, v), -1e18) >= self.window]
        if not free:
            return None
        return rng.choice(free)

    def use(self, cls: str, v: int, t: float) -> None:
        self.last[(cls, v)] = t


# ---------------------------------------------------------------- planner

def plan(seq: Sequence, analyses: dict[str, SceneAnalysis], rules: Rules | None = None) -> CueSheet:
    r = rules or Rules()
    sheet = CueSheet(dialogue_lufs=seq.dialogue_lufs, duration=seq.end)
    blocks = blocked_for_spots(seq, r)
    tblocks = blocked_for_spots(seq, r, Layer.TEXTURE)
    pool = _VariantPool(r.variants_per_class, r.repeat_window_s)
    n = 0

    def nid(prefix: str) -> str:
        nonlocal n
        n += 1
        return f"{prefix}{n:04d}"

    runs = _chapter_runs(seq.stills)
    beds = chapter_beds(seq.stills, analyses)

    def room_of(s: Still, a: SceneAnalysis) -> str | None:
        """The still's own space if it has one, else the chapter's (a train far off sits in the town's air)."""
        return a.room or ROOM_FOR_BED.get(beds[s.chapter][0])

    # 1. beds: one per chapter run, crossfaded, cut only by designed silences
    for i, run in enumerate(runs):
        bed, tod = beds[run[0].chapter]
        xf = r.bed_crossfade_s if i > 0 else 0.0
        span = Interval(max(0.0, run[0].start - xf), run[-1].end)
        for part in _subtract(span, list(seq.silences)):
            sheet.cues.append(Cue(nid("b"), Layer.BED, bed, part.start, part.dur,
                                  target_lufs(Layer.BED, seq.dialogue_lufs),
                                  fade_in=min(r.bed_crossfade_s, part.dur / 2),
                                  fade_out=min(r.bed_crossfade_s if i < len(runs) - 1 else 2.0, part.dur / 2),
                                  prompt=bed_prompt(bed, tod), room=ROOM_FOR_BED.get(bed)))

    # 2. transitions: hand-requested only, <= N per chapter run, straddling the cut
    transitions: list[Cue] = []
    for run in runs:
        used = 0
        for s in run:
            if not s.transition:
                continue
            cls = TRANSITION_CLASSES.get(s.transition)
            if cls is None:
                sheet.decisions.append(Decision(s.id, s.transition, False, "unknown transition class"))
                continue
            if used >= r.max_transitions_per_chapter:
                sheet.decisions.append(Decision(s.id, cls.key, False, "transition budget for chapter spent"))
                continue
            at = max(0.0, s.start - cls.dur * 0.5)
            why = _first_block([b for b in blocks if b[1] != "live sync sound"], at, at + cls.dur)
            if why:
                sheet.decisions.append(Decision(s.id, cls.key, False, f"blocked: {why}"))
                continue
            c = Cue(nid("x"), Layer.TRANSITION, cls.key, at, cls.dur,
                    target_lufs(Layer.TRANSITION, seq.dialogue_lufs),
                    fade_in=r.spot_fade_in_s, fade_out=r.spot_fade_out_s, still_id=s.id, prompt=cls.prompt)
            transitions.append(c)
            used += 1
            sheet.decisions.append(Decision(s.id, cls.key, True, "hand-requested transition"))
    sheet.cues += transitions

    # 3. textures: at most one at a time; they change with the scene, not the still
    tex_end = -1e18
    for run in runs:
        ch_end = run[-1].end
        for s in run:
            a = analyses.get(s.id)
            if a is None or a.no_sound_reason or s.start < tex_end:
                continue
            cands = [c for c in a.candidates if c.layer is Layer.TEXTURE]
            if not cands:
                continue
            cand = max(cands, key=lambda c: c.confidence)
            at = s.start + r.spot_offset_s
            if _first_block(tblocks, at, at + r.texture_min_s):
                sheet.decisions.append(Decision(s.id, cand.sound_class, False,
                                                f"blocked: {_first_block(tblocks, at, at + r.texture_min_s)}"))
                continue
            end = min(at + r.texture_max_s, ch_end)
            for iv, _why in tblocks:   # stop before the next blocked interval
                if iv.start >= at:
                    end = min(end, iv.start)
            if end - at < r.texture_min_s:
                sheet.decisions.append(Decision(s.id, cand.sound_class, False, "too little room for a texture"))
                continue
            v = pool.pick(cand.sound_class, at, _rng(seq.seed, s.id + "t"))
            if v is None:
                sheet.decisions.append(Decision(s.id, cand.sound_class, False, "all variants used in repeat window"))
                continue
            pool.use(cand.sound_class, v, at)
            dur = end - at
            while at + dur > end:   # float rounding must not push the texture's end past a block's start
                dur = math.nextafter(dur, 0.0)
            sheet.cues.append(Cue(nid("t"), Layer.TEXTURE, cand.sound_class, at, dur,
                                  target_lufs(Layer.TEXTURE, seq.dialogue_lufs),
                                  fade_in=r.texture_fade_s, fade_out=r.texture_fade_s,
                                  still_id=s.id, prompt=cand.prompt, variant=v, room=room_of(s, a)))
            sheet.decisions.append(Decision(s.id, cand.sound_class, True, "texture"))
            tex_end = end

    # 4. spots: <= 1 per still, placed off the cut, inside every density budget
    spots: list[Cue] = []
    for s in seq.stills:
        a = analyses.get(s.id)
        if a is None:
            continue
        if a.no_sound_reason:
            sheet.decisions.append(Decision(s.id, None, False, f"no-sound class: {a.no_sound_reason}"))
            continue
        cands = sorted((c for c in a.candidates if c.layer is Layer.SPOT), key=lambda c: -c.confidence)
        if not cands:
            continue
        rng = _rng(seq.seed, s.id)
        placed = False
        last_reason = ""
        for cand in cands[:3]:
            for at in (s.start + r.spot_offset_s + rng.uniform(-r.spot_jitter_s, r.spot_jitter_s),
                       s.start + s.dur * 0.5):
                at = max(at, s.start + r.min_from_cut_s)
                dur = min(cand.dur, s.end - r.end_before_cut_s - at)
                reason = _spot_reject(seq, r, blocks, spots + transitions, cand.sound_class, at, dur)
                if reason:
                    last_reason = reason
                    continue
                v = pool.pick(cand.sound_class, at, rng)
                if v is None:
                    last_reason = "all variants used in repeat window"
                    continue
                pool.use(cand.sound_class, v, at)
                spots.append(Cue(nid("s"), Layer.SPOT, cand.sound_class, round(at, 3), round(dur, 3),
                                 target_lufs(Layer.SPOT, seq.dialogue_lufs, hero=s.hero),
                                 fade_in=r.spot_fade_in_s, fade_out=min(r.spot_fade_out_s, dur / 2),
                                 still_id=s.id, prompt=cand.prompt, variant=v, room=room_of(s, a)))
                sheet.decisions.append(Decision(s.id, cand.sound_class, True, "spot"))
                placed = True
                break
            if placed:
                break
        if not placed:
            sheet.decisions.append(Decision(s.id, cands[0].sound_class, False, last_reason or "no slot"))
    sheet.cues += spots
    sheet.cues.sort(key=lambda c: (c.at, c.layer.value))
    return sheet


def _spot_reject(seq: Sequence, r: Rules, blocks, placed: list[Cue], cls: str, at: float, dur: float) -> str | None:
    if dur < r.min_spot_dur_s:
        return "too close to the next cut"
    why = _first_block(blocks, at, at + dur)
    if why:
        return f"blocked: {why}"
    for c in placed:
        gap = abs(c.at - at)
        if gap < r.min_spot_gap_s:
            return "too close to another spot"
        if c.sound_class == cls and gap < r.same_class_gap_s:
            return "same sound class too recently"
    spots = [(c.at, c.end) for c in placed if c.layer is Layer.SPOT] + [(at, at + dur)]
    # the new spot must not overflow its own minute, nor the minute of any later spot
    for t, e in spots:
        if not (at <= t < at + 60.0):
            continue
        limit = _rate_limit(seq, r, t, e)
        if sum(1 for u, _ in spots if t - 60.0 < u <= t) > limit:
            return f"density budget ({limit}/min) spent"
    return None


def _rate_limit(seq: Sequence, r: Rules, at: float, end: float) -> int:
    if _under_narration(seq, at - 60.0, end):
        return r.max_spots_per_min_narration
    return r.max_spots_per_min_free


# ---------------------------------------------------------------- independent checker

def check(sheet: CueSheet, seq: Sequence, rules: Rules | None = None) -> list[str]:
    """Re-verify the documentary invariants on any cue sheet. Returns violations (empty = pass)."""
    r = rules or Rules()
    errs: list[str] = []
    blocks = blocked_for_spots(seq, r)
    tblocks = blocked_for_spots(seq, r, Layer.TEXTURE)
    stills = {s.id: s for s in seq.stills}
    spots = sorted(sheet.by_layer(Layer.SPOT), key=lambda c: c.at)
    textures = sorted(sheet.by_layer(Layer.TEXTURE), key=lambda c: c.at)

    per_still = Counter(c.still_id for c in spots)
    for sid, k in per_still.items():
        if k > r.max_spots_per_still:
            errs.append(f"{sid}: {k} spots on one still")
    for c in spots:
        s = stills.get(c.still_id or "")
        if s is None:
            errs.append(f"{c.id}: spot without a still")
            continue
        if c.at < s.start + r.min_from_cut_s - 1e-6:
            errs.append(f"{c.id}: spot within {r.min_from_cut_s}s of the cut")
        if c.end > s.end - r.end_before_cut_s + 1e-3:
            errs.append(f"{c.id}: spot runs over the next cut")
        why = _first_block(blocks, c.at, c.end)
        if why:
            errs.append(f"{c.id}: spot in a blocked interval ({why})")
        if c.target_lufs > sheet.dialogue_lufs - 4.0 + 1e-6:
            errs.append(f"{c.id}: spot louder than dialogue - 4 LU")
    for a, b in zip(spots, spots[1:]):
        if b.at - a.at < r.min_spot_gap_s - 1e-6:
            errs.append(f"{a.id}/{b.id}: spots closer than {r.min_spot_gap_s}s")
    for c in spots:
        limit = _rate_limit(seq, r, c.at, c.end)
        k = sum(1 for d in spots if c.at - 60.0 < d.at <= c.at)
        if k > limit:
            errs.append(f"{c.id}: {k} spots in the minute ending here (limit {limit})")
    for a, b in zip(textures, textures[1:]):
        if b.at < a.end - 1e-6:
            errs.append(f"{a.id}/{b.id}: two textures at once")
    for c in textures:
        why = _first_block(tblocks, c.at, c.end)
        if why:
            errs.append(f"{c.id}: texture in a blocked interval ({why})")
    seen: dict[tuple[str, int], float] = {}
    for c in sorted(spots + textures, key=lambda c: c.at):
        k = (c.sound_class, c.variant)
        if k in seen and c.at - seen[k] < r.repeat_window_s - 1e-6:
            errs.append(f"{c.id}: same sample repeated within {r.repeat_window_s}s")
        seen[k] = c.at
    for c in sheet.cues:
        if c.layer is not Layer.BED and any(iv.overlaps(c.at, c.end) for iv in seq.silences):
            errs.append(f"{c.id}: sound inside a designed silence")
        if c.layer is Layer.BED and any(iv.overlaps(c.at, c.end) for iv in seq.silences):
            errs.append(f"{c.id}: bed inside a designed silence")
        if c.target_lufs >= sheet.dialogue_lufs:
            errs.append(f"{c.id}: target at or above dialogue level")
    tr = Counter()
    runs = _chapter_runs(seq.stills)
    run_of = {s.id: i for i, run in enumerate(runs) for s in run}
    for c in sheet.by_layer(Layer.TRANSITION):
        tr[run_of.get(c.still_id or "", -1)] += 1
    for k, v in tr.items():
        if v > r.max_transitions_per_chapter:
            errs.append(f"chapter run {k}: {v} transitions")
    return errs
