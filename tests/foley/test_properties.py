"""Property-based tests: for any timeline, the planner's output satisfies every invariant.

`check()` is an independent re-implementation of the rules as assertions, so this
is "plan agrees with spec" over thousands of random timelines, not over the
handful of examples in test_rules.py.
"""
from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from opendubstage.foley.analysis import KeywordAnalyzer
from opendubstage.foley.levels import LEVEL_OFFSETS_LU
from opendubstage.foley.model import Interval, Layer, MusicSegment, NarrationSegment, Sequence, Still
from opendubstage.foley.rules import Rules, check, plan

CAPTIONS = [
    "cars on a highway", "a portrait of a man", "a creek in the woods", "people walking on a gravel road",
    "wine glasses raised in a toast", "a freight train across open country", "children playing in a field",
    "a blurry photo of a dog", "a busy city street with traffic", "a boat at the dock on a lake",
    "a quiet kitchen interior", "birds over a mountain ridge", "a campfire at night", "", "a typewriter on a desk",
]


@st.composite
def sequences(draw):
    n = draw(st.integers(1, 40))
    durs = draw(st.lists(st.floats(0.8, 30.0), min_size=n, max_size=n))
    stills, t = [], 0.0
    chapters = draw(st.lists(st.sampled_from("abc"), min_size=n, max_size=n))
    for i, d in enumerate(durs):
        stills.append(Still(
            id=f"s{i}", start=t, dur=d, caption=draw(st.sampled_from(CAPTIONS)), chapter=chapters[i],
            hero=draw(st.booleans()), has_sync_audio=draw(st.booleans()) and draw(st.booleans()),
            transition=draw(st.sampled_from([None, None, None, "door_close", "car_pass", "nonsense"])),
            time_of_day=draw(st.sampled_from([None, "day", "dusk", "night"])),
        ))
        t += d
    total = t

    def ivs(kind, maxn):
        out = []
        for _ in range(draw(st.integers(0, maxn))):
            a = draw(st.floats(0, max(total - 0.1, 0)))
            b = a + draw(st.floats(0.1, 40))
            out.append(kind(a, b))
        return out

    narration = [NarrationSegment(iv.start, iv.end, key=draw(st.booleans()) and draw(st.booleans()))
                 for iv in ivs(Interval, 8)]
    music = [MusicSegment(iv.start, iv.end, level=draw(st.sampled_from(["up", "under"]))) for iv in ivs(Interval, 3)]
    return Sequence(stills=stills, narration=narration, music=music, live=ivs(Interval, 2),
                    silences=ivs(Interval, 1), seed=draw(st.text(min_size=1, max_size=5)))


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(sequences())
def test_planner_output_always_passes_checker(seq):
    a = KeywordAnalyzer()
    sheet = plan(seq, {s.id: a.analyze(s) for s in seq.stills})
    assert check(sheet, seq) == []


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(sequences())
def test_nothing_reaches_dialogue_level(seq):
    a = KeywordAnalyzer()
    sheet = plan(seq, {s.id: a.analyze(s) for s in seq.stills})
    loudest = max(LEVEL_OFFSETS_LU.values())
    for c in sheet.cues:
        assert c.target_lufs <= seq.dialogue_lufs + loudest
        assert c.dur > 0


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(sequences(), st.floats(0.0, 1.0))
def test_stricter_rules_never_add_spots(seq, _):
    """Monotonicity: tightening a density rule can only remove spots."""
    a = KeywordAnalyzer()
    an = {s.id: a.analyze(s) for s in seq.stills}
    loose = plan(seq, an, Rules())
    tight = plan(seq, an, Rules(max_spots_per_min_narration=1, max_spots_per_min_free=1, min_spot_gap_s=20.0))
    assert len(tight.by_layer(Layer.SPOT)) <= len(loose.by_layer(Layer.SPOT))
