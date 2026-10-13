"""Example-based tests: one test per documentary rule."""
from __future__ import annotations

from foley_helpers import analyze_all, mk_still

from opendubstage.foley.model import Interval, Layer, MusicSegment, NarrationSegment, Sequence
from opendubstage.foley.rules import Rules, check, plan


def spots(sheet):
    return sheet.by_layer(Layer.SPOT)


def test_plan_passes_its_own_checker(street_sequence, analyzer):
    sheet = plan(street_sequence, analyze_all(street_sequence, analyzer))
    assert check(sheet, street_sequence) == []


def test_portrait_gets_bed_only(street_sequence, analyzer):
    sheet = plan(street_sequence, analyze_all(street_sequence, analyzer))
    assert not [c for c in sheet.cues if c.still_id == "s1" and c.layer is not Layer.BED]
    assert any(d.still_id == "s1" and "no-sound" in d.reason for d in sheet.decisions)


def test_spot_is_never_on_the_cut(analyzer):
    seq = Sequence(stills=[mk_still(i, "cars on a highway", i * 40.0, dur=40.0) for i in range(5)], seed="x")
    sheet = plan(seq, analyze_all(seq, analyzer))
    assert spots(sheet)
    for c in spots(sheet):
        still = next(s for s in seq.stills if s.id == c.still_id)
        assert c.at - still.start >= Rules().min_from_cut_s
        assert c.end <= still.end - Rules().end_before_cut_s + 1e-9


def test_at_most_one_spot_per_still(analyzer):
    seq = Sequence(stills=[mk_still(0, "cars on the road and wine glasses raised in a toast", 0.0, dur=30.0)])
    sheet = plan(seq, analyze_all(seq, analyzer))
    assert len(spots(sheet)) == 1


def test_no_spots_under_music_up(analyzer):
    stills = [mk_still(i, "cars on a highway", i * 10.0, dur=10.0) for i in range(6)]
    seq = Sequence(stills=stills, music=[MusicSegment(0.0, 60.0, level="up")])
    sheet = plan(seq, analyze_all(seq, analyzer))
    assert spots(sheet) == []
    assert sheet.by_layer(Layer.BED), "the bed stays under music"


def test_music_under_dialogue_does_not_block(analyzer):
    stills = [mk_still(i, "cars on a highway", i * 40.0, dur=40.0) for i in range(3)]
    seq = Sequence(stills=stills, music=[MusicSegment(0.0, 120.0, level="under")])
    assert spots(plan(seq, analyze_all(seq, analyzer)))


def test_no_spots_over_live_sync_sound(analyzer):
    stills = [mk_still(i, "cars on a highway", i * 10.0, dur=10.0) for i in range(6)]
    seq = Sequence(stills=stills, live=[Interval(0.0, 60.0)])
    assert spots(plan(seq, analyze_all(seq, analyzer))) == []


def test_footage_with_sync_audio_is_left_alone(analyzer):
    seq = Sequence(stills=[mk_still(0, "cars on a highway", 0.0, dur=20.0, kind="footage", has_sync_audio=True)])
    sheet = plan(seq, analyze_all(seq, analyzer))
    assert spots(sheet) == [] and sheet.by_layer(Layer.TEXTURE) == []


def test_pre_onset_guard(analyzer):
    # narration starts 1.0 s into the still: the jittered slot (0.5-1.1 s) would mask the first word
    seq = Sequence(stills=[mk_still(0, "cars on a highway", 0.0, dur=2.0)],
                   narration=[NarrationSegment(1.0, 5.0)])
    sheet = plan(seq, analyze_all(seq, analyzer))
    for c in spots(sheet):
        assert not (c.at < 1.0 + Rules().first_word_guard_s and c.end > 1.0 - Rules().pre_onset_guard_s)


def test_key_line_window_is_bed_only(analyzer):
    stills = [mk_still(i, "cars on a highway near a creek", i * 8.0, dur=8.0) for i in range(10)]
    seq = Sequence(stills=stills, narration=[NarrationSegment(30.0, 36.0, key=True)])
    sheet = plan(seq, analyze_all(seq, analyzer))
    r = Rules()
    for c in sheet.cues:
        if c.layer is Layer.BED:
            continue
        assert not (c.at < 36.0 + r.key_post_s and c.end > 30.0 - r.key_pre_s), c


def test_density_under_narration(analyzer):
    # a car in every 6 s still for 3 minutes, all under narration
    stills = [mk_still(i, f"car number {i} on a highway", i * 6.0) for i in range(30)]
    seq = Sequence(stills=stills, narration=[NarrationSegment(0.0, 180.0)])
    sheet = plan(seq, analyze_all(seq, analyzer), Rules(same_class_gap_s=0.0, repeat_window_s=0.0))
    ss = spots(sheet)
    assert ss
    for c in ss:
        assert sum(1 for d in ss if c.at - 60 < d.at <= c.at) <= 4
    assert check(sheet, seq, Rules(same_class_gap_s=0.0, repeat_window_s=0.0)) == []


def test_density_free_is_higher_but_bounded(analyzer):
    stills = [mk_still(i, f"car number {i} on a highway", i * 6.0) for i in range(30)]
    seq = Sequence(stills=stills)
    r = Rules(same_class_gap_s=0.0, repeat_window_s=0.0)
    ss = spots(plan(seq, analyze_all(seq, analyzer), r))
    worst = max(sum(1 for d in ss if c.at - 60 < d.at <= c.at) for c in ss)
    assert 4 < worst <= r.max_spots_per_min_free


def test_same_sample_never_repeats_within_window(analyzer):
    stills = [mk_still(i, "cars on a highway", i * 40.0, dur=40.0) for i in range(12)]
    seq = Sequence(stills=stills)
    sheet = plan(seq, analyze_all(seq, analyzer))
    ss = spots(sheet)
    # 3 variants, 10 min window, 40 s stills => at most 3 car spots in any 10 minutes
    for c in ss:
        same = [d for d in ss if d.variant == c.variant and 0 < c.at - d.at < 600]
        assert not same
    assert len(ss) <= 3


def test_textures_never_overlap(analyzer):
    caps = ["a creek in the woods", "birds over a creek", "a creek again", "children playing in a field"]
    stills = [mk_still(i, c, i * 15.0, dur=15.0) for i, c in enumerate(caps)]
    seq = Sequence(stills=stills)
    tex = sorted(plan(seq, analyze_all(seq, analyzer)).by_layer(Layer.TEXTURE), key=lambda c: c.at)
    assert tex
    for a, b in zip(tex, tex[1:]):
        assert b.at >= a.end


def test_designed_silence_is_total(analyzer):
    stills = [mk_still(i, "cars on a highway", i * 10.0, dur=10.0) for i in range(6)]
    seq = Sequence(stills=stills, silences=[Interval(20.0, 30.0)])
    sheet = plan(seq, analyze_all(seq, analyzer))
    assert all(not (c.at < 30.0 and c.end > 20.0) for c in sheet.cues)
    assert check(sheet, seq) == []


def test_beds_crossfade_between_chapters(analyzer):
    stills = [mk_still(0, "a pine forest", 0.0, 20.0, chapter="one"),
              mk_still(1, "a busy city street", 20.0, 20.0, chapter="two")]
    seq = Sequence(stills=stills)
    beds = plan(seq, analyze_all(seq, analyzer)).by_layer(Layer.BED)
    assert [b.sound_class for b in beds] == ["forest", "city_street"]
    assert beds[1].at < beds[0].end, "second bed starts before the first ends"


def test_transition_budget_per_chapter(analyzer):
    stills = [mk_still(i, "a quiet room", i * 10.0, 10.0, transition="door_close") for i in range(4)]
    seq = Sequence(stills=stills)
    sheet = plan(seq, analyze_all(seq, analyzer))
    assert len(sheet.by_layer(Layer.TRANSITION)) == 1


def test_hero_spot_is_louder_but_below_dialogue(analyzer):
    seq = Sequence(stills=[mk_still(0, "cars on a highway", 0.0, 10.0, hero=True),
                           mk_still(1, "cars on a highway", 60.0, 10.0)])
    ss = spots(plan(seq, analyze_all(seq, analyzer)))
    assert ss[0].target_lufs == -28.0 and ss[1].target_lufs == -32.0


def test_deterministic_for_a_seed(street_sequence, analyzer):
    a = plan(street_sequence, analyze_all(street_sequence, analyzer))
    b = plan(street_sequence, analyze_all(street_sequence, analyzer))
    assert [c.to_dict() for c in a.cues] == [c.to_dict() for c in b.cues]


def test_checker_catches_hand_edits(street_sequence, analyzer):
    sheet = plan(street_sequence, analyze_all(street_sequence, analyzer))
    s = spots(sheet)[0]
    still = next(x for x in street_sequence.stills if x.id == s.still_id)
    s.at = still.start  # move it onto the cut
    s.target_lufs = -20.0  # and make it louder than dialogue - 4 LU
    errs = check(sheet, street_sequence)
    assert any("cut" in e for e in errs) and any("louder" in e for e in errs)


def test_cue_without_own_space_takes_the_chapter_room(analyzer):
    stills = [mk_still(0, "a freight train at a station", 0.0, 15.0, chapter="town"),
              mk_still(1, "a busy city street", 15.0, 15.0, chapter="town")]
    seq = Sequence(stills=stills)
    tex = plan(seq, analyze_all(seq, analyzer)).by_layer(Layer.TEXTURE)
    assert tex and tex[0].room == "street"


def test_texture_end_never_rounds_past_a_block():
    """Regression (found by the property test): a texture truncated at a live block's start must not overrun it
    by one ulp. With these values at + (end - at) > end in binary floating point."""
    from opendubstage.foley.analysis import KeywordAnalyzer
    a, b = 6.044, 26.82581278276368
    seq = Sequence(stills=[mk_still(0, "a freight train across open country", a, dur=b - a),
                           mk_still(1, "a street", b, dur=5.0, has_sync_audio=True)], seed="0")
    sheet = plan(seq, analyze_all(seq, KeywordAnalyzer()))
    tex = sheet.by_layer(Layer.TEXTURE)
    assert tex and tex[0].end <= b
    assert check(sheet, seq) == []
