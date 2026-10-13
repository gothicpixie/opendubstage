from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from opendubstage.foley.analysis import KeywordAnalyzer
from opendubstage.foley.io import cuesheet_from_dict, cuesheet_to_dict, sequence_from_dict
from opendubstage.foley.levels import target_lufs
from opendubstage.foley.model import Layer, Still
from opendubstage.foley.post import filter_chain, render_file
from opendubstage.foley.prompts import PromptParts, build_prompt, lint
from opendubstage.foley.rules import plan
from opendubstage.foley.taxonomy import SOUND_CLASSES, TRANSITION_CLASSES, bed_prompt


# ---------------------------------------------------------------- levels
def test_level_table_relative_to_dialogue():
    assert target_lufs(Layer.BED) == -40.0
    assert target_lufs(Layer.TEXTURE) == -36.0
    assert target_lufs(Layer.SPOT) == -32.0
    assert target_lufs(Layer.SPOT, hero=True) == -28.0
    assert target_lufs(Layer.TRANSITION) == -26.0
    assert target_lufs(Layer.SPOT, dialogue_lufs=-27.0) == -35.0


# ---------------------------------------------------------------- prompts
def test_every_built_in_prompt_is_lint_clean():
    for cls in [*SOUND_CLASSES.values(), *TRANSITION_CLASSES.values()]:
        assert lint(cls.prompt) == [], (cls.key, lint(cls.prompt))
    for bed in ["forest", "city_street", "unknown"]:
        for tod in [None, "dusk", "night"]:
            assert lint(bed_prompt(bed, tod)) == []


@pytest.mark.parametrize("bad", [
    "whoosh as the camera pushes in", "a cinematic riser", "sitcom laugh track",
    "piano music in the background", "a man saying hello", "epic explosion",
])
def test_lint_rejects_cartoon_prompts(bad):
    assert lint(bad)


def test_lint_allows_negations():
    assert lint("rain on a tin roof, no music, no voices") == []


def test_prompt_template_shape():
    p = build_prompt(PromptParts("a door closing", "wooden", "in a farmhouse", "distant"))
    assert p.startswith("a door closing wooden in a farmhouse, distant,")
    assert "no music" in p and "no voices" in p


# ---------------------------------------------------------------- analysis
@pytest.mark.parametrize("caption,expected", [
    ("a busy main street with cars parked", "car_far"),
    ("a freight train at a small station", "train_far"),
    ("rowboats at a boat landing on the lake", "water_lap"),
    ("men walking down a gravel road", "steps_gravel"),
])
def test_keyword_analyzer_finds_class(caption, expected):
    a = KeywordAnalyzer().analyze(Still(id="x", start=0, dur=5, caption=caption))
    assert expected in [c.sound_class for c in a.candidates]


@pytest.mark.parametrize("caption,quality,tags", [
    ("portrait of a woman with her children", {}, []),
    ("close-up of hands", {}, []),
    ("cars on a highway", {"blurry": True}, []),
    ("cars on a highway", {}, ["no_sound"]),
])
def test_no_sound_classes(caption, quality, tags):
    a = KeywordAnalyzer().analyze(Still(id="x", start=0, dur=5, caption=caption, quality=quality, tags=tags))
    assert a.no_sound_reason and a.candidates == []


def test_objects_that_make_no_sound_get_nothing():
    a = KeywordAnalyzer().analyze(Still(id="x", start=0, dur=5, caption="a lantern and a cake on a table"))
    assert a.candidates == []


# ---------------------------------------------------------------- io
SEQ = {
    "seed": "demo",
    "stills": [
        {"id": "a", "start": 0, "dur": 8, "caption": "cars on a main street", "chapter": "town"},
        {"id": "b", "start": 8, "dur": 8, "caption": "portrait of a farmer", "chapter": "town"},
    ],
    "narration": [{"start": 9, "end": 14, "key": True}],
    "music": [{"start": 0, "end": 4, "level": "under"}],
}


def test_sequence_and_cuesheet_round_trip():
    seq = sequence_from_dict(SEQ)
    a = KeywordAnalyzer()
    sheet = plan(seq, {s.id: a.analyze(s) for s in seq.stills})
    d = cuesheet_to_dict(sheet)
    d2 = cuesheet_to_dict(cuesheet_from_dict(json.loads(json.dumps(d))))
    assert d == d2
    assert d["schema"].startswith("opendubstage.foley.cuesheet/")


def test_sequence_rejects_unknown_fields_and_duplicate_ids():
    with pytest.raises(ValueError):
        sequence_from_dict({"stills": [{"id": "a", "start": 0, "dur": 1, "colour": "red"}]})
    with pytest.raises(ValueError):
        sequence_from_dict({"stills": [{"id": "a", "start": 0, "dur": 1}, {"id": "a", "start": 1, "dur": 1}]})


# ---------------------------------------------------------------- post
def test_filter_chain_contents():
    f = filter_chain(-32.0, 2.0, room="outdoor_trees")
    assert "highpass=f=80" in f and "lowpass=f=9000" in f and "aecho" in f and "loudnorm=I=-32.0" in f
    with pytest.raises(ValueError):
        filter_chain(-32.0, 2.0, room="cathedral")
    with pytest.raises(ValueError):
        filter_chain(-32.0, 0.0)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_render_file_lands_near_target(tmp_path):
    src = tmp_path / "noise.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "anoisesrc=d=6:c=pink:a=0.3",
                    "-ar", "48000", str(src)], check=True)
    dst = render_file(src, tmp_path / "out.wav", target_lufs=-32.0, dur=5.0, room="small_room")
    out = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(dst), "-af", "ebur128", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    integrated = float(out.rsplit("I:", 1)[1].split("LUFS")[0])
    assert abs(integrated - (-32.0)) < 2.0
