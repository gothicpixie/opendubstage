from __future__ import annotations

import json
from pathlib import Path

import pytest

from opendubstage.foley.analysis_vision import VisionAnalyzer, VisionResult, labels_to_text
from opendubstage.foley.backends import SoundRequest, get_backend
from opendubstage.foley.cli import main
from opendubstage.foley.model import Still
from opendubstage.foley.render import stem_command

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "foley" / "public_domain" / "sequence.json"


# ---------------------------------------------------------------- vision analyser (runner injected)
def fake_runner(labels, face=0.0):
    return lambda path, min_conf: VisionResult(labels, face)


def test_vision_labels_feed_the_keyword_matcher():
    va = VisionAnalyzer(runner=fake_runner([("railroad", 0.4), ("train", 0.3)]))
    a = va.analyze(Still(id="x", start=0, dur=5, path="/nope.jpg"))
    assert [c.sound_class for c in a.candidates] == ["train_far"]
    assert a.candidates[0].source == "vision"


def test_vision_large_face_means_portrait():
    va = VisionAnalyzer(runner=fake_runner([("car", 0.9), ("street", 0.8)], face=0.12))
    a = va.analyze(Still(id="x", start=0, dur=5, path="/nope.jpg", caption="a family"))
    assert a.no_sound_reason and not a.candidates


def test_vision_small_faces_do_not_block():
    va = VisionAnalyzer(runner=fake_runner([("car", 0.9), ("street", 0.8)], face=0.005))
    a = va.analyze(Still(id="x", start=0, dur=5, path="/nope.jpg"))
    assert "car_far" in [c.sound_class for c in a.candidates]
    assert a.bed == "city_street"


def test_vision_without_a_path_falls_back_to_caption():
    va = VisionAnalyzer(runner=lambda *a: pytest.fail("vision should not run"))
    a = va.analyze(Still(id="x", start=0, dur=5, caption="cars on a highway"))
    assert a.candidates


def test_labels_to_text():
    assert labels_to_text([("water_body", 0.9), ("lake", 0.8)]) == "water body lake"


# ---------------------------------------------------------------- library backend
def test_library_backend_rotates_variants(tmp_path):
    for v in range(2):
        (tmp_path / f"car_far_{v}.wav").write_bytes(b"RIFF")
    be = get_backend("library", root=tmp_path)
    got = [be.generate(SoundRequest("car_far", "", 2.0, v), tmp_path).name for v in range(3)]
    assert got == ["car_far_0.wav", "car_far_1.wav", "car_far_0.wav"]
    with pytest.raises(FileNotFoundError):
        be.generate(SoundRequest("train_far", "", 2.0), tmp_path)


def test_unknown_backend():
    with pytest.raises(ValueError):
        get_backend("nope")


# ---------------------------------------------------------------- render graph
def test_stem_command_delays_each_input():
    cmd = stem_command([(Path("a.wav"), 1.5), (Path("b.wav"), 10.0)], Path("out.wav"), 30.0)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "adelay=1500|1500" in graph and "adelay=10000|10000" in graph
    assert "amix=inputs=3:normalize=0" in graph


# ---------------------------------------------------------------- CLI round trip on the public-domain example
def test_cli_plan_then_check(tmp_path, capsys):
    out = tmp_path / "cues.json"
    assert main(["plan", str(EXAMPLE), "-o", str(out)]) == 0
    sheet = json.loads(out.read_text())
    layers = {c["layer"] for c in sheet["cues"]}
    assert {"bed", "spot"} <= layers
    assert main(["check", str(EXAMPLE), str(out)]) == 0
    # a hand edit that moves a spot onto its cut must fail the check
    spot = next(c for c in sheet["cues"] if c["layer"] == "spot")
    spot["at"] = 8.0
    out.write_text(json.dumps(sheet))
    assert main(["check", str(EXAMPLE), str(out)]) == 1
    assert "cut" in capsys.readouterr().out


def test_example_never_sounds_on_the_portrait(tmp_path):
    out = tmp_path / "cues.json"
    main(["plan", str(EXAMPLE), "-o", str(out)])
    cues = json.loads(out.read_text())["cues"]
    assert not [c for c in cues if c["still_id"] == "mother"]
