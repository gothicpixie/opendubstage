"""The `opendubstage` entry point and each tool's command line, on small synthetic files."""
import json
import shutil

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

from opendubstage.cli import TOOLS, main

SR = 48000


def noise(seconds, seed=0, level_db=-30.0):
    x = np.random.default_rng(seed).standard_normal(int(seconds * SR))
    return x / np.sqrt(np.mean(x ** 2)) * 10 ** (level_db / 20)


def test_help_lists_every_tool(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert all(t in out for t in TOOLS)


def test_unknown_tool_is_an_error(capsys):
    assert main(["nope"]) == 2
    assert "unknown tool" in capsys.readouterr().err


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_each_tool_has_help(tool, capsys):
    with pytest.raises(SystemExit) as e:
        main([tool, "--help"])
    assert e.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_specrepair_cli_roundtrip(tmp_path):
    x = noise(4.0, level_db=-50)
    t = np.arange(int(0.2 * SR)) / SR
    n0 = int(2.0 * SR); x[n0:n0 + len(t)] += 0.1 * np.sin(2 * np.pi * 1000 * t) * np.hanning(len(t))
    src, dst, rep = tmp_path / "in.wav", tmp_path / "out.wav", tmp_path / "rep.json"
    sf.write(src, x, SR, subtype="PCM_24")
    (tmp_path / "ev.json").write_text(json.dumps([[1.9, 2.3]]))
    (tmp_path / "words.json").write_text(json.dumps([{"start": 0.5, "end": 1.2}, {"start": 3.0, "end": 3.6}]))
    rc = main(["specrepair", str(src), str(dst), "--events", str(tmp_path / "ev.json"),
               "--protected", str(tmp_path / "words.json"), "--report", str(rep)])
    assert rc == 0
    y, sr = sf.read(dst, dtype="int32")
    x0, _ = sf.read(src, dtype="int32")
    assert sr == SR and len(y) == len(x0)
    assert np.array_equal(y[:int(1.5 * SR)], x0[:int(1.5 * SR)])        # untouched far from the event
    assert isinstance(json.loads(rep.read_text()), list)


def test_ltasmatch_cli_prints_an_equalizer_chain(tmp_path, capsys):
    ref = noise(6.0, 1)
    b, a = signal.butter(2, [800, 3000], "band", fs=SR)
    other = 0.5 * ref + signal.lfilter(b, a, noise(6.0, 2))                # a mid-heavy colouration
    sf.write(tmp_path / "ref.wav", ref, SR); sf.write(tmp_path / "other.wav", other, SR)
    assert main(["ltasmatch", str(tmp_path / "ref.wav"), str(tmp_path / "other.wav")]) == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("equalizer=") or out == "anull"
    assert main(["ltasmatch", str(tmp_path / "ref.wav"), str(tmp_path / "other.wav"), "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["after"]["rms_dev_db"] <= d["before"]["rms_dev_db"]


def test_carve_cli_keeps_length_and_layout(tmp_path):
    mx = np.stack([noise(3.0, 3), noise(3.0, 4)], 1)
    dx = np.zeros(int(2.5 * SR)); dx[SR:2 * SR] = noise(1.0, 5, -20)       # shorter dialogue file is zero-padded
    sf.write(tmp_path / "mx.wav", mx, SR); sf.write(tmp_path / "dx.wav", dx, SR)
    assert main(["carve", str(tmp_path / "mx.wav"), str(tmp_path / "dx.wav"), str(tmp_path / "out.wav"),
                 "--block-s", "1"]) == 0
    y, sr = sf.read(tmp_path / "out.wav")
    assert y.shape == mx.shape and sr == SR


def test_memofix_cli(tmp_path, capsys):
    sf.write(tmp_path / "m.wav", noise(3.0, 6), SR)
    assert main(["memofix", str(tmp_path / "m.wav"), str(tmp_path / "o.wav"), "--no-eq"]) == 0
    assert json.loads(capsys.readouterr().out)["out"].endswith("o.wav")
    assert len(sf.read(tmp_path / "o.wav")[0]) == 3 * SR


def test_transitions_cli(tmp_path, capsys):
    cues = {"total": 200.0, "dx": [], "live": [],
            "mx": [{"src": "a.mp3", "ss": 0.0, "dur": 100.0, "at": 0.0, "mode": "under"}],
            "amb": [{"src": "r1.flac", "ss": 0.0, "dur": 100.0, "at": 0.0},
                    {"src": "r2.flac", "ss": 0.0, "dur": 100.0, "at": 100.0}]}
    (tmp_path / "c.json").write_text(json.dumps(cues))
    (tmp_path / "ch.json").write_text(json.dumps({"chapters": [{"t0": 0.0}, {"t0": 100.0}]}))
    (tmp_path / "durations.json").write_text(json.dumps({"a.mp3": 300.0, "r1.flac": 600.0, "r2.flac": 600.0}))
    assert main(["transitions", str(tmp_path / "c.json"), str(tmp_path / "ch.json"), str(tmp_path / "o.json"),
                 "--probe-cache", str(tmp_path / "durations.json")]) == 0
    new = json.loads((tmp_path / "o.json").read_text())
    assert new["total"] == 200.0
    assert "check_transitions" in capsys.readouterr().out


def test_qc_cli_writes_reports(tmp_path):
    sf.write(tmp_path / "mix.wav", np.stack([noise(20.0, 7, -24)] * 2, 1), SR)
    rc = main(["qc", str(tmp_path / "mix.wav"), "--json", str(tmp_path / "q.json"), "--md", str(tmp_path / "q.md")])
    assert rc in (0, 1)
    assert "metrics" in json.loads((tmp_path / "q.json").read_text())
    assert (tmp_path / "q.md").read_text().strip()


def test_foley_cli_plan_and_check(tmp_path):
    from pathlib import Path
    seq = Path(__file__).resolve().parents[1] / "examples" / "foley" / "public_domain" / "sequence.json"
    out = tmp_path / "cues.json"
    assert main(["foley", "plan", str(seq), "-o", str(out)]) == 0
    assert json.loads(out.read_text())["schema"] == "opendubstage.foley.cuesheet/0.1"
    assert main(["foley", "check", str(seq), str(out)]) == 0


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")
def test_ltasmatch_decode_fallback_reads_what_soundfile_cannot(tmp_path):
    import subprocess
    sf.write(tmp_path / "a.wav", noise(1.0, 8), SR)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp_path / "a.wav"), "-c:a", "aac",
                    str(tmp_path / "a.m4a")], check=True)
    from opendubstage.match.ltasmatch import _mono
    x, sr = _mono(tmp_path / "a.m4a")
    assert sr == 48000 and len(x) > 0.9 * SR
