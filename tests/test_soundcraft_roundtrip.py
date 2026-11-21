"""Exercise the companion against an optional SoundCraft CLI installation."""
import json
import os
import shutil
import struct
import subprocess

import numpy as np
import pytest
import soundfile as sf

from opendubstage.cli import main


@pytest.fixture
def soundcraft_bin():
    executable = shutil.which(os.environ.get("SOUNDCRAFT_CLI", "soundcraft-cli"))
    if not executable:
        pytest.skip("set SOUNDCRAFT_CLI to run the SoundCraft interchange tests")
    return executable


def make_bwf(path, container, rate):
    audio = np.zeros(1001, dtype=np.int32)
    audio[501] = 1_048_577 << 8
    sf.write(path, audio, rate, format=container, subtype="PCM_24")
    header = bytearray(602)
    header[:6] = b"Scene "
    header[256:259] = b"ODS"
    struct.pack_into("<QH", header, 338, 864_001, 2)
    xml = b"<BWFXML><PROJECT>ODS</PROJECT></BWFXML>"
    wav = bytearray(path.read_bytes())
    for tag, payload in ((b"bext", header), (b"iXML", xml)):
        wav.extend(struct.pack("<4sI", tag, len(payload)) + payload + b"\0" * (len(payload) % 2))
    if container == "RF64":
        struct.pack_into("<Q", wav, 20, len(wav) - 8)
    else:
        struct.pack_into("<I", wav, 4, len(wav) - 8)
    path.write_bytes(wav)
    return bytes(wav)


@pytest.mark.parametrize(("container", "rate"), [("WAV", 48000), ("RF64", 44100)])
def test_soundcraft_roundtrip_keeps_the_entire_mono_bwf(tmp_path, soundcraft_bin, container, rate):
    source = tmp_path / "cleaned recording.wav"
    expected = make_bwf(source, container, rate)
    work = tmp_path / "dialogue project"
    output = tmp_path / "delivered.wav"

    assert main(["soundcraft", "prepare", str(source), str(work), "--soundcraft-bin", soundcraft_bin]) == 0
    source.unlink()  # The project owns an authoritative copy even after the source is moved.
    assert main(["soundcraft", "export", str(work), str(output), "--soundcraft-bin", soundcraft_bin]) == 0

    assert output.read_bytes() == expected


def test_soundcraft_export_rejects_a_moved_clip(tmp_path, soundcraft_bin, capsys):
    source = tmp_path / "cleaned.wav"
    make_bwf(source, "WAV", 48000)
    work = tmp_path / "session"
    assert main(["soundcraft", "prepare", str(source), str(work), "--soundcraft-bin", soundcraft_bin]) == 0
    session = work / "dialogue.scraft"
    state = json.loads(subprocess.check_output(
        [soundcraft_bin, "run", "--in", str(session), "--inspect", "--compact"], text=True,
    ))
    clip = state["tracks"][0]["clips"][0]
    subprocess.run([
        soundcraft_bin, "run", "--in", str(session), "--save", str(session), "--in-place",
        "--cmd", f'edit.move_clips={{"clips":[{clip["id"]}],"by":1}}',
    ], check=True, capture_output=True)
    output = tmp_path / "incorrect.wav"

    with pytest.raises(SystemExit) as error:
        main(["soundcraft", "export", str(work), str(output), "--soundcraft-bin", soundcraft_bin])

    assert error.value.code == 2
    assert "timeline position" in capsys.readouterr().err
    assert not output.exists()


def test_soundcraft_export_rejects_same_shape_with_changed_samples(tmp_path, soundcraft_bin, capsys):
    source = tmp_path / "cleaned.wav"
    make_bwf(source, "WAV", 48000)
    work = tmp_path / "session"
    assert main(["soundcraft", "prepare", str(source), str(work), "--soundcraft-bin", soundcraft_bin]) == 0
    media = work / "Audio Files" / "reference.wav"
    audio, rate = sf.read(media)
    sf.write(media, -audio, rate, subtype=sf.info(media).subtype)
    output = tmp_path / "incorrect.wav"

    with pytest.raises(SystemExit) as error:
        main(["soundcraft", "export", str(work), str(output), "--soundcraft-bin", soundcraft_bin])

    assert error.value.code == 2
    assert "changed the audio samples" in capsys.readouterr().err
    assert not output.exists()


def test_soundcraft_export_rejects_an_altered_metadata_reference(tmp_path, soundcraft_bin, capsys):
    source = tmp_path / "cleaned.wav"
    make_bwf(source, "WAV", 48000)
    work = tmp_path / "session"
    assert main(["soundcraft", "prepare", str(source), str(work), "--soundcraft-bin", soundcraft_bin]) == 0
    reference = work / "reference.wav"
    reference.write_bytes(reference.read_bytes().replace(b"Scene ", b"Other "))
    output = tmp_path / "incorrect.wav"

    with pytest.raises(SystemExit) as error:
        main(["soundcraft", "export", str(work), str(output), "--soundcraft-bin", soundcraft_bin])

    assert error.value.code == 2
    assert "reference changed" in capsys.readouterr().err
    assert not output.exists()
