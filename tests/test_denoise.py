"""Exercise the CLI with a local stand-in for the optional DeepFilterNet binary."""
import struct
import sys

import numpy as np
import pytest
import soundfile as sf

from opendubstage.cli import main


@pytest.fixture
def deep_filter(tmp_path):
    binary = tmp_path / "deep-filter"
    binary.write_text(f"#!{sys.executable}\n" + """
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

_, flag, out_flag, directory, src, *mask = sys.argv
assert flag == "--compensate-delay" and out_flag == "--output-dir"
assert not mask or mask == ["--reduce-mask", "0"]
audio, rate = sf.read(src, always_2d=True)
assert rate == 48000
# The v0.5.6 binary skips both output and state advancement on quiet hops.
pending = [np.zeros((480, audio.shape[1])) for _ in range(3)]
delayed = np.zeros_like(audio)
for start in range(0, len(audio) - 479, 480):
    hop = audio[start:start + 480]
    if np.mean(hop ** 2) >= 1e-7:
        pending.append(hop)
        delayed[start:start + 480] = pending.pop(0)
Path(directory).mkdir()
sf.write(Path(directory) / Path(src).name, delayed[1440:] * 0.5, rate, subtype="FLOAT")
""")
    binary.chmod(0o755)
    return binary


@pytest.mark.parametrize(("rate", "frames", "channels"), [(48000, 4999, 2), (44100, 4999, 1), (48000, 17, 1)])
def test_denoise_preserves_sync_through_the_final_sample(tmp_path, deep_filter, rate, frames, channels):
    src, dst = tmp_path / "in.wav", tmp_path / "out.wav"
    audio = np.zeros((frames, channels))
    audio[3, 0] = 0.25
    audio[-1, -1] = 0.75
    if frames > 480:
        audio[-100, 0] = 0.20
    sf.write(src, audio, rate, subtype="PCM_24")

    assert main(["denoise", str(src), str(dst), "--backend", "deepfilternet3",
                 "--deep-filter-bin", str(deep_filter)]) == 0

    result, out_rate = sf.read(dst, always_2d=True)
    assert out_rate == rate
    assert result.shape == audio.shape
    assert np.argmax(np.abs(result[:-2, 0])) == 3
    assert np.argmax(np.abs(result[:, -1])) == frames - 1
    assert result[-1, -1] == pytest.approx(0.375, abs=0.05)
    if channels == 2:
        assert np.allclose(result, audio * 0.5, atol=1 / 32768)


def test_denoise_rejects_an_unexpected_backend_length(tmp_path, deep_filter, capsys):
    src, dst = tmp_path / "in.wav", tmp_path / "out.wav"
    sf.write(src, np.zeros(1000), 48000)
    dst.write_bytes(b"existing output")
    source = deep_filter.read_text()
    deep_filter.write_text(source.replace("delayed[1440:]", "delayed[1441:]"))

    with pytest.raises(SystemExit) as exc:
        main(["denoise", str(src), str(dst), "--backend", "deepfilternet3",
              "--deep-filter-bin", str(deep_filter)])

    assert exc.value.code == 2
    assert "expected" in capsys.readouterr().err
    assert dst.read_bytes() == b"existing output"


@pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24"])
def test_denoise_rejects_a_16_bit_backend_for_24_bit_input(tmp_path, deep_filter, capsys, subtype):
    src, dst = tmp_path / "in.wav", tmp_path / "out.wav"
    sf.write(src, np.full(1000, 0.25), 48000, subtype=subtype)
    deep_filter.write_text(deep_filter.read_text().replace('subtype="FLOAT"', 'subtype="PCM_16"'))
    command = ["denoise", str(src), str(dst), "--deep-filter-bin", str(deep_filter)]

    if subtype == "PCM_16":
        assert main(command) == 0
        assert sf.info(dst).subtype == "PCM_16"
    else:
        dst.write_bytes(b"existing output")
        with pytest.raises(SystemExit) as exc:
            main(command)
        assert exc.value.code == 2
        assert "deep-filter-f32" in capsys.readouterr().err
        assert dst.read_bytes() == b"existing output"


def _chunk(tag, payload):
    return struct.pack("<4sI", tag, len(payload)) + payload + b"\0" * (len(payload) % 2)


def _wav_chunks(blob):
    """Read the simple RIFF and RF64 fixtures used in these tests."""
    rf64 = blob[:4] == b"RF64"
    size_pos, size_fmt = (20, "<Q") if rf64 else (4, "<I")
    assert struct.unpack_from(size_fmt, blob, size_pos)[0] == len(blob) - 8
    offset = 12
    while offset < len(blob):
        tag, size = struct.unpack_from("<4sI", blob, offset)
        if rf64 and tag == b"data":
            assert size == 0xFFFFFFFF
            size = struct.unpack_from("<Q", blob, 28)[0]
        start = offset + 8
        yield tag, blob[start:start + size]
        offset = start + size + size % 2
    assert offset == len(blob)


@pytest.mark.parametrize(("wave_format", "bext_version"), [("WAVEX", 2), ("RF64", 1)])
def test_denoise_keeps_broadcast_timecode_and_recording_metadata(tmp_path, deep_filter, wave_format, bext_version):
    src, dst = tmp_path / "recording.wav", tmp_path / "cleaned.wav"
    audio = np.zeros((1001, 2))
    audio[501, 1] = 2_097_154 / 2**23
    rate = 96000 if wave_format == "RF64" else 48000
    sf.write(src, audio, rate, format=wave_format, subtype="PCM_24")

    bext = bytearray(602)
    bext[:9] = b"Scene 104"
    bext[256:262] = b"ODS-01"
    struct.pack_into("<QH", bext, 338, 5_000_000_123, bext_version)
    bext[348:412] = bytes(range(64))
    if bext_version == 2:
        struct.pack_into("<5h", bext, 412, -2300, 400, -100, -2000, -2100)
    history = b",T=Shot" if wave_format == "RF64" else b""
    bext.extend(f"A=PCM,F={rate},W=24".encode() + history + b"\r\n")
    info = b"INFO" + _chunk(b"INAM", b"Scene 104\0")
    cue = struct.pack("<III4sIII", 1, 42, 501, b"data", 0, 0, 501)
    labels = b"adtl" + _chunk(b"labl", struct.pack("<I", 42) + b"Slate\0")
    tags = [(b"bext", bytes(bext)), (b"iXML", b"<BWFXML />\n"), (b"axml", b"<ebuCoreMain/>"),
            (b"LIST", info), (b"cue ", cue), (b"LIST", labels)]

    source = src.read_bytes()
    parts = []
    for tag, payload in _wav_chunks(source):
        if tag == b"fmt ":
            # WAVEFORMATEXTENSIBLE speaker positions, different from the libsndfile default.
            payload = payload[:20] + struct.pack("<I", 0x30) + payload[24:]
        if tag == b"data" and wave_format == "RF64":
            parts.append(struct.pack("<4sI", tag, 0xFFFFFFFF) + payload)
        else:
            parts.append(_chunk(tag, payload))
    # Exercise metadata after the audio and payloads with odd lengths.
    source = bytearray(source[:12] + b"".join(parts) + b"".join(_chunk(*entry) for entry in tags)
                       + _chunk(b"MD5 ", bytes(16)) + _chunk(b"levl", bytes(16)))
    struct.pack_into("<Q" if wave_format == "RF64" else "<I", source,
                     20 if wave_format == "RF64" else 4, len(source) - 8)
    src.write_bytes(source)

    assert main(["denoise", str(src), str(dst), "--deep-filter-bin", str(deep_filter)]) == 0

    chunks = list(_wav_chunks(dst.read_bytes()))
    fmt = dict(chunks)[b"fmt "]
    assert sf.info(dst).format == wave_format
    assert sf.info(dst).subtype == "PCM_24"
    assert struct.unpack_from("<HI", fmt, 18) == (24, 0x30)
    assert sf.info(dst).frames == len(audio)
    assert sf.info(dst).samplerate == rate
    samples, _ = sf.read(dst, dtype="int32")
    assert np.argmax(np.abs(samples[:, 1])) == 501
    if rate == 48000:
        assert samples[501, 1] >> 8 == 1_048_577  # 24-bit integer, with a nonzero low byte.
    if bext_version == 2:
        struct.pack_into("<5H", bext, 412, *([0x7FFF] * 5))
    if wave_format == "RF64":
        bext += b"\0"  # NUL terminates odd-length coding history before the RF64 data marker.
    kept = [(tag, payload) for tag, payload in chunks if tag not in {b"ds64", b"fmt ", b"fact", b"data"}]
    assert sorted(kept) == sorted([(b"bext", bytes(bext)), *tags[1:]])
    assert [tag for tag, _ in chunks].index(b"bext") < [tag for tag, _ in chunks].index(b"data")
