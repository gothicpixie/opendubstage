"""Round-trip one full mono 24-bit Broadcast WAV through a SoundCraft session."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from shutil import copyfile, rmtree, which
from struct import unpack
import subprocess
from tempfile import TemporaryDirectory

import soundfile as sf

from opendubstage.restore._wave_metadata import _wave_chunks


SESSION = "dialogue.scraft"
REFERENCE = "reference.wav"
CHECKSUM = "reference.sha256"
BLOCK_BYTES = 1024 * 1024


def _bwf_info(path):
    """Return the sample rate, frame count and recording start in samples."""
    with path.open("rb") as wav:
        container, chunks = _wave_chunks(wav)
        bext = [chunk for chunk in chunks if chunk.tag == b"bext"]
        if container is None or len(bext) != 1 or bext[0].size < 602:
            raise ValueError("the source must be a RIFF or RF64 Broadcast WAV with one bext chunk")
        wav.seek(bext[0].offset + 338)
        (start,) = unpack("<Q", wav.read(8))
    info = sf.info(path)
    if info.channels != 1 or info.subtype != "PCM_24" or info.frames == 0:
        raise ValueError("the source must be a nonempty mono 24-bit PCM Broadcast WAV")
    if start + info.frames >= 1 << 63:
        raise ValueError("the BWF time reference is outside SoundCraft's sample timeline")
    return info.samplerate, info.frames, start


def _checksum(path):
    digest = sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(executable, *args):
    result = subprocess.run([executable, "run", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"SoundCraft failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout


def _clip(state, rate, frames, start):
    message = "the SoundCraft project must contain one full mono dialogue clip"
    tracks = state["tracks"]
    sources = state["sources"]
    if len(tracks) != 1 or len(sources) != 1:
        raise ValueError(message)
    track, source = tracks[0], sources[0]
    clips = track["clips"]
    if (track["format"] != "Mono" or track["playlists"] != 1 or len(clips) != 1
            or source["channels"] != 1 or source["frames"] != frames or not source["loaded"]):
        raise ValueError(message)
    clip = clips[0]
    if state["sample_rate"] != rate:
        raise ValueError("the SoundCraft session sample rate changed")
    if clip["start"] != start:
        raise ValueError("the clip no longer starts at its original BWF timeline position")
    if clip["offset"] != 0 or clip["length"] != frames or clip["end"] != start + frames:
        raise ValueError("the clip was trimmed or its length changed")
    if clip["source"] != source["id"] or clip["muted"] or track["mute"]:
        raise ValueError("the clip source or mute state changed")
    if clip["gain_db"] or clip["fade_in"] or clip["fade_out"]:
        raise ValueError("the clip has gain changes or fades")
    return clip["id"]


def prepare(source, workspace, executable):
    """Create a new project and keep an authoritative, byte-for-byte copy of the source."""
    rate, frames, start = _bwf_info(source)
    workspace.mkdir()
    try:
        reference = workspace / REFERENCE
        copyfile(source, reference)
        if _bwf_info(reference) != (rate, frames, start):
            raise ValueError("the source changed while it was being copied")
        (workspace / CHECKSUM).write_text(_checksum(reference) + "\n", encoding="ascii")
        result = _run(executable,
                      "--cmd", "session.new=" + json.dumps({"sample_rate": rate, "bit_depth": 24}),
                      "--cmd", "file.import_audio=" + json.dumps({"path": str(reference), "at": start}),
                      "--save", str(workspace / SESSION), "--inspect", "--compact")
        _clip(json.loads(result.splitlines()[-1]), rate, frames, start)
    except Exception:
        rmtree(workspace, ignore_errors=True)
        raise
    return workspace / SESSION


def export(workspace, output, executable):
    """Verify the reopened SoundCraft clip and return the original WAV bytes."""
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    session = workspace / SESSION
    project = session.read_bytes()
    if json.loads(project)["timecode_start"] != 0:
        raise ValueError("the SoundCraft session timecode start changed")
    with TemporaryDirectory(dir=output.parent) as scratch:
        directory = Path(scratch)
        reference = directory / REFERENCE
        copyfile(workspace / REFERENCE, reference)
        expected = (workspace / CHECKSUM).read_text(encoding="ascii").strip()
        if _checksum(reference) != expected:
            raise ValueError("the project's Broadcast WAV reference changed")
        rate, frames, start = _bwf_info(reference)
        result = _run(executable, "--in", str(session), "--inspect", "--compact")
        clip_id = _clip(json.loads(result.splitlines()[-1]), rate, frames, start)

        clips = directory / "clips"
        result = _run(executable, "--in", str(session), "--cmd",
                      "file.export_clips=" + json.dumps({"dir": str(clips), "clips": [clip_id]}),
                      "--inspect", "--compact")
        if session.read_bytes() != project:
            raise ValueError("the SoundCraft project was saved during export; try again")
        if _clip(json.loads(result.splitlines()[-1]), rate, frames, start) != clip_id:
            raise ValueError("the SoundCraft clip changed during export")
        files = list(clips.glob("*.wav"))
        if len(files) != 1:
            raise ValueError("SoundCraft did not export exactly one WAV clip")
        with sf.SoundFile(reference) as original, sf.SoundFile(files[0]) as rendered:
            if (rendered.channels, rendered.subtype, rendered.samplerate, rendered.frames) != (
                1, "PCM_24", rate, frames
            ):
                raise ValueError("SoundCraft changed the audio format or sample count")
            while rendered.tell() < frames:
                expected_audio = original.read(65536, dtype="int32")
                received = rendered.read(65536, dtype="int32")
                if not (len(received) and received.shape == expected_audio.shape
                        and (received == expected_audio).all()):
                    raise ValueError("SoundCraft changed the audio samples")
        # The verified PCM is identical. Use the original container so every metadata chunk,
        # including fields SoundCraft does not understand, survives without re-encoding.
        os.link(reference, output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(prog="opendubstage soundcraft", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare", help="import a cleaned BWF into a new SoundCraft project")
    setup.add_argument("source", type=Path, help="cleaned mono 24-bit Broadcast WAV")
    setup.add_argument("workspace", type=Path, help="new project directory")
    finish = commands.add_parser("export", help="verify the saved clip and write a metadata-complete WAV")
    finish.add_argument("workspace", type=Path, help="project directory created by prepare")
    finish.add_argument("output", type=Path, help="new output file")
    for command in (setup, finish):
        command.add_argument("--soundcraft-bin", default="soundcraft-cli", help="path to soundcraft-cli")
    args = parser.parse_args(argv)
    executable = which(args.soundcraft_bin)
    if executable is None:
        parser.error(f"cannot find {args.soundcraft_bin}; install SoundCraft CLI or use --soundcraft-bin")
    try:
        workspace = args.workspace.absolute()
        if args.command == "prepare":
            path = prepare(args.source, workspace, executable)
        else:
            path = export(workspace, args.output.absolute(), executable)
    except (OSError, ValueError, sf.LibsndfileError) as exc:
        parser.error(str(exc))
    except (KeyError, IndexError, TypeError):
        parser.error("unrecognized SoundCraft project or CLI response")
    print(path)
    return 0
