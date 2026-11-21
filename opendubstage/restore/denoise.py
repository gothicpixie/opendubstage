"""Denoise dialogue WAVs with the standalone DeepFilterNet 3 executable.

The upstream v0.5.6 CLI drops the fixed STFT and model delay with
``--compensate-delay`` but doesn't flush it, and skips incomplete hops.
Feed trailing silence for both, then crop from sample zero after compensation.
Use a temporary extra channel to keep the model advancing through silence.
See upstream ``libDF/src/bin/enhance_wav.rs`` and ``libDF/src/tract.rs`` at v0.5.6.
"""
from __future__ import annotations

import argparse
import json
from math import gcd, sqrt
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

from ._wave_metadata import write_cleaned_wave


MODEL_RATE = 48_000
HOP = 480
# STFT window minus hop (960 - 480), plus two model lookahead hops.
DELAY = 1440
# The upstream process() returns before advancing its state below mean-square 1e-7.
GUARD_AMPLITUDE = 0.0004
WAVE_SAMPLE_FORMATS = {"PCM_U8", "PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"}
FLOAT_SAMPLE_FORMATS = {"FLOAT", "DOUBLE"}


def _resample(audio, source_rate, target_rate):
    if source_rate == target_rate:
        return audio
    divisor = gcd(source_rate, target_rate)
    return resample_poly(audio, target_rate // divisor, source_rate // divisor, axis=0)


def denoise_file(src, dst, *, executable=None):
    """Write a WAV with the source rate, channel count and frame count."""
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise ValueError("input and output must be different files")
    if executable is None:
        binary = shutil.which("deep-filter-f32") or shutil.which("deep-filter")
    else:
        binary = shutil.which(executable)
    if binary is None:
        name = executable or "deep-filter-f32 or deep-filter"
        raise ValueError(f"cannot find {name}; run scripts/install_deepfilternet.sh or use --deep-filter-bin")
    with sf.SoundFile(src) as recording:
        audio = recording.read(dtype="float32", always_2d=True)
        rate = recording.samplerate
        subtype = "PCM_16"
        if recording.format in ("WAV", "WAVEX", "RF64") and recording.subtype in WAVE_SAMPLE_FORMATS:
            subtype = recording.subtype
    frames, channels = audio.shape
    if not np.isfinite(audio).all():
        raise ValueError("input contains non-finite samples")

    with TemporaryDirectory(prefix=".ods-denoise-", dir=dst.parent) as directory:
        temp = Path(directory)
        if frames:
            model_audio = _resample(audio, rate, MODEL_RATE)
            model_frames = len(model_audio)
            aligned_frames = model_frames + (-model_frames % HOP)
            backend_channels = channels + 1
            padded = np.zeros((aligned_frames + DELAY, backend_channels), dtype="float32")
            padded[:model_frames, :channels] = model_audio
            # The guard channel alone clears the gate even for a silent input hop.
            # Independent masks keep it out of the dialogue channels.
            guard = GUARD_AMPLITUDE * sqrt(backend_channels)
            padded[::2, -1] = guard
            padded[1::2, -1] = -guard
            staged = temp / "input.wav"
            sf.write(staged, padded, MODEL_RATE, subtype="FLOAT")
            enhanced_dir = temp / "enhanced"
            run = subprocess.run(
                [binary, "--compensate-delay", "--output-dir", str(enhanced_dir), str(staged), "--reduce-mask", "0"],
                capture_output=True, text=True,
            )
            if run.returncode:
                message = run.stderr.strip() or run.stdout.strip()
                raise ValueError(f"deep-filter exited with status {run.returncode}: {message}")
            with sf.SoundFile(enhanced_dir / staged.name) as result:
                if subtype not in ("PCM_U8", "PCM_16") and result.subtype not in FLOAT_SAMPLE_FORMATS:
                    raise ValueError(
                        f"{subtype} output requires floating-point backend audio, got {result.subtype}; "
                        "build deep-filter-f32 with scripts/install_deepfilternet.sh and use --deep-filter-bin if needed"
                    )
                enhanced_rate = result.samplerate
                enhanced = result.read(dtype="float32", always_2d=True)
            if enhanced_rate != MODEL_RATE or enhanced.shape != (aligned_frames, backend_channels):
                raise ValueError(
                    f"unexpected deep-filter output: expected {aligned_frames} frames, {backend_channels} channels "
                    f"at {MODEL_RATE} Hz, got {enhanced.shape[0]} frames, {enhanced.shape[1]} channels "
                    f"at {enhanced_rate} Hz"
                )
            if not np.isfinite(enhanced).all():
                raise ValueError("deep-filter output contains non-finite samples")
            audio = _resample(enhanced[:model_frames, :channels], MODEL_RATE, rate)[:frames]
            if audio.shape != (frames, channels):
                raise ValueError("resampling failed to restore the original frame count")

        output = temp / "output.wav"
        write_cleaned_wave(src, output, audio, rate, subtype=subtype)
        output.replace(dst)
    return {"sample_rate": rate, "frames": frames, "channels": channels}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="opendubstage denoise", description=__doc__.split("\n\n")[0])
    parser.add_argument("inp", help="input dialogue WAV")
    parser.add_argument("out", help="output WAV")
    parser.add_argument("--backend", choices=("deepfilternet3",), default="deepfilternet3")
    parser.add_argument("--deep-filter-bin", help="path to the backend (default: deep-filter-f32, then deep-filter)")
    args = parser.parse_args(argv)
    try:
        report = denoise_file(args.inp, args.out, executable=args.deep_filter_bin)
    except (OSError, ValueError, sf.LibsndfileError) as exc:
        parser.error(str(exc))
    print(json.dumps({"out": args.out, "backend": args.backend, **report}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
