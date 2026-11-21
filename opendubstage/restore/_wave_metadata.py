"""Carry source recording metadata onto a newly rendered RIFF or RF64 WAV."""
from collections import defaultdict, deque
from shutil import copyfileobj
from struct import Struct, pack, unpack
from typing import NamedTuple

import soundfile as sf


CHUNK_HEADER = Struct("<4sI")
SIZE_SENTINEL = 0xFFFFFFFF
# These describe the recording, the channel assignment, or positions on the
# unchanged sample timeline. Audio checksums, peak envelopes and codec chunks
# are intentionally excluded.
RECORDING_CHUNKS = {b"bext", b"iXML", b"axml", b"chna", b"cue ", b"plst", b"smpl", b"inst",
                    b"cart", b"ID3 ", b"id3 ", b"_PMX", b"bxml", b"sxml", b"r64m"}


class _Chunk(NamedTuple):
    tag: bytes
    offset: int  # Payload offset
    size: int    # Payload size without the pad byte


def _wave_chunks(file):
    header = file.read(12)
    container = header[:4]
    if container not in (b"RIFF", b"RF64") or header[8:] != b"WAVE":
        return None, []

    file_size = file.seek(0, 2)
    file.seek(12)
    end = 8 + unpack("<I", header[4:8])[0]
    large_sizes = defaultdict(deque)
    if container == b"RF64":
        if file_size < 48:
            raise ValueError("truncated RF64 header")
        tag, size = CHUNK_HEADER.unpack(file.read(8))
        if tag != b"ds64" or size < 28 or 20 + size > file_size:
            raise ValueError("invalid RF64 size chunk")
        riff_size, data_size, _, entries = unpack("<QQQI", file.read(28))
        if entries > (size - 28) // 12:
            raise ValueError("invalid RF64 size table")
        large_sizes[b"data"].append(data_size)
        for _ in range(entries):
            chunk_tag, chunk_size = unpack("<4sQ", file.read(12))
            large_sizes[chunk_tag].append(chunk_size)
        end = riff_size + 8
        file.seek(20 + size + size % 2)

    if not file.tell() <= end <= file_size:
        raise ValueError("invalid WAV container size")
    chunks = []
    while file.tell() + CHUNK_HEADER.size <= end:
        tag, size = CHUNK_HEADER.unpack(file.read(CHUNK_HEADER.size))
        if size == SIZE_SENTINEL and container == b"RF64":
            if not large_sizes[tag]:
                raise ValueError(f"missing RF64 size for {tag!r}")
            size = large_sizes[tag].popleft()
        offset = file.tell()
        next_chunk = offset + size + size % 2
        if offset + size > end:
            raise ValueError(f"truncated WAV chunk {tag!r}")
        chunks.append(_Chunk(tag, offset, size))
        file.seek(next_chunk)
    return container, chunks


def _read_at(file, offset, size):
    file.seek(offset)
    return file.read(size)


def _write_metadata(source, dest, chunk, *, include_pad=False):
    declared_size = chunk.size + int(include_pad)
    if declared_size >= SIZE_SENTINEL:
        raise ValueError(f"WAV metadata chunk {chunk.tag!r} exceeds the supported size")
    dest.write(CHUNK_HEADER.pack(chunk.tag, declared_size))
    source.seek(chunk.offset)
    remaining = chunk.size
    if chunk.tag == b"bext" and chunk.size >= 602:
        prefix = source.read(422)
        # EBU Tech 3285 v2 marks unavailable loudness values with 0x7FFF.
        if prefix[346:348] == b"\x02\x00":
            prefix = prefix[:412] + pack("<5H", *([0x7FFF] * 5))
        dest.write(prefix)
        remaining -= len(prefix)
    while remaining:
        block = source.read(min(remaining, 1024 * 1024))
        if not block:
            raise ValueError(f"truncated WAV metadata {chunk.tag!r}")
        dest.write(block)
        remaining -= len(block)
    if chunk.size % 2:
        dest.write(b"\0")


def write_cleaned_wave(src, dst, audio, rate, *, subtype):
    """Write new samples and retain source metadata valid for the new audio."""
    with src.open("rb") as source:
        container, chunks = _wave_chunks(source)
        metadata = [c for c in chunks if c.tag in RECORDING_CHUNKS or (
            c.tag == b"LIST" and c.size >= 4 and _read_at(source, c.offset, 4) in (b"INFO", b"adtl")
        )]
        fmt = next((c for c in chunks if c.tag == b"fmt "), None)
        channel_mask = None
        if fmt and fmt.size >= 40 and _read_at(source, fmt.offset, 2) == b"\xFE\xFF":
            channel_mask = _read_at(source, fmt.offset + 20, 4)

        wave_format = "WAVEX" if channel_mask is not None else "WAV"
        if container == b"RF64":
            wave_format = "RF64"
        sf.write(dst, audio, rate, format=wave_format, subtype=subtype)
        if not metadata and channel_mask is None:
            return

        with dst.open("r+b") as rendered:
            _, rendered_chunks = _wave_chunks(rendered)
            if channel_mask is not None:
                generated_fmt = next(c for c in rendered_chunks if c.tag == b"fmt ")
                rendered.seek(generated_fmt.offset + 20)
                rendered.write(channel_mask)
            if not metadata:
                return

            data_start = next(c.offset - CHUNK_HEADER.size for c in rendered_chunks if c.tag == b"data")
            # libsndfile 1.2.2 can miss RF64's data marker after an odd-sized chunk.
            # Defer odd chunks, except bext, whose pad can terminate its coding history.
            trailing = []
            merged = dst.with_name("with-metadata.wav")
            with merged.open("wb") as dest:
                dest.write(_read_at(rendered, 0, data_start))
                for chunk in metadata:
                    odd_rf64 = container == b"RF64" and chunk.size % 2 != 0
                    if odd_rf64 and chunk.tag != b"bext":
                        trailing.append(chunk)
                    else:
                        _write_metadata(source, dest, chunk, include_pad=odd_rf64)
                copyfileobj(rendered, dest)
                for chunk in trailing:
                    _write_metadata(source, dest, chunk)
                riff_size = dest.tell() - 8
                if container == b"RF64":
                    dest.seek(20)  # ds64 payload starts with the 64-bit container size.
                    dest.write(pack("<Q", riff_size))
                else:
                    if riff_size >= SIZE_SENTINEL:
                        raise ValueError("WAV with metadata exceeds the RIFF size limit")
                    dest.seek(4)
                    dest.write(pack("<I", riff_size))
        merged.replace(dst)
