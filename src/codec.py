"""Lossless compress/decompress harness for float32 traces (Week 4 skeleton)."""

from __future__ import annotations

import struct

import numpy as np
import zstandard as zstd

from .io import Trace, validate_trace

MAGIC = b"TRCE"
VERSION = 1
FLAG_ZSTD = 1
_HEADER_SIZE = 10  # magic(4) + version(1) + flags(1) + n_samples(4)


def compress(trace: Trace, level: int = 3, use_zstd: bool = True) -> bytes:
    """Compress a trace to a self-describing byte blob (bit-exact round-trip)."""
    x = validate_trace(trace)
    payload = x.tobytes()
    flags = 0
    if use_zstd:
        cctx = zstd.ZstdCompressor(level=level)
        payload = cctx.compress(payload)
        flags |= FLAG_ZSTD

    header = MAGIC + struct.pack("<BBI", VERSION, flags, len(x))
    return header + payload


def decompress(data: bytes) -> Trace:
    """Restore a trace from bytes produced by compress."""
    if len(data) < _HEADER_SIZE:
        raise ValueError("compressed data too short for header")

    magic = data[:4]
    if magic != MAGIC:
        raise ValueError(f"invalid magic {magic!r}; expected {MAGIC!r}")

    version, flags, n_samples = struct.unpack("<BBI", data[4:_HEADER_SIZE])
    if version != VERSION:
        raise ValueError(f"unsupported container version {version}")

    payload = data[_HEADER_SIZE:]
    if flags & FLAG_ZSTD:
        dctx = zstd.ZstdDecompressor()
        payload = dctx.decompress(payload)

    expected_len = n_samples * 4
    if len(payload) != expected_len:
        raise ValueError(
            f"payload length {len(payload)} != expected {expected_len} for n={n_samples}"
        )

    x = np.frombuffer(payload, dtype=np.float32).copy()
    return validate_trace(x)
