"""Timing utilities for encode/decode benchmarks."""

from __future__ import annotations

import platform
import statistics
import sys
import time
from collections.abc import Callable
from typing import Any, TypeVar

from . import codec
from .io import Trace

T = TypeVar("T")


def median_time(fn: Callable[[], T], repeats: int = 5, warmup: int = 1) -> float:
    """Return median wall-clock seconds for fn over repeats runs (after warmup)."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    for _ in range(warmup):
        fn()

    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def system_info() -> dict[str, str]:
    """Record environment metadata for experiment logs."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "machine": platform.machine(),
        "python": sys.version.split()[0],
    }


def benchmark_codec(
    trace: Trace,
    *,
    repeats: int = 5,
    zstd_level: int = 3,
) -> dict[str, Any]:
    """Benchmark compress/decompress on one trace; return summary dict."""
    blob_box: list[bytes] = []

    def encode() -> None:
        blob_box.clear()
        blob_box.append(codec.compress(trace, level=zstd_level))

    encode_s = median_time(encode, repeats=repeats)
    blob = blob_box[0]

    decode_s = median_time(lambda: codec.decompress(blob), repeats=repeats)

    return {
        "n": len(trace),
        "compressed_bytes": len(blob),
        "encode_s": encode_s,
        "decode_s": decode_s,
        "system": system_info(),
    }
