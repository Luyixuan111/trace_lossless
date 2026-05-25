"""Tests for timing helpers."""

from __future__ import annotations

import time

import pytest

from src import synthetic, timing


def test_median_time_positive() -> None:
    def work() -> None:
        time.sleep(0.001)

    med = timing.median_time(work, repeats=3, warmup=0)
    assert med > 0


def test_median_time_rejects_zero_repeats() -> None:
    with pytest.raises(ValueError, match="repeats"):
        timing.median_time(lambda: None, repeats=0)


def test_benchmark_codec_keys() -> None:
    trace = synthetic.smooth(1000, seed=0)
    stats = timing.benchmark_codec(trace, repeats=3)
    assert stats["n"] == 1000
    assert stats["compressed_bytes"] > 0
    assert stats["encode_s"] > 0
    assert stats["decode_s"] > 0
    assert "platform" in stats["system"]
