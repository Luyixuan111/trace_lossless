"""Bit-exact compress/decompress round-trip tests."""

from __future__ import annotations

import numpy as np
import pytest

from src import codec, synthetic


@pytest.mark.parametrize(
    "trace",
    [
        synthetic.smooth(10_000, seed=0),
        synthetic.oscillatory(10_000, seed=1),
        np.random.default_rng(42).standard_normal(10_000).astype(np.float32),
    ],
    ids=["smooth", "oscillatory", "random"],
)
def test_roundtrip_bit_exact(trace: np.ndarray) -> None:
    blob = codec.compress(trace)
    restored = codec.decompress(blob)
    assert np.array_equal(trace, restored)


def test_roundtrip_without_zstd() -> None:
    trace = synthetic.smooth(256, seed=2)
    blob = codec.compress(trace, use_zstd=False)
    restored = codec.decompress(blob)
    assert np.array_equal(trace, restored)


def test_compress_is_deterministic() -> None:
    trace = synthetic.smooth(512, seed=3)
    a = codec.compress(trace)
    b = codec.compress(trace)
    assert a == b
