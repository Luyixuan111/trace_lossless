"""Tests for trace I/O and validation."""

from __future__ import annotations

import numpy as np
import pytest

from src import io, synthetic


def test_validate_trace_rejects_nan() -> None:
    x = np.array([1.0, np.nan], dtype=np.float32)
    with pytest.raises(ValueError, match="NaN or Inf"):
        io.validate_trace(x)


def test_validate_trace_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        io.validate_trace(np.array([], dtype=np.float32))


def test_save_load_roundtrip(tmp_path) -> None:
    x = synthetic.smooth(128, seed=1)
    path = tmp_path / "trace.f32"
    io.save_trace(path, x)
    y = io.load_trace(path)
    assert np.array_equal(x, y)


def test_load_rejects_bad_file_size(tmp_path) -> None:
    path = tmp_path / "bad.f32"
    path.write_bytes(b"\x00\x00\x00")  # 3 bytes
    with pytest.raises(ValueError, match="multiple of"):
        io.load_trace(path)
