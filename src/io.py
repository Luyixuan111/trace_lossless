"""I/O and validation for 1-D float32 traces."""

from __future__ import annotations

from pathlib import Path

import numpy as np

Trace = np.ndarray  # 1-D float32, shape (n,)

TRACE_DTYPE = np.dtype(np.float32)
BYTES_PER_SAMPLE = TRACE_DTYPE.itemsize  # 4


def validate_trace(x: np.ndarray) -> Trace:
    """Return a contiguous 1-D float32 trace with no NaN/Inf.

    Policy:
    - dtype must be float32 after coercion
    - at least one sample
    - all values finite (no NaN, no Inf)
    """
    out = np.asarray(x, dtype=np.float32).reshape(-1)
    if out.size < 1:
        raise ValueError("trace must have at least one sample")
    if out.ndim != 1:
        raise ValueError("trace must be one-dimensional")
    if not np.isfinite(out).all():
        raise ValueError("trace contains NaN or Inf")
    return np.ascontiguousarray(out)


def save_trace(path: str | Path, x: Trace) -> None:
    """Write trace as raw little-endian float32 bytes (row-major, no header)."""
    trace = validate_trace(x)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(trace.tobytes())


def load_trace(path: str | Path) -> Trace:
    """Load trace from raw float32 byte file produced by save_trace."""
    path = Path(path)
    data = path.read_bytes()
    if len(data) == 0:
        raise ValueError("empty trace file")
    if len(data) % BYTES_PER_SAMPLE != 0:
        raise ValueError(
            f"file size {len(data)} is not a multiple of {BYTES_PER_SAMPLE} bytes per sample"
        )
    x = np.frombuffer(data, dtype=TRACE_DTYPE).copy()
    return validate_trace(x)
