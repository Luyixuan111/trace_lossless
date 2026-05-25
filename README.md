# trace_lossless

Lossless compression of float32 trace-like data (MSc dissertation: predictor + limited context + zstd on integer residuals). **Week 4** provides I/O, a bit-exact codec harness, timing, and tests.

## Requirements

- Python 3.10+
- See `requirements.txt`

## Setup

```bash
cd trace_lossless
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## NaN / Inf policy

- Traces must be **1-D `float32`** with **at least one sample**.
- **NaN and Inf are not supported**; `validate_trace()` raises `ValueError`.
- No silent replacement or filtering.

## Byte layout

- In memory: contiguous 1-D `numpy.float32`.
- On disk (`.f32`): raw `trace.tobytes()`, row-major, **native little-endian** (typical on x86/ARM64 macOS/Linux).
- Codec container: 10-byte header (`TRCE` magic + version + flags + `n`) + payload (optionally zstd-compressed raw float32 bytes).

## Commands

```bash
# Synthetic traces
python -m src.synthetic --regime smooth --n 10000 --seed 0

# Week 4 baseline stub (codec benchmark)
python scripts/baseline.py --regime smooth --n 10000

# Tests
pytest -q
```

## Layout

- `src/io.py` — validate, save/load `.f32`
- `src/codec.py` — `compress` / `decompress` (Week 4 skeleton)
- `src/timing.py` — median timing + `benchmark_codec`
- `src/synthetic.py` — smooth / oscillatory / noisy_spikes generators
- `scripts/baseline.py` — Week 5 zstd table placeholder
