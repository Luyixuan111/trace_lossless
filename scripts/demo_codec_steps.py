#!/usr/bin/env python3
"""Step-by-step demo of compress / decompress (run from repo root)."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import zstandard as zstd

from src import codec
from src.codec import FLAG_ZSTD, MAGIC, VERSION, _HEADER_SIZE

# --- small trace: easy to read ---
x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
print("=" * 60)
print("INPUT TRACE (what we compress)")
print("=" * 60)
print(f"  x = {x}")
print(f"  dtype = {x.dtype}, shape = {x.shape}, n = {len(x)}")
print(f"  nbytes = {x.nbytes}  (n * 4 = {len(x) * 4})")

# --- COMPRESS steps ---
print("\n" + "=" * 60)
print("COMPRESS - step by step")
print("=" * 60)

print("\n[1] validate_trace(x)")
print(f"  output: same array, still {x}")

print("\n[2] payload = x.tobytes()")
payload_raw = x.tobytes()
print(f"  output: bytes object, len = {len(payload_raw)}")
print(f"  hex (12 bytes, 4 per float): {payload_raw.hex(' ')}")
for i in range(len(x)):
    chunk = payload_raw[i * 4 : (i + 1) * 4]
    print(f"    sample[{i}] = {x[i]!r}  ->  bytes {chunk.hex(' ')}")

print("\n[3] zstd.compress(payload)  [optional, use_zstd=True]")
cctx = zstd.ZstdCompressor(level=3)
payload_zstd = cctx.compress(payload_raw)
flags = FLAG_ZSTD
print(f"  input len  = {len(payload_raw)}")
print(f"  output len = {len(payload_zstd)}")
print(f"  flags = {flags}  (bit 0 set = used zstd)")

print("\n[4] header = TRCE + version + flags + n")
header = MAGIC + struct.pack("<BBI", VERSION, flags, len(x))
print(f"  MAGIC     = {MAGIC!r}")
print(f"  version   = {VERSION}")
print(f"  flags     = {flags}")
print(f"  n_samples = {len(x)}")
print(f"  header hex: {header.hex(' ')}  (len={len(header)})")

print("\n[5] blob = header + payload")
blob = header + payload_zstd
print(f"  total blob len = {len(blob)} = {_HEADER_SIZE} + {len(payload_zstd)}")

print("\n[6] codec.compress(x) - full function")
blob_api = codec.compress(x, use_zstd=True)
print(f"  blob len = {len(blob_api)}")

# --- DECOMPRESS steps ---
print("\n" + "=" * 60)
print("DECOMPRESS - step by step (using blob from compress)")
print("=" * 60)

data = blob_api
print(f"\n[0] input: bytes blob, len = {len(data)}")

print("\n[1] read header (first 10 bytes)")
magic = data[:4]
version, flags_d, n_samples = struct.unpack("<BBI", data[4:_HEADER_SIZE])
print(f"  magic     = {magic!r}")
print(f"  version   = {version}")
print(f"  flags     = {flags_d}")
print(f"  n_samples = {n_samples}")

print("\n[2] payload = data[10:]")
payload_in = data[_HEADER_SIZE:]
print(f"  payload len = {len(payload_in)}")

print("\n[3] if flags & ZSTD: zstd.decompress(payload)")
if flags_d & FLAG_ZSTD:
    dctx = zstd.ZstdDecompressor()
    payload_out = dctx.decompress(payload_in)
    print(f"  decompressed len = {len(payload_out)}  (expect {n_samples * 4})")
else:
    payload_out = payload_in

print("\n[4] check len(payload) == n_samples * 4")
print(f"  {len(payload_out)} == {n_samples * 4} ? {len(payload_out) == n_samples * 4}")

print("\n[5] y = np.frombuffer(payload, dtype=np.float32).copy()")
y = np.frombuffer(payload_out, dtype=np.float32).copy()
print(f"  output: {y}")

print("\n[6] codec.decompress(blob) - full function")
y_api = codec.decompress(blob_api)
print(f"  output: {y_api}")

print("\n" + "=" * 60)
print("FINAL CHECK")
print("=" * 60)
print(f"  np.array_equal(x, y_api) = {np.array_equal(x, y_api)}")
