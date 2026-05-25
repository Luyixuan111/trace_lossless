#!/usr/bin/env python3
"""Baseline runner stub (Week 4). Full zstd BPS table arrives in Week 5."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as: python scripts/baseline.py from repo root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import synthetic, timing  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Baseline stub: generate trace and benchmark codec harness."
    )
    parser.add_argument(
        "--regime",
        choices=list(synthetic.REGIMES.keys()),
        default="smooth",
        help="synthetic data regime",
    )
    parser.add_argument("--n", type=int, default=10_000, help="number of samples")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--repeats", type=int, default=5, help="timing repeats (median)")
    args = parser.parse_args(argv)

    trace = synthetic.generate(args.regime, args.n, seed=args.seed)
    stats = timing.benchmark_codec(trace, repeats=args.repeats)

    bps = (8.0 * stats["compressed_bytes"]) / stats["n"]
    print(f"regime={args.regime} n={stats['n']} seed={args.seed}")
    print(f"compressed_bytes={stats['compressed_bytes']}  bps={bps:.2f}")
    print(f"encode_median_s={stats['encode_s']:.6f}  decode_median_s={stats['decode_s']:.6f}")
    print(f"system={stats['system']}")
    print()
    print("Week 5 TODO: zstd/lz4 on raw float32 bytes + results table for 3+ regimes.")


if __name__ == "__main__":
    main()
