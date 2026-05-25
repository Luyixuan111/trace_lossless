"""Synthetic float32 trace generators for lossless compression experiments."""

from __future__ import annotations

import argparse
from typing import Callable

import numpy as np

from .io import Trace, validate_trace


def smooth(n: int, seed: int = 0) -> Trace:
    """Slow drift plus small Gaussian noise (easy to predict with lag-1)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    x = 1.0 + 0.1 * t + 0.002 * rng.standard_normal(n)
    return validate_trace(x)


def oscillatory(n: int, seed: int = 0, period: int = 64) -> Trace:
    """Sinusoid plus small noise (structured, still predictable)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if period < 4:
        raise ValueError("period must be >= 4")
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float32)
    x = np.sin(2.0 * np.pi * t / float(period)) + 0.01 * rng.standard_normal(n)
    return validate_trace(x)


def noisy_spikes(
    n: int,
    seed: int = 0,
    spike_every: int = 500,
    spike_amp: float = 5.0,
) -> Trace:
    """Smooth trace with occasional large spikes (harder for simple predictors)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if spike_every < 1:
        raise ValueError("spike_every must be >= 1")
    x = smooth(n, seed).copy()
    spike_idx = np.arange(0, n, spike_every, dtype=np.intp)
    x[spike_idx] += np.float32(spike_amp)
    return validate_trace(x)


REGIMES: dict[str, Callable[..., Trace]] = {
    "smooth": smooth,
    "oscillatory": oscillatory,
    "noisy_spikes": noisy_spikes,
}


def generate(regime: str, n: int, seed: int = 0, **kwargs) -> Trace:
    """Generate a trace by regime name."""
    if regime not in REGIMES:
        raise KeyError(f"unknown regime {regime!r}; choose from {list(REGIMES)}")
    return REGIMES[regime](n, seed=seed, **kwargs)


def _summarize(name: str, x: Trace) -> None:
    print(f"\n=== {name} ===")
    print(f"  shape: {x.shape}, dtype: {x.dtype}")
    print(f"  min: {x.min():.6f}, max: {x.max():.6f}, mean: {x.mean():.6f}")
    print(f"  first 8 samples: {x[:8]}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic float32 traces (smooth / oscillatory / noisy_spikes)."
    )
    parser.add_argument(
        "--regime",
        choices=["all", *REGIMES.keys()],
        default="all",
        help="which generator to run (default: all)",
    )
    parser.add_argument("--n", type=int, default=10_000, help="number of samples")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument(
        "--period",
        type=int,
        default=64,
        help="sine period for oscillatory (samples)",
    )
    parser.add_argument(
        "--spike-every",
        type=int,
        default=500,
        help="spike interval for noisy_spikes",
    )
    args = parser.parse_args(argv)

    names = list(REGIMES.keys()) if args.regime == "all" else [args.regime]
    for name in names:
        extra = {}
        if name == "oscillatory":
            extra["period"] = args.period
        if name == "noisy_spikes":
            extra["spike_every"] = args.spike_every
        x = generate(name, args.n, seed=args.seed, **extra)
        _summarize(name, x)


if __name__ == "__main__":
    main()
