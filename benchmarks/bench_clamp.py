"""Clamp benchmark, ported from benchmarks/fol-code/asr-clamp.fol.

Exercises branch-shaped reconstruction (_try_branch_reconstruction):
each iteration either resets x to 0.0 (boundary clamp) or advances x/y
normally, so the loop's single reconstruction site is an if/else, not an
unconditional constructor call.
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr import asr, guard
from benchmarks.harness import run_benchmark


@dataclasses.dataclass(frozen=True)
class ClampPoint(object):
    x: float
    y: float


def run_clamp_original(n):
    p = ClampPoint(0.0, 0.0)
    i = 0
    while i < n:
        if p.x > 100.0:
            p = ClampPoint(0.0, p.y)
        else:
            p = ClampPoint(p.x + 1.0, p.y + 0.5)
        i += 1
    return p


def run_clamp(n):
    p = ClampPoint(0.0, 0.0)
    i = 0
    while i < n:
        if p.x > 100.0:
            p = ClampPoint(0.0, p.y)
        else:
            p = ClampPoint(p.x + 1.0, p.y + 0.5)
        i += 1
    return p


run_clamp = asr(run_clamp)
assert getattr(run_clamp, "__asr_transformed__", False), "Clamp benchmark failed to transform"


def main():
    cell = guard._registry[(run_clamp.__module__, "ClampPoint")]
    return run_benchmark(
        "Clamp (if/else branch-shaped reconstruction)",
        run_clamp_original,
        run_clamp,
        cell,
        ClampPoint,
        arg=200_000,
    )


if __name__ == "__main__":
    main()
