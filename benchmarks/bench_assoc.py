"""Assoc benchmark, ported from benchmarks/fol-code/asr-assoc.fol.

Exercises partial reconstruction via `dataclasses.replace` (the `assoc`
analog) rather than a full constructor call: only `vx` changes each
iteration, x/y/vy pass through unchanged from their current scalar
values. No helper function involved in the FOL original either, so this
one ports directly with no inlining simplification needed.
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr import asr, guard
from benchmarks.harness import run_benchmark


@dataclasses.dataclass(frozen=True)
class AssocParticle(object):
    x: float
    y: float
    vx: float
    vy: float


def run_assoc_original(n):
    acc = AssocParticle(0.0, 0.0, 0.0, 0.0)
    i = 0
    while i < n:
        acc = dataclasses.replace(acc, vx=acc.vx + 0.1)
        i += 1
    return acc


def run_assoc(n):
    acc = AssocParticle(0.0, 0.0, 0.0, 0.0)
    i = 0
    while i < n:
        acc = dataclasses.replace(acc, vx=acc.vx + 0.1)
        i += 1
    return acc


run_assoc = asr(run_assoc)
assert getattr(run_assoc, "__asr_transformed__", False), "Assoc benchmark failed to transform"


def main():
    cell = guard._registry[(run_assoc.__module__, "AssocParticle")]
    return run_benchmark(
        "Assoc (partial reconstruction via dataclasses.replace)",
        run_assoc_original,
        run_assoc,
        cell,
        AssocParticle,
        arg=200_000,
    )


if __name__ == "__main__":
    main()
