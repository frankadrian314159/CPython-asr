"""Particle benchmark, ported from benchmarks/fol-code/asr-particle.fol.

FOL's version threads the accumulator through an inlinable helper
(update-particle); v1 of this port excludes interprocedural inlining
(the plan's locked-in scope decision), so the update is inlined directly
in the loop body here -- the same simplification FOL's own paper applies
before sec:inline is introduced.
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr import asr, guard
from benchmarks.harness import run_benchmark


@dataclasses.dataclass(frozen=True)
class Particle(object):
    x: float
    y: float


def run_particle_original(iterations):
    p = Particle(0.0, 0.0)
    i = 0
    while i < iterations:
        p = Particle(p.x + 0.1, p.y + 0.2)
        i += 1
    return p


def run_particle(iterations):
    p = Particle(0.0, 0.0)
    i = 0
    while i < iterations:
        p = Particle(p.x + 0.1, p.y + 0.2)
        i += 1
    return p


run_particle = asr(run_particle)
assert getattr(run_particle, "__asr_transformed__", False), "Particle benchmark failed to transform"


def main():
    cell = guard._registry[(run_particle.__module__, "Particle")]
    return run_benchmark(
        "Particle (full reconstruction)",
        run_particle_original,
        run_particle,
        cell,
        Particle,
        arg=200_000,
    )


if __name__ == "__main__":
    main()
