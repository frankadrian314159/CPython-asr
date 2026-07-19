"""Lorenz benchmark, ported from benchmarks/fol-code/asr-lorenz.fol.

Semi-explicit Euler integration of the Lorenz system (sigma=10, rho=28,
beta=8/3, dt=0.01). The accumulator {x, y, z} is a THREE-field record
rebuilt every step. The dynamics are chaotic but the trajectory stays on
the (bounded) attractor.
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr import asr, guard
from benchmarks.harness import run_benchmark


@dataclasses.dataclass(frozen=True)
class Lvec3(object):
    x: float
    y: float
    z: float


def run_lorenz_original(iterations):
    p = Lvec3(1.0, 1.0, 1.0)
    i = 0
    while i < iterations:
        dx = 10.0 * (p.y - p.x)
        dy = p.x * (28.0 - p.z) - p.y
        dz = p.x * p.y - 2.6666667 * p.z
        p = Lvec3(p.x + dx * 0.01, p.y + dy * 0.01, p.z + dz * 0.01)
        i += 1
    return p


def run_lorenz(iterations):
    p = Lvec3(1.0, 1.0, 1.0)
    i = 0
    while i < iterations:
        dx = 10.0 * (p.y - p.x)
        dy = p.x * (28.0 - p.z) - p.y
        dz = p.x * p.y - 2.6666667 * p.z
        p = Lvec3(p.x + dx * 0.01, p.y + dy * 0.01, p.z + dz * 0.01)
        i += 1
    return p


run_lorenz = asr(run_lorenz)
assert getattr(run_lorenz, "__asr_transformed__", False), "Lorenz benchmark failed to transform"


def main():
    cell = guard._registry[(run_lorenz.__module__, "Lvec3")]
    return run_benchmark(
        "Lorenz (chaotic ODE integrator, three fields)",
        run_lorenz_original,
        run_lorenz,
        cell,
        Lvec3,
        arg=200_000,
    )


if __name__ == "__main__":
    main()
