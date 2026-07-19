"""Phase benchmark, ported from benchmarks/fol-code/asr-phase.fol.

Exercises match/case literal-dispatch reconstruction
(_try_match_reconstruction): the iteration index modulo 3 selects among
three distinct reconstructions, one per phase, with a mandatory `case
_:` default -- FOL's own `case`, dispatching a single key against a
fixed set of literal values.
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr import asr, guard
from benchmarks.harness import run_benchmark


@dataclasses.dataclass(frozen=True)
class Phase(object):
    x: float
    y: float


def run_phase_original(n):
    p = Phase(0.0, 0.0)
    i = 0
    while i < n:
        match i % 3:
            case 0:
                p = Phase(p.x + 1.0, p.y)
            case 1:
                p = Phase(p.x, p.y + 2.0)
            case _:
                p = Phase(p.x + 0.5, p.y + 0.5)
        i += 1
    return p


def run_phase(n):
    p = Phase(0.0, 0.0)
    i = 0
    while i < n:
        match i % 3:
            case 0:
                p = Phase(p.x + 1.0, p.y)
            case 1:
                p = Phase(p.x, p.y + 2.0)
            case _:
                p = Phase(p.x + 0.5, p.y + 0.5)
        i += 1
    return p


run_phase = asr(run_phase)
assert getattr(run_phase, "__asr_transformed__", False), "Phase benchmark failed to transform"


def main():
    cell = guard._registry[(run_phase.__module__, "Phase")]
    return run_benchmark(
        "Phase (match/case literal-dispatch reconstruction)",
        run_phase_original,
        run_phase,
        cell,
        Phase,
        arg=200_000,
    )


if __name__ == "__main__":
    main()
