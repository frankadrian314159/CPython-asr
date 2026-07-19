"""Bounce benchmark, ported from benchmarks/fol-code/asr-bounce.fol.

Exercises a three-way branch-shaped reconstruction (if/elif/else, Python
parses `elif` as a nested If in orelse, which _try_branch_reconstruction
handles by recursing): bounce off the high wall, bounce off the low
wall, or move normally.
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr import asr, guard
from benchmarks.harness import run_benchmark


@dataclasses.dataclass(frozen=True)
class Bounce(object):
    x: float
    y: float


def run_bounce_original(n):
    p = Bounce(0.0, 0.0)
    i = 0
    while i < n:
        if p.x > 100.0:
            p = Bounce(0.0, p.y)
        elif p.x < -100.0:
            p = Bounce(0.0, p.y)
        else:
            p = Bounce(p.x + 1.0, p.y + 0.5)
        i += 1
    return p


def run_bounce(n):
    p = Bounce(0.0, 0.0)
    i = 0
    while i < n:
        if p.x > 100.0:
            p = Bounce(0.0, p.y)
        elif p.x < -100.0:
            p = Bounce(0.0, p.y)
        else:
            p = Bounce(p.x + 1.0, p.y + 0.5)
        i += 1
    return p


run_bounce = asr(run_bounce)
assert getattr(run_bounce, "__asr_transformed__", False), "Bounce benchmark failed to transform"


def main():
    cell = guard._registry[(run_bounce.__module__, "Bounce")]
    return run_benchmark(
        "Bounce (three-way if/elif/else branch-shaped reconstruction)",
        run_bounce_original,
        run_bounce,
        cell,
        Bounce,
        arg=200_000,
    )


if __name__ == "__main__":
    main()
