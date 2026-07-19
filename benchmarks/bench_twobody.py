"""Two-body benchmark, ported from benchmarks/fol-code/asr-twobody.fol.

TWO persistent-record accumulators (a, b : Vec2), both advanced together
every step, each reading BOTH records: every point moves 1% of the way
toward the other. Exercises the multi-accumulator fixpoint -- both
records get unboxed into scalar loop vars (four in total).

FOL's `recur` updates a and b SIMULTANEOUSLY (both new values computed
from the SAME pre-iteration a, b). Plain sequential Python statements
(`a = ...` then `b = ...`) don't have that property -- b's update sees
the ALREADY-updated a, not the pre-iteration one -- and this transform
can't recognize a staged/parallel update shape (`a = new_a` isn't a
recognized reconstruction; the right-hand side has to be a direct
constructor call). So this port is written sequentially, same as
test_multi_accumulator.py's test_two_coupled_accumulators: the
correctness bar is transformed == this port's own original Python, not
bit-identical output vs. FOL's parallel-update semantics.
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr import asr, guard
from benchmarks.harness import run_benchmark


@dataclasses.dataclass(frozen=True)
class Vec2(object):
    x: float
    y: float


def run_twobody_original(iterations):
    a = Vec2(0.0, 0.0)
    b = Vec2(1.0, 1.0)
    i = 0
    while i < iterations:
        a = Vec2(a.x + 0.01 * (b.x - a.x), a.y + 0.01 * (b.y - a.y))
        b = Vec2(b.x + 0.01 * (a.x - b.x), b.y + 0.01 * (a.y - b.y))
        i += 1
    return a, b


def run_twobody(iterations):
    a = Vec2(0.0, 0.0)
    b = Vec2(1.0, 1.0)
    i = 0
    while i < iterations:
        a = Vec2(a.x + 0.01 * (b.x - a.x), a.y + 0.01 * (b.y - a.y))
        b = Vec2(b.x + 0.01 * (a.x - b.x), b.y + 0.01 * (a.y - b.y))
        i += 1
    return a, b


run_twobody = asr(run_twobody)
assert getattr(run_twobody, "__asr_transformed__", False), "Two-body benchmark failed to transform"


def main():
    cell = guard._registry[(run_twobody.__module__, "Vec2")]
    return run_benchmark(
        "Two-body (coupled multi-accumulator relaxation)",
        run_twobody_original,
        run_twobody,
        cell,
        Vec2,
        arg=200_000,
    )


if __name__ == "__main__":
    main()
