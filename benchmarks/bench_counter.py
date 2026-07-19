"""Counter benchmark, ported from benchmarks/fol-code/asr-counter.fol.

The simplest possible case: a single-field record advanced through what
was, in FOL, an inlinable helper (bump-ctr); inlined directly here for
the same reason as bench_particle.py (no interprocedural inlining in
v1). Isolates the pass's fixed per-iteration overhead from field count
and domain-specific arithmetic, same role as FOL's own Counter benchmark.
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr import asr, guard
from benchmarks.harness import run_benchmark


@dataclasses.dataclass(frozen=True)
class Ctr(object):
    n: float


def run_counter_original(iterations):
    c = Ctr(0.0)
    i = 0
    while i < iterations:
        c = Ctr(c.n + 1.0)
        i += 1
    return c


def run_counter(iterations):
    c = Ctr(0.0)
    i = 0
    while i < iterations:
        c = Ctr(c.n + 1.0)
        i += 1
    return c


run_counter = asr(run_counter)
assert getattr(run_counter, "__asr_transformed__", False), "Counter benchmark failed to transform"


def main():
    cell = guard._registry[(run_counter.__module__, "Ctr")]
    return run_benchmark(
        "Counter (single field, simplest case)",
        run_counter_original,
        run_counter,
        cell,
        Ctr,
        arg=200_000,
    )


if __name__ == "__main__":
    main()
