"""Ballistic benchmark, ported from benchmarks/fol-code/asr-projectile.fol.

A projectile (x, y, vy) advanced by semi-implicit Euler under gravity.
FOL's version binds the fresh velocity `nvy` once (via `bind`) and reuses
it in two field expressions -- exercising ASR's peeling of a single-form
`bind` layer. try_transform's inlining only recognizes a bare `return
<reconstruction>` helper body (no intermediate `bind`-style local), so
this ports as a direct inline loop body instead of a separate helper
function: an ordinary Python local (`nvy`) computed once before the
reconstruction assignment gets field-substituted for free by the same
generic pass every other non-reconstruction statement goes through --
no engine change needed, and it's a closer match to `bind`'s "computed
once, reused" semantics than duplicating the expression would be.
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr import asr, guard
from benchmarks.harness import run_benchmark


@dataclasses.dataclass(frozen=True)
class State3(object):
    x: float
    y: float
    vy: float


def run_ballistic_original(iterations):
    s = State3(0.0, 0.0, 20.0)
    i = 0
    while i < iterations:
        nvy = s.vy - 0.098
        s = State3(s.x + 1.0, s.y + nvy, nvy)
        i += 1
    return s


def run_ballistic(iterations):
    s = State3(0.0, 0.0, 20.0)
    i = 0
    while i < iterations:
        nvy = s.vy - 0.098
        s = State3(s.x + 1.0, s.y + nvy, nvy)
        i += 1
    return s


run_ballistic = asr(run_ballistic)
assert getattr(run_ballistic, "__asr_transformed__", False), "Ballistic benchmark failed to transform"


def main():
    cell = guard._registry[(run_ballistic.__module__, "State3")]
    return run_benchmark(
        "Ballistic (three-field, inter-field coupling via a local temp)",
        run_ballistic_original,
        run_ballistic,
        cell,
        State3,
        arg=200_000,
    )


if __name__ == "__main__":
    main()
