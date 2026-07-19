"""GridPosition benchmark, adapted from pymunk's own performance benchmark
suite (corpus-study/corpus/viblo__pymunk/benchmarks/chipmunk.py's
`Pyramid`/`MostlyStaticMultiBody` scene-construction loops, which lay out
a grid of physics bodies for a stress-test scene).

The REAL corpus site uses pymunk.Vec2d's own operator overloading:

    position = pymunk.Vec2d(0, 0)
    for j in range(M):
        position = pymunk.Vec2d(-N * a, position.y)
        for i in range(N):
            ...
            position += (2 * a, 0)
        position -= (0, 2 * a)

`position += (2*a, 0)` is `Vec2d.__iadd__`/`__add__` operator dispatch on
a bare Name target (`ast.AugAssign`), not a plain `p = ClassName(...)`
reconstruction or a `p.field = ...` mutation -- a third accumulator-
update shape this project's transform doesn't recognize at all. Adapted
here two ways: the `+=`/`-=` operator calls become direct field
mutation (`position.x = ...`), which also sidesteps a SEPARATE
restriction reconstruction mode has that mutate mode doesn't --
_analyze_loop_body declines outright when an accumulator has more than
one reconstruction site in its governing loop, and this pattern
genuinely has three (before, inside, and after the inner loop) once
nested two loops deep. Mutate mode's escape check has no such
restriction (any number of field writes, anywhere in the loop body,
including nested loops), so this is also a more natural fit for a
loop shape whose original form was already mutation-flavored (`+=`),
not reconstruction-flavored. The real nested-grid-layout control flow
(outer loop resets x per row, inner loop advances x per column, a
decrement after each row) is preserved exactly.
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from asr import asr, guard
from benchmarks.harness import run_benchmark

_A = 0.5  # matches chipmunk.py's own `a = 0.5` half-extent constant


@dataclasses.dataclass
class Vec2d(object):  # not frozen -- mutate mode; see module docstring
    x: float
    y: float


def run_grid_position_original(size):
    n = size
    m = size
    position = Vec2d(0.0, 0.0)
    j = 0
    while j < m:
        position.x = -n * _A
        i = 0
        while i < n:
            position.x = position.x + 2.0 * _A
            i += 1
        position.y = position.y - 2.0 * _A
        j += 1
    return position


def run_grid_position(size):
    n = size
    m = size
    position = Vec2d(0.0, 0.0)
    j = 0
    while j < m:
        position.x = -n * _A
        i = 0
        while i < n:
            position.x = position.x + 2.0 * _A
            i += 1
        position.y = position.y - 2.0 * _A
        j += 1
    return position


run_grid_position = asr(run_grid_position)
assert getattr(run_grid_position, "__asr_transformed__", False), "GridPosition benchmark failed to transform"


def main():
    cell = guard._registry[(run_grid_position.__module__, "Vec2d")]
    return run_benchmark(
        "GridPosition (pymunk chipmunk.py, += operator rewritten to field mutation)",
        run_grid_position_original,
        run_grid_position,
        cell,
        Vec2d,
        arg=450,  # 450*450 ~= 200,000 inner-loop iterations, matching the other benchmarks' scale
    )


if __name__ == "__main__":
    main()
