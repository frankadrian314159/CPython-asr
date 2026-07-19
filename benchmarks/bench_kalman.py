"""Kalman benchmark, ported from benchmarks/fol-code/asr-kalman.fol.

A 1D constant-velocity Kalman filter processing a noiseless constant
position measurement (z=10). Carries TWO coupled record accumulators:
the state estimate {x, v} (KState) and the error covariance {p00, p01,
p11} (KCov, symmetric 2x2). Each step's predict+update reads both
records and rebuilds both -- the multi-accumulator case, unboxed by the
fixpoint (four scalars for KState/KCov's five fields total... six,
across two classes).

FOL's version binds ~13 intermediate values in one `bind` form before
recur; ported here as ordinary Python locals computed before both
reconstruction assignments, same reasoning as Ballistic/Mandelbrot:
these get field-substituted for free by the generic (non-reconstruction
statement) pass, chaining correctly across the multi-accumulator
fixpoint's two passes (verified: pass 1 unboxes KState first, rewriting
every s.field reference throughout the WHOLE loop body -- including
inside statements that feed KCov's own reconstruction -- before pass 2
scans the now-partially-rewritten body for KCov).
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr import asr, guard
from benchmarks.harness import run_benchmark


@dataclasses.dataclass(frozen=True)
class KState(object):
    x: float
    v: float


@dataclasses.dataclass(frozen=True)
class KCov(object):
    p00: float
    p01: float
    p11: float


def run_kalman_original(iterations):
    s = KState(0.0, 0.0)
    c = KCov(1.0, 0.0, 1.0)
    i = 0
    while i < iterations:
        x = s.x
        v = s.v
        p00 = c.p00
        p01 = c.p01
        p11 = c.p11
        xp = x + v
        pp00 = (p00 + 2.0 * p01) + (p11 + 0.001)
        pp01 = p01 + p11
        pp11 = p11 + 0.001
        y = 10.0 - xp
        sden = pp00 + 0.1
        k0 = pp00 / sden
        k1 = pp01 / sden
        s = KState(xp + k0 * y, v + k1 * y)
        c = KCov((1.0 - k0) * pp00, (1.0 - k0) * pp01, pp11 - k1 * pp01)
        i += 1
    return s, c


def run_kalman(iterations):
    s = KState(0.0, 0.0)
    c = KCov(1.0, 0.0, 1.0)
    i = 0
    while i < iterations:
        x = s.x
        v = s.v
        p00 = c.p00
        p01 = c.p01
        p11 = c.p11
        xp = x + v
        pp00 = (p00 + 2.0 * p01) + (p11 + 0.001)
        pp01 = p01 + p11
        pp11 = p11 + 0.001
        y = 10.0 - xp
        sden = pp00 + 0.1
        k0 = pp00 / sden
        k1 = pp01 / sden
        s = KState(xp + k0 * y, v + k1 * y)
        c = KCov((1.0 - k0) * pp00, (1.0 - k0) * pp01, pp11 - k1 * pp01)
        i += 1
    return s, c


run_kalman = asr(run_kalman)
assert getattr(run_kalman, "__asr_transformed__", False), "Kalman benchmark failed to transform"


def main():
    cell_s = guard._registry[(run_kalman.__module__, "KState")]
    cell_c = guard._registry[(run_kalman.__module__, "KCov")]
    return run_benchmark(
        "Kalman (two coupled accumulators, 13 local bind-style temps)",
        run_kalman_original,
        run_kalman,
        [cell_s, cell_c],
        [KState, KCov],
        arg=200_000,
    )


if __name__ == "__main__":
    main()
