"""Co-moments benchmark, ported from benchmarks/fol-code/asr-comoments.fol.

Welford-style online covariance of two constant streams (x=1, y=2). The
accumulator {n, mx, my, cxy} is a FOUR-field record rebuilt every
element, using running divisions (dx/n, dy/n) -- a real, recognizable
streaming-statistics kernel.
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr import asr, guard
from benchmarks.harness import run_benchmark


@dataclasses.dataclass(frozen=True)
class Comoments(object):
    n: float
    mx: float
    my: float
    cxy: float


def run_comoments_original(iterations):
    st = Comoments(0.0, 0.0, 0.0, 0.0)
    i = 0
    while i < iterations:
        n1 = st.n + 1.0
        dx = 1.0 - st.mx
        mx1 = st.mx + dx / n1
        dy = 2.0 - st.my
        my1 = st.my + dy / n1
        dy2 = 2.0 - my1
        st = Comoments(n1, mx1, my1, st.cxy + dx * dy2)
        i += 1
    return st


def run_comoments(iterations):
    st = Comoments(0.0, 0.0, 0.0, 0.0)
    i = 0
    while i < iterations:
        n1 = st.n + 1.0
        dx = 1.0 - st.mx
        mx1 = st.mx + dx / n1
        dy = 2.0 - st.my
        my1 = st.my + dy / n1
        dy2 = 2.0 - my1
        st = Comoments(n1, mx1, my1, st.cxy + dx * dy2)
        i += 1
    return st


run_comoments = asr(run_comoments)
assert getattr(run_comoments, "__asr_transformed__", False), "Co-moments benchmark failed to transform"


def main():
    cell = guard._registry[(run_comoments.__module__, "Comoments")]
    return run_benchmark(
        "Co-moments (Welford-style streaming statistics, four fields)",
        run_comoments_original,
        run_comoments,
        cell,
        Comoments,
        arg=200_000,
    )


if __name__ == "__main__":
    main()
