"""Biquad benchmark, ported from benchmarks/fol-code/asr-biquad.fol.

A Direct-Form-I biquad IIR section filtering a constant input. The
filter state {x1, x2, y1, y2} is a FOUR-field record rebuilt every
sample. Stable coefficients (b = .1,.2,.1 ; a1 = -.9, a2 = .2), so the
output settles to a bounded DC value.
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr import asr, guard
from benchmarks.harness import run_benchmark


@dataclasses.dataclass(frozen=True)
class Biquad(object):
    x1: float
    x2: float
    y1: float
    y2: float


def run_biquad_original(iterations):
    st = Biquad(0.0, 0.0, 0.0, 0.0)
    i = 0
    while i < iterations:
        xin = 1.0
        y = (0.1 * xin + 0.2 * st.x1 + 0.1 * st.x2 + 0.9 * st.y1) - 0.2 * st.y2
        st = Biquad(xin, st.x1, y, st.y1)
        i += 1
    return st


def run_biquad(iterations):
    st = Biquad(0.0, 0.0, 0.0, 0.0)
    i = 0
    while i < iterations:
        xin = 1.0
        y = (0.1 * xin + 0.2 * st.x1 + 0.1 * st.x2 + 0.9 * st.y1) - 0.2 * st.y2
        st = Biquad(xin, st.x1, y, st.y1)
        i += 1
    return st


run_biquad = asr(run_biquad)
assert getattr(run_biquad, "__asr_transformed__", False), "Biquad benchmark failed to transform"


def main():
    cell = guard._registry[(run_biquad.__module__, "Biquad")]
    return run_benchmark(
        "Biquad (four-field IIR filter state)",
        run_biquad_original,
        run_biquad,
        cell,
        Biquad,
        arg=200_000,
    )


if __name__ == "__main__":
    main()
