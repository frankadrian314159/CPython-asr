"""Mandelbrot benchmark, ported from benchmarks/fol-code/asr-mandelbrot.fol.

Iterates the Mandelbrot map z <- z^2 + c for a fixed c inside the set
(the "rabbit" parameter -0.123 + 0.745i), whose critical orbit is
bounded. Reads both fields and multiplies them -- a coupled, quadratic
update. Direct inline loop body (locals zr/zi), same reasoning as
Ballistic: mirrors FOL's `bind` more closely than duplicating the field
reads would.
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr import asr, guard
from benchmarks.harness import run_benchmark


@dataclasses.dataclass(frozen=True)
class Cplx(object):
    re: float
    im: float


def run_mandelbrot_original(iterations):
    z = Cplx(0.0, 0.0)
    i = 0
    while i < iterations:
        zr = z.re
        zi = z.im
        z = Cplx(zr * zr - zi * zi + -0.123, 2.0 * (zr * zi) + 0.745)
        i += 1
    return z


def run_mandelbrot(iterations):
    z = Cplx(0.0, 0.0)
    i = 0
    while i < iterations:
        zr = z.re
        zi = z.im
        z = Cplx(zr * zr - zi * zi + -0.123, 2.0 * (zr * zi) + 0.745)
        i += 1
    return z


run_mandelbrot = asr(run_mandelbrot)
assert getattr(run_mandelbrot, "__asr_transformed__", False), "Mandelbrot benchmark failed to transform"


def main():
    cell = guard._registry[(run_mandelbrot.__module__, "Cplx")]
    return run_benchmark(
        "Mandelbrot (coupled quadratic update)",
        run_mandelbrot_original,
        run_mandelbrot,
        cell,
        Cplx,
        arg=200_000,
    )


if __name__ == "__main__":
    main()
