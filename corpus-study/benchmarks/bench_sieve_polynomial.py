"""SievePolynomial benchmark, adapted from sympy's quadratic sieve
factorization implementation (corpus-study/corpus/sympy__sympy/sympy/
ntheory/qs.py -- see corpus-study/README.md's case studies table).

The REAL corpus site declines ASR classification because __init__
computes derived fields:

    class SievePolynomial:
        def __init__(self, a, b, N):
            self.a = a
            self.b = b
            self.a2 = a**2
            self.ab = 2*a*b
            self.b2 = b**2 - N

guard._infer_plain_class_fields requires a flat `self.field = param`
passthrough -- computing a2/ab/b2 from a/b/N aborts inference. Adapted
here by moving the derived-field computation to the CALL SITE (the same
pattern this project's Ballistic/Mandelbrot benchmarks already use for
FOL's own `bind`-shaped sources), which preserves the exact same field
values and the exact same downstream eval_u/eval_v computation while
making the class itself a plain, ASR-addressable dataclass.

The real qs.py loop drives `b`'s update via a Gray-code walk over a
factor base (`bit_scan1(i)`, a per-iteration sign flip indexed by
factor-base position, `b = g.b + 2*neg_pow*B[v]`) -- simplified here to
a deterministic alternating increment, since the factor-base/Gray-code
bookkeeping isn't the record-accumulator pattern under test; the
computed-field reconstruction shape that actually blocked
classification is preserved exactly, unsimplified.
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from asr import asr, guard
from benchmarks.harness import run_benchmark


@dataclasses.dataclass(frozen=True)
class SievePolynomial(object):
    a: float
    b: float
    a2: float
    ab: float
    b2: float


_N = 8051.0  # a small composite, standing in for the number being factored
_A = 5.0  # fixed sieve-polynomial parameter, matching qs.py's per-outer-loop `a`


def run_sieve_polynomial_original(iterations):
    a = _A
    b = 1.0
    g = SievePolynomial(a, b, a * a, 2.0 * a * b, b * b - _N)
    i = 0
    while i < iterations:
        neg_pow = 1.0 if i % 2 == 0 else -1.0
        b = g.b + 2.0 * neg_pow
        a = g.a
        a2 = a * a
        ab = 2.0 * a * b
        b2 = b * b - _N
        g = SievePolynomial(a, b, a2, ab, b2)
        i += 1
    return g


def run_sieve_polynomial(iterations):
    a = _A
    b = 1.0
    g = SievePolynomial(a, b, a * a, 2.0 * a * b, b * b - _N)
    i = 0
    while i < iterations:
        neg_pow = 1.0 if i % 2 == 0 else -1.0
        b = g.b + 2.0 * neg_pow
        a = g.a
        a2 = a * a
        ab = 2.0 * a * b
        b2 = b * b - _N
        g = SievePolynomial(a, b, a2, ab, b2)
        i += 1
    return g


run_sieve_polynomial = asr(run_sieve_polynomial)
assert getattr(run_sieve_polynomial, "__asr_transformed__", False), "SievePolynomial benchmark failed to transform"


def main():
    cell = guard._registry[(run_sieve_polynomial.__module__, "SievePolynomial")]
    return run_benchmark(
        "SievePolynomial (sympy qs.py, computed-field dataclass)",
        run_sieve_polynomial_original,
        run_sieve_polynomial,
        cell,
        SievePolynomial,
        arg=200_000,
    )


if __name__ == "__main__":
    main()
