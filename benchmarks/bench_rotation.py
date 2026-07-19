"""Rotation benchmark, ported from benchmarks/fol-code/asr-rotation.fol.

A unit 2-vector rotated N times by a fixed angle (theta = 0.1 rad),
threaded through an inlinable helper (rotate) whose reconstruction reads
BOTH fields and multiplies them together -- a coupled, multiplicative
update, distinct in shape from Particle's independent additive one.
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr import asr, guard
from benchmarks.harness import run_benchmark


@dataclasses.dataclass(frozen=True)
class Rot(object):
    re: float
    im: float


def rotate(z):
    return Rot(
        z.re * 0.9950041652780258 - z.im * 0.09983341664682815,
        z.re * 0.09983341664682815 + z.im * 0.9950041652780258,
    )


def run_rotation_original(iterations):
    z = Rot(1.0, 0.0)
    i = 0
    while i < iterations:
        z = rotate(z)
        i += 1
    return z


def run_rotation(iterations):
    z = Rot(1.0, 0.0)
    i = 0
    while i < iterations:
        z = rotate(z)
        i += 1
    return z


run_rotation = asr(run_rotation)
assert getattr(run_rotation, "__asr_transformed__", False), "Rotation benchmark failed to transform"


def main():
    cell = guard._registry[(run_rotation.__module__, "Rot")]
    return run_benchmark(
        "Rotation (inlined helper, coupled multiplicative update)",
        run_rotation_original,
        run_rotation,
        cell,
        Rot,
        arg=200_000,
    )


if __name__ == "__main__":
    main()
