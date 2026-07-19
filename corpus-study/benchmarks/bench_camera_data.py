"""CameraData benchmark, adapted from arcade's camera system
(corpus-study/corpus/pythonarcade__arcade/arcade/camera/data_types.py and
.../grips/strafe.py -- see corpus-study/README.md's case studies table).

The REAL corpus site (arcade/tests/unit/camera/test_camera_controller_
methods.py) declines ASR classification for a genuine escape:

    camera_data.position = grips.strafe(camera_data, dirs)

`camera_data` is passed as a bare argument to `grips.strafe(...)` inside
the loop -- exactly the aliasing hazard the escape check exists to catch
(the Python-native instance of the Clojure/FOL papers'
`:aliased-reference` category, structurally unfixable by any recognition
fix). Adapted here by inlining `strafe`'s own logic directly into the
loop body, reading `camera_data`'s fields instead of passing the object
itself -- the same transformation a human would make to actually get
this code ASR-addressable, preserving strafe's real vector-math
computation (forward/up cross product, direction-weighted offset)
exactly. `pyglet.math.Vec3` (an external dependency this project doesn't
otherwise need) is replaced by a tiny local normalize/cross helper doing
the same 3-component arithmetic.
"""

import dataclasses
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from asr import asr, guard
from benchmarks.harness import run_benchmark


def _normalize3(v):
    x, y, z = v
    length = math.sqrt(x * x + y * y + z * z)
    if length == 0.0:
        return (0.0, 0.0, 0.0)
    return (x / length, y / length, z / length)


def _cross3(a, b):
    ax, ay, az = a
    bx, by, bz = b
    return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)


@dataclasses.dataclass(frozen=True)
class CameraData(object):
    x: float
    y: float
    z: float
    fx: float
    fy: float
    fz: float
    ux: float
    uy: float
    uz: float


_DIRECTIONS = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (0.5, 0.5))


def run_camera_data_original(iterations):
    camera_data = CameraData(0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0)
    i = 0
    while i < iterations:
        dirs = _DIRECTIONS[i % len(_DIRECTIONS)]
        # strafe(camera_data, dirs), inlined:
        _forward = _normalize3((camera_data.fx, camera_data.fy, camera_data.fz))
        _up = _normalize3((camera_data.ux, camera_data.uy, camera_data.uz))
        _right = _cross3(_forward, _up)
        offset_x = _right[0] * dirs[0] + _up[0] * dirs[1]
        offset_y = _right[1] * dirs[0] + _up[1] * dirs[1]
        offset_z = _right[2] * dirs[0] + _up[2] * dirs[1]
        camera_data = CameraData(
            camera_data.x + offset_x,
            camera_data.y + offset_y,
            camera_data.z + offset_z,
            camera_data.fx,
            camera_data.fy,
            camera_data.fz,
            camera_data.ux,
            camera_data.uy,
            camera_data.uz,
        )
        i += 1
    return camera_data


def run_camera_data(iterations):
    camera_data = CameraData(0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0)
    i = 0
    while i < iterations:
        dirs = _DIRECTIONS[i % len(_DIRECTIONS)]
        _forward = _normalize3((camera_data.fx, camera_data.fy, camera_data.fz))
        _up = _normalize3((camera_data.ux, camera_data.uy, camera_data.uz))
        _right = _cross3(_forward, _up)
        offset_x = _right[0] * dirs[0] + _up[0] * dirs[1]
        offset_y = _right[1] * dirs[0] + _up[1] * dirs[1]
        offset_z = _right[2] * dirs[0] + _up[2] * dirs[1]
        camera_data = CameraData(
            camera_data.x + offset_x,
            camera_data.y + offset_y,
            camera_data.z + offset_z,
            camera_data.fx,
            camera_data.fy,
            camera_data.fz,
            camera_data.ux,
            camera_data.uy,
            camera_data.uz,
        )
        i += 1
    return camera_data


run_camera_data = asr(run_camera_data)
assert getattr(run_camera_data, "__asr_transformed__", False), "CameraData benchmark failed to transform"


def main():
    cell = guard._registry[(run_camera_data.__module__, "CameraData")]
    return run_benchmark(
        "CameraData (arcade camera strafe, inlined to avoid the escape)",
        run_camera_data_original,
        run_camera_data,
        cell,
        CameraData,
        arg=200_000,
    )


if __name__ == "__main__":
    main()
