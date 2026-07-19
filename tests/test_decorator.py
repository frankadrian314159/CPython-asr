"""End-to-end tests of the @asr decorator itself (not the transform
internals) -- exercised the way real user code would use it. Module-
level dataclass/function definitions for the same reason as
test_transform.py: func.__globals__ is the defining module's namespace,
never an enclosing function's locals.
"""

import dataclasses

from asr import asr


@dataclasses.dataclass(frozen=True)
class _DecoratorPoint(object):
    x: float
    y: float


def _undecorated(n):
    p = _DecoratorPoint(0.0, 0.0)
    i = 0
    while i < n:
        p = _DecoratorPoint(p.x + 0.1, p.y + 0.2)
        i += 1
    return p


def test_decorator_transforms_and_runs_correctly():
    expected = _undecorated(20)
    decorated = asr(_undecorated)
    assert decorated.__asr_transformed__ is True
    assert decorated(20) == expected


def _undecoratable(n):
    # no while loop at all -- must decline
    return n * 2


def test_decorator_falls_back_on_decline():
    decorated = asr(_undecoratable)
    assert not hasattr(decorated, "__asr_transformed__")
    assert decorated is _undecoratable
    assert decorated(21) == 42
