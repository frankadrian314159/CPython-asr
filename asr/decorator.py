"""The @asr entry point.

Attempts Aggregate Scalar Replacement on a function at decoration time.
On success, returns a new function with a guarded dual path (the fast,
unboxed path when the world guard holds, the untouched original path
when it doesn't or hasn't been decided yet). On any decline, returns the
original function completely unchanged -- opt-in, source-to-source,
never silently wrong.
"""

from . import transform


def asr(func):
    transformed = transform.try_transform(func)
    return transformed if transformed is not None else func
