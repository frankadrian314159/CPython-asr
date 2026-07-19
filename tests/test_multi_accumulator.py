"""v1.2: the multi-accumulator fixpoint (FOL's maybe-scalar-replace-loop
/ %sr-replace-one). Module-level definitions throughout, same reason as
the other test files: func.__globals__ resolution.
"""

import dataclasses

from asr import guard
from asr.transform import try_transform


def _cell_for(func_module, class_name):
    return guard._registry[(func_module, class_name)]


def _check_dual_path(transformed, cell, arg, expected):
    cell.valid = True
    assert transformed(arg) == expected
    cell.valid = False
    assert transformed(arg) == expected
    cell.valid = True


# --------------------------------------------------------------------------
# Positive cases
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _Vec2(object):
    x: float
    y: float


def _run_two_independent(n):
    p = _Vec2(0.0, 0.0)
    q = _Vec2(10.0, 10.0)
    i = 0
    while i < n:
        p = _Vec2(p.x + 1.0, p.y + 2.0)
        q = _Vec2(q.x - 1.0, q.y - 2.0)
        i += 1
    return p, q


def test_two_independent_accumulators_tuple_return():
    expected = _run_two_independent(23)
    transformed = try_transform(_run_two_independent)
    assert transformed is not None
    p_cell = _cell_for(_run_two_independent.__module__, "_Vec2")
    _check_dual_path(transformed, p_cell, 23, expected)


def _run_two_body(n):
    """FOL's Two-body shape: a coupled relaxation where each
    accumulator's update reads the other's CURRENT fields. Python's
    sequential statement execution means q's update (the second
    statement) naturally sees p's value AFTER p's own update on the
    preceding line -- unlike FOL's recur, which has true parallel-update
    semantics across separate loop variables. The transform must match
    whatever the ORIGINAL, unoptimized Python actually does, not FOL's
    semantics -- this test's correctness check (transformed == original)
    verifies exactly that, regardless of which convention is "more
    correct" in the abstract."""
    p = _Vec2(1.0, 1.0)
    q = _Vec2(2.0, 2.0)
    i = 0
    while i < n:
        p = _Vec2(p.x + q.x, p.y + q.y)
        q = _Vec2(q.x + p.x, q.y + p.y)
        i += 1
    return p, q


def test_two_coupled_accumulators():
    expected = _run_two_body(15)
    transformed = try_transform(_run_two_body)
    assert transformed is not None
    cell = _cell_for(_run_two_body.__module__, "_Vec2")
    _check_dual_path(transformed, cell, 15, expected)


@dataclasses.dataclass(frozen=True)
class _StateA(object):
    x: float


@dataclasses.dataclass(frozen=True)
class _StateB(object):
    y: float


def _run_two_different_classes(n):
    a = _StateA(0.0)
    b = _StateB(100.0)
    i = 0
    while i < n:
        a = _StateA(a.x + 1.0)
        b = _StateB(b.y - 1.0)
        i += 1
    return a, b


def test_two_accumulators_different_classes():
    expected = _run_two_different_classes(12)
    transformed = try_transform(_run_two_different_classes)
    assert transformed is not None
    cell_a = _cell_for(_run_two_different_classes.__module__, "_StateA")
    cell_b = _cell_for(_run_two_different_classes.__module__, "_StateB")
    cell_a.valid = True
    cell_b.valid = True
    assert transformed(12) == expected
    # The fast path must depend on BOTH classes' guards: invalidating
    # either one alone must fall back to the correct, original path.
    cell_a.valid = False
    assert transformed(12) == expected
    cell_a.valid = True
    cell_b.valid = False
    assert transformed(12) == expected
    cell_a.valid = True
    cell_b.valid = True


def _run_single_still_works(n):
    """Sanity check that the fixpoint-based driver didn't regress the
    plain single-accumulator case."""
    p = _Vec2(0.0, 0.0)
    i = 0
    while i < n:
        p = _Vec2(p.x + 1.0, p.y + 1.0)
        i += 1
    return p


def test_single_accumulator_still_works_under_fixpoint_driver():
    expected = _run_single_still_works(9)
    transformed = try_transform(_run_single_still_works)
    assert transformed is not None
    cell = _cell_for(_run_single_still_works.__module__, "_Vec2")
    _check_dual_path(transformed, cell, 9, expected)


# --------------------------------------------------------------------------
# Negative / abort-safe cases
# --------------------------------------------------------------------------

def _run_returns_non_accumulator_in_tuple(n):
    p = _Vec2(0.0, 0.0)
    i = 0
    while i < n:
        p = _Vec2(p.x + 1.0, p.y + 1.0)
        i += 1
    return p, i  # `i` was never a tracked accumulator


def test_declines_tuple_return_naming_a_non_accumulator():
    assert try_transform(_run_returns_non_accumulator_in_tuple) is None


def _run_second_accumulator_escapes(n):
    p = _Vec2(0.0, 0.0)
    q = _Vec2(0.0, 0.0)
    i = 0
    while i < n:
        p = _Vec2(p.x + 1.0, p.y + 1.0)
        print(q)  # q itself never reconstructs and escapes into a call
        i += 1
    return p, q


def test_declines_when_second_qualifying_initializer_never_reconstructs():
    """q qualifies as a candidate initializer (a frozen-dataclass
    constructor call before the loop) but never gets reconstructed in
    the loop body and escapes into an arbitrary call instead. p alone
    would normally transform fine, but the post-loop `return p, q`
    requires q to have been successfully processed too -- since it
    wasn't, the whole thing declines rather than silently returning a
    stale/wrong q."""
    assert try_transform(_run_second_accumulator_escapes) is None
