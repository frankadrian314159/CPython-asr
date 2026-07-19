"""Positive and negative/abort-safe correctness tests for the ASR
transform. Every positive case checks both the guarded fast path and the
guarded slow path independently (by toggling the registered validity
cell) agree with the untransformed original -- the property that
actually matters, since the paper's whole ethos is "never wrong, only
sometimes inapplicable."

Classes and the functions under test are deliberately defined at MODULE
level, not nested inside the test functions: a function's __globals__ is
always its *defining module's* namespace, never an enclosing function's
locals, so a class only visible as a local of the test function would
never be resolvable by the transform's `_find_accumulator` -- this is
also exactly how real @asr usage looks (module-level dataclasses).
"""

import dataclasses

from asr import guard
from asr.transform import try_transform


def _cell_for(func_module, class_name):
    return guard._registry[(func_module, class_name)]


# --------------------------------------------------------------------------
# Positive cases
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _Point(object):
    x: float
    y: float


def _full_reconstruction(n):
    p = _Point(0.0, 0.0)
    i = 0
    while i < n:
        p = _Point(p.x + 0.1, p.y + 0.2)
        i += 1
    return p


def test_full_reconstruction_correctness():
    expected = _full_reconstruction(50)
    transformed = try_transform(_full_reconstruction)
    assert transformed is not None
    assert transformed.__asr_transformed__ is True

    cell = _cell_for(_full_reconstruction.__module__, "_Point")
    cell.valid = True
    assert transformed(50) == expected
    cell.valid = False
    assert transformed(50) == expected
    cell.valid = True  # leave in the "hot" state for subsequent tests


@dataclasses.dataclass(frozen=True)
class _Counter(object):
    n: int
    total: float


def _partial_reconstruction(n):
    acc = _Counter(0, 0.0)
    i = 0
    while i < n:
        acc = dataclasses.replace(acc, n=acc.n + 1, total=acc.total + acc.n)
        i += 1
    return acc


def test_partial_reconstruction_dataclasses_replace():
    expected = _partial_reconstruction(30)
    transformed = try_transform(_partial_reconstruction)
    assert transformed is not None

    cell = _cell_for(_partial_reconstruction.__module__, "_Counter")
    cell.valid = True
    assert transformed(30) == expected
    cell.valid = False
    assert transformed(30) == expected
    cell.valid = True


@dataclasses.dataclass(frozen=True)
class _Pair(object):
    a: int
    b: int


def _partial_reconstruction_unsupplied_field(n):
    p = _Pair(0, 100)
    i = 0
    while i < n:
        p = dataclasses.replace(p, a=p.a + 1)  # b is never touched
        i += 1
    return p


def test_partial_reconstruction_carries_unsupplied_field():
    """A field NOT named in the dataclasses.replace(...) call must keep
    its prior value across iterations -- this is the one place the
    Python port's design genuinely differs from FOL's parallel
    recur/psetq update (ordinary Python local persistence handles it
    for free, see transform.py's _rewrite_loop_body docstring)."""
    expected = _partial_reconstruction_unsupplied_field(10)
    assert expected == _Pair(10, 100)
    transformed = try_transform(_partial_reconstruction_unsupplied_field)
    assert transformed is not None

    cell = _cell_for(_partial_reconstruction_unsupplied_field.__module__, "_Pair")
    cell.valid = True
    assert transformed(10) == expected
    cell.valid = False
    assert transformed(10) == expected
    cell.valid = True


@dataclasses.dataclass(frozen=True)
class _SideStatePoint(object):
    x: float
    y: float


def _run_with_side_state(n):
    p = _SideStatePoint(0.0, 0.0)
    i = 0
    touched = 0
    while i < n:
        p = _SideStatePoint(p.x + 1.0, p.y + 1.0)
        touched += 1  # unrelated to `p` -- must survive the rewrite untouched
        i += 1
    return p


def test_unrelated_statements_pass_through():
    """Statements in the loop body that don't touch the accumulator
    (here, a running side counter) must survive the rewrite untouched."""
    transformed = try_transform(_run_with_side_state)
    assert transformed is not None
    cell = _cell_for(_run_with_side_state.__module__, "_SideStatePoint")
    cell.valid = True
    assert transformed(7) == _SideStatePoint(7.0, 7.0)
    cell.valid = False
    assert transformed(7) == _SideStatePoint(7.0, 7.0)
    cell.valid = True


# --------------------------------------------------------------------------
# Negative / abort-safe cases -- try_transform must return None, and the
# decorator (tested separately) must then fall back to the original
# function untouched.
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _LeakPoint(object):
    x: float
    y: float


def _leaks(n):
    p = _LeakPoint(0.0, 0.0)
    i = 0
    while i < n:
        print(p)  # bare accumulator escapes into an arbitrary call
        p = _LeakPoint(p.x + 1.0, p.y + 1.0)
        i += 1
    return p


def test_declines_bare_accumulator_passed_to_call():
    assert try_transform(_leaks) is None


@dataclasses.dataclass(frozen=True)
class _StoredPoint(object):
    x: float
    y: float


def _stores(n):
    p = _StoredPoint(0.0, 0.0)
    i = 0
    history = []
    while i < n:
        history.append(p)  # bare accumulator escapes into a container
        p = _StoredPoint(p.x + 1.0, p.y + 1.0)
        i += 1
    return p


def test_declines_accumulator_stored_in_container():
    assert try_transform(_stores) is None


@dataclasses.dataclass  # not frozen
class _MutablePoint(object):
    x: float
    y: float


def _run_mutable(n):
    p = _MutablePoint(0.0, 0.0)
    i = 0
    while i < n:
        p = _MutablePoint(p.x + 1.0, p.y + 1.0)
        i += 1
    return p


def test_declines_non_frozen_dataclass():
    assert try_transform(_run_mutable) is None


class _PlainPoint(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


def _run_plain(n):
    p = _PlainPoint(0.0, 0.0)
    i = 0
    while i < n:
        p = _PlainPoint(p.x + 1.0, p.y + 1.0)
        i += 1
    return p


def test_declines_plain_class():
    assert try_transform(_run_plain) is None


@dataclasses.dataclass(frozen=True)
class _BranchPoint(object):
    x: float
    y: float


def _run_branched(n):
    p = _BranchPoint(0.0, 0.0)
    i = 0
    while i < n:
        if i % 2 == 0:
            p = _BranchPoint(p.x + 1.0, p.y)
        else:
            p = _BranchPoint(p.x, p.y + 1.0)
        i += 1
    return p


def test_declines_branched_reconstruction():
    assert try_transform(_run_branched) is None


def _no_loop(n):
    return n * 2


def test_declines_no_while_loop():
    assert try_transform(_no_loop) is None


@dataclasses.dataclass(frozen=True)
class _TailShapePoint(object):
    x: float
    y: float


def _run_unsupported_tail(n):
    p = _TailShapePoint(0.0, 0.0)
    i = 0
    while i < n:
        p = _TailShapePoint(p.x + 1.0, p.y + 1.0)
        i += 1
    result = p.x + p.y  # extra computation after the loop -- unsupported
    return result


def test_declines_unsupported_post_loop_shape():
    assert try_transform(_run_unsupported_tail) is None


@dataclasses.dataclass(frozen=True)
class _MultiPoint(object):
    x: float
    y: float


def _run_multi_accumulator_boundary(n):
    p = _MultiPoint(0.0, 0.0)
    q = _MultiPoint(1.0, 1.0)
    i = 0
    while i < n:
        p = _MultiPoint(p.x + q.x, p.y + q.y)
        i += 1
    return p


def test_transforms_around_untracked_second_dataclass():
    """`q` is a second, untracked dataclass-shaped local that is only
    ever read, never reconstructed, inside the loop. v1 recognizes a
    single accumulator (the first qualifying one found before the loop)
    and simply treats every other variable as ordinary surrounding code
    -- it does not need to understand `q` to safely rewrite `p`. This
    documents that boundary rather than asserting a decline: FOL's own
    fixpoint would unbox both; v1 here unboxes only `p`, correctly."""
    expected = _run_multi_accumulator_boundary(5)
    transformed = try_transform(_run_multi_accumulator_boundary)
    assert transformed is not None
    cell = _cell_for(_run_multi_accumulator_boundary.__module__, "_MultiPoint")
    cell.valid = True
    assert transformed(5) == expected
    cell.valid = False
    assert transformed(5) == expected
    cell.valid = True
