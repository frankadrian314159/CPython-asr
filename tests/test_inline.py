"""v1.1: interprocedural reach by inlining (FOL's sec:inline analog).

Module-level dataclass/function definitions throughout, for the same
reason as test_transform.py: func.__globals__ is the defining module's
namespace, never an enclosing function's locals, and a helper looked up
via globalns.get(name) needs to actually be there.
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
class _InlinePoint(object):
    x: float
    y: float


def _update_inline_point(p):
    return _InlinePoint(p.x + 0.1, p.y + 0.2)


def _run_inline_basic(n):
    p = _InlinePoint(0.0, 0.0)
    i = 0
    while i < n:
        p = _update_inline_point(p)
        i += 1
    return p


def test_inlines_basic_helper_call():
    expected = _run_inline_basic(25)
    transformed = try_transform(_run_inline_basic)
    assert transformed is not None
    cell = _cell_for(_run_inline_basic.__module__, "_InlinePoint")
    _check_dual_path(transformed, cell, 25, expected)


@dataclasses.dataclass(frozen=True)
class _StepPoint(object):
    x: float
    y: float


def _update_step_point(p, dx, dy):
    return _StepPoint(p.x + dx, p.y + dy)


def _run_inline_literal_args(n):
    p = _StepPoint(0.0, 0.0)
    i = 0
    while i < n:
        p = _update_step_point(p, 0.1, 0.2)
        i += 1
    return p


def test_inlines_helper_with_literal_other_args():
    """The callee's OTHER (non-accumulator) parameters, here dx/dy, get
    substituted with the actual call-site argument expressions."""
    expected = _run_inline_literal_args(25)
    transformed = try_transform(_run_inline_literal_args)
    assert transformed is not None
    cell = _cell_for(_run_inline_literal_args.__module__, "_StepPoint")
    _check_dual_path(transformed, cell, 25, expected)


def _run_inline_name_args(n):
    p = _StepPoint(0.0, 0.0)
    dx = 0.5
    dy = 1.5
    i = 0
    while i < n:
        p = _update_step_point(p, dx, dy)
        i += 1
    return p


def test_inlines_helper_with_name_other_args():
    """OTHER parameters can also be substituted with a caller-local
    variable Name, not just a literal constant -- both are allowed by
    FOL's "arguments are symbols or literals" restriction."""
    expected = _run_inline_name_args(25)
    transformed = try_transform(_run_inline_name_args)
    assert transformed is not None
    cell = _cell_for(_run_inline_name_args.__module__, "_StepPoint")
    _check_dual_path(transformed, cell, 25, expected)


@dataclasses.dataclass(frozen=True)
class _BumpCtr(object):
    n: float


def _bump_ctr(c):
    return dataclasses.replace(c, n=c.n + 1.0)


def _run_inline_replace(n):
    c = _BumpCtr(0.0)
    i = 0
    while i < n:
        c = _bump_ctr(c)
        i += 1
    return c


def test_inlines_dataclasses_replace_helper():
    expected = _run_inline_replace(25)
    transformed = try_transform(_run_inline_replace)
    assert transformed is not None
    cell = _cell_for(_run_inline_replace.__module__, "_BumpCtr")
    _check_dual_path(transformed, cell, 25, expected)


# --------------------------------------------------------------------------
# Negative / abort-safe cases
# --------------------------------------------------------------------------

def _update_compound_arg(p, dx):
    return _StepPoint(p.x + dx, p.y)


def _run_inline_compound_arg(n):
    p = _StepPoint(0.0, 0.0)
    x = 1.0
    i = 0
    while i < n:
        p = _update_compound_arg(p, x + 1.0)  # not a bare Name or literal
        i += 1
    return p


def test_declines_helper_call_with_compound_other_arg():
    assert try_transform(_run_inline_compound_arg) is None


def _update_with_local_binding(p):
    step = 0.1
    return _InlinePoint(p.x + step, p.y + step)


def _run_inline_multi_statement_helper(n):
    p = _InlinePoint(0.0, 0.0)
    i = 0
    while i < n:
        p = _update_with_local_binding(p)
        i += 1
    return p


def test_declines_helper_with_more_than_one_statement():
    """FOL's inliner only reaches a helper whose body is exactly one
    reconstruction; a local binding before the return is out of scope."""
    assert try_transform(_run_inline_multi_statement_helper) is None


def _inner_update(p):
    return _InlinePoint(p.x + 0.1, p.y + 0.2)


def _outer_update(p):
    return _inner_update(p)  # helper calling ANOTHER helper


def _run_inline_two_levels(n):
    p = _InlinePoint(0.0, 0.0)
    i = 0
    while i < n:
        p = _outer_update(p)
        i += 1
    return p


def test_declines_two_level_inline_chain():
    """One-level inlining only, matching FOL's own stated scope: the
    callee's return expression must itself be a direct reconstruction,
    not a call to a further helper."""
    assert try_transform(_run_inline_two_levels) is None


def _update_twice(p, q):
    return _InlinePoint(p.x + q.x, p.y + q.y)


def _run_inline_accumulator_passed_twice(n):
    p = _InlinePoint(0.0, 0.0)
    i = 0
    while i < n:
        p = _update_twice(p, p)  # the accumulator passed to BOTH params
        i += 1
    return p


def test_declines_accumulator_passed_twice():
    assert try_transform(_run_inline_accumulator_passed_twice) is None


def _get_a_point():
    return _InlinePoint(9.0, 9.0)


def _run_inline_non_bare_accumulator_arg(n):
    p = _InlinePoint(0.0, 0.0)
    i = 0
    while i < n:
        p = _update_inline_point(_get_a_point())  # not the accumulator itself
        i += 1
    return p


def test_declines_when_accumulator_arg_is_not_the_bare_accumulator():
    assert try_transform(_run_inline_non_bare_accumulator_arg) is None


def _run_inline_already_transformed_helper(n):
    p = _InlinePoint(0.0, 0.0)
    i = 0
    while i < n:
        p = _update_inline_point(p)
        i += 1
    return p


def test_does_not_inline_an_already_transformed_helper():
    """If the helper itself was already @asr-transformed (has a guarded
    dual-path body), inlining it would mean splicing that whole
    if/else into the caller -- explicitly declined rather than
    attempted. Sets the marker attribute directly rather than running a
    real transform-and-rebind cycle: try_transform's use of the live
    globals dict (needed for the world guard, see transform.py) means
    actually calling try_transform(_update_inline_point) here would
    rebind the module-level name as a side effect, which would then
    contaminate test_inlines_basic_helper_call's use of the same helper
    depending on test order -- this is a more surgical, order-independent
    way to exercise the same check in _try_inline_call."""
    _update_inline_point.__asr_transformed__ = True
    try:
        assert try_transform(_run_inline_already_transformed_helper) is None
    finally:
        del _update_inline_point.__asr_transformed__
