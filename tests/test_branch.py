"""v1.2: branch-shaped (if/elif/.../else) reconstruction, FOL's
Reconstruct if/cond cases. Module-level definitions throughout, same
reason as the other test files: func.__globals__ resolution.
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
class _ClampPoint(object):
    x: float
    y: float


def _run_if_else(n):
    p = _ClampPoint(0.0, 0.0)
    i = 0
    while i < n:
        if i % 2 == 0:
            p = _ClampPoint(p.x + 1.0, p.y)
        else:
            p = _ClampPoint(p.x, p.y + 1.0)
        i += 1
    return p


def test_if_else_full_reconstruction_both_branches():
    """FOL's Clamp shape: both branches are full reconstructions, each
    leaving one field's own value passed through explicitly."""
    expected = _run_if_else(37)
    transformed = try_transform(_run_if_else)
    assert transformed is not None
    cell = _cell_for(_run_if_else.__module__, "_ClampPoint")
    _check_dual_path(transformed, cell, 37, expected)


def _run_elif_chain(n):
    p = _ClampPoint(0.0, 0.0)
    i = 0
    while i < n:
        if i % 3 == 0:
            p = _ClampPoint(p.x + 1.0, p.y)
        elif i % 3 == 1:
            p = _ClampPoint(p.x, p.y + 1.0)
        else:
            p = _ClampPoint(p.x + 1.0, p.y + 1.0)
        i += 1
    return p


def test_elif_chain_three_branches():
    """FOL's Phase shape (case-branched, here as a 3-way elif chain --
    Python has no direct `case` analog with the same key-dispatch
    semantics, so if/elif/elif/else is the natural port). Python parses
    `elif` as a nested `If` in `orelse`, which _try_branch_reconstruction
    handles by recursing."""
    expected = _run_elif_chain(41)
    transformed = try_transform(_run_elif_chain)
    assert transformed is not None
    cell = _cell_for(_run_elif_chain.__module__, "_ClampPoint")
    _check_dual_path(transformed, cell, 41, expected)


@dataclasses.dataclass(frozen=True)
class _MixedPoint(object):
    x: float
    y: float


def _run_mixed_full_and_partial(n):
    p = _MixedPoint(0.0, 0.0)
    i = 0
    while i < n:
        if i % 2 == 0:
            p = _MixedPoint(p.x + 1.0, p.y)  # full reconstruction
        else:
            p = dataclasses.replace(p, y=p.y + 1.0)  # partial -- only y touched
        i += 1
    return p


def test_mixed_full_and_partial_reconstruction_across_branches():
    """Different branches can touch different fields (one branch a full
    constructor, the other a partial dataclasses.replace touching only
    y) -- each field needs an explicit passthrough default for whichever
    branch doesn't touch it, unlike the simpler single-branch case."""
    expected = _run_mixed_full_and_partial(29)
    transformed = try_transform(_run_mixed_full_and_partial)
    assert transformed is not None
    cell = _cell_for(_run_mixed_full_and_partial.__module__, "_MixedPoint")
    _check_dual_path(transformed, cell, 29, expected)


def _run_with_unrelated_if(n):
    p = _ClampPoint(0.0, 0.0)
    i = 0
    total_evens = 0
    while i < n:
        if i % 2 == 0:  # an `if` that has NOTHING to do with p
            total_evens += 1
        p = _ClampPoint(p.x + 1.0, p.y + 1.0)
        i += 1
    return p


def test_transforms_with_unrelated_if_statement_in_loop():
    """An `if` statement in the loop body that never mentions the
    accumulator at all must not force a decline -- only an `if` that
    actually looks like an attempted reconstruction goes through branch
    validation; anything else is just ordinary code to recurse into."""
    expected = _run_with_unrelated_if(19)
    transformed = try_transform(_run_with_unrelated_if)
    assert transformed is not None
    cell = _cell_for(_run_with_unrelated_if.__module__, "_ClampPoint")
    _check_dual_path(transformed, cell, 19, expected)


# --------------------------------------------------------------------------
# Negative / abort-safe cases
# --------------------------------------------------------------------------

def _run_if_without_else(n):
    p = _ClampPoint(0.0, 0.0)
    i = 0
    while i < n:
        if i % 2 == 0:
            p = _ClampPoint(p.x + 1.0, p.y)
        i += 1
    return p


def test_declines_if_without_mandatory_else():
    """FOL requires every branch to reconstruct; an `if` with no `else`
    leaves p unmodified on the false path, which isn't a reconstruction
    at all on that path."""
    assert try_transform(_run_if_without_else) is None


def _run_branch_with_extra_statement(n):
    p = _ClampPoint(0.0, 0.0)
    i = 0
    while i < n:
        if i % 2 == 0:
            print(i)  # extra statement alongside the reconstruction
            p = _ClampPoint(p.x + 1.0, p.y)
        else:
            p = _ClampPoint(p.x, p.y + 1.0)
        i += 1
    return p


def test_declines_branch_with_extra_statement():
    """v1.2 requires each branch to be exactly one statement (the
    reconstruction itself, or a further nested if for elif chains) --
    keeps the shape bounded and simple."""
    assert try_transform(_run_branch_with_extra_statement) is None


def _update_branch_point(p):
    return _ClampPoint(p.x + 1.0, p.y)


def _run_branch_with_inlined_leaf(n):
    p = _ClampPoint(0.0, 0.0)
    i = 0
    while i < n:
        if i % 2 == 0:
            p = _update_branch_point(p)  # a helper call, not a direct reconstruction
        else:
            p = _ClampPoint(p.x, p.y + 1.0)
        i += 1
    return p


def test_declines_branch_leaf_that_needs_inlining():
    """FOL's own restriction: a branch needing inlining (or bind/do
    peeling) aborts the whole reconstruction -- branch leaves must be
    direct reconstructions only."""
    assert try_transform(_run_branch_with_inlined_leaf) is None


def _run_branch_with_escape(n):
    p = _ClampPoint(0.0, 0.0)
    i = 0
    while i < n:
        if i % 2 == 0:
            p = _ClampPoint(p.x + 1.0, p.y)
        else:
            print(p)  # bare accumulator escape in the else branch
            p = _ClampPoint(p.x, p.y + 1.0)
        i += 1
    return p


def test_declines_branch_with_bare_accumulator_escape():
    assert try_transform(_run_branch_with_escape) is None
