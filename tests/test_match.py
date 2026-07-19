"""v1.3: match/case reconstruction (Python 3.10+, PEP 634-636), FOL's own
`case` construct's closest Python analog. Module-level definitions
throughout, same reason as the other test files: func.__globals__
resolution.
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
class _PhasePoint(object):
    x: float
    y: float


def _run_match_literal_dispatch(n):
    p = _PhasePoint(0.0, 0.0)
    i = 0
    while i < n:
        match i % 3:
            case 0:
                p = _PhasePoint(p.x + 1.0, p.y)
            case 1:
                p = _PhasePoint(p.x, p.y + 1.0)
            case _:
                p = _PhasePoint(p.x + 1.0, p.y + 1.0)
        i += 1
    return p


def test_match_case_literal_dispatch_with_wildcard():
    """FOL's Phase shape (case-branched), ported to Python's own
    match/case instead of if/elif/else -- see test_branch.py's
    test_elif_chain_three_branches for the same shape the other way."""
    expected = _run_match_literal_dispatch(41)
    transformed = try_transform(_run_match_literal_dispatch)
    assert transformed is not None
    cell = _cell_for(_run_match_literal_dispatch.__module__, "_PhasePoint")
    _check_dual_path(transformed, cell, 41, expected)


@dataclasses.dataclass(frozen=True)
class _MixedMatchPoint(object):
    x: float
    y: float


def _run_match_mixed_full_and_partial(n):
    p = _MixedMatchPoint(0.0, 0.0)
    i = 0
    while i < n:
        match i % 2:
            case 0:
                p = _MixedMatchPoint(p.x + 1.0, p.y)  # full reconstruction
            case _:
                p = dataclasses.replace(p, y=p.y + 1.0)  # partial -- only y touched
        i += 1
    return p


def test_match_case_mixed_full_and_partial_reconstruction():
    """Different cases can touch different fields, same as if/elif
    branches -- each field needs an explicit passthrough default for
    whichever case doesn't touch it."""
    expected = _run_match_mixed_full_and_partial(29)
    transformed = try_transform(_run_match_mixed_full_and_partial)
    assert transformed is not None
    cell = _cell_for(_run_match_mixed_full_and_partial.__module__, "_MixedMatchPoint")
    _check_dual_path(transformed, cell, 29, expected)


@dataclasses.dataclass(frozen=True)
class _SingletonPoint(object):
    x: float
    y: float


def _run_match_singleton_pattern(n):
    p = _SingletonPoint(0.0, 0.0)
    i = 0
    while i < n:
        match i % 2 == 0:
            case True:
                p = _SingletonPoint(p.x + 1.0, p.y)
            case _:
                p = _SingletonPoint(p.x, p.y + 1.0)
        i += 1
    return p


def test_match_case_singleton_pattern():
    """MatchSingleton patterns (`case True:`, `case False:`, `case
    None:`) are recognized the same way as MatchValue literals."""
    expected = _run_match_singleton_pattern(19)
    transformed = try_transform(_run_match_singleton_pattern)
    assert transformed is not None
    cell = _cell_for(_run_match_singleton_pattern.__module__, "_SingletonPoint")
    _check_dual_path(transformed, cell, 19, expected)


_match_subject_calls = []


def _side_effecting_key(i):
    """A subject expression with an observable side effect (records
    every call to a module-level list) -- used to verify the rewrite
    doesn't re-evaluate the match subject once per case comparison.
    Module-level, not threaded through the accumulator's return value,
    since the transform's supported post-loop tail shapes are only
    `return p` / `return p, q, ...` naming bare accumulators."""
    _match_subject_calls.append(i)
    return i % 3


@dataclasses.dataclass(frozen=True)
class _SubjectOncePoint(object):
    x: float


def _run_match_subject_evaluated_once(n):
    p = _SubjectOncePoint(0.0)
    i = 0
    while i < n:
        match _side_effecting_key(i):
            case 0:
                p = _SubjectOncePoint(p.x + 1.0)
            case 1:
                p = _SubjectOncePoint(p.x + 2.0)
            case _:
                p = _SubjectOncePoint(p.x + 3.0)
        i += 1
    return p


def test_match_subject_evaluated_exactly_once_per_iteration():
    """Python's match statement evaluates its subject expression
    exactly once no matter how many cases it has; the rewrite must
    preserve that call count, not re-invoke a side-effecting subject
    expression once per case comparison."""
    global _match_subject_calls
    _match_subject_calls = []
    expected = _run_match_subject_evaluated_once(17)
    assert len(_match_subject_calls) == 17

    transformed = try_transform(_run_match_subject_evaluated_once)
    assert transformed is not None
    cell = _cell_for(_run_match_subject_evaluated_once.__module__, "_SubjectOncePoint")

    cell.valid = True
    _match_subject_calls = []
    assert transformed(17) == expected
    assert len(_match_subject_calls) == 17

    cell.valid = False
    _match_subject_calls = []
    assert transformed(17) == expected
    assert len(_match_subject_calls) == 17
    cell.valid = True


@dataclasses.dataclass(frozen=True)
class _UnrelatedMatchPoint(object):
    x: float
    y: float


def _run_with_unrelated_match(n):
    p = _UnrelatedMatchPoint(0.0, 0.0)
    i = 0
    total_evens = 0
    while i < n:
        match i % 2:
            case 0:
                total_evens += 1  # a match statement that has NOTHING to do with p
            case _:
                pass
        p = _UnrelatedMatchPoint(p.x + 1.0, p.y + 1.0)
        i += 1
    return p


def test_transforms_with_unrelated_match_statement_in_loop():
    """A match statement that never mentions the accumulator at all
    must not force a decline -- mirrors test_branch.py's analogous
    unrelated-if regression test."""
    expected = _run_with_unrelated_match(19)
    transformed = try_transform(_run_with_unrelated_match)
    assert transformed is not None
    cell = _cell_for(_run_with_unrelated_match.__module__, "_UnrelatedMatchPoint")
    _check_dual_path(transformed, cell, 19, expected)


# --------------------------------------------------------------------------
# Negative / abort-safe cases
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _NoDefaultPoint(object):
    x: float


def _run_match_without_wildcard(n):
    p = _NoDefaultPoint(0.0)
    i = 0
    while i < n:
        match i % 2:
            case 0:
                p = _NoDefaultPoint(p.x + 1.0)
            case 1:
                p = _NoDefaultPoint(p.x + 2.0)
        i += 1
    return p


def test_declines_match_without_mandatory_wildcard_default():
    """FOL requires a mandatory default clause; a match with no `case
    _:` leaves p unmodified whenever no pattern matches (impossible
    here since i % 2 is exhaustive over {0, 1}, but the transform can't
    know that -- same safe-by-abort discipline as the if-without-else
    case)."""
    assert try_transform(_run_match_without_wildcard) is None


@dataclasses.dataclass(frozen=True)
class _GuardedPoint(object):
    x: float


def _run_match_with_guard(n):
    p = _GuardedPoint(0.0)
    i = 0
    while i < n:
        match i % 3:
            case 0 if i > 10:
                p = _GuardedPoint(p.x + 1.0)
            case _:
                p = _GuardedPoint(p.x + 2.0)
        i += 1
    return p


def test_declines_match_case_with_guard():
    """FOL's own `case` has no per-clause guard; a `case ... if ...:`
    is out of scope."""
    assert try_transform(_run_match_with_guard) is None


@dataclasses.dataclass(frozen=True)
class _StructuralPoint(object):
    x: float
    y: float


def _run_match_with_structural_pattern(n):
    p = _StructuralPoint(0.0, 0.0)
    key = _StructuralPoint(0.0, 0.0)
    i = 0
    while i < n:
        match key:
            case _StructuralPoint(x=0.0):
                p = _StructuralPoint(p.x + 1.0, p.y)
            case _:
                p = _StructuralPoint(p.x, p.y + 1.0)
        i += 1
    return p


def test_declines_match_case_with_structural_pattern():
    """Class/structural patterns are well beyond FOL's own `case`,
    which only ever dispatches on literal values."""
    assert try_transform(_run_match_with_structural_pattern) is None


@dataclasses.dataclass(frozen=True)
class _CapturePoint(object):
    x: float


def _run_match_with_capture_pattern(n):
    p = _CapturePoint(0.0)
    i = 0
    while i < n:
        match i % 3:
            case 0:
                p = _CapturePoint(p.x + 1.0)
            case captured:  # binds, doesn't wildcard -- not `_`
                p = _CapturePoint(p.x + float(captured))
        i += 1
    return p


def test_declines_match_case_with_capture_pattern_instead_of_wildcard():
    """A trailing `case name:` capture pattern is not the same as a
    true wildcard `case _:` -- it binds a new local, which is out of
    scope; only the parameterless wildcard counts as FOL's default."""
    assert try_transform(_run_match_with_capture_pattern) is None


@dataclasses.dataclass(frozen=True)
class _InlineLeafPoint(object):
    x: float


def _update_match_leaf(p):
    return _InlineLeafPoint(p.x + 1.0)


def _run_match_leaf_needs_inlining(n):
    p = _InlineLeafPoint(0.0)
    i = 0
    while i < n:
        match i % 2:
            case 0:
                p = _update_match_leaf(p)  # a helper call, not a direct reconstruction
            case _:
                p = _InlineLeafPoint(p.x + 2.0)
        i += 1
    return p


def test_declines_match_leaf_that_needs_inlining():
    """Same restriction as branch leaves: a case body needing inlining
    aborts the whole reconstruction -- case leaves must be direct
    reconstructions only."""
    assert try_transform(_run_match_leaf_needs_inlining) is None


@dataclasses.dataclass(frozen=True)
class _EscapePoint(object):
    x: float


def _run_match_with_escape(n):
    p = _EscapePoint(0.0)
    i = 0
    while i < n:
        match i % 2:
            case 0:
                p = _EscapePoint(p.x + 1.0)
            case _:
                print(p)  # bare accumulator escape in a case body
                p = _EscapePoint(p.x + 2.0)
        i += 1
    return p


def test_declines_match_case_with_bare_accumulator_escape():
    assert try_transform(_run_match_with_escape) is None


@dataclasses.dataclass(frozen=True)
class _SubjectEscapePoint(object):
    x: float


def _run_match_subject_references_accumulator(n):
    p = _SubjectEscapePoint(0.0)
    i = 0
    while i < n:
        match p.x:  # subject reads the accumulator itself -- not FOL's case shape
            case 0.0:
                p = _SubjectEscapePoint(p.x + 1.0)
            case _:
                p = _SubjectEscapePoint(p.x + 2.0)
        i += 1
    return p


def test_declines_match_subject_referencing_accumulator():
    """FOL's own `case` dispatches on an unrelated key, never on the
    accumulator's own fields."""
    assert try_transform(_run_match_subject_references_accumulator) is None


@dataclasses.dataclass(frozen=True)
class _ExtraStatementPoint(object):
    x: float


def _run_match_case_with_extra_statement(n):
    p = _ExtraStatementPoint(0.0)
    i = 0
    while i < n:
        match i % 2:
            case 0:
                print(i)  # extra statement alongside the reconstruction
                p = _ExtraStatementPoint(p.x + 1.0)
            case _:
                p = _ExtraStatementPoint(p.x + 2.0)
        i += 1
    return p


def test_declines_match_case_with_extra_statement():
    """v1.3 requires each case body to be exactly one statement, same
    as if/elif branches."""
    assert try_transform(_run_match_case_with_extra_statement) is None
