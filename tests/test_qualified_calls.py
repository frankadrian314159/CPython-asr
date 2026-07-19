"""v1.6: module-qualified constructor calls (`p = module.ClassName(...)`,
from `import module`, not just the bare `p = ClassName(...)` shape from
`from module import ClassName`). Motivated directly by cpython-asr's own
corpus study (corpus-study/README.md): `import module` + `module.Class(...)`
turned out to be common real-world Python, and the pass couldn't see
through it at all.

Classes live in _qualified_call_fixtures.py (a real, separate module) and
are referenced here as `fixtures.ClassName(...)` -- module-level
definitions throughout, same reason as the other test files:
func.__globals__ resolution.
"""

import dataclasses

import _qualified_call_fixtures as fixtures

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

def _run_qualified_reconstruct(n):
    p = fixtures.QualPoint(0.0, 0.0)
    i = 0
    while i < n:
        p = fixtures.QualPoint(p.x + 1.0, p.y + 2.0)
        i += 1
    return p


def test_qualified_call_reconstruct_mode():
    """Both the pre-loop init AND the in-loop reconstruction use the
    qualified `fixtures.QualPoint(...)` spelling -- exercises
    _resolve_ctor_class (init) and _call_target_matches_class (update)
    together."""
    expected = _run_qualified_reconstruct(37)
    transformed = try_transform(_run_qualified_reconstruct)
    assert transformed is not None
    cell = _cell_for(_run_qualified_reconstruct.__module__, "QualPoint")
    _check_dual_path(transformed, cell, 37, expected)


def _run_qualified_init_bare_update(n):
    """Only the INIT is qualified; the in-loop reconstruction uses the
    bare name -- a mix that should work, since __globals__ has both
    `fixtures` and (via `from ... import` below) QualPoint itself
    available for the update site's own bare-name match."""
    p = fixtures.QualPoint(0.0, 0.0)
    i = 0
    while i < n:
        p = QualPoint(p.x + 1.0, p.y + 2.0)
        i += 1
    return p


from _qualified_call_fixtures import QualPoint  # noqa: E402  (needed for the bare-name update site above)


def test_qualified_init_with_bare_name_update():
    expected = _run_qualified_init_bare_update(19)
    transformed = try_transform(_run_qualified_init_bare_update)
    assert transformed is not None
    cell = _cell_for(_run_qualified_init_bare_update.__module__, "QualPoint")
    _check_dual_path(transformed, cell, 19, expected)


def _run_qualified_mutate(n):
    p = fixtures.QualMutablePoint(0.0, 0.0)
    i = 0
    while i < n:
        p.x = p.x + 1.0
        p.y = p.y + 2.0
        i += 1
    return p


def test_qualified_call_mutate_mode():
    """The qualified-call fix applies uniformly to _find_accumulator,
    which both reconstruct- and mutate-mode qualification go through --
    exercises it specifically for a non-frozen (mutate-mode) dataclass."""
    expected = _run_qualified_mutate(23)
    transformed = try_transform(_run_qualified_mutate)
    assert transformed is not None
    cell = _cell_for(_run_qualified_mutate.__module__, "QualMutablePoint")
    _check_dual_path(transformed, cell, 23, expected)


def _run_qualified_plain_class(n):
    c = fixtures.QualPlainCounter(0, 0.0)
    i = 0
    while i < n:
        c.n = c.n + 1
        c.total = c.total + c.n
        i += 1
    return c


def test_qualified_call_plain_class():
    expected = _run_qualified_plain_class(15)
    transformed = try_transform(_run_qualified_plain_class)
    assert transformed is not None
    cell = _cell_for(_run_qualified_plain_class.__module__, "QualPlainCounter")
    _check_dual_path(transformed, cell, 15, expected)


def _update_qualified_point(p):
    return fixtures.QualPoint(p.x + 1.0, p.y)


def _run_qualified_inlined_helper(n):
    """The helper CALL SITE (`_update_qualified_point(p)`) is a bare
    name -- _try_inline_call's own helper resolution stays Name-only by
    design (see transform.py's v1.6 docstring) -- but the helper's OWN
    return expression uses the qualified spelling, which
    _reconstruction_field_values (shared by direct and inlined
    reconstruction) now recognizes."""
    p = fixtures.QualPoint(0.0, 0.0)
    i = 0
    while i < n:
        p = _update_qualified_point(p)
        i += 1
    return p


def test_qualified_reconstruction_inside_an_inlined_helper():
    expected = _run_qualified_inlined_helper(11)
    transformed = try_transform(_run_qualified_inlined_helper)
    assert transformed is not None
    cell = _cell_for(_run_qualified_inlined_helper.__module__, "QualPoint")
    _check_dual_path(transformed, cell, 11, expected)


# --------------------------------------------------------------------------
# Negative / abort-safe cases
# --------------------------------------------------------------------------

def _run_qualified_field_mismatch(n):
    p = fixtures.QualPoint(0.0, 0.0)
    i = 0
    while i < n:
        p = fixtures.QualPoint(p.x + 1.0)  # missing the y field
        i += 1
    return p


def test_declines_qualified_call_with_field_mismatch():
    """_ctor_supplies_all_fields still applies to a qualified call --
    the qualification fix only changes HOW the class is resolved, not
    the field-completeness requirement."""
    assert try_transform(_run_qualified_field_mismatch) is None


def _run_qualified_unresolvable_module(n):
    p = nonexistent_module.QualPoint(0.0, 0.0)  # not a real global -- no such accumulator init exists
    i = 0
    while i < n:
        p = fixtures.QualPoint(p.x + 1.0, p.y + 2.0)
        i += 1
    return p


def test_declines_qualified_call_to_unresolvable_module_at_init():
    """_resolve_ctor_class returns (None, None) when the qualifying
    module name isn't bound in __globals__ at all -- _find_accumulator
    then finds no qualifying pre-loop initializer, so the whole
    transform declines rather than raising."""
    assert try_transform(_run_qualified_unresolvable_module) is None
