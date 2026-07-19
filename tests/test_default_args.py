"""v1.7: a pre-loop initializer that leaves one or more fields to
__init__'s own default value (`p = CameraData()`, relying on defaults for
all fields, rather than supplying every one explicitly). Motivated
directly by cpython-asr's own corpus study (corpus-study/README.md):
arcade's CameraData is constructed with zero explicit arguments in its
only real usage. Module-level definitions throughout, same reason as the
other test files: func.__globals__ resolution.
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
class DefaultedPoint(object):
    x: float = 0.0
    y: float = 0.0


def _run_all_fields_defaulted(n):
    p = DefaultedPoint()  # zero args -- both fields left to their defaults
    i = 0
    while i < n:
        p = DefaultedPoint(p.x + 1.0, p.y + 2.0)
        i += 1
    return p


def test_frozen_dataclass_with_all_fields_defaulted():
    expected = _run_all_fields_defaulted(23)
    transformed = try_transform(_run_all_fields_defaulted)
    assert transformed is not None
    cell = _cell_for(_run_all_fields_defaulted.__module__, "DefaultedPoint")
    _check_dual_path(transformed, cell, 23, expected)


@dataclasses.dataclass(frozen=True)
class PartiallyDefaultedPoint(object):
    x: float
    y: float = 5.0


def _run_partial_defaults(n):
    p = PartiallyDefaultedPoint(1.0)  # x explicit, y defaulted
    i = 0
    while i < n:
        p = PartiallyDefaultedPoint(p.x + 1.0, p.y + 2.0)
        i += 1
    return p


def test_frozen_dataclass_with_some_fields_defaulted():
    expected = _run_partial_defaults(17)
    transformed = try_transform(_run_partial_defaults)
    assert transformed is not None
    cell = _cell_for(_run_partial_defaults.__module__, "PartiallyDefaultedPoint")
    _check_dual_path(transformed, cell, 17, expected)


@dataclasses.dataclass
class DefaultedMutablePoint(object):  # not frozen
    x: float = 0.0
    y: float = 0.0


def _run_mutate_with_defaults(n):
    p = DefaultedMutablePoint()
    i = 0
    while i < n:
        p.x = p.x + 1.0
        p.y = p.y + 2.0
        i += 1
    return p


def test_mutate_mode_with_all_fields_defaulted():
    """The defaults fix applies uniformly to _find_accumulator, shared
    by both reconstruct and mutate mode's init-stmt building."""
    expected = _run_mutate_with_defaults(19)
    transformed = try_transform(_run_mutate_with_defaults)
    assert transformed is not None
    cell = _cell_for(_run_mutate_with_defaults.__module__, "DefaultedMutablePoint")
    _check_dual_path(transformed, cell, 19, expected)


class DefaultedPlainCounter(object):
    def __init__(self, n=0, total=0.0):
        self.n = n
        self.total = total

    def __eq__(self, other):
        return isinstance(other, DefaultedPlainCounter) and self.n == other.n and self.total == other.total


def _run_plain_class_with_defaults(n):
    c = DefaultedPlainCounter()
    i = 0
    while i < n:
        c.n = c.n + 1
        c.total = c.total + c.n
        i += 1
    return c


def test_plain_class_with_defaulted_constructor():
    expected = _run_plain_class_with_defaults(15)
    transformed = try_transform(_run_plain_class_with_defaults)
    assert transformed is not None
    cell = _cell_for(_run_plain_class_with_defaults.__module__, "DefaultedPlainCounter")
    _check_dual_path(transformed, cell, 15, expected)


@dataclasses.dataclass(frozen=True)
class MutableDefaultPoint(object):
    tag: str = "default-tag"


def _run_with_string_default(n):
    """A non-numeric default (a string) -- exercises injecting an
    arbitrary Python object as a global, not just floats."""
    p = MutableDefaultPoint()
    i = 0
    while i < n:
        p = MutableDefaultPoint(p.tag)
        i += 1
    return p


def test_default_value_is_a_non_numeric_object():
    expected = _run_with_string_default(9)
    transformed = try_transform(_run_with_string_default)
    assert transformed is not None
    cell = _cell_for(_run_with_string_default.__module__, "MutableDefaultPoint")
    _check_dual_path(transformed, cell, 9, expected)


@dataclasses.dataclass(frozen=True)
class TwoDefaultedPoint(object):
    x: float = 0.0
    y: float = 0.0


def _run_two_accumulators_defaulted(n):
    a = TwoDefaultedPoint()
    b = TwoDefaultedPoint(100.0, 100.0)
    i = 0
    while i < n:
        a = TwoDefaultedPoint(a.x + 1.0, a.y + 1.0)
        b = TwoDefaultedPoint(b.x - 1.0, b.y - 1.0)
        i += 1
    return a, b


def test_two_accumulators_one_defaulted_one_explicit():
    """The unique-per-accumulator-and-function default-key prefix
    (default_key_prefix) must not clash between two accumulators of the
    SAME class in the same function."""
    expected = _run_two_accumulators_defaulted(11)
    transformed = try_transform(_run_two_accumulators_defaulted)
    assert transformed is not None
    cell = _cell_for(_run_two_accumulators_defaulted.__module__, "TwoDefaultedPoint")
    _check_dual_path(transformed, cell, 11, expected)


# --------------------------------------------------------------------------
# Negative / abort-safe cases
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class NoDefaultPoint(object):
    x: float
    y: float


def _run_missing_field_no_default(n):
    p = NoDefaultPoint(0.0)  # y is missing and has no default at all
    i = 0
    while i < n:
        p = NoDefaultPoint(p.x + 1.0, p.y + 2.0)
        i += 1
    return p


def test_declines_missing_field_with_no_default():
    """Pre-v1.7 behavior preserved: a field left unsupplied with no
    known default still declines, exactly as before."""
    assert try_transform(_run_missing_field_no_default) is None


@dataclasses.dataclass(frozen=True)
class ReconstructDefaultPoint(object):
    x: float = 0.0
    y: float = 0.0


def _run_reconstruction_relies_on_default(n):
    p = ReconstructDefaultPoint(0.0, 0.0)
    i = 0
    while i < n:
        p = ReconstructDefaultPoint(p.x + 1.0)  # y silently reset to its default every iteration
        i += 1
    return p


def test_declines_in_loop_reconstruction_relying_on_a_default():
    """v1.7 is deliberately scoped to the pre-loop initializer only --
    an in-loop reconstruction relying on a default would silently RESET
    that field every iteration rather than preserving it (matching real
    Python semantics, but easy to misread), so this still declines, by
    design (see transform.py's v1.7 docstring and
    _ctor_init_defaults's)."""
    assert try_transform(_run_reconstruction_relies_on_default) is None
