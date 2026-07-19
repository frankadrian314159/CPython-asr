"""v1.4: mutation-based unboxing for non-frozen (mutable) dataclasses and
plain classes -- FOL has no analog, since its persistent records are
always immutable. Module-level definitions throughout, same reason as
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

@dataclasses.dataclass
class _MutablePoint(object):  # not frozen
    x: float
    y: float


def _run_mutable_dataclass(n):
    p = _MutablePoint(0.0, 0.0)
    i = 0
    while i < n:
        p.x = p.x + 1.0
        p.y = p.y + 2.0
        i += 1
    return p


def test_mutable_dataclass_direct_field_writes():
    expected = _run_mutable_dataclass(23)
    transformed = try_transform(_run_mutable_dataclass)
    assert transformed is not None
    cell = _cell_for(_run_mutable_dataclass.__module__, "_MutablePoint")
    _check_dual_path(transformed, cell, 23, expected)


class _PlainCounter(object):
    def __init__(self, n, total):
        self.n = n
        self.total = total

    def __eq__(self, other):
        return isinstance(other, _PlainCounter) and self.n == other.n and self.total == other.total


def _run_plain_class(n):
    c = _PlainCounter(0, 0.0)
    i = 0
    while i < n:
        c.n = c.n + 1
        c.total = c.total + c.n
        i += 1
    return c


def test_plain_class_direct_field_writes():
    """Plain classes have no dataclasses.fields() to consult -- fields
    are inferred from __init__'s own `self.<name> = <name>` assignments
    (guard._infer_plain_class_fields)."""
    expected = _run_plain_class(19)
    transformed = try_transform(_run_plain_class)
    assert transformed is not None
    cell = _cell_for(_run_plain_class.__module__, "_PlainCounter")
    _check_dual_path(transformed, cell, 19, expected)


class _AnnotatedPlainCounter(object):
    """v1.6: fields inferred from ANNOTATED self-assignment
    (`self.n: int = n`), not just the unannotated form -- a standard
    modern, type-hinted idiom, motivated directly by cpython-asr's own
    corpus study (corpus-study/README.md: arcade's CameraData used this
    exact shape and, before this fix, couldn't be recognized at all)."""

    def __init__(self, n: int, total: float):
        self.n: int = n
        self.total: float = total

    def __eq__(self, other):
        return isinstance(other, _AnnotatedPlainCounter) and self.n == other.n and self.total == other.total


def _run_annotated_plain_class(n):
    c = _AnnotatedPlainCounter(0, 0.0)
    i = 0
    while i < n:
        c.n = c.n + 1
        c.total = c.total + c.n
        i += 1
    return c


def test_plain_class_with_annotated_field_assignment():
    expected = _run_annotated_plain_class(19)
    transformed = try_transform(_run_annotated_plain_class)
    assert transformed is not None
    cell = _cell_for(_run_annotated_plain_class.__module__, "_AnnotatedPlainCounter")
    _check_dual_path(transformed, cell, 19, expected)


class _MixedAnnotatedPlain(object):
    """A mix of annotated and unannotated self-assignment in the same
    __init__ -- each statement is checked independently, so this should
    infer correctly too."""

    def __init__(self, x: float, y):
        self.x: float = x
        self.y = y

    def __eq__(self, other):
        return isinstance(other, _MixedAnnotatedPlain) and self.x == other.x and self.y == other.y


def _run_mixed_annotated_plain(n):
    p = _MixedAnnotatedPlain(0.0, 0.0)
    i = 0
    while i < n:
        p.x = p.x + 1.0
        p.y = p.y + 2.0
        i += 1
    return p


def test_plain_class_with_mixed_annotated_and_unannotated_fields():
    expected = _run_mixed_annotated_plain(11)
    transformed = try_transform(_run_mixed_annotated_plain)
    assert transformed is not None
    cell = _cell_for(_run_mixed_annotated_plain.__module__, "_MixedAnnotatedPlain")
    _check_dual_path(transformed, cell, 11, expected)


@dataclasses.dataclass
class _ConditionalPoint(object):
    x: float
    y: float


def _run_conditional_mutation(n):
    p = _ConditionalPoint(0.0, 0.0)
    i = 0
    while i < n:
        if i % 2 == 0:
            p.x = p.x + 1.0
        else:
            p.y = p.y + 1.0
        i += 1
    return p


def test_mutate_mode_handles_conditional_field_writes_with_no_special_casing():
    """Unlike reconstruct mode's branch support, mutation mode needs no
    mandatory-else or per-branch-touches-every-field bookkeeping at
    all -- an `if` with no `else`, touching only one field per branch,
    just works, since each write is an ordinary in-place substitution
    exactly where the original statement was."""
    expected = _run_conditional_mutation(19)
    transformed = try_transform(_run_conditional_mutation)
    assert transformed is not None
    cell = _cell_for(_run_conditional_mutation.__module__, "_ConditionalPoint")
    _check_dual_path(transformed, cell, 19, expected)


@dataclasses.dataclass
class _ReadOnlyFieldPoint(object):
    x: float
    tag: str


_read_only_field_side_total = 0


def _run_field_never_written(n):
    """total_len isn't returned -- the transform's tail-shape support
    is only `return p` / `return p, q, ...` naming exactly the
    processed accumulators, so a plain int alongside p in the return
    would be declined for an unrelated reason. Recorded as a
    module-level side effect instead, purely to confirm p.tag's
    read-only value is still correctly visible throughout the loop."""
    global _read_only_field_side_total
    p = _ReadOnlyFieldPoint(0.0, "hello")
    i = 0
    while i < n:
        p.x = p.x + 1.0
        _read_only_field_side_total += len(p.tag)  # p.tag is only ever read, never written
        i += 1
    return p


def test_field_that_is_only_ever_read_stays_correct():
    global _read_only_field_side_total
    _read_only_field_side_total = 0
    expected = _run_field_never_written(11)
    expected_total = _read_only_field_side_total

    transformed = try_transform(_run_field_never_written)
    assert transformed is not None
    cell = _cell_for(_run_field_never_written.__module__, "_ReadOnlyFieldPoint")

    cell.valid = True
    _read_only_field_side_total = 0
    assert transformed(11) == expected
    assert _read_only_field_side_total == expected_total

    cell.valid = False
    _read_only_field_side_total = 0
    assert transformed(11) == expected
    assert _read_only_field_side_total == expected_total
    cell.valid = True


@dataclasses.dataclass
class _WhileTestFieldPoint(object):
    x: float


def _run_while_test_reads_field(n):
    p = _WhileTestFieldPoint(0.0)
    while p.x < n:
        p.x = p.x + 1.0
    return p


def test_while_test_referencing_a_field_is_recognized():
    """The while loop's own condition, not just its body, can read a
    field -- exercises the hardening added to the escape-check walk
    alongside this feature (see _analyze_mutation_loop_body's docstring)."""
    expected = _run_while_test_reads_field(7)
    transformed = try_transform(_run_while_test_reads_field)
    assert transformed is not None
    cell = _cell_for(_run_while_test_reads_field.__module__, "_WhileTestFieldPoint")
    _check_dual_path(transformed, cell, 7, expected)


@dataclasses.dataclass(frozen=True)
class _FrozenHalf(object):
    x: float


@dataclasses.dataclass
class _MutableHalf(object):
    y: float


def _run_mixed_reconstruct_and_mutate(n):
    p = _FrozenHalf(0.0)
    q = _MutableHalf(0.0)
    i = 0
    while i < n:
        p = _FrozenHalf(p.x + 1.0)  # reconstruct mode
        q.y = q.y + 2.0  # mutate mode
        i += 1
    return p, q


def test_mixed_reconstruct_and_mutate_accumulators_in_one_function():
    """The multi-accumulator fixpoint composes both unboxing strategies
    in the same function for free -- each accumulator dispatches on its
    own class's mode independently."""
    expected = _run_mixed_reconstruct_and_mutate(17)
    transformed = try_transform(_run_mixed_reconstruct_and_mutate)
    assert transformed is not None
    cell_p = _cell_for(_run_mixed_reconstruct_and_mutate.__module__, "_FrozenHalf")
    cell_q = _cell_for(_run_mixed_reconstruct_and_mutate.__module__, "_MutableHalf")
    cell_p.valid = True
    cell_q.valid = True
    assert transformed(17) == expected
    cell_p.valid = False
    assert transformed(17) == expected
    cell_p.valid = True
    cell_q.valid = False
    assert transformed(17) == expected
    cell_p.valid = True
    cell_q.valid = True


@dataclasses.dataclass
class _TwoMutablePoint(object):
    x: float


def _run_two_independent_mutate_accumulators(n):
    """Regression test for the bug caught while building this feature:
    mutate mode deliberately keeps the original `p = ClassName(...)`
    assign in place (the real object must survive to be returned), so
    without _find_accumulator's `already_processed` tracking, the
    fixpoint would keep re-discovering the FIRST mutate-mode
    accumulator's still-present assign and never reach a second one."""
    p = _TwoMutablePoint(0.0)
    q = _TwoMutablePoint(100.0)
    i = 0
    while i < n:
        p.x = p.x + 1.0
        q.x = q.x - 1.0
        i += 1
    return p, q


def test_two_independent_mutate_accumulators():
    expected = _run_two_independent_mutate_accumulators(13)
    transformed = try_transform(_run_two_independent_mutate_accumulators)
    assert transformed is not None
    cell = _cell_for(_run_two_independent_mutate_accumulators.__module__, "_TwoMutablePoint")
    _check_dual_path(transformed, cell, 13, expected)


_identity_init_calls = 0


@dataclasses.dataclass(init=False)
class _IdentityPoint(object):
    x: float

    def __init__(self, x):
        global _identity_init_calls
        _identity_init_calls += 1
        self.x = x


def _run_identity_check(n):
    p = _IdentityPoint(0.0)
    i = 0
    while i < n:
        p.x = p.x + 1.0
        i += 1
    return p


def test_mutate_mode_never_reconstructs_only_writes_back():
    """__init__ must be invoked exactly once per call -- at the
    original pre-loop construction -- never again at the end for the
    final writeback. Unlike reconstruct mode's rebox (which calls the
    constructor again every time), mutate mode writes the final
    scalars directly into the real, already-existing object, which is
    what preserves its identity across the transform."""
    global _identity_init_calls
    transformed = try_transform(_run_identity_check)
    assert transformed is not None
    cell = _cell_for(_run_identity_check.__module__, "_IdentityPoint")

    cell.valid = True
    _identity_init_calls = 0
    result = transformed(5)
    assert result == _IdentityPoint(5.0)  # a second __init__ call here, not counted below
    assert _identity_init_calls == 2  # 1 from transformed(5), 1 from the comparison operand above

    cell.valid = False
    _identity_init_calls = 0
    result = transformed(5)
    assert result.x == 5.0
    assert _identity_init_calls == 1  # fallback path: also constructs exactly once


# --------------------------------------------------------------------------
# Negative / abort-safe cases
# --------------------------------------------------------------------------

@dataclasses.dataclass
class _EscapeMutablePoint(object):
    x: float


def _run_mutable_with_escape(n):
    p = _EscapeMutablePoint(0.0)
    i = 0
    while i < n:
        print(p)  # bare accumulator escape -- a real hazard for a mutable object
        p.x = p.x + 1.0
        i += 1
    return p


def test_declines_mutable_accumulator_with_bare_escape():
    """Critical for mutable objects specifically: if p aliased out to
    something that reads it mid-loop, scalarizing its fields would let
    that alias observe stale data -- not just a missed optimization,
    an actual correctness hazard. Same escape check as reconstruct
    mode, applied here too."""
    assert try_transform(_run_mutable_with_escape) is None


@dataclasses.dataclass
class _UnwrittenMutablePoint(object):
    x: float
    y: float


def _run_mutable_never_written(n):
    p = _UnwrittenMutablePoint(0.0, 0.0)
    i = 0
    total = 0.0
    while i < n:
        total += p.x  # only ever read, never written -- not loop-carried
        i += 1
    return p


def test_declines_mutable_accumulator_never_written():
    """A mutable-classed local that's never actually mutated in the
    loop isn't a loop-carried accumulator -- nothing to scalarize."""
    assert try_transform(_run_mutable_never_written) is None


class _ComputedInitPlain(object):
    def __init__(self, x):
        self.x = x * 2  # computed, not a flat passthrough -- can't infer fields safely


def _run_plain_with_computed_init(n):
    p = _ComputedInitPlain(0.0)
    i = 0
    while i < n:
        p.x = p.x + 1.0
        i += 1
    return p


def test_declines_plain_class_with_computed_init_body():
    """guard._infer_plain_class_fields only recognizes a flat sequence
    of `self.<name> = <name>` assignments; anything else (computed
    values here) aborts inference, so the class doesn't qualify at
    all."""
    assert try_transform(_run_plain_with_computed_init) is None


class _ComputedAnnotatedInitPlain(object):
    def __init__(self, x: float):
        self.x: float = x * 2  # computed, still aborts inference even though annotated


def _run_plain_with_computed_annotated_init(n):
    p = _ComputedAnnotatedInitPlain(0.0)
    i = 0
    while i < n:
        p.x = p.x + 1.0
        i += 1
    return p


def test_declines_plain_class_with_computed_annotated_init_body():
    """v1.6's AnnAssign support only extends WHICH assignment shape is
    recognized (`self.x: T = x`), not what counts as "flat" -- a
    computed value is still declined even when annotated."""
    assert try_transform(_run_plain_with_computed_annotated_init) is None


class _BareAnnotationPlain(object):
    def __init__(self, x: float):
        self.x: float  # annotation with no value -- not an assignment at all
        self.x = x


def _run_plain_with_bare_annotation(n):
    p = _BareAnnotationPlain(0.0)
    i = 0
    while i < n:
        p.x = p.x + 1.0
        i += 1
    return p


def test_declines_plain_class_with_bare_annotation_no_value():
    """`self.x: float` with no `= value` is `ast.AnnAssign(value=None)`
    -- not an assignment, so it can't be a flat self.x = x passthrough;
    correctly declined rather than crashing on a None value."""
    assert try_transform(_run_plain_with_bare_annotation) is None


class _CustomSetattrPlain(object):
    def __init__(self, x):
        self.x = x

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)  # a real override, even if benign here


def _run_plain_with_custom_setattr(n):
    p = _CustomSetattrPlain(0.0)
    i = 0
    while i < n:
        p.x = p.x + 1.0
        i += 1
    return p


def test_declines_class_with_custom_setattr():
    """A custom __setattr__ might have side effects (validation,
    logging, cache invalidation) that mutate mode's batched final
    writeback would only invoke once instead of once per original
    write -- guard.mutation_safe rules this out at qualification,
    regardless of what the override actually does."""
    assert try_transform(_run_plain_with_custom_setattr) is None


@dataclasses.dataclass
class _TypoFieldPoint(object):
    x: float


def _run_mutable_with_unknown_field(n):
    p = _TypoFieldPoint(0.0)
    i = 0
    while i < n:
        p.x = p.x + 1.0
        p.y = 0.0  # not a declared field
        i += 1
    return p


def test_declines_write_to_field_not_in_class():
    assert try_transform(_run_mutable_with_unknown_field) is None


@dataclasses.dataclass
class _DeletedFieldPoint(object):
    x: float


def _run_mutable_with_deletion(n):
    p = _DeletedFieldPoint(0.0)
    i = 0
    while i < n:
        p.x = p.x + 1.0
        if i == 0:
            del p.x
            p.x = 0.0
        i += 1
    return p


def test_declines_field_deletion():
    assert try_transform(_run_mutable_with_deletion) is None


@dataclasses.dataclass
class _ReconstructStyleMutablePoint(object):  # not frozen
    x: float


def _run_reconstruction_style_on_mutable(n):
    p = _ReconstructStyleMutablePoint(0.0)
    i = 0
    while i < n:
        p = _ReconstructStyleMutablePoint(p.x + 1.0)  # rebind, not mutation
        i += 1
    return p


def test_declines_reconstruction_style_rebind_on_non_frozen_dataclass():
    """Non-frozen dataclasses get mutate mode only, matching their
    idiomatic usage -- a reconstruction-style rebind is a bare
    reference to p under mutate-mode's escape analysis (Name(p,
    Store)), so it's declined the same way any other escape would be.
    Documents a deliberate scope boundary, not a bug: someone using a
    mutable class who wants reconstruction-style updates should use a
    frozen dataclass instead."""
    assert try_transform(_run_reconstruction_style_on_mutable) is None


class _NoInitPlain(object):
    pass


def _run_plain_no_init(n):
    p = _NoInitPlain()
    i = 0
    while i < n:
        p.x = i
        i += 1
    return p


def test_declines_plain_class_with_no_init():
    """No __init__ to infer fields from at all -- and the bare
    constructor call `_NoInitPlain()` wouldn't supply any fields
    either way."""
    assert try_transform(_run_plain_no_init) is None
