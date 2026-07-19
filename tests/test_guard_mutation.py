"""World-guard tests specific to v1.4's mutate mode: real
importlib.reload()-driven invalidation covering the two new hazards
mutate mode introduces that reconstruct mode never had --
guard.mutation_safe (a class becoming frozen, or gaining a custom
__setattr__, after a fast path was already compiled to write its
instances' attributes directly) -- on top of the field-set check
test_guard.py already covers. Same real file-backed module + reload
pattern as test_guard.py, for the same reason: reload() requires an
actual importable module.
"""

import importlib
import sys

import pytest

from asr import guard
from asr.decorator import asr  # noqa: F401  (ensures guard.install() ran)


MODULE_NAME = "_asr_guard_mutation_fixture"

SOURCE_V1 = '''
import dataclasses
from asr import asr

@dataclasses.dataclass
class Counter:
    n: float

@asr
def run_counter(iters):
    c = Counter(0.0)
    i = 0
    while i < iters:
        c.n = c.n + 1.0
        i += 1
    return c
'''

# v2: Counter becomes a FROZEN dataclass, same field set -- the fast
# path's final writeback (a direct c.n = ... attribute assignment)
# would raise dataclasses.FrozenInstanceError if this weren't caught.
SOURCE_V2_FROZEN = '''
import dataclasses
from asr import asr

@dataclasses.dataclass(frozen=True)
class Counter:
    n: float

@asr
def run_counter(iters):
    c = Counter(0.0)
    i = 0
    while i < iters:
        c.n = c.n + 1.0
        i += 1
    return c
'''

# v2: Counter gains a custom __setattr__, same field set -- the fast
# path's batched writeback would silently invoke it once instead of
# once per original assignment if this weren't caught.
SOURCE_V2_CUSTOM_SETATTR = '''
import dataclasses
from asr import asr

@dataclasses.dataclass
class Counter:
    n: float

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

@asr
def run_counter(iters):
    c = Counter(0.0)
    i = 0
    while i < iters:
        c.n = c.n + 1.0
        i += 1
    return c
'''

# v2: Counter's field set changes -- the ordinary case test_guard.py
# already covers for reconstruct mode, repeated here for mutate mode.
SOURCE_V2_FIELDS_CHANGED = '''
import dataclasses
from asr import asr

@dataclasses.dataclass
class Counter:
    n: float
    extra: float = 0.0

@asr
def run_counter(iters):
    c = Counter(0.0, 0.0)
    i = 0
    while i < iters:
        c.n = c.n + 1.0
        i += 1
    return c
'''


@pytest.fixture
def fixture_module(tmp_path):
    sys.path.insert(0, str(tmp_path))
    guard.reset()
    (tmp_path / f"{MODULE_NAME}.py").write_text(SOURCE_V1)
    module = importlib.import_module(MODULE_NAME)
    yield module, tmp_path
    del sys.modules[MODULE_NAME]
    sys.path.remove(str(tmp_path))
    guard.reset()


def test_mutate_mode_no_reload_stays_hot(fixture_module):
    module, _ = fixture_module
    cell = guard._registry[(MODULE_NAME, "Counter")]
    assert cell.valid is True
    assert module.run_counter(5) == module.Counter(5.0)
    assert cell.valid is True


def test_mutate_mode_reload_to_frozen_invalidates(fixture_module):
    """Reconstruct mode doesn't care if a class becomes frozen (calling
    the constructor again still works either way), but mutate mode's
    writeback does a direct attribute assignment -- unsafe the moment
    the class becomes frozen, even with the same field set. Note this
    reload's own reloaded source keeps the OLD mutation-style body
    (`c.n = ...`), which is invalid Python for a frozen dataclass with
    or without ASR involved at all -- so this test only checks that the
    guard correctly flags the transition, not that calling the
    (unavoidably now-broken) reloaded function succeeds."""
    module, tmp_path = fixture_module
    cell = guard._registry[(MODULE_NAME, "Counter")]
    assert cell.valid is True

    (tmp_path / f"{MODULE_NAME}.py").write_text(SOURCE_V2_FROZEN)
    importlib.reload(module)
    assert cell.valid is False


def test_mutate_mode_reload_to_custom_setattr_invalidates(fixture_module):
    module, tmp_path = fixture_module
    cell = guard._registry[(MODULE_NAME, "Counter")]
    assert cell.valid is True

    (tmp_path / f"{MODULE_NAME}.py").write_text(SOURCE_V2_CUSTOM_SETATTR)
    importlib.reload(module)
    assert cell.valid is False

    assert module.run_counter(4) == module.Counter(4.0)


def test_mutate_mode_reload_with_changed_fields_invalidates(fixture_module):
    module, tmp_path = fixture_module
    cell = guard._registry[(MODULE_NAME, "Counter")]
    assert cell.valid is True

    (tmp_path / f"{MODULE_NAME}.py").write_text(SOURCE_V2_FIELDS_CHANGED)
    importlib.reload(module)
    assert cell.valid is False

    assert module.run_counter(3) == module.Counter(3.0, 0.0)


def test_mutate_mode_reload_after_fast_path_ran_falls_back_correctly(fixture_module):
    """Uses the custom-__setattr__ transition rather than the frozen
    one: `c.n = ...` stays valid Python either way (the override just
    delegates to object.__setattr__), so this can actually exercise
    the reloaded fallback path running to completion and producing the
    right answer -- unlike the frozen-transition test above, where the
    reloaded module's own mutation-style source is inherently broken
    regardless of ASR."""
    module, tmp_path = fixture_module
    cell = guard._registry[(MODULE_NAME, "Counter")]

    first = module.run_counter(4)
    assert first == module.Counter(4.0)
    assert cell.valid is True

    (tmp_path / f"{MODULE_NAME}.py").write_text(SOURCE_V2_CUSTOM_SETATTR)
    importlib.reload(module)
    assert cell.valid is False

    second = module.run_counter(4)
    assert second == module.Counter(4.0)
