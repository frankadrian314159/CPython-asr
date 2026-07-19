"""World-guard tests: real importlib.reload()-driven invalidation against
an actual file-backed module, since reload() requires one. Mirrors FOL's
own redefinition tests: reload before first call -> fast path never
taken; reload after the fast path has already run once -> next call
falls back correctly; no reload -> fast path stays hot throughout.
"""

import importlib
import sys

import pytest

from asr import guard
from asr.decorator import asr  # noqa: F401  (ensures guard.install() ran)


MODULE_NAME = "_asr_guard_fixture"

SOURCE_V1 = '''
import dataclasses
from asr import asr

@dataclasses.dataclass(frozen=True)
class Point:
    x: float
    y: float

@asr
def run_simulation(n):
    p = Point(0.0, 0.0)
    i = 0
    while i < n:
        p = Point(p.x + 1.0, p.y + 2.0)
        i += 1
    return p
'''

# v2 changes Point's field set -- the guard must catch this.
SOURCE_V2 = '''
import dataclasses
from asr import asr

@dataclasses.dataclass(frozen=True)
class Point:
    x: float
    y: float
    z: float = 0.0

@asr
def run_simulation(n):
    p = Point(0.0, 0.0, 0.0)
    i = 0
    while i < n:
        p = Point(p.x + 1.0, p.y + 2.0, p.z)
        i += 1
    return p
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


def test_no_reload_stays_hot(fixture_module):
    module, _ = fixture_module
    cell = guard._registry[(MODULE_NAME, "Point")]
    assert cell.valid is True
    result = module.run_simulation(5)
    assert result == module.Point(5.0, 10.0)
    assert cell.valid is True


def test_reload_before_first_call_invalidates(fixture_module):
    module, tmp_path = fixture_module
    cell = guard._registry[(MODULE_NAME, "Point")]
    assert cell.valid is True

    (tmp_path / f"{MODULE_NAME}.py").write_text(SOURCE_V2)
    importlib.reload(module)

    # register() reuses the same cell for a given (module, class) key
    # rather than minting a fresh one per call, and reload() both
    # redefines Point AND re-decorates run_simulation in one atomic
    # re-execution -- so the freshly-recompiled fast path (which would
    # in fact be correct for the new 3-field layout) gets conservatively
    # invalidated too, since the guard only compares against the
    # originally-tracked field set. That's a missed optimization, not a
    # correctness bug -- matches the paper's own "inapplicable is a
    # cheap failure mode" stance. What must hold is that the result is
    # still right regardless of which path is taken.
    assert guard._registry[(MODULE_NAME, "Point")].valid is False

    result = module.run_simulation(3)
    assert result == module.Point(3.0, 6.0, 0.0)


def test_reload_after_fast_path_ran_falls_back_correctly(fixture_module):
    module, tmp_path = fixture_module
    cell = guard._registry[(MODULE_NAME, "Point")]

    # Run once while still valid -- exercises the fast path.
    first = module.run_simulation(4)
    assert first == module.Point(4.0, 8.0)
    assert cell.valid is True

    (tmp_path / f"{MODULE_NAME}.py").write_text(SOURCE_V2)
    importlib.reload(module)
    assert cell.valid is False

    second = module.run_simulation(4)
    assert second == module.Point(4.0, 8.0, 0.0)
