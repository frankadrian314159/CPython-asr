"""v1.5: automatic application via a sys.meta_path import hook. Real
file-backed module + fresh import (and, separately, importlib.reload())
to verify functions get ASR'd with no @asr decoration in the source at
all, once their module name has been enable()'d -- same real-module
pattern as test_guard.py/test_guard_mutation.py, for the same reason:
import and reload() both require an actual importable module, not a
module-level fixture def.
"""

import importlib
import sys

import pytest

from asr import autopatch, guard


MODULE_NAME = "_asr_autopatch_fixture"

SOURCE_NO_DECORATOR = '''
import dataclasses

@dataclasses.dataclass(frozen=True)
class Point:
    x: float
    y: float

def run_simulation(n):
    p = Point(0.0, 0.0)
    i = 0
    while i < n:
        p = Point(p.x + 1.0, p.y + 2.0)
        i += 1
    return p

def not_a_loop(n):
    return n * 2
'''

# v2 changes Point's field set -- same shape test_guard.py's reload
# tests use, repeated here to confirm the world guard still protects an
# auto-transformed function exactly the same way it protects a manually
# @asr-decorated one.
SOURCE_V2_FIELDS_CHANGED = '''
import dataclasses

@dataclasses.dataclass(frozen=True)
class Point:
    x: float
    y: float
    z: float = 0.0

def run_simulation(n):
    p = Point(0.0, 0.0, 0.0)
    i = 0
    while i < n:
        p = Point(p.x + 1.0, p.y + 2.0, p.z)
        i += 1
    return p

def not_a_loop(n):
    return n * 2
'''

SUBPACKAGE_MODULE_NAME = "_asr_autopatch_pkg.sub"


@pytest.fixture
def fixture_module(tmp_path):
    sys.path.insert(0, str(tmp_path))
    guard.reset()
    autopatch.reset()
    (tmp_path / f"{MODULE_NAME}.py").write_text(SOURCE_NO_DECORATOR)
    yield tmp_path
    if MODULE_NAME in sys.modules:
        del sys.modules[MODULE_NAME]
    sys.path.remove(str(tmp_path))
    guard.reset()
    autopatch.reset()


def test_disabled_module_is_not_transformed(fixture_module):
    module = importlib.import_module(MODULE_NAME)
    assert getattr(module.run_simulation, "__asr_transformed__", False) is False
    assert module.run_simulation(5) == module.Point(5.0, 10.0)


def test_enabled_module_is_transformed_automatically_on_import(fixture_module):
    autopatch.enable(MODULE_NAME)
    module = importlib.import_module(MODULE_NAME)
    assert module.run_simulation.__asr_transformed__ is True
    assert module.run_simulation(5) == module.Point(5.0, 10.0)


def test_enabled_module_leaves_non_qualifying_functions_alone(fixture_module):
    autopatch.enable(MODULE_NAME)
    module = importlib.import_module(MODULE_NAME)
    assert getattr(module.not_a_loop, "__asr_transformed__", False) is False
    assert module.not_a_loop(5) == 10


def test_auto_transformed_function_still_falls_back_correctly(fixture_module):
    autopatch.enable(MODULE_NAME)
    module = importlib.import_module(MODULE_NAME)
    cell = guard._registry[(MODULE_NAME, "Point")]
    assert cell.valid is True
    assert module.run_simulation(5) == module.Point(5.0, 10.0)

    cell.valid = False
    assert module.run_simulation(5) == module.Point(5.0, 10.0)
    cell.valid = True


def test_reload_of_auto_transformed_module_re_transforms(fixture_module):
    """The module's __spec__.loader IS the wrapping loader installed at
    first import, so importlib.reload() naturally re-triggers the
    auto-transform pass too -- no separate hook needed for reload."""
    autopatch.enable(MODULE_NAME)
    module = importlib.import_module(MODULE_NAME)
    assert module.run_simulation.__asr_transformed__ is True

    importlib.reload(module)
    assert module.run_simulation.__asr_transformed__ is True
    assert module.run_simulation(3) == module.Point(3.0, 6.0)


def test_reload_with_changed_fields_invalidates_the_auto_transformed_fast_path(fixture_module):
    autopatch.enable(MODULE_NAME)
    module = importlib.import_module(MODULE_NAME)
    cell = guard._registry[(MODULE_NAME, "Point")]
    assert cell.valid is True

    (fixture_module / f"{MODULE_NAME}.py").write_text(SOURCE_V2_FIELDS_CHANGED)
    importlib.reload(module)
    assert cell.valid is False

    assert module.run_simulation(3) == module.Point(3.0, 6.0, 0.0)


def test_disabling_stops_future_transforms_but_not_already_loaded_ones(fixture_module):
    autopatch.enable(MODULE_NAME)
    module = importlib.import_module(MODULE_NAME)
    assert module.run_simulation.__asr_transformed__ is True

    autopatch.disable(MODULE_NAME)
    # A fresh reload after disabling re-executes the plain source with
    # no auto-transform pass applied -- the freshly (re)defined
    # run_simulation is the untransformed function again.
    importlib.reload(module)
    assert getattr(module.run_simulation, "__asr_transformed__", False) is False
    assert module.run_simulation(3) == module.Point(3.0, 6.0)


@pytest.fixture
def fixture_package(tmp_path):
    sys.path.insert(0, str(tmp_path))
    guard.reset()
    autopatch.reset()
    pkg_dir = tmp_path / "_asr_autopatch_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "sub.py").write_text(SOURCE_NO_DECORATOR)
    yield tmp_path
    for name in ("_asr_autopatch_pkg", SUBPACKAGE_MODULE_NAME):
        if name in sys.modules:
            del sys.modules[name]
    sys.path.remove(str(tmp_path))
    guard.reset()
    autopatch.reset()


def test_enabling_a_package_covers_its_submodules(fixture_package):
    """enable("pkg") matches "pkg.sub" too, by dotted-prefix -- you opt
    a whole package in once, not each of its modules individually."""
    autopatch.enable("_asr_autopatch_pkg")
    module = importlib.import_module(SUBPACKAGE_MODULE_NAME)
    assert module.run_simulation.__asr_transformed__ is True
    assert module.run_simulation(5) == module.Point(5.0, 10.0)
