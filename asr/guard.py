"""World guard for CPython ASR.

A global registry of (module_name, class_name) -> validity cell,
invalidated when importlib.reload() changes a tracked dataclass's field
set. Direct analog of FOL's emit-defclass hook / register-region: the
guard a transformed function's fast path checks is a single attribute
read, and invalidation is monotonic (valid -> invalid, never revived),
exactly as in FOL. No thread-safety claims beyond what the paper itself
now (honestly) makes -- this is unverified under concurrent load.
"""

import dataclasses
import importlib


class ValidityCell:
    __slots__ = ("valid",)

    def __init__(self):
        self.valid = True


_registry = {}       # (module_name, class_name) -> ValidityCell
_tracked_fields = {}  # (module_name, class_name) -> frozenset of field names


def register(module_name, class_name, fields):
    """Called once per transformed function that depends on this class.
    Returns the ValidityCell the fast path should check."""
    key = (module_name, class_name)
    cell = _registry.get(key)
    if cell is None:
        cell = ValidityCell()
        _registry[key] = cell
        _tracked_fields[key] = frozenset(fields)
    return cell


def _invalidate_if_changed(module_name, class_name, cls):
    key = (module_name, class_name)
    cell = _registry.get(key)
    if cell is None or not cell.valid:
        return
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
        cell.valid = False
        return
    current_fields = frozenset(f.name for f in dataclasses.fields(cls))
    if current_fields != _tracked_fields[key]:
        cell.valid = False


_real_reload = importlib.reload
_installed = False


def _guarded_reload(module):
    result = _real_reload(module)
    module_name = module.__name__
    for name, value in vars(module).items():
        if isinstance(value, type):
            _invalidate_if_changed(module_name, name, value)
    return result


def install():
    """Monkeypatch importlib.reload so class redefinitions become
    visible to every registered fast path. Idempotent."""
    global _installed
    if not _installed:
        importlib.reload = _guarded_reload
        _installed = True


def uninstall():
    """Restore the real importlib.reload. Mainly for tests."""
    global _installed
    importlib.reload = _real_reload
    _installed = False


def reset():
    """Clear the registry. For tests only -- never called by the guard
    itself, since invalidation must stay monotonic in real use."""
    _registry.clear()
    _tracked_fields.clear()
