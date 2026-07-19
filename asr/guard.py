"""World guard for CPython ASR.

A global registry of (module_name, class_name) -> validity cell,
invalidated when importlib.reload() changes a tracked class's shape.
Direct analog of FOL's emit-defclass hook / register-region: the guard
a transformed function's fast path checks is a single attribute read,
and invalidation is monotonic (valid -> invalid, never revived), exactly
as in FOL. No thread-safety claims beyond what the paper itself now
(honestly) makes -- this is unverified under concurrent load.

class_fields/mutation_safe (v1.4) are the single source of truth for
"what are this class's fields" and "is direct attribute mutation on its
instances safe" -- transform.py's qualification and this module's own
reload-time invalidation both build on them, so they always agree.
"""

import ast
import dataclasses
import importlib
import inspect
import textwrap


class ValidityCell:
    __slots__ = ("valid",)

    def __init__(self):
        self.valid = True


_registry = {}       # (module_name, class_name) -> ValidityCell
_tracked_fields = {}  # (module_name, class_name) -> frozenset of field names
_tracked_mode = {}    # (module_name, class_name) -> "reconstruct" | "mutate"


def register(module_name, class_name, fields, mode):
    """Called once per transformed function that depends on this class.
    Returns the ValidityCell the fast path should check."""
    key = (module_name, class_name)
    cell = _registry.get(key)
    if cell is None:
        cell = ValidityCell()
        _registry[key] = cell
        _tracked_fields[key] = frozenset(fields)
        _tracked_mode[key] = mode
    return cell


def _infer_plain_class_fields(cls):
    """Infer a plain (non-dataclass) class's field set by introspecting
    __init__'s source for a flat sequence of `self.<name> = <name>`
    assignments -- one per parameter, in order, name for name. Returns
    an ordered tuple of field names, or None if __init__ is missing,
    its source isn't available, or its body does anything more complex
    (computed values, conditional logic, a call to super().__init__,
    reordered or partial assignment, etc.). Deliberately narrow and
    conservative -- good enough to recognize the common case,
    safe-by-abort otherwise, same discipline as every other shape this
    project recognizes."""
    init = cls.__dict__.get("__init__")
    if init is None or not inspect.isfunction(init):
        return None
    try:
        src = textwrap.dedent(inspect.getsource(init))
        tree = ast.parse(src)
    except (OSError, TypeError, SyntaxError):
        return None
    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
        return None
    init_def = tree.body[0]
    params = [a.arg for a in init_def.args.args]
    if not params or params[0] != "self" or len(params) == 1:
        return None
    param_names = params[1:]

    body = init_def.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    fields = []
    for stmt in body:
        if not (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Attribute)
            and isinstance(stmt.targets[0].value, ast.Name)
            and stmt.targets[0].value.id == "self"
            and isinstance(stmt.value, ast.Name)
            and stmt.value.id in param_names
        ):
            return None  # anything beyond a flat self.field = param aborts inference
        fields.append(stmt.targets[0].attr)

    if fields != param_names or len(set(fields)) != len(fields):
        return None  # keep field order == __init__'s own parameter order, no reordering
    return tuple(fields)


def class_fields(cls):
    """Ordered tuple of cls's field names -- a dataclass's declared
    fields (declaration order), or a plain class's inferred __init__
    parameter order -- or None if cls qualifies as neither."""
    if not isinstance(cls, type):
        return None
    if dataclasses.is_dataclass(cls):
        return tuple(f.name for f in dataclasses.fields(cls))
    return _infer_plain_class_fields(cls)


def mutation_safe(cls):
    """True when direct attribute writes on instances of cls are safe
    to batch into a single final writeback instead of one write per
    original assignment: plain object.__setattr__, no override (rules
    out a custom __setattr__ with validation/logging/etc. side effects
    the batched rewrite would then under-invoke, and rules out frozen
    dataclasses, whose generated __setattr__ raises)."""
    if not isinstance(cls, type):
        return False
    if cls.__setattr__ is not object.__setattr__:
        return False
    params = getattr(cls, "__dataclass_params__", None)
    return not (params is not None and params.frozen)


def _current_fields(cls):
    fields = class_fields(cls)
    return frozenset(fields) if fields is not None else None


def _invalidate_if_changed(module_name, class_name, cls):
    key = (module_name, class_name)
    cell = _registry.get(key)
    if cell is None or not cell.valid:
        return
    current_fields = _current_fields(cls)
    if current_fields is None or current_fields != _tracked_fields[key]:
        cell.valid = False
        return
    if _tracked_mode[key] == "mutate" and not mutation_safe(cls):
        # The fast path's final writeback does a direct attribute
        # write, batched instead of one write per original assignment
        # -- only safe while the class's __setattr__ stays the default
        # and it hasn't become a frozen dataclass (see mutation_safe).
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
    _tracked_mode.clear()
