"""Automatic application (v1.5): a sys.meta_path import hook that
applies ASR to every qualifying while loop in an enabled package's
modules, with no per-function @asr decoration needed -- the "no opt-in
per callsite" ideal FOL itself has at the language level (it applies to
every loop the compiler sees), brought as close as CPython's import
system reasonably allows.

Deliberately scoped per-package rather than truly global: enabling this
for the entire interpreter -- every import, including the standard
library and arbitrary third-party (often C-extension-backed) packages
-- would be both slow (an inspect.getsource + ast.parse attempt on
every function in every imported module) and unsafe (attempting to
transform code whose semantics this project's narrow recognizer was
never validated against). enable(name) opts a package or module in by
name, plus any dotted submodule of it; nothing outside that set is ever
touched.

Only module-level plain functions are considered -- not methods, nested
functions, or anything imported from elsewhere into the enabled
module's namespace (checked via __module__) -- the same scope the @asr
decorator itself has always had; this hook is just a different way of
reaching try_transform, not a broader recognizer.
"""

import importlib.abc
import inspect
import sys

from . import transform

_enabled_prefixes = set()
_installed = False


def enable(name):
    """Opt `name` (and any dotted submodule of it, e.g. "pkg.sub") into
    automatic ASR: every module-level plain function recognized as a
    qualifying accumulator loop gets transformed on import, with no
    @asr needed. Must be called before `name` is first imported --
    like any import hook, this can't retroactively affect an
    already-loaded module (re-importing an already-cached module is a
    no-op in Python; use importlib.reload if it must be re-processed)."""
    _enabled_prefixes.add(name)
    install()


def disable(name):
    """Stop auto-transforming `name` on future imports. Does not affect
    a module already imported and transformed."""
    _enabled_prefixes.discard(name)


def _is_enabled(module_name):
    return any(module_name == p or module_name.startswith(p + ".") for p in _enabled_prefixes)


def _transform_module_functions(module):
    for attr_name, value in list(vars(module).items()):
        if (
            inspect.isfunction(value)
            and getattr(value, "__module__", None) == module.__name__
            and not getattr(value, "__asr_transformed__", False)
        ):
            new_func = transform.try_transform(value)
            if new_func is not None:
                setattr(module, attr_name, new_func)


class _AutoAsrLoader:
    """Wraps a real loader, running the auto-transform pass right after
    the module's own exec_module finishes -- including on a later
    importlib.reload(), which re-invokes the module's __spec__.loader,
    i.e. this same wrapper, so a reload of an auto-patched module
    re-transforms its (freshly redefined) functions automatically too.

    Not an importlib.abc.Loader subclass on purpose: every attribute
    this wrapper doesn't explicitly override is delegated to the real
    loader via __getattr__, so tooling that introspects the loader
    (inspect.getsource, pickling, is_package, get_source, ...) keeps
    working exactly as it would without this wrapper -- inheriting the
    ABC would risk its own default method implementations shadowing
    that delegation for names this wrapper doesn't define."""

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def create_module(self, spec):
        return self._wrapped.create_module(spec)

    def exec_module(self, module):
        self._wrapped.exec_module(module)
        _transform_module_functions(module)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


class _AutoAsrFinder(importlib.abc.MetaPathFinder):
    """Only intercepts imports of enabled names; everything else is
    left to the rest of sys.meta_path untouched, so this is a cheap
    early-exit for the vast majority of imports in a real process.

    Delegates spec-finding to the REST of sys.meta_path (skipping
    itself) rather than reimplementing path search -- this must never
    call importlib.util.find_spec or similar, which walks the FULL
    meta_path again, including this finder, and would recurse forever."""

    def find_spec(self, name, path, target=None):
        if not _is_enabled(name):
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec = find_spec(name, path, target)
            if spec is not None:
                if spec.loader is not None:
                    spec.loader = _AutoAsrLoader(spec.loader)
                return spec
        return None


_finder = _AutoAsrFinder()


def install():
    """Insert the auto-ASR finder into sys.meta_path. Idempotent;
    called automatically by enable(), and by asr/__init__.py at import
    time (harmless no-op until something is actually enabled, since
    find_spec returns None immediately for anything not enabled)."""
    global _installed
    if not _installed:
        sys.meta_path.insert(0, _finder)
        _installed = True


def uninstall():
    """Remove the auto-ASR finder. Mainly for tests."""
    global _installed
    if _finder in sys.meta_path:
        sys.meta_path.remove(_finder)
    _installed = False


def reset():
    """Test-only: clear the enabled-name registry and uninstall."""
    _enabled_prefixes.clear()
    uninstall()
