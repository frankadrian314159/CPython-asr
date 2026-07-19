"""Gate-faithful second pass: ports the ACTUAL qualification gates
cpython-asr's transform.py/guard.py apply (not just a syntactic-shape
proxy) as closely as static (non-executing) analysis of arbitrary
third-party source allows, and reports what fraction of analyze.py's
candidate sites the real rules would actually accept. Mirrors
docs/cgo2027/corpus-study/src/asr_corpus/classify.clj in the FOL repo.

What's ported, closely mirroring the real implementation:
  - guard.class_fields: dataclass field order from annotations, or a
    plain class's fields inferred from a flat `self.name = name` OR
    `self.name: Type = name` (v1.6) sequence in __init__ (same
    restriction as guard._infer_plain_class_fields -- computed values,
    reordering, or anything else aborts inference).
  - guard.mutation_safe: a class with its own __setattr__ is never
    mutate-mode-safe.
  - _classify_accumulator_class: frozen dataclass -> reconstruct mode;
    non-frozen dataclass or inferable plain class (mutation-safe) ->
    mutate mode; anything else doesn't qualify.
  - _ctor_supplies_all_fields: the init call (and, for reconstruct
    mode, every record-ctor update site) must supply EXACTLY the
    class's fields, no more, no fewer.
  - _resolve_ctor_class / _call_target_matches_class (v1.6): a
    module-qualified `module.ClassName(...)` call is recognized at both
    the pre-loop init site and every reconstruction site, not just the
    bare `ClassName(...)` shape -- this was the previous version's
    single largest source of undercount, verified against real corpus
    code (see corpus-study/README.md's original run). Statically, we
    can't verify `module` actually resolves to the class's own defining
    module (no live import graph here) -- matched by attribute name
    alone against the project-wide registry, the same simplification
    the bare-Name case already makes (no identity check beyond a name
    match either) -- see analyze_project's own collision caveat, which
    now applies equally to qualified calls.
  - The escape check _analyze_loop_body/_analyze_mutation_loop_body
    both apply: every reference to the accumulator name, anywhere in
    the loop, must be a `.field` access for a known field (mutate mode:
    read or write; reconstruct mode: read only, outside the single
    recognized update site) -- a bare reference anywhere disqualifies
    the whole site.
  - dataclasses.replace(acc, ...)/replace(acc, ...): every keyword must
    name a known field (_is_replace_call).
  - One-level helper inlining (_try_inline_call): a call to a
    LOCALLY-DEFINED (same file only -- no cross-file/import resolution)
    helper function whose body is exactly `return <reconstruction>`,
    with the accumulator passed as exactly one bare argument.
    _try_inline_call's own helper-CALL-SITE resolution stays Name-only
    by design, matching transform.py -- v1.6 only extended constructor-
    call resolution, not helper-function resolution.

What's deliberately NOT replicated (documented lower-bound reasons,
same spirit as classify.clj's own header):
  1. No macroexpansion-equivalent: decorators/metaclasses that
     dynamically add fields are invisible (we only see the literal
     source).
  2. No cross-file helper resolution: an inlinable helper defined in a
     different module is scored as an escape, exactly as an un-inlined
     call would be for cpython-asr itself without that helper visible
     in the same __globals__.
  3. Branch/match-shaped reconstruction isn't validated against the
     REAL mandatory-else/every-case-reconstructs restriction -- we
     accept a site as long as EVERY update site found (across however
     many branches) is individually a recognized reconstruction/
     mutation shape, which is necessary but not sufficient for the
     real _try_branch_reconstruction/_try_match_reconstruction gates.
  4. No post-loop return-shape check (`return p` / `return p, q, ...`)
     -- this measures whether the LOOP ITSELF would qualify, not
     whether a complete function around it would transform end to end.
  5. No multi-accumulator fixpoint interaction effects (each
     accumulator is judged independently).
All four push toward UNDERCOUNT relative to a hypothetical perfect
static replica, never overcount -- except (3)/(4), which could in
principle admit a site the real pass would reject; read the qualifying
fraction as an ESTIMATE bracketed by analyze.py's syntactic-shape
proxy above it, not a hard floor or ceiling either direction.

Usage: python classify.py [corpus_dir] [manifest.json] [out.json]
"""

import ast
import dataclasses
import json
import sys
from pathlib import Path

from analyze import (
    EXCLUDE_DIR_NAMES,
    python_files,
    parse_file,
)


# --------------------------------------------------------------------------
# Class registry: ports guard.class_fields / guard.mutation_safe /
# _classify_accumulator_class, statically.
# --------------------------------------------------------------------------

def _decorator_name(dec):
    node = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_dataclass_decorator(dec):
    return _decorator_name(dec) == "dataclass"


def _decorator_is_frozen(dec):
    if not isinstance(dec, ast.Call):
        return False
    for kw in dec.keywords:
        if kw.arg == "frozen" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
            return True
    return False


def _dataclass_fields(class_def):
    """Class-level annotated attributes, in declaration order -- the
    static analog of dataclasses.fields(cls)."""
    fields = []
    for stmt in class_def.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            fields.append(stmt.target.id)
    return tuple(fields)


def _infer_plain_class_fields(class_def):
    """Ports guard._infer_plain_class_fields exactly: __init__'s body
    must be a flat sequence of `self.<name> = <name>` or
    `self.<name>: Type = <name>` (v1.6 -- ast.AnnAssign, a standard
    modern, type-hinted idiom) assignments, one per parameter, in order
    -- name for name, no reordering, nothing computed. Returns an
    ordered tuple, or None if not inferable."""
    init = next(
        (s for s in class_def.body if isinstance(s, ast.FunctionDef) and s.name == "__init__"),
        None,
    )
    if init is None:
        return None
    params = [a.arg for a in init.args.args]
    if not params or params[0] != "self" or len(params) == 1:
        return None
    param_names = params[1:]

    body = init.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    fields = []
    for stmt in body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target, value = stmt.targets[0], stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            target, value = stmt.target, stmt.value
        else:
            return None
        if not (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and isinstance(value, ast.Name)
            and value.id in param_names
        ):
            return None
        fields.append(target.attr)

    if fields != param_names or len(set(fields)) != len(fields):
        return None
    return tuple(fields)


def _defines_own_setattr(class_def):
    return any(
        isinstance(s, ast.FunctionDef) and s.name == "__setattr__" for s in class_def.body
    )


@dataclasses.dataclass
class ClassInfo:
    name: str
    mode: str  # "reconstruct" | "mutate"
    fields: tuple
    helper_defs: dict  # populated later: name -> FunctionDef, for inlining


def build_class_registry(tree):
    """{class_name: ClassInfo} for every LOCALLY-DEFINED class in this
    file that qualifies under _classify_accumulator_class -- mirrors
    guard.class_fields + guard.mutation_safe + the frozen/mutate-mode
    split, statically."""
    registry = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        dataclass_decs = [d for d in node.decorator_list if _is_dataclass_decorator(d)]
        if dataclass_decs:
            fields = _dataclass_fields(node)
            if not fields:
                continue
            frozen = any(_decorator_is_frozen(d) for d in dataclass_decs)
            if frozen:
                registry[node.name] = ClassInfo(node.name, "reconstruct", fields, {})
                continue
            if not _defines_own_setattr(node):
                registry[node.name] = ClassInfo(node.name, "mutate", fields, {})
            continue
        # Plain class -- only mutate mode is possible, and only if
        # fields are inferable and __setattr__ isn't overridden.
        fields = _infer_plain_class_fields(node)
        if fields and not _defines_own_setattr(node):
            registry[node.name] = ClassInfo(node.name, "mutate", fields, {})
    return registry


def build_helper_registry(tree):
    """{function_name: FunctionDef} for every module-level plain
    function in this file -- the same-file-only helper resolution
    _try_inline_call would use via func.__globals__, minus cross-file
    import resolution (see module docstring, limitation 2)."""
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


# --------------------------------------------------------------------------
# Constructor-call field matching (ports _ctor_supplies_all_fields)
# --------------------------------------------------------------------------

def ctor_supplies_all_fields(call_node, fields):
    if any(kw.arg is None for kw in call_node.keywords):
        return False
    n_pos = len(call_node.args)
    if n_pos > len(fields):
        return False
    kw_names = [kw.arg for kw in call_node.keywords]
    if len(set(kw_names)) != len(kw_names):
        return False
    supplied = set(fields[:n_pos]) | set(kw_names)
    return supplied == set(fields) and n_pos + len(kw_names) == len(fields)


def is_replace_call(node, alias):
    if not isinstance(node, ast.Call):
        return False
    fn = node.func
    is_replace_name = (isinstance(fn, ast.Attribute) and fn.attr == "replace") or (
        isinstance(fn, ast.Name) and fn.id == "replace"
    )
    if not is_replace_name:
        return False
    if not node.args or not (isinstance(node.args[0], ast.Name) and node.args[0].id == alias):
        return False
    if len(node.args) > 1:
        return False
    return not any(kw.arg is None for kw in node.keywords)


def replace_keywords_known(node, fields):
    return all(kw.arg in fields for kw in node.keywords)


# --------------------------------------------------------------------------
# One-level helper inlining (ports _try_inline_call's shape check)
# --------------------------------------------------------------------------

def call_matches_inlinable_helper(call_node, var_name, helper_registry):
    """True when call_node is `helper(..., var_name, ...)` where helper
    is a same-file function whose body is exactly `return
    <reconstruction-shaped-expr>` and var_name appears as exactly one
    bare positional/keyword argument (the accumulator param)."""
    if not isinstance(call_node.func, ast.Name):
        return None
    helper = helper_registry.get(call_node.func.id)
    if helper is None:
        return None
    body = helper.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if not (len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is not None):
        return None
    args = list(call_node.args) + [kw.value for kw in call_node.keywords]
    matches = [a for a in args if isinstance(a, ast.Name) and a.id == var_name]
    if len(matches) != 1:
        return None
    return helper, body[0].value


# --------------------------------------------------------------------------
# Escape-checked qualification for one accumulator candidate
# --------------------------------------------------------------------------

def _call_target_matches_class(call_node, class_name):
    """True when call_node's callee is a bare `class_name(...)` or a
    module-qualified `module.class_name(...)` call (v1.6, mirrors
    transform.py's real _call_target_matches_class). Matched by
    attribute name alone -- see module docstring for why identity
    beyond a name match isn't (and can't be, statically) verified."""
    fn = call_node.func
    if isinstance(fn, ast.Name):
        return fn.id == class_name
    if isinstance(fn, ast.Attribute):
        return fn.attr == class_name
    return False


def qualifies(loop_node, var_name, info, registry, helper_registry):
    """Statically replays _analyze_loop_body / _analyze_mutation_loop_body's
    escape check + reconstruction/mutation recognition against one
    candidate accumulator. Returns True/False."""
    fields = set(info.fields)
    ok = True
    reconstruction_seen = [False]  # single-slot mutable cell (reconstruct mode only)

    def is_known_reconstruction(value):
        if isinstance(value, ast.Call) and _call_target_matches_class(value, info.name):
            if ctor_supplies_all_fields(value, info.fields):
                return True
        if is_replace_call(value, var_name) and replace_keywords_known(value, fields):
            return True
        if isinstance(value, ast.Call):
            inlined = call_matches_inlinable_helper(value, var_name, helper_registry)
            if inlined is not None:
                _helper, ret_expr = inlined
                # The inlined callee's own return expression must itself
                # be a recognized reconstruction of ITS OWN accumulator
                # param -- approximate by checking the same shape with
                # info's class/fields (cross-parameter-name rewrite is
                # not attempted; see module docstring limitation 2's
                # spirit -- this is a same-file-only, best-effort check).
                if isinstance(ret_expr, ast.Call) and _call_target_matches_class(ret_expr, info.name):
                    if ctor_supplies_all_fields(ret_expr, info.fields):
                        return True
                if isinstance(ret_expr, ast.Call):
                    helper_arg_names = {
                        a.arg for a in _helper.args.args
                    }
                    for hp in helper_arg_names:
                        if is_replace_call(ret_expr, hp) and replace_keywords_known(ret_expr, fields):
                            return True
        return False

    def visit(node):
        nonlocal ok
        if not ok:
            return
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == var_name:
            if node.attr not in fields or isinstance(node.ctx, ast.Del):
                ok = False
            return
        if isinstance(node, ast.Name) and node.id == var_name:
            ok = False
            return
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == var_name
        ):
            if info.mode != "reconstruct":
                ok = False  # a rebind on a mutate-mode class is an escape
                return
            if is_known_reconstruction(node.value):
                reconstruction_seen[0] = True
                visit(node.value)
                return
            ok = False
            return
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == var_name
        ):
            if info.mode != "mutate" or node.targets[0].attr not in fields:
                ok = False
                return
            visit(node.value)
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    if isinstance(loop_node, ast.While):
        visit(loop_node.test)
    else:  # ast.For -- no .test; the accumulator could appear in the
        # iterable expression (e.g. `for x in acc.items:`) or, unusually,
        # as the loop target itself
        visit(loop_node.iter)
        visit(loop_node.target)
    for stmt in loop_node.body:
        visit(stmt)

    if not ok:
        return False
    if info.mode == "reconstruct":
        return reconstruction_seen[0]
    return True  # mutate mode: any escape-free write already confirmed by the caller


# --------------------------------------------------------------------------
# Site discovery, reusing analyze.py's loop/candidate-name machinery
# --------------------------------------------------------------------------

def _candidate_names(loop_body):
    names = set()
    for n in ast.walk(ast.Module(body=loop_body, type_ignores=[])):
        if isinstance(n, ast.Assign) and len(n.targets) == 1:
            t = n.targets[0]
            if isinstance(t, ast.Name):
                names.add(t.id)
            elif isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name):
                names.add(t.value.id)
    return names


def _find_pre_loop_init(preceding_stmts, name):
    init = None
    for stmt in preceding_stmts:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == name
        ):
            init = stmt.value
    return init


def _has_any_write(loop_body, var_name):
    found = [False]

    def visit(n):
        if isinstance(n, (ast.While, ast.For)):
            return
        if (
            isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and (
                (isinstance(n.targets[0], ast.Name) and n.targets[0].id == var_name)
                or (
                    isinstance(n.targets[0], ast.Attribute)
                    and isinstance(n.targets[0].value, ast.Name)
                    and n.targets[0].value.id == var_name
                )
            )
        ):
            found[0] = True
            return
        for child in ast.iter_child_nodes(n):
            visit(child)

    for stmt in loop_body:
        visit(stmt)
    return found[0]


def candidate_sites_for_loop(loop_node, preceding_stmts, registry, helper_registry):
    """Yields (var_name, ClassInfo) for every candidate accumulator on
    this loop whose init is a call -- bare `ClassName(...)` or
    module-qualified `module.ClassName(...)` (v1.6, mirrors
    transform.py's real _resolve_ctor_class) -- naming a class this
    project's registry recognizes, supplying exactly its fields, and
    actually reassigned/mutated somewhere in the loop body."""
    for name in _candidate_names(loop_node.body):
        init_expr = _find_pre_loop_init(preceding_stmts, name)
        if init_expr is None or not isinstance(init_expr, ast.Call):
            continue
        if not _has_any_write(loop_node.body, name):
            continue
        fn = init_expr.func
        if isinstance(fn, ast.Name):
            class_name = fn.id
        elif isinstance(fn, ast.Attribute):
            class_name = fn.attr
        else:
            continue
        info = registry.get(class_name)
        if info is not None and ctor_supplies_all_fields(init_expr, info.fields):
            yield name, info


# --------------------------------------------------------------------------
# Per-project analysis
# --------------------------------------------------------------------------

def analyze_project(project, domain, root):
    """Class/helper registries are built PROJECT-WIDE (across every file,
    not per-file) -- matching analyze.py's own local_class_names scope,
    and much closer to what the REAL cpython-asr would see: a live
    func.__globals__ resolves an imported class or helper defined in a
    different module of the same project just fine, so restricting
    classify.py to same-file-only definitions would be an artificial
    (and, as discovered while building this, badly undercounting)
    restriction relative to what the actual tool does. Collision caveat:
    if two files define an unrelated class with the same name, the
    later one wins in the merged registry -- the same "assumed same
    project, rare false match" caveat analyze.clj's own README states
    for its record-name matching (now applies equally to qualified
    `module.ClassName(...)` calls, matched by attribute name alone --
    see this module's docstring)."""
    files = list(python_files(root))
    trees = []
    for path in files:
        tree, err = parse_file(path)
        if not err:
            trees.append(tree)

    registry = {}
    helper_registry = {}
    for tree in trees:
        registry.update(build_class_registry(tree))
        helper_registry.update(build_helper_registry(tree))

    total_candidates = 0
    qualified = 0
    forms_with_a_candidate = 0
    forms_qualified = 0

    for tree in trees:
        if not registry:
            break

        def walk_block(stmts):
            nonlocal total_candidates, qualified, forms_with_a_candidate, forms_qualified
            for i, stmt in enumerate(stmts):
                if isinstance(stmt, (ast.While, ast.For)):
                    sites = list(candidate_sites_for_loop(stmt, stmts[:i], registry, helper_registry))
                    if sites:
                        forms_with_a_candidate += 1
                        form_ok = False
                        for var_name, info in sites:
                            total_candidates += 1
                            if qualifies(stmt, var_name, info, registry, helper_registry):
                                qualified += 1
                                form_ok = True
                        if form_ok:
                            forms_qualified += 1

                for field_name in ("body", "orelse", "finalbody"):
                    sub = getattr(stmt, field_name, None)
                    if isinstance(sub, list) and sub and isinstance(sub[0], ast.stmt):
                        walk_block(sub)
                for handler in getattr(stmt, "handlers", None) or []:
                    walk_block(handler.body)
                for case in getattr(stmt, "cases", None) or []:
                    walk_block(case.body)

        walk_block(tree.body)

    return {
        "project": project,
        "domain": domain,
        "files": len(files),
        "candidate_bindings": total_candidates,
        "qualified_bindings": qualified,
        "candidate_forms": forms_with_a_candidate,
        "qualified_forms": forms_qualified,
    }


def pct(num, den):
    return f"{100.0 * num / den:.2f}%" if den > 0 else "n/a"


def print_summary(stats):
    forms_c = sum(s["candidate_forms"] for s in stats)
    forms_q = sum(s["qualified_forms"] for s in stats)
    binds_c = sum(s["candidate_bindings"] for s in stats)
    binds_q = sum(s["qualified_bindings"] for s in stats)

    print("\n======== ASR pattern corpus study -- gate-faithful pass (Python) ========")
    print(f"Projects: {len(stats)}   Files: {sum(s['files'] for s in stats)}")
    print(f"Forms with >=1 syntactic record-accumulator candidate: {forms_c}")
    print(f"  of which qualify under the REAL gates: {forms_q} ({pct(forms_q, forms_c)})")
    print(f"Individual accumulator-binding candidates: {binds_c}")
    print(f"  of which qualify under the REAL gates: {binds_q} ({pct(binds_q, binds_c)})")
    print("\n--- by domain (qualified forms / candidate forms) ---")
    domains = sorted({s["domain"] for s in stats})
    for dom in domains:
        ss = [s for s in stats if s["domain"] == dom]
        c = sum(s["candidate_forms"] for s in ss)
        q = sum(s["qualified_forms"] for s in ss)
        print(f"  {dom:<18} {q:5d} / {c:<6d}  {pct(q, c)}")
    print("===========================================================================\n")


def main():
    corpus_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "corpus")
    manifest_path = Path(sys.argv[2] if len(sys.argv) > 2 else "manifest.json")
    out_path = Path(sys.argv[3] if len(sys.argv) > 3 else "results-classify.json")

    manifest = json.loads(manifest_path.read_text())
    dom_of = {e["org_repo"].replace("/", "__"): e["domain"] for e in manifest["repos"]}

    projects = sorted(p for p in corpus_dir.iterdir() if p.is_dir())
    stats = []
    for p in projects:
        print(f"classifying {p.name}", file=sys.stderr)
        stats.append(analyze_project(p.name, dom_of.get(p.name, "unknown"), p))

    out_path.write_text(json.dumps({"projects": stats}, indent=2) + "\n")
    print_summary(stats)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
