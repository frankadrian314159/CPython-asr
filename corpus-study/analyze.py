"""Reader-level (AST-parse, no execution/import) detector for the ASR
pattern -- a record accumulator carried through a loop and rebuilt each
iteration -- over a corpus of real Python projects. Mirrors
docs/cgo2027/corpus-study/src/asr_corpus/analyze.clj in the FOL repo as
closely as Python's loop constructs allow. See README.md for
methodology and caveats.

Unlike the Clojure study (a *proxy* for FOL, a different language),
this is a *direct* measurement: cpython-asr targets Python natively.

Usage: python analyze.py [corpus_dir] [manifest.json] [out.json]
"""

import ast
import json
import sys
from pathlib import Path

EXCLUDE_DIR_NAMES = {
    ".git", "__pycache__", "build", "dist", "node_modules",
    ".tox", ".mypy_cache", ".pytest_cache", ".venv", "venv", ".eggs",
}

MAP_CTOR_NAMES = {"dict", "defaultdict", "OrderedDict", "Counter", "ChainMap"}
COLL_CTOR_NAMES = {"list", "set", "frozenset", "tuple", "deque", "array"}


# --------------------------------------------------------------------------
# File discovery
# --------------------------------------------------------------------------

def python_files(root):
    for path in Path(root).rglob("*.py"):
        if any(part in EXCLUDE_DIR_NAMES or part.endswith(".egg-info") for part in path.parts):
            continue
        yield path


def parse_file(path):
    """Returns (ast.Module or None, read_error: bool)."""
    try:
        src = path.read_text(encoding="utf-8", errors="strict")
        return ast.parse(src), False
    except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
        return None, True


# --------------------------------------------------------------------------
# Record (class) definitions, file-local
# --------------------------------------------------------------------------

def local_class_names(tree):
    """Every class name defined anywhere in this file -- permissive, like
    analyze.clj's defrecord/deftype scan: any class counts as a
    candidate 'record' regardless of internal shape (the syntactic pass
    is a proxy; classify.py checks the real shape)."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            names.add(node.name)
    return names


# --------------------------------------------------------------------------
# Init/update classification (mirrors classify-init/classify-update)
# --------------------------------------------------------------------------

def _call_name(node):
    """Unqualified callee name of a Call node, or None -- 'dict(...)' ->
    'dict', 'dataclasses.replace(...)' -> 'replace', 'Point(...)' -> 'Point'."""
    if not isinstance(node, ast.Call):
        return None
    fn = node.func
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return None


def classify_init(expr, class_names):
    if isinstance(expr, ast.Call):
        name = _call_name(expr)
        if name in class_names:
            return "record"
        if name in MAP_CTOR_NAMES:
            return "map"
        if name in COLL_CTOR_NAMES:
            return "coll"
        return "other"
    if isinstance(expr, ast.Dict) or isinstance(expr, ast.DictComp):
        return "map"
    if isinstance(expr, (ast.List, ast.Set, ast.Tuple, ast.ListComp, ast.SetComp)):
        return "coll"
    if isinstance(expr, ast.Constant) and isinstance(expr.value, (int, float, bool, complex)) and not isinstance(expr.value, str):
        return "primitive"
    if isinstance(expr, ast.Constant) and expr.value is None:
        return "primitive"
    return "other"


def classify_update(expr, acc_name, class_names):
    """How is accumulator `acc_name` produced by `expr` (the value side of
    an assignment/mutation targeting it)?"""
    if isinstance(expr, ast.Name) and expr.id == acc_name:
        return "passthrough"
    if isinstance(expr, ast.Call):
        name = _call_name(expr)
        if name in class_names:
            return "record-ctor"
        if name == "replace":
            # dataclasses.replace(acc, ...) -- the assoc analog.
            if expr.args and isinstance(expr.args[0], ast.Name) and expr.args[0].id == acc_name:
                return "record-assoc"
        for arg in list(expr.args) + [kw.value for kw in expr.keywords]:
            if isinstance(arg, ast.Name) and arg.id == acc_name:
                return "helper"
    return "other"


# --------------------------------------------------------------------------
# Loop analysis
# --------------------------------------------------------------------------

def _iter_stmts_shallow(body):
    """Statements at this block's own level -- does not descend into
    nested compound statements (if/for/while/try/with/...), matching
    find-recurs's 'do not descend into nested loop*/fn*' boundary: we
    want assignments/mutations at THIS loop's own nesting depth plus one
    (branches directly inside it), not inside a further-nested loop."""
    return body


def _assigned_names_in(node, target_name):
    """Every RHS expr assigned to bare Name(target_name), or to
    Attribute(Name(target_name), field) (mutation), found anywhere in
    node (including inside nested if/match, but not inside a further
    nested loop, which would be its own independent accumulator site)."""
    updates = []  # list of ("rebind"|"mutate", value_expr)

    def visit(n):
        if isinstance(n, (ast.While, ast.For)):
            return  # a nested loop has its own, independent accumulators
        if isinstance(n, ast.Assign) and len(n.targets) == 1:
            t = n.targets[0]
            if isinstance(t, ast.Name) and t.id == target_name:
                updates.append(("rebind", n.value))
                return
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == target_name:
                updates.append(("mutate", None))
                return
        for child in ast.iter_child_nodes(n):
            visit(child)

    for stmt in node:
        visit(stmt)
    return updates


def _candidate_names(loop_body):
    """Every bare Name ever assigned (rebind) or mutated (Name.attr = ...)
    somewhere in the loop body -- candidate accumulators."""
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
    """Last plain `name = expr` assignment among the statements preceding
    the loop in the same block, or None."""
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


_UPDATE_PRIORITY = ["record-ctor", "record-assoc", "record-mutate", "helper", "passthrough", "other"]


def analyze_loop_or_body(loop_node, preceding_stmts, class_names):
    """Returns a list of {"init": kw, "update": kw} per accumulator
    candidate found in this loop -- Python's while/for analog of
    analyze-loop, minus the structural guarantee Clojure's explicit
    loop-binding vector gives us (any pre-loop local could coincidentally
    share a name with something reassigned in the body; we require an
    ACTUAL pre-loop assignment to treat it as a candidate at all)."""
    candidates = _candidate_names(loop_node.body)
    results = []
    for name in candidates:
        init_expr = _find_pre_loop_init(preceding_stmts, name)
        if init_expr is None:
            continue  # not threaded from before the loop -- not an accumulator
        updates = _assigned_names_in(loop_node.body, name)
        if not updates:
            continue
        update_kinds = set()
        for kind, value in updates:
            if kind == "mutate":
                update_kinds.add("record-mutate")
            else:
                update_kinds.add(classify_update(value, name, class_names))
        update_kind = next((k for k in _UPDATE_PRIORITY if k in update_kinds), "other")
        results.append({"init": classify_init(init_expr, class_names), "update": update_kind})
    return results


def analyze_reduce_call(call_node, class_names):
    """functools.reduce(lambda acc, x: ..., iterable, init) or
    reduce(...) -- classified only when the function arg is a lambda
    literal with an explicit init (matching analyze.clj's restriction to
    fn-literal + explicit init, 3-arg reduce only)."""
    name = _call_name(call_node)
    if name != "reduce":
        return None
    args = call_node.args
    if len(args) != 3 or not isinstance(args[0], ast.Lambda):
        return None
    fn, _iterable, init = args
    if len(fn.args.args) < 1:
        return None
    acc_name = fn.args.args[0].arg
    body_exprs = _lambda_tail_exprs(fn.body)
    update_kinds = {classify_update(e, acc_name, class_names) for e in body_exprs}
    update_kind = next((k for k in _UPDATE_PRIORITY if k in update_kinds), "other")
    return {"init": classify_init(init, class_names), "update": update_kind}


def _lambda_tail_exprs(expr):
    """A lambda body is a single expression; if it's a conditional
    (ast.IfExp, Python's only expression-level branch), collect both
    arms -- the closest analog to tail-exprs for an expression body."""
    if isinstance(expr, ast.IfExp):
        return _lambda_tail_exprs(expr.body) + _lambda_tail_exprs(expr.orelse)
    return [expr]


# --------------------------------------------------------------------------
# Site tagging / tallying
# --------------------------------------------------------------------------

def site_tags(bindings):
    def is_record_acc(b):
        return b["init"] == "record" and b["update"] in ("record-ctor", "record-assoc", "record-mutate", "helper")

    def is_strong_acc(b):
        return b["init"] == "record" and b["update"] in ("record-ctor", "record-assoc", "record-mutate")

    record = any(is_record_acc(b) for b in bindings)
    return {
        "record": record,
        "record_strong": any(is_strong_acc(b) for b in bindings),
        "ctor": any(b["init"] == "record" and b["update"] == "record-ctor" for b in bindings),
        "assoc": any(b["init"] == "record" and b["update"] == "record-assoc" for b in bindings),
        "mutate": any(b["init"] == "record" and b["update"] == "record-mutate" for b in bindings),
        "map": any(b["init"] == "map" for b in bindings),
        "coll": any(b["init"] == "coll" for b in bindings),
        "primitive": (
            not record
            and not any(b["init"] in ("map", "coll") for b in bindings)
            and any(b["init"] == "primitive" for b in bindings)
        ),
    }


EMPTY_TALLY = {"sites": 0, "record": 0, "record_strong": 0, "ctor": 0, "assoc": 0, "mutate": 0, "map": 0, "coll": 0, "primitive": 0}


def tally(tags_list):
    t = dict(EMPTY_TALLY)
    for tags in tags_list:
        t["sites"] += 1
        for key in ("record", "record_strong", "ctor", "assoc", "mutate", "map", "coll", "primitive"):
            if tags[key]:
                t[key] += 1
    return t


# --------------------------------------------------------------------------
# Per-project analysis
# --------------------------------------------------------------------------

def analyze_project(project, domain, root):
    files = list(python_files(root))
    read_errors = 0
    class_names = set()
    trees = []
    for path in files:
        tree, err = parse_file(path)
        if err:
            read_errors += 1
            continue
        trees.append(tree)
        class_names |= local_class_names(tree)

    loop_sites = []
    reduce_sites = []

    for tree in trees:

        def walk_block(stmts):
            """Scans While/For at THIS block's own nesting level, using the
            block's own preceding statements as the pre-loop-init search
            space, then recurses into every nested statement block exactly
            once (function/class/if/for/while/try/with bodies, plus
            try's except handlers and match's cases) -- each nested block
            is a fully independent walk, so a loop is neither missed nor
            counted twice."""
            for i, stmt in enumerate(stmts):
                if isinstance(stmt, (ast.While, ast.For)):
                    bindings = analyze_loop_or_body(stmt, stmts[:i], class_names)
                    if bindings:
                        loop_sites.append(site_tags(bindings))

                for field_name in ("body", "orelse", "finalbody"):
                    sub = getattr(stmt, field_name, None)
                    if isinstance(sub, list) and sub and isinstance(sub[0], ast.stmt):
                        walk_block(sub)
                for handler in getattr(stmt, "handlers", None) or []:
                    walk_block(handler.body)
                for case in getattr(stmt, "cases", None) or []:
                    walk_block(case.body)

        walk_block(tree.body)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                result = analyze_reduce_call(node, class_names)
                if result is not None:
                    reduce_sites.append(site_tags([result]))

    return {
        "project": project,
        "domain": domain,
        "files": len(files),
        "read_errors": read_errors,
        "records": len(class_names),
        "loop": tally(loop_sites),
        "reduce": tally(reduce_sites),
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def add_tallies(a, b):
    return {k: a[k] + b[k] for k in EMPTY_TALLY}


def combine(stats, key):
    total = dict(EMPTY_TALLY)
    for s in stats:
        total = add_tallies(total, s[key])
    return total


def pct(num, den):
    return f"{100.0 * num / den:.2f}%" if den > 0 else "n/a"


def print_summary(stats):
    loop_tot = combine(stats, "loop")
    red_tot = combine(stats, "reduce")
    sites = loop_tot["sites"] + red_tot["sites"]
    rec = loop_tot["record"] + red_tot["record"]
    rec_strong = loop_tot["record_strong"] + red_tot["record_strong"]
    prim = loop_tot["primitive"] + red_tot["primitive"]
    mapc = loop_tot["map"] + red_tot["map"]
    collc = loop_tot["coll"] + red_tot["coll"]
    ctor = loop_tot["ctor"] + red_tot["ctor"]
    assc = loop_tot["assoc"] + red_tot["assoc"]
    mut = loop_tot["mutate"] + red_tot["mutate"]

    print("\n================ ASR pattern corpus study (Python, direct) ================")
    print(f"Projects: {len(stats)}   Files: {sum(s['files'] for s in stats)}   "
          f"Read errors: {sum(s['read_errors'] for s in stats)}   "
          f"Classes defined: {sum(s['records'] for s in stats)}")
    print(f"Loop sites: {loop_tot['sites']}   reduce() sites (classified): {red_tot['sites']}   Total: {sites}")
    print(f"\n(a) record accumulator rebuilt : {rec} ({pct(rec, sites)} of sites)")
    print(f"      of which strong (ctor/assoc/mutate): {rec_strong}")
    print(f"      rebuild via constructor            : {ctor}")
    print(f"      rebuild via dataclasses.replace     : {assc}")
    print(f"      rebuild via direct mutation (p.x=..): {mut}")
    print(f"(b) map/dict accumulator rebuilt : {mapc} ({pct(mapc, sites)})")
    print(f"(c) collection accumulator grown : {collc} ({pct(collc, sites)})")
    print(f"(d) primitive-scalar loop        : {prim} ({pct(prim, sites)})")
    print(f"\nSuppression signal (d):(a) = {prim / rec if rec else 0.0:.1f} : 1")
    print("\n--- by domain (record sites / total sites) ---")
    domains = sorted({s["domain"] for s in stats})
    for dom in domains:
        ss = [s for s in stats if s["domain"] == dom]
        lt = combine(ss, "loop")
        rt = combine(ss, "reduce")
        s_total = lt["sites"] + rt["sites"]
        r_total = lt["record"] + rt["record"]
        print(f"  {dom:<18} {r_total:5d} / {s_total:<6d}  {pct(r_total, s_total)}")
    print("=============================================================================\n")


def main():
    corpus_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "corpus")
    manifest_path = Path(sys.argv[2] if len(sys.argv) > 2 else "manifest.json")
    out_path = Path(sys.argv[3] if len(sys.argv) > 3 else "results.json")

    manifest = json.loads(manifest_path.read_text())
    dom_of = {e["org_repo"].replace("/", "__"): e["domain"] for e in manifest["repos"]}

    projects = sorted(p for p in corpus_dir.iterdir() if p.is_dir())
    stats = []
    for p in projects:
        print(f"analyzing {p.name}", file=sys.stderr)
        stats.append(analyze_project(p.name, dom_of.get(p.name, "unknown"), p))

    out_path.write_text(json.dumps({"projects": stats}, indent=2) + "\n")
    print_summary(stats)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
