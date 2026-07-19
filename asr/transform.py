"""Aggregate Scalar Replacement (ASR) for CPython.

A minimal AST-level port of FOL's loop-carried classify-and-rewrite walk:
given a `while` loop that threads a frozen-dataclass accumulator through
its own back-edge, rebuilding it every iteration via a full constructor
call, `dataclasses.replace`, or a one-level helper-function call, split
the accumulator into one scalar local per field and re-box only once, at
the loop's exit.

v1.1 adds interprocedural reach by inlining (FOL's sec:inline): when the
loop's reconstruction is a call to a plain helper function whose own
body is exactly `return <reconstruction>`, and exactly one of the call's
arguments is the accumulator itself, the callee's body is inlined in
place and its parameter joins the alias set -- reads of that parameter
inside the inlined body are recognized as field reads exactly like reads
of the accumulator's own name. The callee's OTHER parameters (if any)
are substituted with the actual argument expressions from the call site;
FOL's "arguments are symbols or literals" restriction is enforced here
too, keeping the substitution capture-free. This is deliberately a
ONE-LEVEL inliner, same as FOL's: the callee's return expression must
itself be a direct reconstruction, not another call to a further helper.

Scope still deliberately narrow otherwise: single accumulator only, no
if/cond/case-branched reconstruction, and the only supported post-loop
shape is a single trailing `return p`. Every unrecognized shape is
declined, never miscompiled -- the same safe-by-abort discipline FOL's
own walk uses.
"""

import ast
import copy
import dataclasses
import inspect
import textwrap

from . import guard


class AsrDecline(Exception):
    """Raised internally whenever the walk hits a shape it doesn't
    recognize. Never escapes `try_transform`."""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _mangled_name(var_name, field_name):
    return f"__asr_{var_name}_{field_name}"


def _ctor_supplies_all_fields(call_node, fields):
    """True when a `ClassName(...)` call supplies exactly `fields`, by
    position (in field-declaration order) or keyword, with no **kwargs
    spread and no field left to a default."""
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


def _ctor_field_value(call_node, fields, field_name):
    idx = fields.index(field_name)
    if idx < len(call_node.args):
        return call_node.args[idx]
    for kw in call_node.keywords:
        if kw.arg == field_name:
            return kw.value
    raise AsrDecline(f"field {field_name!r} not supplied in constructor call")


def _is_replace_call(node, alias, fields):
    """`dataclasses.replace(alias, ...)` or `replace(alias, ...)` (if
    imported directly) -- the assoc analog. Every keyword must name a
    known field and there must be no **kwargs spread."""
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
        return False  # dataclasses.replace only takes the instance positionally
    if any(kw.arg is None for kw in node.keywords):
        return False
    return all(kw.arg in fields for kw in node.keywords)


def _collect_all_names(func_def):
    names = set()
    for node in ast.walk(func_def):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


def _strip_docstring(body):
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _references_only_as_field_reads(node, alias_names, fields):
    """True when every bare reference to a name in `alias_names` inside
    `node` is part of a recognized `alias.field` attribute read (for a
    field in `fields`). Used to validate a reconstruction's own value
    expressions don't smuggle a bare accumulator reference through some
    other channel."""
    ok = True

    def visit(n):
        nonlocal ok
        if not ok:
            return
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in alias_names:
            if n.attr not in fields:
                ok = False
            return
        if isinstance(n, ast.Name) and n.id in alias_names:
            ok = False
            return
        for child in ast.iter_child_nodes(n):
            visit(child)

    visit(node)
    return ok


def _reconstruction_field_values(value, alias_names, class_name, fields):
    """If `value` is a `ClassName(...)` full reconstruction or a
    `dataclasses.replace(alias, ...)` partial reconstruction (for some
    alias in alias_names), and every reference to an alias anywhere in
    `value` is a recognized `.field` read, return {field_name:
    value_expr} for the fields actually touched (all of them, for a
    full reconstruction). Returns None if `value` isn't one of these two
    recognized shapes, or if an alias escapes some other way."""
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == class_name
        and not any(kw.arg is None for kw in value.keywords)
        and _ctor_supplies_all_fields(value, fields)
    ):
        if not _references_only_as_field_reads(value, alias_names, fields):
            return None
        return {f: _ctor_field_value(value, fields, f) for f in fields}
    for alias in alias_names:
        if _is_replace_call(value, alias, fields):
            # Check only the keyword VALUES for escapes -- value.args[0]
            # is the call's own required `replace(alias, ...)` reference
            # to the alias itself, a recognized part of this shape, not
            # a bare-reference escape.
            if not all(
                _references_only_as_field_reads(kw.value, alias_names, fields) for kw in value.keywords
            ):
                return None
            return {kw.arg: kw.value for kw in value.keywords}
    return None


# --------------------------------------------------------------------------
# Interprocedural reach by inlining (v1.1, FOL's sec:inline)
# --------------------------------------------------------------------------

class _ParamSubstituter(ast.NodeTransformer):
    """Replaces bare `Name(id=param)` with the corresponding call-site
    argument expression, for the callee's non-accumulator parameters."""

    def __init__(self, bindings):
        self.bindings = bindings  # param_name -> ast.expr (Name or Constant)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.bindings:
            new = copy.deepcopy(self.bindings[node.id])
            return ast.copy_location(new, node)
        return node


def _match_call_to_accumulator_param(call_node, param_names, var_name):
    """If exactly one of call_node's arguments is `Name(id=var_name)`
    (the accumulator itself, unmodified) and every other argument is a
    plain Name or literal Constant (FOL's "arguments are symbols or
    literals" restriction -- keeps the substitution capture-free and
    side-effect-safe), return (accumulator_param_name, {other_param:
    arg_expr, ...}). Otherwise return None."""
    if len(call_node.args) > len(param_names):
        return None
    if any(kw.arg is None for kw in call_node.keywords):
        return None  # **kwargs spread
    bound = {}
    for i, a in enumerate(call_node.args):
        bound[param_names[i]] = a
    for kw in call_node.keywords:
        if kw.arg not in param_names or kw.arg in bound:
            return None
        bound[kw.arg] = kw.value
    if len(bound) != len(param_names):
        return None  # a defaulted param left unsupplied -- decline for v1.1 simplicity

    accumulator_params = [p for p, expr in bound.items() if isinstance(expr, ast.Name) and expr.id == var_name]
    if len(accumulator_params) != 1:
        return None
    accumulator_param = accumulator_params[0]
    others = {p: expr for p, expr in bound.items() if p != accumulator_param}
    if not all(isinstance(expr, (ast.Name, ast.Constant)) for expr in others.values()):
        return None
    return accumulator_param, others


def _try_inline_call(call_node, var_name, class_name, fields, globalns):
    """Attempt a one-level inline of a `var_name = helper(...)`
    reconstruction. Returns (accumulator_param, {field: substituted_value_expr})
    on success, or None to decline (never raises -- an unrecognized
    helper shape is exactly as safe as an unrecognized inline expression,
    the caller just falls through to the ordinary decline path)."""
    if not isinstance(call_node.func, ast.Name):
        return None
    helper = globalns.get(call_node.func.id)
    if helper is None or not inspect.isfunction(helper):
        return None
    if getattr(helper, "__asr_transformed__", False):
        return None  # don't inline an already-@asr-transformed function

    try:
        helper_src = textwrap.dedent(inspect.getsource(helper))
        helper_tree = ast.parse(helper_src)
    except (OSError, TypeError, SyntaxError):
        return None
    if not helper_tree.body or not isinstance(helper_tree.body[0], ast.FunctionDef):
        return None
    helper_def = helper_tree.body[0]

    body = _strip_docstring(helper_def.body)
    if not (len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is not None):
        return None  # FOL's "single-clause function whose body reduces to a constructor"

    param_names = [a.arg for a in helper_def.args.args]
    matched = _match_call_to_accumulator_param(call_node, param_names, var_name)
    if matched is None:
        return None
    accumulator_param, other_bindings = matched

    field_values = _reconstruction_field_values(body[0].value, {accumulator_param}, class_name, fields)
    if field_values is None:
        return None  # callee's return expr isn't itself a recognized reconstruction

    param_subst = _ParamSubstituter(other_bindings)
    substituted = {f: param_subst.visit(copy.deepcopy(expr)) for f, expr in field_values.items()}
    return accumulator_param, substituted


# --------------------------------------------------------------------------
# Phase 1: qualification
# --------------------------------------------------------------------------

def _find_accumulator(pre_loop_stmts, globalns):
    """Scan the statements before the while loop for `p = ClassName(...)`
    where ClassName is a frozen dataclass and the call supplies exactly
    its fields. Returns (index, var_name, cls, fields)."""
    for i, stmt in enumerate(pre_loop_stmts):
        if not (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            continue
        var_name = stmt.targets[0].id
        call = stmt.value
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)):
            continue
        cls = globalns.get(call.func.id)
        if cls is None or not isinstance(cls, type) or not dataclasses.is_dataclass(cls):
            continue
        params = getattr(cls, "__dataclass_params__", None)
        if params is None or not params.frozen:
            continue
        fields = tuple(f.name for f in dataclasses.fields(cls))
        if not _ctor_supplies_all_fields(call, fields):
            continue
        return i, var_name, cls, fields
    raise AsrDecline("no qualifying accumulator initializer found before the loop")


def _analyze_loop_body(while_node, var_name, class_name, fields, globalns):
    """Walk the while loop's body looking for exactly one recognized
    reconstruction of `var_name` -- direct, dataclasses.replace-based,
    or a one-level inlined helper call -- and no other bare reference to
    it. Returns (reconstruction_assign, alias_names: frozenset,
    field_values: dict[str, ast.expr])."""
    reconstruction_assign = None
    result = None  # (alias_names, field_values)
    ok = True

    def visit(node):
        nonlocal reconstruction_assign, result, ok
        if not ok:
            return
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == var_name:
            if node.attr not in fields:
                ok = False
            return  # do not recurse into the Name('p') itself
        if isinstance(node, ast.Name) and node.id == var_name:
            ok = False  # a bare reference we don't recognize -> escape
            return
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == var_name
        ):
            if reconstruction_assign is not None:
                ok = False  # more than one reconstruction -- out of v1 scope
                return
            value = node.value

            field_values = _reconstruction_field_values(value, frozenset({var_name}), class_name, fields)
            if field_values is not None:
                reconstruction_assign = node
                result = (frozenset({var_name}), field_values)
                for expr in field_values.values():
                    visit(expr)
                return

            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                inlined = _try_inline_call(value, var_name, class_name, fields, globalns)
                if inlined is not None:
                    accumulator_param, substituted = inlined
                    reconstruction_assign = node
                    result = (frozenset({var_name, accumulator_param}), substituted)
                    for expr in substituted.values():
                        visit(expr)
                    return

            ok = False
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    for stmt in while_node.body:
        visit(stmt)

    if not ok or result is None:
        raise AsrDecline("loop body has an unrecognized accumulator use")
    alias_names, field_values = result
    return reconstruction_assign, alias_names, field_values


# --------------------------------------------------------------------------
# Phase 2: the rewrite
# --------------------------------------------------------------------------

class _FieldSubstituter(ast.NodeTransformer):
    """Replaces `alias.field` with the corresponding scalar Name, for
    any alias in the alias set (the accumulator's own name, plus -- when
    a reconstruction was inlined -- the inlined callee's own parameter
    name for that same accumulator)."""

    def __init__(self, alias_names, scalar_names):
        self.alias_names = alias_names
        self.scalar_names = scalar_names

    def visit_Attribute(self, node):
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in self.alias_names
            and node.attr in self.scalar_names
        ):
            new = ast.Name(id=self.scalar_names[node.attr], ctx=ast.Load())
            return ast.copy_location(new, node)
        return self.generic_visit(node)


def _mk_assign(target_name, value_expr, loc_from):
    stmt = ast.Assign(targets=[ast.Name(id=target_name, ctx=ast.Store())], value=value_expr)
    ast.copy_location(stmt, loc_from)
    ast.fix_missing_locations(stmt)
    return stmt


def _rewrite_loop_body(while_node, alias_names, scalar_names, reconstruction_assign, field_values):
    subst = _FieldSubstituter(alias_names, scalar_names)
    new_body = []
    for stmt in while_node.body:
        if stmt is reconstruction_assign:
            # Parallel-update semantics (mirrors FOL's psetq/recur):
            # evaluate every new value against the CURRENT scalars first,
            # via temporaries, before reassigning any of them. Without
            # this, a reconstruction that reads one field while updating
            # another in the same call -- e.g.
            # `replace(acc, n=acc.n+1, total=acc.total+acc.n)`, or an
            # inlined helper doing the same -- would silently see the
            # just-updated value instead of the value at the start of
            # the iteration. Fields NOT touched (a partial
            # reconstruction) keep their prior scalar value automatically
            # -- ordinary Python local persistence across loop
            # iterations, unlike FOL's parallel-update recur/psetq,
            # which has to thread every loop variable explicitly.
            touched = list(field_values.keys())
            substituted_values = {f: subst.visit(copy.deepcopy(field_values[f])) for f in touched}
            tmp_names = {f: f"__asr_tmp_{scalar_names[f]}" for f in touched}
            for f in touched:
                new_body.append(_mk_assign(tmp_names[f], substituted_values[f], stmt))
            for f in touched:
                new_body.append(
                    _mk_assign(scalar_names[f], ast.Name(id=tmp_names[f], ctx=ast.Load()), stmt)
                )
        else:
            new_body.append(subst.visit(copy.deepcopy(stmt)))
    new_while = ast.While(
        test=subst.visit(copy.deepcopy(while_node.test)), body=new_body, orelse=[]
    )
    ast.copy_location(new_while, while_node)
    ast.fix_missing_locations(new_while)
    return new_while


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def try_transform(func):
    """Attempt Aggregate Scalar Replacement on `func`. Returns a new,
    guarded-dual-path function on success, or None if the shape isn't
    recognized (the caller should then just use `func` unchanged)."""
    try:
        return _try_transform_inner(func)
    except AsrDecline:
        return None
    except (OSError, TypeError):
        # inspect.getsource fails for builtins, REPL/exec-defined
        # functions, etc. -- decline rather than error.
        return None


def _try_transform_inner(func):
    src = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(src)
    func_def = tree.body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise AsrDecline("not a plain function definition")
    func_def.decorator_list = []  # strip @asr etc. before recompiling

    globalns = func.__globals__

    loop_idx = next((i for i, s in enumerate(func_def.body) if isinstance(s, ast.While)), None)
    if loop_idx is None:
        raise AsrDecline("no while loop in function body")
    while_node = func_def.body[loop_idx]

    accum_idx, var_name, cls, fields = _find_accumulator(func_def.body[:loop_idx], globalns)

    for stmt in func_def.body[accum_idx + 1 : loop_idx]:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and node.id == var_name:
                raise AsrDecline("accumulator referenced between its init and the loop")

    reconstruction_assign, alias_names, field_values = _analyze_loop_body(
        while_node, var_name, cls.__name__, fields, globalns
    )

    tail = func_def.body[loop_idx + 1 :]
    if not (
        len(tail) == 1
        and isinstance(tail[0], ast.Return)
        and isinstance(tail[0].value, ast.Name)
        and tail[0].value.id == var_name
    ):
        raise AsrDecline("unsupported post-loop shape (v1 only supports a trailing 'return p')")

    scalar_names = {f: _mangled_name(var_name, f) for f in fields}
    existing_names = _collect_all_names(func_def)
    candidate_names = list(scalar_names.values()) + [
        f"__asr_tmp_{n}" for n in scalar_names.values()
    ]
    for name in candidate_names:
        if name in existing_names:
            raise AsrDecline(f"scalar name collision: {name}")
    # An inlined helper's substituted field-value expressions can, in
    # principle, mention a free name that happens to collide with a
    # synthesized scalar/temp name too (vanishingly unlikely given the
    # __asr_ prefix, but cheap to rule out -- never trust an inlined
    # expression more than a local one).
    if len(alias_names) > 1:
        inlined_names = set()
        for expr in field_values.values():
            for node in ast.walk(expr):
                if isinstance(node, ast.Name):
                    inlined_names.add(node.id)
        for name in candidate_names:
            if name in inlined_names:
                raise AsrDecline(f"scalar name collision in inlined expression: {name}")

    accum_assign = func_def.body[accum_idx]
    init_stmts = [
        _mk_assign(
            scalar_names[field],
            copy.deepcopy(_ctor_field_value(accum_assign.value, fields, field)),
            accum_assign,
        )
        for field in fields
    ]

    new_while = _rewrite_loop_body(while_node, alias_names, scalar_names, reconstruction_assign, field_values)

    # Per-function-unique injected names: two @asr-decorated functions in
    # the same module must not clobber each other's guard cell/class ref.
    cell_key = f"__asr_cell_{func_def.name}"
    cls_key = f"__asr_cls_{func_def.name}"

    rebox_call = ast.Call(
        func=ast.Name(id=cls_key, ctx=ast.Load()),
        args=[ast.Name(id=scalar_names[f], ctx=ast.Load()) for f in fields],
        keywords=[],
    )
    rebox_return = ast.Return(value=rebox_call)
    ast.copy_location(rebox_return, tail[0])
    ast.fix_missing_locations(rebox_return)

    fast_body = (
        list(func_def.body[:accum_idx])
        + init_stmts
        + list(func_def.body[accum_idx + 1 : loop_idx])
        + [new_while, rebox_return]
    )
    original_body = copy.deepcopy(func_def.body)

    guard_test = ast.Attribute(
        value=ast.Name(id=cell_key, ctx=ast.Load()), attr="valid", ctx=ast.Load()
    )
    guarded_if = ast.If(test=guard_test, body=fast_body, orelse=original_body)
    ast.copy_location(guarded_if, func_def)
    ast.fix_missing_locations(guarded_if)

    # Deliberately no lineno/col_offset here: ast.fix_missing_locations
    # below fills in a full, self-consistent (lineno, col_offset,
    # end_lineno, end_col_offset) tuple for a wholly location-less node.
    # Setting only lineno/col_offset by hand (as an earlier version of
    # this code did) left end_lineno unset, since fix_missing_locations
    # skips any node that already has *a* location -- producing an
    # invalid, inverted line range at compile() time.
    new_func_def = ast.FunctionDef(
        name=func_def.name,
        args=func_def.args,
        body=[guarded_if],
        decorator_list=[],
        returns=func_def.returns,
        type_comment=None,
    )
    module = ast.Module(body=[new_func_def], type_ignores=[])
    ast.fix_missing_locations(module)

    cell = guard.register(func.__module__, cls.__name__, fields)
    # IMPORTANT: use the function's *actual* __globals__ dict, not a copy.
    # importlib.reload() re-execs a module's source against its existing
    # __dict__ in place; a copy would silently stop seeing redefinitions,
    # defeating the whole world guard.
    namespace = globalns
    namespace[cell_key] = cell
    namespace[cls_key] = cls

    code = compile(module, filename=f"<asr:{func.__qualname__}>", mode="exec")
    exec(code, namespace)
    new_func = namespace[func_def.name]
    new_func.__asr_transformed__ = True
    return new_func
