# cpython-asr

A minimal AST-level port of FOL's Aggregate Scalar Replacement (ASR) to
CPython. Built as the second-language existence-proof referenced in the
CGO 2027 paper *"Objects Without Allocation"*'s Threats to Validity
section, which asserts the mechanism "transfers to other transpiled
dynamic languages" without demonstrating it on a second one.

Given a `while` loop that threads one or more frozen-dataclass
accumulators through its own back-edge -- rebuilt every iteration via a
full constructor call, `dataclasses.replace`, a call to a
one-level-inlinable helper function, an if/elif/.../else chain, or a
literal-dispatch match/case block -- `@asr` splits each accumulator
into one scalar local per field and re-boxes only once, at the loop's
exit, behind a guarded dual path that falls back safely if a tracked
dataclass is redefined out from under it (via `importlib.reload`).

## Status: v1 + v1.1 (interprocedural inlining) + v1.2 (branching, multi-accumulator) + v1.3 (match/case)

| FOL concept | This port |
|---|---|
| `sec:cand` -- candidate qualification | `asr/transform.py::_find_accumulator` |
| `sec:loop` -- the classify-and-rewrite walk | `asr/transform.py::_analyze_loop_body` / `_rewrite_loop_body` |
| `sec:inline` -- interprocedural reach by inlining | `asr/transform.py::_try_inline_call` (one-level, same restriction as FOL: callee arguments must be symbols or literals) |
| Reconstruct's `if`/`cond` cases -- branch-shaped reconstruction | `asr/transform.py::_try_branch_reconstruction` (if/elif/.../else, mandatory terminal else, no inlining inside a branch -- same restriction as FOL) |
| FOL's `case` -- literal-dispatch reconstruction | `asr/transform.py::_try_match_reconstruction` (Python's own `match`/`case`, 3.10+; literal/singleton patterns only, mandatory trailing `case _:`, no guards or structural patterns; subject bound to a one-time temporary since Python evaluates it exactly once) |
| `maybe-scalar-replace-loop`/`%sr-replace-one` -- the multi-accumulator fixpoint | `asr/transform.py::_try_transform_inner`'s fixpoint loop over `_process_one_accumulator`; `return p, q, ...` for FOL's Two-body/Kalman shape |
| `sec:world` -- the world guard | `asr/guard.py`, keyed on `(module, class)`, invalidated by wrapping `importlib.reload`; a multi-accumulator fast path is guarded by the AND of every tracked class's cell |
| Figure 3 -- guarded dual path | every transformed function's body is `if <cell>.valid: <fast path> else: <original path>` |

**Deliberately out of scope**: mutable (non-frozen) dataclasses, plain
classes, and automatic/non-opt-in application -- this is an opt-in
`@asr` decorator (source-to-source via `inspect.getsource` + `ast`), not
a `sys.meta_path` import hook, closer in spirit to how Numba's `@jit`
works than to something living inside CPython's own compiler.

## Layout

- `asr/transform.py` -- qualification + rewrite (phases 1 and 2), inlining, branching, and the multi-accumulator fixpoint
- `asr/guard.py` -- the world guard and `importlib.reload` wrapper
- `asr/decorator.py` -- the `@asr` entry point
- `tests/` -- 53 pytest cases across `test_transform.py` (core v1), `test_inline.py` (v1.1), `test_branch.py` and `test_multi_accumulator.py` (v1.2), `test_match.py` (v1.3), `test_decorator.py`, `test_guard.py`
- `benchmarks/` -- Particle, Counter, and Assoc, ported from FOL's `benchmarks/fol-code/*.fol` (Clamp/Bounce/Phase and Two-body/Kalman are portable now that branching and multi-accumulator support exist, but aren't ported yet)

## Running

```bash
pip install -e ".[dev]"
pytest
python -m benchmarks.run_all
```

## Honest caveats

This is an existence proof, not a claim of parity with the paper's
Table 1/2 rigor: single machine, opt-in decorator only, and the
allocation figures reported by the
benchmark harness are exact constructor-call counts (not FOL's
`bytes-consed` counter) -- an earlier version of the harness used
`tracemalloc` snapshot-diffing and it was actively misleading on this
workload, since Python's allocator reuses freed same-size slots within
a tight loop of identically-shaped objects; see `benchmarks/harness.py`
for the full account.
