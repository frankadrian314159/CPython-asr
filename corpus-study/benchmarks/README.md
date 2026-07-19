# Corpus-derived benchmarks

Five benchmarks adapted from real sites found while building
`corpus-study/`'s gate-faithful pass — not from the CGO 2027 paper's own
Table 1 (that set lives in `../../benchmarks/`). None of these five
already qualified for ASR verbatim; each needed a documented adaptation to
become ASR-addressable, the same way FOL's own corpus study needed a real
"whole-program port" for its 3 hand-audited Clojure sites (reitit,
fastmath, datascript) rather than finding them already passing. Every
benchmark's own module docstring explains exactly what changed from the
real corpus source and why — read those before quoting a number here.

## Results (mean over 20 trials, 200,000 iterations)

```
Benchmark                                                           Base ms   ASR ms     Time  Constructions
----------------------------------------------------------------------------------------------------------------
SievePolynomial (sympy qs.py, computed-field dataclass)              314.64    72.73   4.33x       200001x
CameraData (arcade camera strafe, inlined to avoid the escape)       817.05   399.71   2.04x       200001x
Production (PLY yacc.py grammar-rule bookkeeping)                    354.65    97.70   3.63x       200001x
LineState (black linegen.py, method-call API converted to mutation)   74.69    59.85   1.25x            1x
GridPosition (pymunk chipmunk.py, += rewritten to field mutation)     21.56    20.60   1.05x            1x
```

Correctness (baseline vs. ASR fast path vs. ASR fallback path, all
bit-identical) is checked before any timing is trusted, same protocol as
`../../benchmarks/harness.py`.

## Two different kinds of win

The five split cleanly by which unboxing mode they exercise, and that
split is directly visible in the numbers:

- **Reconstruction mode** (`SievePolynomial`, `CameraData`, `Production`
  — all frozen dataclasses, rebuilt via a full constructor call each
  iteration): baseline conses one object per iteration; ASR conses none
  (200,001 → 1 construction). The **2–4x wall-time speedup comes
  primarily from eliminating that allocation**, the same effect the
  paper's own Table 1 measures.
- **Mutation mode** (`LineState`, `GridPosition` — non-frozen dataclasses,
  updated via direct field assignment): the ORIGINAL, untransformed code
  was *already* allocation-free (that's the whole point of using
  mutation instead of reconstruction) — both baseline and ASR show 1
  construction. There is **no allocation win available here at all**.
  The 1.05–1.25x speedup that remains is purely from replacing repeated
  Python attribute get/set (`__dict__`/descriptor-protocol overhead) with
  plain local-variable access once scalarized — real, but a
  qualitatively different and much smaller effect than reconstruction
  mode's headline allocation-elimination story.

This is a useful, unplanned finding in its own right: it makes concrete,
with real numbers, exactly what corpus-study/README.md's "Two different
kinds of win" from v1.4's design docstring predicted in the abstract —
mutation mode exists to cover a real Python idiom reconstruction mode
can't reach at all, but it does NOT deliver the same category of
performance benefit, precisely because well-written mutation-style code
was never paying the allocation cost reconstruction-style code pays.

## What each benchmark adapts, briefly

- **`bench_sieve_polynomial.py`** — sympy's quadratic sieve factorization
  (`ntheory/qs.py`). Declined because `__init__` computes derived fields
  (`self.a2 = a**2`, etc.); adapted by moving that computation to the
  call site (matching `../../benchmarks/bench_ballistic.py`'s existing
  pattern for the same FOL-side shape). The Gray-code factor-base walk
  that drives the real `b` update is simplified to a deterministic
  alternation — not the record-accumulator shape under test.
- **`bench_camera_data.py`** — arcade's camera `strafe` movement.
  Declined for a genuine escape (`camera_data` passed as a bare argument
  to `grips.strafe(...)`); adapted by inlining `strafe`'s own vector math
  directly, reading fields instead of passing the object. `pyglet.math.
  Vec3` is replaced by a tiny local normalize/cross helper (no new
  dependency).
- **`bench_production.py`** — PLY (Python Lex-Yacc)'s grammar-rule
  bookkeeping, vendored into astropy. Declined because `__init__` has 10+
  statements including a symbol-dedup loop; adapted by keeping the two
  scalar derived fields (`length`, `rule_repr`) computed at the call
  site and dropping the list-valued `usyms` field (a collection, not a
  scalar — out of scope for a different, structural reason).
- **`bench_line_state.py`** — black's `Line`/`current_line.append_safe`
  formatting accumulator. Declined because it's a method call, not a
  field access, on an object whose own fields are partly list-valued;
  adapted to the scalar bookkeeping (depth, length, bracket depth) black
  tracks per line, with the method-call API converted to direct field
  mutation.
- **`bench_grid_position.py`** — pymunk's own benchmark suite
  (`benchmarks/chipmunk.py`), laying out a grid of physics bodies.
  Declined because `position += (2*a, 0)` is operator-overload dispatch
  (`ast.AugAssign`), a shape this project doesn't recognize at all, and
  because the real site has three separate reconstruction points for one
  accumulator (reconstruct mode's `_analyze_loop_body` allows only one).
  Adapted to direct field mutation, which has neither restriction.

## Running

```bash
cd corpus-study/benchmarks
python run_all.py
```

Or run any one directly, e.g. `python bench_sieve_polynomial.py`.
