# ASR pattern corpus study (Python, direct measurement)

Estimates how often the **aggregate-scalar-replacement (ASR) pattern** — a
record accumulator carried through a loop and rebuilt each iteration —
occurs in real Python code, and separately, what fraction of those sites
`cpython-asr`'s *actual* qualification gates would accept.

This mirrors `docs/cgo2027/corpus-study/` in the FOL repo (a Clojure corpus,
analyzed as a *proxy* for FOL, a different language) as closely as Python's
own constructs allow — but it is not a proxy study. `cpython-asr` targets
Python natively, so this is a **direct measurement** of the pattern's
incidence in the language the tool actually runs against.

## What counts as the pattern

For each `while`/`for` loop and each `functools.reduce(lambda ...)` call, we
classify every loop-carried local (assigned before the loop, then
reassigned or mutated inside it) into:

- **(a) record accumulator, rebuilt** — init is a project-local class
  constructor call, and it is rebuilt at the back-edge via a full
  constructor call, `dataclasses.replace`, **direct field mutation**
  (`p.x = ...`), or passed into a helper call. This is the ASR-addressable
  pattern — and unlike the Clojure study, direct mutation is a real,
  ASR-addressable rebuild mechanism here (`transform.py`'s mutate mode,
  v1.4), not just reconstruction; Clojure's persistent records have no
  equivalent to offer.
- **(b) map/dict accumulator, rebuilt** — `{}`, `dict()`, `defaultdict`, …
- **(c) collection accumulator, grown** — `[]`, `list()`, `set()`, …
- **(d) primitive-scalar loop** — every accumulator is a numeric/bool/None
  literal. The hand-optimized form: what a performance-aware programmer
  writes instead of threading a record.

Two passes, same split as the FOL repo's study:

- **`analyze.py`** — a syntactic-shape proxy (`ast.parse`, no execution):
  any locally-defined class counts as a candidate "record" regardless of
  its internal shape, mirroring `analyze.clj`'s permissive `defrecord`/
  `deftype` recognition. This is the **upper-bound, necessary-condition**
  estimate.
- **`classify.py`** — ports the *actual* gates `asr/transform.py` and
  `asr/guard.py` apply (field-set matching, frozen/mutation-safety,
  full escape analysis, one-level helper inlining), statically, as closely
  as non-executing analysis of arbitrary third-party source allows. This
  *measures*, rather than bounds, the qualifying fraction — see its own
  module docstring for exactly what is and isn't replicated, and why every
  simplification pushes toward undercount.

## Corpus

27 real, actively-maintained Python projects, 10,074 files, chosen to
mirror the Clojure study's 7 domains as closely as Python's own ecosystem
allows:

| domain | projects |
|---|---|
| numeric | sympy, astropy, pint, mpmath, statsmodels |
| graphics/games | pyglet, arcade, pymunk, manim |
| data structures | pyrsistent, sortedcontainers, attrs, boltons |
| language/compilers | mypy, black, LibCST, parso, astroid |
| web | flask, fastapi, starlette, pydantic, httpx |
| tooling | click, rich, poetry, pytest |

Full list with URLs in `manifest.json`; exact commits cloned in
`manifest.lock.json` for reproducibility.

## Results — syntactic-shape pass (`analyze.py`)

```
Projects: 27   Files: 10074   Read errors: 22   Classes defined: 15595
Loop sites: 2784   reduce() sites (classified): 27   Total: 2811

(a) record accumulator rebuilt : 142 (5.05% of sites)
      of which strong (ctor/assoc/mutate): 130
      rebuild via constructor            : 76
      rebuild via dataclasses.replace     : 0
      rebuild via direct mutation (p.x=..): 61
(b) map/dict accumulator rebuilt : 19 (0.68%)
(c) collection accumulator grown : 180 (6.40%)
(d) primitive-scalar loop        : 1099 (39.10%)

Suppression signal (d):(a) = 7.7 : 1

--- by domain (record sites / total sites) ---
  data_structures        1 / 66      1.52%
  graphics_games        57 / 317     17.98%
  language_compilers    18 / 386     4.66%
  numeric               63 / 1822    3.46%
  tooling                0 / 121     0.00%
  web                    3 / 99      3.03%
```

**Notably higher incidence than the Clojure proxy** (5.05% vs. 0.55% of
sites; 142 sites vs. 8) — and the gap is not just noise. Two real,
structural reasons:

1. Python natively supports **direct field mutation** (`p.x = p.x + 1.0`)
   as an idiomatic way to update a record accumulator — 61 of the 142
   sites (43%) use it, with zero using `dataclasses.replace`. Clojure has
   no equivalent at all (persistent records can only be rebuilt via
   `assoc`/`update`/a fresh constructor); the Clojure study's `record-assoc`
   category is the closest analog, and it's near-zero there too. Direct
   mutation is Python's own, structurally distinct idiom for this pattern.
2. **The suppression signal is weaker**: 7.7:1 primitive-vs-record here,
   vs. "27 to 1" reported for the Clojure corpus (e.g. fastmath alone:
   39 primitive loops to 2 record ones). Python programmers appear to
   reach for a primitive-scalar loop less defensively than performance-
   conscious Clojure code does — consistent with, though not proof of,
   the pattern being less aggressively hand-optimized-around in Python.

Domain concentration matches the Clojure finding closely: graphics/games
code carries the pattern far more than any other domain (18.0% of sites,
vs. numeric's 3.5% and tooling's 0%) — physics/position state is exactly
the shape this technique targets.

## Results — gate-faithful pass (`classify.py`)

**v1.6 update**: two of the gaps this study originally surfaced —
module-qualified construction and annotated self-assignment — have since
been fixed in `asr/transform.py`/`asr/guard.py` (see `_resolve_ctor_class`,
`_call_target_matches_class`, and `guard._infer_plain_class_fields`'s
`ast.AnnAssign` support), verified against the real `@asr` decorator with
new passing tests (`tests/test_qualified_calls.py`,
`tests/test_mutation.py`'s new annotated-field cases), and `classify.py`
was updated to match — it no longer has separate "blocked by X" diagnostic
buckets for these two, because they're no longer blocked; they're part of
real qualification now. The corpus was then re-run:

```
Projects: 27   Files: 10074
Forms with >=1 syntactic record-accumulator candidate: 1
  of which qualify under the REAL gates: 0 (0.00%)
Individual accumulator-binding candidates: 1
  of which qualify under the REAL gates: 0 (0.00%)
```

**Unchanged from the pre-fix run**: still only the one mypy `class_info`
site (a genuine escape, see below) reaches real-gate evaluation, and nothing
newly qualifies. This is not a null result on the two fixes — both are
independently verified, tested, and confirmed to work against the exact
shapes they targeted (see the `CameraData` case study below) — it means
this specific 27-project corpus's near-miss sites were blocked by *other*
factors that happened to co-occur, discovered while re-auditing after the
fix. The syntactic-shape proxy (142 sites, 5.05%) still dramatically
overstates real applicability; the reasons why are now better understood,
not resolved.

### Why almost nothing reaches the real gate

Sampling 53 record-accumulator classes across 5 of the largest projects
(astropy, arcade, pymunk, mypy, sympy): **zero use `@dataclass`.** Every
one is a hand-written class with its own `__init__`. This matters because
`guard._infer_plain_class_fields` — confirmed against the actual
`asr/guard.py` source, not assumed — only accepts an `__init__` body that
is a *flat, unconditional* sequence of `self.<name> = <name>` or (v1.6)
`self.<name>: Type = <name>` assignments, one per parameter, in order.
Real `__init__` bodies routinely do more: compute derived fields, apply
defaults, coerce types, call `super().__init__()`, branch, or (newly
discovered, see below) intersperse per-attribute documentation strings —
every one of those aborts inference.

Three gaps have now been isolated and confirmed by hand against
`transform.py`/`guard.py`'s actual source (not assumed from the syntactic
pass) — two fixed in v1.6, one newly found while re-auditing after the fix
and left unfixed (out of the scope of what was asked for this round):

- **Module-qualified construction** (`p = arcade.SpriteSolidColor(...)`,
  from `import arcade` rather than `from arcade import SpriteSolidColor`)
  — **fixed in v1.6.**
- **Annotated self-assignment** (`self.position: tuple[float, ...] =
  position`, a standard modern, type-hinted idiom) — **fixed in v1.6.**
- **Interspersed attribute docstrings** (a bare string-literal statement
  immediately after each field assignment — Sphinx-style per-attribute
  documentation, e.g. `self.x: float = x` followed on the next line by
  `"""A vector which..."""`) — **newly discovered, not fixed.** `guard.
  _infer_plain_class_fields` only strips a single docstring at
  `body[0]` (the function-level one); it doesn't recognize or skip a bare
  string `Expr` statement appearing *between* field assignments, so any
  `__init__` using this documentation style fails inference regardless of
  the first two fixes. Confirmed directly against the real `@asr`
  decorator with a reduced repro of `CameraData`'s exact shape (see case
  study below) — this is precisely why fixing the first two gaps didn't
  change this corpus's measured result: `CameraData` needed all three to
  register, and still only has two.

All three are the same *kind* of gap — narrow, additive, and safely
fixable without touching the escape-analysis or field-matching logic that
actually does the safety-critical work — unlike the Clojure PLDI study's
`:aliased-reference` category, which is structural. The third one is a
natural next candidate if this line of work continues.

### Case studies (hand-audited)

| site | file | verdict | reason |
|---|---|---|---|
| `SievePolynomial` | `sympy/ntheory/qs.py` | correctly declined | `__init__` computes derived fields (`self.a2 = a**2`), not a flat passthrough |
| `Production` | `astropy/extern/ply/yacc.py` (vendored PLY parser) | correctly declined | `__init__` has 10+ statements: list construction, string formatting, more locals than parameters |
| `CameraData` | `arcade/camera/data_types.py` | correctly declined, for a **third, previously-undiscovered reason** | originally blocked by TWO now-fixed gaps (module-qualified `camera.CameraData(...)` construction, annotated `self.position: tuple[...] = position`) — re-auditing after the v1.6 fix revealed a THIRD, independent blocker: each field assignment is immediately followed by its own string-literal "attribute docstring," which `guard._infer_plain_class_fields` doesn't recognize as skippable. Confirmed directly against the real decorator with a reduced repro matching this exact shape. |
| `class_info` (`ClassInfo`) | `mypy/stubgenc.py:827` | the one real candidate; **correctly declined for a genuine escape** | `class_info` is passed as a bare argument to several helper methods (`is_method`, `is_staticmethod`, `generate_function_stub`) inside the loop — exactly the aliasing hazard the escape check exists to catch. This is the Python-native instance of the Clojure/FOL papers' `:aliased-reference` category: **structurally unfixable**, not an analysis gap. |

The `class_info` case is worth dwelling on: it's a real, independent
confirmation — in a completely different language and corpus — of the
same phenomenon the FOL papers call the "quicksort-swap" shape and the
Clojure PLDI study's `classify.clj` measures directly (51.6% of its own
genuine collection-init failures are this exact kind of unfixable
aliasing). Even reaching for a *correct* rejection this cleanly, on the
very first real candidate this corpus surfaced, is a small but genuine
piece of evidence that the escape check generalizes.

The `CameraData` case is worth dwelling on too, for the opposite reason:
it's a concrete illustration of how a single real-world class can be
blocked by *multiple, independently-fixable* gaps stacked on top of each
other — fixing two of three doesn't change the outcome for that specific
site, even though both fixes are individually correct and each unlocks
other, different hypothetical code. A corpus of one is not evidence either
fix was pointless; it's evidence that this particular corpus's few
near-miss sites happen to be unusually gap-stacked, which is itself
informative about what real-world `__init__` bodies actually look like.

## Interpretation

Bracket the honest answer the same way the Clojure PLDI study did: **at
least ~0%, at most 5.05%**, of real Python loop/reduce sites in this
corpus carry a genuinely ASR-addressable accumulator. The upper bound
(syntactic shape) is *higher* than Clojure's proxy figure — Python code
uses the record-accumulator pattern more, not less, in absolute terms,
and does so via a mutation idiom Clojure cannot express at all. But the
measured lower bound collapsed almost entirely once the real gates were
applied, for reasons that are mostly (not entirely) fixable: real-world
`__init__` bodies are rarely the trivial shape `guard.
_infer_plain_class_fields` requires, and `@dataclass` — which would sidestep
that requirement entirely — is essentially never used for the numeric/
state classes that carry this pattern in these particular 27 projects.

**Caveats, read before quoting any number:**

1. This corpus is small by web-scale standards (27 projects) and was
   hand-picked to mirror the Clojure study's domain categories, not
   randomly sampled — the same selection-bias caveat any 27-repo sample
   carries.
2. `classify.py`'s gate-faithful pass found only 1 raw candidate in this
   specific corpus; treat its 0% as a qualitative finding ("the syntactic
   proxy overstates applicability, for specific and partly fixable
   reasons"), not a precise measured rate the way the Clojure PLDI study's
   1,166-form sample supports.
3. Both passes are syntactic/AST-only — no macroexpansion-equivalent, no
   import resolution beyond "defined somewhere in this project," no
   dynamic execution. `classify.py`'s own module docstring lists five
   specific, further simplifications, all but two of which push toward
   undercount only (never overcount).
4. `classify.py`'s project-wide class/helper registries (built across
   every file in a project, not per-file) are a closer match to what a
   live `func.__globals__` would actually resolve than a per-file
   restriction would be — verified necessary during development (an
   earlier per-file version undercounted `arcade` by conflating "class
   defined in a different file" with "class doesn't exist").
5. This study has now been re-run once after fixing two of its own
   findings (module-qualified construction, annotated self-assignment),
   and the corpus-measured result didn't move (`CameraData` needed a
   third, still-unfixed gap too) — a reminder that a single-corpus
   before/after comparison can look like "the fix didn't matter" even
   when the fix is independently correct and tested; it just means this
   particular corpus's remaining near-misses aren't the fix's target.

## Usage

```bash
cd corpus-study

# 1. Fetch the corpus (shallow-clones into ./corpus, writes
#    manifest.lock.json with the exact SHAs analyzed).
python fetch.py manifest.json corpus

# 2. Syntactic-shape pass. Writes results.json and prints a summary.
python analyze.py corpus manifest.json results.json

# 3. Gate-faithful pass. Writes results-classify.json and prints a summary.
python classify.py corpus manifest.json results-classify.json
```

To reproduce an earlier run exactly, re-fetch from `manifest.lock.json`
(which pins SHAs) instead of `manifest.json`.

## Output

`results.json`/`results-classify.json` contain per-project counts. The
printed summaries (reproduced above) report totals, the (a)/(b)/(c)/(d)
breakdown and rebuild-mechanism split (analyze.py), and the
qualified/blocked-by-reason breakdown (classify.py), each with a
by-domain table.
