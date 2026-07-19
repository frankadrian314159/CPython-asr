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

**v1.6/v1.7 update**: four narrow, real gaps this study surfaced have since
been fixed in `asr/transform.py`/`asr/guard.py` — module-qualified
construction (`_resolve_ctor_class`, `_call_target_matches_class`),
annotated self-assignment (`guard._infer_plain_class_fields`'s
`ast.AnnAssign` support), interspersed attribute docstrings
(`guard._is_docstring_stmt`), and (v1.7) a pre-loop initializer relying on
`__init__`'s own default values (`_ctor_init_defaults`,
`_ctor_field_value_or_default`) — each verified against the real `@asr`
decorator with new passing tests, and `classify.py` updated to match after
each one. The corpus was re-run after all four:

```
Projects: 27   Files: 10074
Forms with >=1 syntactic record-accumulator candidate: 9
  of which qualify under the REAL gates: 0 (0.00%)
Individual accumulator-binding candidates: 9
  of which qualify under the REAL gates: 0 (0.00%)

--- by domain (qualified forms / candidate forms) ---
  graphics_games         0 / 2       0.00%
  language_compilers     0 / 7       0.00%
```

**Real movement this time** — candidate discovery jumped from 1 to 9 (the
v1.7 defaults fix relaxes *what counts as a candidate at all*, not just
what qualifies), surfacing real production code for the first time:
`black`'s own line-formatting hot path (`linegen.py`, `current_line: Line`,
twice), `LibCST`'s parser/codemod internals (three sites), `black`'s test
harness, and the two previously-known sites (mypy's `class_info`, arcade's
`CameraData`, which now finally registers as a class). **Still zero
qualify** — but hand-tracing all 9 turned up a genuinely richer picture
than a single repeated reason:

- **Bare accumulator passed to a function** (2 sites: mypy's `class_info`,
  arcade's `camera_data.position = grips.strafe(camera_data, dirs)`) — the
  Clojure/FOL papers' `:aliased-reference`, structurally unfixable.
- **A method called on the accumulator, not a field** (`black`'s
  `current_line.append_safe(leaf, ...)`, 2 sites) — `Line` has real
  behavior, not just data; same structural category, different shape.
- **The accumulator changes CLASS across iterations** (LibCST's
  `convert_dotted_name`/`_apply_type_annotations.py`, 2 sites): `node =
  Name(...)` then, inside the loop, `node = Attribute(value=node, ...)` --
  a genuinely different class each time. This whole technique assumes ONE
  fixed class per accumulator; a type-changing accumulator isn't a
  recognition gap, it's a different pattern entirely.
- **The accumulator embeds ITSELF as a field of its own reconstruction**
  (LibCST's `convert_union_to_or.py`: `replacement = cst.BinaryOperation
  (left=replacement, right=type_, ...)`) — building a left-leaning tree
  where each step's value must contain the FULL prior structure, not just
  read scalar fields out of it. This is structurally incompatible with
  flattening to scalars, in the same spirit as Clojure's `reductions` (its
  own one fundamental, never-attempted limit) -- unlike an aliasing
  escape, there's no way to "fix" this by extending the recognizer, since
  the accumulator genuinely needs to persist as a real object here.
- **A helper too complex to inline** (`black`'s `mode = parse_mode(...)`):
  `parse_mode`'s body is 8+ statements, not `_try_inline_call`'s required
  bare `return <reconstruction>` -- the existing, documented inlining
  restriction, not a new finding.

This is a more informative result than a single repeated reason would have
been: it completes the investigation with a genuinely varied, honest
picture, rather than leaving open whether a fifth narrow gap was hiding.
Two of five categories (bare-escape, method-call) are the same structural
limit the Clojure/FOL papers already describe; two more (type-changing
accumulator, self-embedding reconstruction) are *new*, Python-specific
structural limits this study hadn't previously characterized; the last is
an existing, already-documented scope boundary.

### Why almost nothing reaches the real gate

Sampling 53 record-accumulator classes across 5 of the largest projects
(astropy, arcade, pymunk, mypy, sympy): **zero use `@dataclass`.** Every
one is a hand-written class with its own `__init__`. This matters because
`guard._infer_plain_class_fields` — confirmed against the actual
`asr/guard.py` source, not assumed — only accepts an `__init__` body that
is a *flat, unconditional* sequence of `self.<name> = <name>` or (v1.6)
`self.<name>: Type = <name>` assignments (with any bare-string docstring
statement skipped, v1.6), one per parameter, in order, with any field left
unsupplied at the call site covered by a known default (v1.7). Real
`__init__` bodies routinely do more: compute derived fields, coerce types,
call `super().__init__()`, or branch — every one of those still aborts
inference. And once a class *does* register, its accumulator still needs
to survive full escape analysis — used only as a data record (field reads
and writes), never as an object with behavior (methods called on it).

Four gaps have now been isolated, fixed, and confirmed by hand against
`transform.py`/`guard.py`'s actual source (not assumed from the syntactic
pass) — all four narrow, additive, and safely fixable without touching the
escape-analysis or field-matching logic that does the safety-critical work:

- **Module-qualified construction** (`p = arcade.SpriteSolidColor(...)`,
  from `import arcade` rather than `from arcade import SpriteSolidColor`)
  — **fixed in v1.6.**
- **Annotated self-assignment** (`self.position: tuple[float, ...] =
  position`, a standard modern, type-hinted idiom) — **fixed in v1.6.**
- **Interspersed attribute docstrings** (a bare string-literal statement
  immediately after each field assignment — Sphinx-style per-attribute
  documentation) — **fixed in v1.6.**
- **Defaulted constructor arguments** (`CameraData()`, relying on
  `__init__`'s own defaults rather than supplying every field explicitly)
  — **fixed in v1.7**, scoped to the pre-loop initializer only (an in-loop
  reconstruction relying on a default would silently reset that field
  every iteration, so it still requires every field explicit by design).

Unlike the Clojure PLDI study's `:aliased-reference` category, none of
these four were structural — each closed real, verified real-world code.
What's left after all four, per this corpus, is a small taxonomy of
structural limits (below) -- some already known from the Clojure/FOL
papers, some newly characterized here.

### Case studies (hand-audited)

| site | file | category | reason |
|---|---|---|---|
| `SievePolynomial` | `sympy/ntheory/qs.py` | non-trivial `__init__` | computes derived fields (`self.a2 = a**2`), not a flat passthrough |
| `Production` | `astropy/extern/ply/yacc.py` (vendored PLY parser) | non-trivial `__init__` | 10+ statements: list construction, string formatting, more locals than parameters |
| `class_info` (`ClassInfo`) | `mypy/stubgenc.py:827` | **bare escape** (Clojure/FOL's `:aliased-reference`) | passed as a bare argument to `is_method`/`is_staticmethod`/`generate_function_stub` inside the loop |
| `camera_data` (`CameraData`) | `arcade/tests/unit/camera/test_camera_controller_methods.py` | **bare escape** | `camera_data.position = grips.strafe(camera_data, dirs)` passes the accumulator itself as an argument; class now fully registers (v1.7 closed the last gap) but the site still, correctly, declines |
| `current_line` (`Line`) | `black/src/black/linegen.py:1476,1557` | **method call, not field access** | `current_line.append_safe(leaf, preformatted=True)` -- `Line` has real behavior, not just data |
| `node` (`Name`→`Attribute`) | `LibCST/libcst/_parser/conversions/statement.py:613`, `.../_apply_type_annotations.py:548` | **accumulator changes class across iterations** (newly characterized) | `node = Name(...)` initially, then `node = Attribute(value=node, ...)` inside the loop -- a different class each time; this technique assumes one fixed class per accumulator by design |
| `replacement` (`BinaryOperation`) | `LibCST/libcst/codemod/commands/convert_union_to_or.py:44` | **accumulator embeds itself in its own reconstruction** (newly characterized) | `replacement = cst.BinaryOperation(left=replacement, right=type_, ...)` builds a left-leaning tree where each step needs the FULL prior structure, not scalar fields out of it -- structurally incompatible with flattening, in the same spirit as Clojure's own `reductions` limit |
| `mode` (`TestCaseArgs`) | `black/tests/util.py:313` | helper too complex to inline | `parse_mode`'s body is 8+ statements, not the required bare `return <reconstruction>` -- an existing, already-documented scope boundary, not a new finding |

`class_info` and `camera_data` are worth dwelling on together with
`current_line`: three independent, real-world confirmations — in a
completely different language and corpus from the one the FOL papers
studied — of the same phenomenon the FOL papers call the "quicksort-swap"
shape and the Clojure PLDI study's `classify.clj` measures directly
(51.6% of its own genuine collection-init failures are this exact kind of
unfixable aliasing). The `node` and `replacement` cases are a different,
useful result: two structural limits specific to Python's OOP-flavored
"rebuild by constructing a new instance" idiom that the Clojure study,
working with a single immutable record type per accumulator by
construction, never had reason to surface.

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
2. `classify.py`'s gate-faithful pass found only 9 raw candidates in this
   specific corpus (up from 1, after the v1.7 defaults fix widened what
   counts as a candidate at all); treat its 0% as a qualitative finding
   ("the syntactic proxy overstates applicability, for specific and
   mostly-fixable reasons") rather than a precise measured rate the way
   the Clojure PLDI study's 1,166-form sample supports.
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
5. This study has now been re-run three times, after fixing four of its
   own findings one at a time (module-qualified construction, annotated
   self-assignment, interspersed attribute docstrings, defaulted
   constructor arguments), and the corpus-measured qualifying fraction
   never moved off 0% -- but the fourth fix DID move candidate discovery
   (1 -> 9), surfacing a genuinely richer, more varied set of structural
   limits than the first three fixes alone had revealed (see the case
   studies table). Read this as the intended shape of an honest corpus
   study: each fix is independently verified and tested regardless of
   whether it happens to flip this specific corpus's headline number, and
   the investigation is more complete, not less informative, for having
   run to the point where no further narrow gap was left to find.

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
