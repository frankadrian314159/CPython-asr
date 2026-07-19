"""LineState benchmark, adapted from black's line-formatting accumulator
(corpus-study/corpus/psf__black/src/black/linegen.py's `current_line:
Line`, and black/src/black/lines.py's `Line` dataclass -- see
corpus-study/README.md's case studies table).

The REAL corpus site declines ASR classification for a genuine escape:

    current_line.append_safe(leaf, preformatted=True)

`current_line.append_safe(...)` calls a METHOD on the accumulator, not a
field access -- `Line` has real behavior (an `append`/`append_safe` API
that also mutates a `BracketTracker` and list-valued `leaves`/`comments`
fields), not just data, so this is out of scope by design: this
technique unboxes DATA records, not objects with methods, and `Line`'s
own `leaves`/`comments` fields are themselves growing collections, not
scalars, which a flat scalar-replacement pass can't unbox regardless of
the method-call issue.

Adapted here to the scalar BOOKKEEPING black's own `Line`/`BracketTracker`
track per output line -- depth, running character length, and bracket
nesting depth -- updated via direct field mutation as a sequence of
synthetic tokens (a length and a bracket-delta each) is "appended" one at
a time, the same per-leaf loop shape `append`/`bracket_tracker.mark`
drive in the real formatter, with the OOP method-call API converted to
the field-mutation idiom this project's mutate mode (v1.4) targets.
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from asr import asr, guard
from benchmarks.harness import run_benchmark

# A fixed cycle of synthetic "tokens": (text_length, bracket_delta),
# standing in for the leaves black appends one at a time while it walks
# a line's token stream (opening a bracket = +1, closing = -1, plain
# text/operators = 0).
_TOKENS = ((4, 0), (1, 1), (3, 0), (1, 0), (1, -1), (2, 0), (1, 1), (1, -1))


@dataclasses.dataclass
class LineState(object):  # not frozen -- mutate mode, matching Line's own non-frozen dataclass shape
    depth: int
    length: int
    bracket_depth: int
    max_bracket_depth: int


def run_line_state_original(iterations):
    current_line = LineState(depth=0, length=0, bracket_depth=0, max_bracket_depth=0)
    i = 0
    while i < iterations:
        text_length, bracket_delta = _TOKENS[i % len(_TOKENS)]
        # current_line.append_safe(leaf, ...), inlined as direct field
        # mutation instead of a method call:
        current_line.length = current_line.length + text_length + 1  # +1 for a separating space
        current_line.bracket_depth = current_line.bracket_depth + bracket_delta
        if current_line.bracket_depth > current_line.max_bracket_depth:
            current_line.max_bracket_depth = current_line.bracket_depth
        if current_line.bracket_depth == 0 and bracket_delta < 0:
            current_line.depth = current_line.depth + 1
        i += 1
    return current_line


def run_line_state(iterations):
    current_line = LineState(depth=0, length=0, bracket_depth=0, max_bracket_depth=0)
    i = 0
    while i < iterations:
        text_length, bracket_delta = _TOKENS[i % len(_TOKENS)]
        current_line.length = current_line.length + text_length + 1
        current_line.bracket_depth = current_line.bracket_depth + bracket_delta
        if current_line.bracket_depth > current_line.max_bracket_depth:
            current_line.max_bracket_depth = current_line.bracket_depth
        if current_line.bracket_depth == 0 and bracket_delta < 0:
            current_line.depth = current_line.depth + 1
        i += 1
    return current_line


run_line_state = asr(run_line_state)
assert getattr(run_line_state, "__asr_transformed__", False), "LineState benchmark failed to transform"


def main():
    cell = guard._registry[(run_line_state.__module__, "LineState")]
    return run_benchmark(
        "LineState (black linegen.py, method-call API converted to field mutation)",
        run_line_state_original,
        run_line_state,
        cell,
        LineState,
        arg=200_000,
    )


if __name__ == "__main__":
    main()
