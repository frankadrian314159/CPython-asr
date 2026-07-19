"""Production benchmark, adapted from PLY (Python Lex-Yacc)'s grammar-rule
bookkeeping (corpus-study/corpus/astropy__astropy/astropy/extern/ply/
yacc.py, vendored into astropy -- see corpus-study/README.md's case
studies table).

The REAL corpus site declines ASR classification because __init__ does
far more than a flat passthrough:

    class Production(object):
        def __init__(self, number, name, prod, precedence=('right', 0),
                     func=None, file='', line=0):
            self.name = name
            self.prod = tuple(prod)
            self.number = number
            ...
            self.len = len(self.prod)
            self.usyms = []
            for s in self.prod:
                if s not in self.usyms:
                    self.usyms.append(s)
            ...
            self.str = '%s -> %s' % (self.name, ' '.join(self.prod))

10+ statements, more locals than parameters, a dedup loop building
`usyms`. Adapted here by keeping the essential record-accumulator shape
(one Production rebuilt per grammar rule processed) with its two
derived fields -- `length` (`len(prod)`) and `rule_repr` (the same
`name -> sym1 sym2 ...` string PLY itself builds) -- computed at the
call site, matching this project's existing Ballistic/Mandelbrot/
SievePolynomial benchmarks' pattern for FOL/corpus sources whose
`__init__` computes derived fields. The `usyms` dedup loop is a
collection-building op (list, not a scalar field) and is left out, same
as this project's `benchmarks/harness.py` note that a benchmark's job is
to isolate ASR's own record-accumulator pattern, not reproduce every
surrounding line of the original.
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from asr import asr, guard
from benchmarks.harness import run_benchmark

# A fixed cycle of grammar productions, standing in for the rules a real
# parser generator processes one at a time while building its rule table.
_GRAMMAR_RULES = (
    ("expr", ("expr", "PLUS", "term")),
    ("expr", ("term",)),
    ("term", ("term", "TIMES", "factor")),
    ("term", ("factor",)),
    ("factor", ("LPAREN", "expr", "RPAREN")),
    ("factor", ("NUMBER",)),
)


@dataclasses.dataclass(frozen=True)
class Production(object):
    number: int
    name: str
    prod: tuple
    length: int
    rule_repr: str


def run_production_original(iterations):
    name0, prod0 = _GRAMMAR_RULES[0]
    p = Production(0, name0, prod0, len(prod0), f"{name0} -> {' '.join(prod0)}")
    i = 0
    while i < iterations:
        name, prod = _GRAMMAR_RULES[i % len(_GRAMMAR_RULES)]
        number = p.number + 1
        length = len(prod)
        rule_repr = f"{name} -> {' '.join(prod)}" if prod else f"{name} -> <empty>"
        p = Production(number, name, prod, length, rule_repr)
        i += 1
    return p


def run_production(iterations):
    name0, prod0 = _GRAMMAR_RULES[0]
    p = Production(0, name0, prod0, len(prod0), f"{name0} -> {' '.join(prod0)}")
    i = 0
    while i < iterations:
        name, prod = _GRAMMAR_RULES[i % len(_GRAMMAR_RULES)]
        number = p.number + 1
        length = len(prod)
        rule_repr = f"{name} -> {' '.join(prod)}" if prod else f"{name} -> <empty>"
        p = Production(number, name, prod, length, rule_repr)
        i += 1
    return p


run_production = asr(run_production)
assert getattr(run_production, "__asr_transformed__", False), "Production benchmark failed to transform"


def main():
    cell = guard._registry[(run_production.__module__, "Production")]
    return run_benchmark(
        "Production (PLY yacc.py grammar-rule bookkeeping)",
        run_production_original,
        run_production,
        cell,
        Production,
        arg=200_000,
    )


if __name__ == "__main__":
    main()
