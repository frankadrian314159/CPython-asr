"""Not a test module itself (no test_ prefix) -- classes referenced via
`fixtures.ClassName(...)` from test_qualified_calls.py, exercising the
`import module` + `module.ClassName(...)` shape v1.6 recognizes, as
opposed to `from module import ClassName` + bare `ClassName(...)`.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class QualPoint(object):
    x: float
    y: float


@dataclasses.dataclass
class QualMutablePoint(object):  # not frozen
    x: float
    y: float


class QualPlainCounter(object):
    def __init__(self, n, total):
        self.n = n
        self.total = total

    def __eq__(self, other):
        return isinstance(other, QualPlainCounter) and self.n == other.n and self.total == other.total
