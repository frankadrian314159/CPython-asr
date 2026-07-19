"""Shared benchmark harness, ported (not reused -- different language) from
FOL's benchmarks/run-asr-bench.lisp: one warm-up call, N timed trials,
mean +/- sample stddev, correctness checked before any timing is trusted.

Allocation is measured by directly counting constructor calls to the
tracked dataclass during the measurement window, not via tracemalloc
snapshot-diffing. An earlier version of this harness used tracemalloc's
net-live-bytes diff and it was actively misleading here (reported ~1.0x
allocation "reduction" alongside a real ~5x wall-clock speedup): Python's
small-object allocator aggressively reuses freed same-size slots within
a tight loop of identically-shaped instances, so a NET snapshot diff sees
almost no growth regardless of how many objects were actually
constructed and discarded. Counting constructor calls directly is exact
and immune to that reuse confound -- the same lesson FOL's own session
learned about small-allocation measurement granularity, just a different
failure mode (allocator reuse vs. counter quantization) for the same
underlying reason: don't trust an allocation proxy you haven't checked
against a workload where you already know the right answer.
"""

import gc
import statistics
import time


def mean_stddev(xs):
    m = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return m, sd


def time_trials(fn, arg, trials=20, warmup=1):
    for _ in range(warmup):
        fn(arg)
    samples = []
    for _ in range(trials):
        gc.collect()
        t0 = time.perf_counter()
        fn(arg)
        t1 = time.perf_counter()
        samples.append(t1 - t0)
    return mean_stddev(samples)


def count_constructions_per_call(fn, arg, cls, batch=1):
    """Temporarily wraps cls.__init__ to count how many instances get
    constructed while `fn` runs, then restores the original. Safe for
    frozen dataclasses: the wrapper calls straight through to the
    generated __init__, which itself uses object.__setattr__ internally
    to bypass the frozen __setattr__ override -- untouched here."""
    count = 0
    real_init = cls.__init__

    def counting_init(self, *a, **kw):
        nonlocal count
        count += 1
        real_init(self, *a, **kw)

    cls.__init__ = counting_init
    try:
        for _ in range(batch):
            fn(arg)
    finally:
        cls.__init__ = real_init
    return count / batch


def run_benchmark(name, original, transformed, cell, cls, arg, n_correctness=5, trials=20):
    """Runs the full protocol for one benchmark: correctness (fast path
    vs. slow path vs. original, all must agree) gates everything else;
    only if that passes do we report timing and allocation."""
    print(f"=== {name} ===")

    baseline_results = [original(arg) for _ in range(n_correctness)]

    cell.valid = True
    fast_results = [transformed(arg) for _ in range(n_correctness)]
    cell.valid = False
    slow_results = [transformed(arg) for _ in range(n_correctness)]
    cell.valid = True

    if not (baseline_results == fast_results == slow_results):
        print("  CORRECTNESS FAILURE -- baseline/fast/slow path disagree, skipping timing")
        print(f"  baseline={baseline_results[0]!r} fast={fast_results[0]!r} slow={slow_results[0]!r}")
        return None
    print(f"  Correctness: baseline, ASR fast path, and ASR fallback path bit-identical across {n_correctness} calls.")

    cell.valid = False  # ensure baseline timing never accidentally takes the fast path
    base_mean, base_sd = time_trials(original, arg, trials=trials)
    cell.valid = True
    asr_mean, asr_sd = time_trials(transformed, arg, trials=trials)

    cell.valid = False
    base_constructions = count_constructions_per_call(original, arg, cls)
    cell.valid = True
    asr_constructions = count_constructions_per_call(transformed, arg, cls)

    speedup = base_mean / asr_mean if asr_mean > 0 else float("inf")
    alloc_ratio = base_constructions / asr_constructions if asr_constructions > 0 else float("inf")

    print(f"  Baseline : {base_mean * 1000:9.4f} +/- {base_sd * 1000:7.4f} ms   {base_constructions:10.1f} constructions/call")
    print(f"  ASR      : {asr_mean * 1000:9.4f} +/- {asr_sd * 1000:7.4f} ms   {asr_constructions:10.1f} constructions/call"
          f"   {speedup:5.2f}x time   {alloc_ratio:8.1f}x fewer constructions")
    return {
        "name": name,
        "base_ms": base_mean * 1000,
        "asr_ms": asr_mean * 1000,
        "base_constructions": base_constructions,
        "asr_constructions": asr_constructions,
        "speedup": speedup,
        "alloc_ratio": alloc_ratio,
    }
