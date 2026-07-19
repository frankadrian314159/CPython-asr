"""Runs all three ported benchmarks and prints a summary table, mirroring
the shape of FOL's benchmarks/run-asr-bench.lisp summary output. This is
the "existence proof" data point for the paper's Threats to Validity
section -- a minimal port, not a claim of Table 1/2 parity (single
machine, opt-in decorator only, no interprocedural inlining, no
multi-accumulator fixpoint).
"""

import platform
import sys

from benchmarks import bench_particle, bench_counter, bench_assoc


def main():
    print(f"Python {sys.version.split()[0]} on {platform.platform()}")
    print("200,000 iterations per call, 1 warm-up + 20 timed trials\n")

    results = []
    for mod in (bench_particle, bench_counter, bench_assoc):
        r = mod.main()
        if r is not None:
            results.append(r)
        print()

    print("=" * 88)
    print("SUMMARY (mean over 20 trials)")
    print("=" * 88)
    header = f"{'Benchmark':<58} {'Base ms':>8} {'ASR ms':>8} {'Time':>8} {'Constructions':>14}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['name']:<58} {r['base_ms']:8.2f} {r['asr_ms']:8.2f} "
            f"{r['speedup']:6.2f}x {r['alloc_ratio']:12.0f}x"
        )


if __name__ == "__main__":
    main()
