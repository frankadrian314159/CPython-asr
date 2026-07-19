"""Runs all 5 corpus-derived benchmarks and prints a summary table.

Unlike benchmarks/ (the paper's own Table 1 set), these five are ADAPTED
from real sites found during the corpus study (corpus-study/README.md's
case studies table) -- not from the FOL paper. Each benchmark's own
module docstring documents exactly what changed and why, mirroring the
FOL repo's own "whole-program port" precedent for its 3 corpus sites
(reitit/fastmath/datascript): none of the 5 already qualified verbatim,
each needed a real, documented adaptation to become ASR-addressable.
"""

import platform
import sys
from pathlib import Path

# corpus-study has a hyphen, so it can't be imported as a dotted package
# name (`corpus_study.benchmarks`) -- add this directory to sys.path
# directly and import the bare module names instead, same effect as
# each individual bench_*.py's own sys.path.insert for asr/benchmarks.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_sieve_polynomial
import bench_camera_data
import bench_production
import bench_line_state
import bench_grid_position


def main():
    print(f"Python {sys.version.split()[0]} on {platform.platform()}")
    print("Adapted from real sites found by corpus-study/classify.py -- see each")
    print("benchmark's own module docstring for what changed and why.\n")

    results = []
    for mod in (
        bench_sieve_polynomial,
        bench_camera_data,
        bench_production,
        bench_line_state,
        bench_grid_position,
    ):
        r = mod.main()
        if r is not None:
            results.append(r)
        print()

    print("=" * 96)
    print("SUMMARY (mean over 20 trials)")
    print("=" * 96)
    header = f"{'Benchmark':<66} {'Base ms':>8} {'ASR ms':>8} {'Time':>8} {'Constructions':>14}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['name']:<66} {r['base_ms']:8.2f} {r['asr_ms']:8.2f} "
            f"{r['speedup']:6.2f}x {r['alloc_ratio']:12.0f}x"
        )


if __name__ == "__main__":
    main()
