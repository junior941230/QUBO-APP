"""Paired patient-level bootstrap for differences between two result CSVs.

Resamples the held-out patients with replacement and reports the mean
difference (B - A) of each per-patient metric with a 95% percentile interval.
Run-to-run (seed) noise is far smaller than this patient-sampling uncertainty,
so an interval that excludes 0 is the bar for calling a difference real.

Usage: python tools/bootstrap_compare.py A.csv B.csv [--metric F1 ...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from todo_acceptance import per_patient  # noqa: E402

N_BOOT = 10_000


def paired_bootstrap(a, b, n_boot=N_BOOT, seed=0):
    """Return (mean difference, 2.5th, 97.5th percentile) of mean(b - a)."""
    diff = (b - a).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    return diff.mean(), *np.percentile(means, [2.5, 97.5])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--metric", nargs="+", default=["AP", "F1", "FP", "qubo_F1", "qubo_FP"])
    args = parser.parse_args()
    a, b = per_patient(args.a), per_patient(args.b)
    for metric in args.metric:
        mean, low, high = paired_bootstrap(a[metric], b[metric])
        flag = "" if low <= 0 <= high else "  *"
        print(f"{metric:8s} B-A {mean:+.4f}  95% CI [{low:+.4f}, {high:+.4f}]{flag}")


if __name__ == "__main__":
    main()
