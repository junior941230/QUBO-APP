"""Apply the todo.md experiment rules to the per-patient result CSVs.

Each step is compared with the current best configuration. A change is kept
only when, on the baseline (inner-validation tuned threshold):
  * per-patient seizure-file macro F1 improves for more patients than it worsens
    (patients whose F1 moves by less than EPS are ties and do not count; several
    patients stay at F1=0 under every configuration),
  * mean per-patient AP and mean macro F1 both improve, and
  * mean non-seizure FP rate rises by at most FP_TOLERANCE.
"""
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

FP_TOLERANCE = 0.01
# todo.md asks for AP to rise; a strict ">" rejects steps whose AP moves by noise.
# AP_TOLERANCE allows a drop below this much (non-inferiority), symmetric to FP.
AP_TOLERANCE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
EPS = 0.005  # per-patient F1 differences within this are ties

# (step label, candidates compared against the current best at that point)
PLAN = [
    ("Phase 1: class weight", ["spw25", "spw100", "spw500"]),
    ("Phase 1: max_delta_step", ["+mds1"]),
    ("Phase 2: patient-balanced weights", ["+pbw"]),
    ("Phase 2: negative downsampling", ["+neg025"]),
    ("Phase 3: log power", ["+logpower"]),
    ("Phase 3: relative power", ["+relative"]),
    ("Phase 3: robust normalization", ["+robust"]),
    ("Phase 3: causal context", ["+context5", "+context10"]),
]
# "+x" candidates resolve to "<current best>-x" when that run exists, so each
# step is built on the current best. Otherwise fall back to the August matrix,
# which built every later step on spw25 (and context on spw25+robust).
NAME = {
    "spw25": "spw25", "spw100": "spw100", "spw500": "spw500",
    "+mds1": "spw25-mds1", "+pbw": "spw25-pbw", "+neg025": "spw25-neg025",
    "+logpower": "spw25-logpower", "+relative": "spw25-relative",
    "+robust": "spw25-robust", "+context5": "spw25-robust-context5",
    "+context10": "spw25-robust-context10",
}
# Which configuration each candidate was actually built on in the matrix.
BUILT_ON = {
    "spw25-mds1": "spw25", "spw25-pbw": "spw25", "spw25-neg025": "spw25",
    "spw25-logpower": "spw25", "spw25-relative": "spw25", "spw25-robust": "spw25",
    "spw25-robust-context5": "spw25-robust", "spw25-robust-context10": "spw25-robust",
}


def per_patient(csv):
    df = pd.read_csv(csv)
    seizure = df[df.has_seizure.astype(bool)]
    g = df.groupby("subject")
    out = pd.DataFrame({
        "AP": g.patient_average_precision.first(),
        "F1": g.patient_seizure_macro_baseline_f1.first(),
        "fixed_F1": seizure.groupby("subject").baseline_fixed_f1.mean(),
        "precision": seizure.groupby("subject").baseline_precision.mean(),
        "recall": seizure.groupby("subject").baseline_recall.mean(),
        "FP": g.patient_nonseizure_baseline_fp_rate.first(),
        "qubo_F1": g.patient_seizure_macro_qubo_f1.first(),
        "qubo_FP": g.patient_nonseizure_qubo_fp_rate.first(),
    })
    return out


def runtime(log):
    stamps = re.findall(r"^\[(\d\d:\d\d:\d\d)\]", Path(log).read_text(errors="ignore"), re.M)
    if len(stamps) < 2:
        return None
    t = [datetime.strptime(s, "%H:%M:%S") for s in stamps]
    total, prev = timedelta(), t[0]
    for cur in t[1:]:
        step = cur - prev
        total += step if step >= timedelta() else step + timedelta(days=1)
        prev = cur
    return total


def compare(best, cand):
    d = cand.F1 - best.F1
    better, worse = int((d > EPS).sum()), int((d < -EPS).sum())
    n = len(d)
    checks = {
        "majority": better > worse,
        "AP_up": cand.AP.mean() > best.AP.mean() - AP_TOLERANCE,
        "F1_up": cand.F1.mean() > best.F1.mean(),
        "FP_ok": cand.FP.mean() <= best.FP.mean() + FP_TOLERANCE,
    }
    return better, worse, n - better - worse, checks


def main(result_dir):
    result_dir = Path(result_dir)
    data = {p.stem: per_patient(p) for p in sorted(result_dir.glob("*.csv"))}
    times = {p.stem: runtime(p) for p in sorted(result_dir.glob("*.log"))}

    print("## Summary (means over 24 held-out patients; baseline uses the tuned threshold)\n")
    rows = []
    for name, d in data.items():
        rows.append({"config": name, "AP": d.AP.mean(), "F1": d.F1.mean(),
                     "fixed_F1": d.fixed_F1.mean(), "precision": d.precision.mean(),
                     "recall": d.recall.mean(), "FP": d.FP.mean(),
                     "QUBO_F1": d.qubo_F1.mean(), "QUBO_FP": d.qubo_FP.mean(),
                     "runtime": str(times.get(name) or "")})
    print(pd.DataFrame(rows).round(4).to_markdown(index=False))

    print(f"\n## Sequential decisions (todo.md rules, AP tolerance {AP_TOLERANCE})\n")
    if "phase0" not in data:
        print("phase0 missing; cannot run the decision chain")
        return
    best = "phase0"
    for label, cands in PLAN:
        names = []
        for c in cands:
            on_best = f"{best}-{c[1:]}" if c.startswith("+") else None
            if on_best in data:
                BUILT_ON[on_best] = best
                names.append(on_best)
            elif NAME[c] in data:
                names.append(NAME[c])
        if not names:
            print(f"- {label}: no results yet")
            continue
        results = []
        for name in names:
            base = BUILT_ON.get(name)
            note = "" if base in (None, best) else f" (built on {base}, current best is {best})"
            better, worse, tie, checks = compare(data[best], data[name])
            results.append((all(checks.values()), data[name].F1.mean(), name, better, worse, tie, checks, note))
            flags = ", ".join(f"{k}={'Y' if v else 'N'}" for k, v in checks.items())
            print(f"- {label}: {name} vs {best}: better={better} worse={worse} tie={tie}; {flags}{note}")
        accepted = [r for r in results if r[0]]
        if accepted:
            best = max(accepted, key=lambda r: r[1])[2]
            print(f"  -> keep {best}")
        else:
            print(f"  -> revert, best stays {best}")
    print(f"\nFinal best baseline: {best}")

    print("\n## QUBO gain on the final best baseline\n")
    d = data[best]
    gain = d.qubo_F1 - d.F1
    print(f"mean baseline F1 {d.F1.mean():.4f}, QUBO F1 {d.qubo_F1.mean():.4f}, "
          f"gain {gain.mean():+.4f}; better={int((gain > EPS).sum())} "
          f"worse={int((gain < -EPS).sum())}; FP {d.FP.mean():.4f} -> {d.qubo_FP.mean():.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results" / "todo-20261015")
