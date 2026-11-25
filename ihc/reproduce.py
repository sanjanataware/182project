"""Regenerate every table and figure in the report from the saved per-prompt scores, then check them.

  python ihc/reproduce.py            # paper tables + figures (a few minutes on a laptop, no GPU)
  python ihc/reproduce.py --all      # also the supplementary analyze.py tables (slower)

Inputs (all in the repository): scored/gpt-oss-20b__*.jsonl, results/eqat/, results/heldout_ce.json.
Before running, it snapshots the committed CSVs; afterwards it compares each regenerated CSV with its
committed version and prints PASS/FAIL (max absolute difference over numeric columns).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"

# (script + args, CSVs it writes, what it is in the report)
STEPS = [
    (["plot_oss_curve.py"], ["oss_curve.csv"],
     "retention tables (RTN dose-response, GPTQ vs RTN, calibration, EfficientQAT) + supporting figures"),
    (["plot_dose_response.py"], ["oss_dose_response_values.csv"], "Figure 1: dose-response"),
    (["calibration_diffs.py"], ["calibration_diffs.csv"], "calibration paired differences"),
    (["difficulty.py"], ["difficulty_bins.csv", "difficulty_matched.csv", "difficulty_counts.csv"],
     "difficulty-matched gaps and items-lost table"),
    (["table_heldout_loss.py"], ["heldout_loss_table.csv"], "held-out loss vs. IHEval table"),
    (["plot_eqat_curves.py"], [], "E2E-QP training curves figure"),
    (["plot_blockap_layers.py"], ["oss_blockap_layers.csv"], "appendix: Block-AP per-layer error"),
]
SUPPLEMENTARY = [
    (["analyze.py", *[str(p) for p in sorted((ROOT / "scored").glob("gpt-oss-20b__*.jsonl"))]],
     ["cells.csv", "delta_vs_base.csv", "noise_floor.csv", "headline.csv", "gap.csv", "did.csv"],
     "supplementary: per-cell rates, paired deltas, noise floor, DiD, regression"),
]


def compare(new: Path, old: Path) -> tuple[bool, str]:
    a, b = pd.read_csv(new), pd.read_csv(old)
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return False, f"shape/columns differ: {a.shape} vs {b.shape}"
    num = a.select_dtypes("number").columns
    other = [c for c in a.columns if c not in num]
    if not a[other].astype(str).equals(b[other].astype(str)):
        return False, "non-numeric columns differ"
    x, y = a[num].to_numpy(float), b[num].to_numpy(float)
    both_nan = np.isnan(x) & np.isnan(y)
    d = np.where(both_nan, 0.0, np.abs(x - y))
    dev = float(np.nanmax(d)) if d.size else 0.0
    return bool(dev < 1e-9 and not np.isnan(d).any()), f"max |diff| {dev:.1e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="also run the supplementary analyze.py tables")
    args = ap.parse_args()
    steps = STEPS + (SUPPLEMENTARY if args.all else [])
    (RESULTS / "figs").mkdir(parents=True, exist_ok=True)

    snap = Path(tempfile.mkdtemp(prefix="ihc_committed_"))
    for _, csvs, _ in steps:
        for c in csvs:
            if (RESULTS / c).exists():
                shutil.copy(RESULTS / c, snap / c)

    for cmd, _, what in steps:
        t = time.time()
        print(f"\n=== {cmd[0]}  ({what})", flush=True)
        subprocess.run([sys.executable, str(HERE / cmd[0]), *cmd[1:]], cwd=HERE, check=True,
                       stdout=subprocess.DEVNULL if cmd[0] == "analyze.py" else None)
        print(f"    done in {time.time() - t:.0f}s", flush=True)

    print("\n=== check against committed results")
    ok_all = True
    for _, csvs, _ in steps:
        for c in csvs:
            if not (snap / c).exists():
                print(f"  NEW   {c} (no committed copy to compare)")
                continue
            ok, msg = compare(RESULTS / c, snap / c)
            ok_all &= ok
            print(f"  {'PASS' if ok else 'FAIL'}  {c:32s} {msg}")
    print("\nALL RESULTS REPRODUCED" if ok_all else "\nSOME RESULTS DIFFER (see above)")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
