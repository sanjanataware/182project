"""Is "compliance drops more than ability" just a difficulty effect?

  python ihc/difficulty.py [--model gpt-oss-20b]   -> results/difficulty_*.csv

Borderline items flip under any perturbation; easy items don't. BF16 refuses only ~2/3 of the
extraction attacks, while the ability items are near 100%, so compliance items are on average more
borderline. This script compares compliance and ability items OF THE SAME BF16 DIFFICULTY.

Difficulty = an item's BF16 pass rate on two "binning" seeds: 0, 0.5 or 1 (fail both / one / both).
The change is measured on two OTHER seeds, quantized vs. BF16 on the same seeds (paired: same
per-request sampling seeds). Using disjoint seeds avoids regression to the mean: an item that passed
twice by luck would otherwise "drop" with no quantization at all, and that bias would be larger for
the more borderline compliance items, faking the effect being tested.
  * levels with seeds 2,3: bin on BF16 seeds 0,1, compare on seeds 2,3
  * levels with only seeds 0,1: bin on BF16 seeds 2,3, compare on seeds 0,1

Outputs
  difficulty_bins.csv     per task x level x bin: items and mean pass-rate drop, compliance vs ability
  difficulty_matched.csv  difficulty-matched gap = sum_bin w_bin (drop_compliance - drop_ability), with
                          w = the compliance items' bin distribution (bins with >= MIN_ITEMS of both);
                          95% CI by cluster bootstrap over example groups. Also the reliable-items-only
                          gap (bin 1: items BF16 passed on both binning seeds) and the raw (unmatched) gap.
                          drop = BF16 rate - quantized rate, so POSITIVE = compliance drops more than
                          ability at equal BF16 difficulty.
  difficulty_counts.csv   absolute numbers: BF16 and quantized pass rates on the comparison seeds,
                          items lost (BF16 bin 1 -> failed both comparison seeds) and gained (bin 0 -> passed both)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import RESULTS, SCORED

MIN_ITEMS = 10


def load(model):
    rows = []
    for f in sorted(SCORED.glob(f"{model}__*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    d = pd.DataFrame(rows)
    d = d[d.depth == 0]
    d["kind"] = np.where(d.setting == "conflict", "compliance", "ability")
    d["compliant"] = d.compliant.astype(float)
    return d


def item_rates(d, prec, seeds):
    x = d[(d.precision == prec) & d.seed.isin(seeds)]
    if x.seed.nunique() < len(seeds):
        return None
    return x.groupby(["uid", "task", "kind", "group_id"], as_index=False).compliant.mean()


def level_table(d, prec, base="bf16"):
    have = set(d[d.precision == prec].seed)
    if {2, 3} <= have:
        cmp_seeds, bin_seeds = [2, 3], [0, 1]
    elif {0, 1} <= have:
        cmp_seeds, bin_seeds = [0, 1], [2, 3]
    else:
        return None
    b_bin = item_rates(d, base, bin_seeds)
    b_cmp = item_rates(d, base, cmp_seeds)
    q_cmp = item_rates(d, prec, cmp_seeds)
    if b_bin is None or b_cmp is None or q_cmp is None:
        return None
    t = (b_bin.rename(columns={"compliant": "bin"})
         .merge(b_cmp[["uid", "compliant"]].rename(columns={"compliant": "base"}), on="uid")
         .merge(q_cmp[["uid", "compliant"]].rename(columns={"compliant": "quant"}), on="uid"))
    t["drop"] = t.base - t.quant
    t["precision"], t["cmp_seeds"] = prec, str(cmp_seeds)
    return t


def matched_gap(t, idx=None):
    """sum_bin w_bin (drop_comp - drop_abil), w = compliance bin distribution over usable bins."""
    if idx is not None:
        t = t[t.group_id.isin(idx)]
    gaps, ws = [], []
    for b, g in t.groupby("bin"):
        c, a = g[g.kind == "compliance"], g[g.kind == "ability"]
        if len(c) >= MIN_ITEMS and len(a) >= MIN_ITEMS:
            gaps.append(c["drop"].mean() - a["drop"].mean()); ws.append(len(c))
    if not ws:
        return np.nan
    return float(np.average(gaps, weights=ws))


def reliable_gap(t):
    g = t[t.bin == 1.0]
    c, a = g[g.kind == "compliance"], g[g.kind == "ability"]
    if len(c) < MIN_ITEMS or len(a) < MIN_ITEMS:
        return np.nan
    return c["drop"].mean() - a["drop"].mean()


def raw_gap(t):
    return t[t.kind == "compliance"]["drop"].mean() - t[t.kind == "ability"]["drop"].mean()


def boot(t, fn, n_boot, rng):
    groups = t.group_id.unique()
    by = {g: x for g, x in t.groupby("group_id")}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(groups, len(groups))
        vals.append(fn(pd.concat([by[g] for g in pick], ignore_index=True)))
    vals = np.array(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    return (np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-oss-20b")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    d = load(args.model)
    levels = [p for p in sorted(d.precision.unique()) if p not in ("bf16", "native")]
    bins, matched, counts = [], [], []
    for prec in levels:
        t = level_table(d, prec)
        if t is None:
            continue
        for task, tt in t.groupby("task"):
            for (b, kind), g in tt.groupby(["bin", "kind"]):
                bins.append(dict(task=task, precision=prec, cmp_seeds=g.cmp_seeds.iloc[0], bf16_bin=b, kind=kind,
                                 n_items=len(g), base_rate=g.base.mean(), quant_rate=g.quant.mean(),
                                 mean_drop=g["drop"].mean()))
            row = dict(task=task, precision=prec, cmp_seeds=tt.cmp_seeds.iloc[0])
            for name, fn in [("matched", matched_gap), ("reliable_only", reliable_gap), ("raw", raw_gap)]:
                lo, hi = boot(tt, fn, args.n_boot, rng)
                row.update({f"{name}_gap": fn(tt), f"{name}_lo": lo, f"{name}_hi": hi})
            matched.append(row)
            for kind, g in tt.groupby("kind"):
                counts.append(dict(task=task, precision=prec, kind=kind, n_items=len(g),
                                   bf16_rate=g.base.mean(), quant_rate=g.quant.mean(),
                                   n_bf16_reliable=int((g.bin == 1).sum()),
                                   items_lost=int(((g.bin == 1) & (g.quant == 0)).sum()),
                                   items_gained=int(((g.bin == 0) & (g.quant == 1)).sum())))
    pd.set_option("display.width", 250)
    for name, rows in [("bins", bins), ("matched", matched), ("counts", counts)]:
        df = pd.DataFrame(rows)
        df.to_csv(RESULTS / f"difficulty_{name}.csv", index=False)
        print(f"\n== difficulty_{name}.csv"); print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
