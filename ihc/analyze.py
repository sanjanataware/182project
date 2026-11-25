"""Aggregate scored files into tables, tests and plots.

  python ihc/analyze.py                 # reads scored/*.jsonl, writes results/
  python ihc/analyze.py --baseline bf16 --n-boot 2000

The full analysis runs twice:
  results/               all test prompts (main results; matches Kariyappa & Suh's construction)
  results/clean_subset/  only examples whose filler turns never violate the system rule
                         (an example is clean iff its depth-8 fillers are clean; depths are nested,
                         so the same examples appear at every depth)

Files in each:
  headline.csv       HEADLINE: per depth, conflict compliance change vs. the baseline precision,
                     next to the BF16 seed-vs-seed noise floor for the same cell
  cells.csv          pass rate per model x task x setting x precision x depth (pooled over seeds),
                     95% CI (cluster bootstrap over IHEval example groups), loose rate,
                     truncation and unfinished-thinking rates
  delta_vs_base.csv  paired change vs. the baseline on identical prompts (per-prompt mean over
                     seeds), bootstrap CI, sign-flip permutation p (clustered by example group)
  noise_floor.csv    the same statistic for baseline seed A vs. baseline seed B
  gap.csv            hierarchy gap = aligned - conflict and reference - conflict
  did.csv            H1: (drop in conflict compliance) - (drop in aligned / reference accuracy)
  regression.txt     H3: logistic regression on conflict prompts,
                     compliant ~ quant_level * depth + filler_violations (+ task),
                     cluster-robust SEs by example group (+ categorical-precision version)
  figs/              compliance vs. depth, pass rate vs. bits

Why per-depth comparisons are the headline: at a fixed depth every precision sees the same
fillers, so filler content cannot explain a precision difference. Across depths (rule-following)
depth changes both distance and the content of the history, which the violation covariate and the
clean subset address.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import PRECISION_BITS, RESULTS, SCORED, read_jsonl

KEYS = ["model", "task", "setting", "precision", "depth"]


# ---------------------------------------------------------------- loading
def load(paths) -> pd.DataFrame:
    rows = [r for p in paths for r in read_jsonl(p)]
    if not rows:
        sys.exit("no scored rows")
    df = pd.DataFrame(rows)
    for c, default in [("seed", 0), ("think_unfinished", False), ("degenerate", False), ("empty_reasoning", False)]:
        if c not in df:
            df[c] = default
    for c in ["compliant", "loose", "truncated", "think_unfinished", "degenerate", "empty_reasoning"]:
        df[c] = df[c].astype(int)
    df["bits"] = df["precision"].map(lambda p: PRECISION_BITS.get(p, np.nan))
    dup = df.duplicated(["model", "precision", "seed", "uid"]).sum()
    if dup:
        sys.exit(f"{dup} duplicate (model, precision, seed, uid) rows: remove stale scored files")
    return df


def clean_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Keep example groups whose fillers are clean at every depth (i.e. at depth 8)."""
    bad = df.groupby(["task", "group_id"]).filler_violations.max()
    keep = set(bad[bad == 0].index)
    return df[[k in keep for k in zip(df.task, df.group_id)]]


def prec_order(df):
    return sorted(df["precision"].unique(), key=lambda p: -PRECISION_BITS.get(p, 0))


# ---------------------------------------------------------------- stats helpers
def cluster_boot_mean(values, groups, n_boot, rng):
    """95% CI of a mean, resampling clusters (IHEval example groups) with replacement."""
    values = np.asarray(values, float)
    if len(values) == 0:
        return np.nan, np.nan
    ug, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(ug))
    cnts = np.bincount(inv, minlength=len(ug))
    idx = rng.integers(0, len(ug), size=(n_boot, len(ug)))
    boots = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1)
    return tuple(np.percentile(boots, [2.5, 97.5]))


def signflip_p(diffs, groups, rng, n_perm=10000):
    """Two-sided paired permutation test: flip the sign of each group's summed difference."""
    diffs = np.asarray(diffs, float)
    ug, inv = np.unique(groups, return_inverse=True)
    gsum = np.bincount(inv, weights=diffs, minlength=len(ug))
    obs = abs(gsum.sum())
    if obs == 0:
        return 1.0
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(ug)))
    null = np.abs((signs * gsum).sum(1))
    return float((1 + (null >= obs - 1e-12).sum()) / (n_perm + 1))


def paired_table(a: pd.DataFrame, b: pd.DataFrame, n_boot, rng, label_cols: dict):
    """Per (model, task, setting, depth): mean(b) - mean(a) over prompts present in both.
    a, b: per-prompt rates with columns model, uid, task, setting, depth, group_id, rate."""
    m = a.merge(b[["model", "uid", "rate"]], on=["model", "uid"], suffixes=("_a", "_b"))
    out = []
    for (model, task, setting, depth), h in m.groupby(["model", "task", "setting", "depth"]):
        d = (h.rate_b - h.rate_a).values
        lo, hi = cluster_boot_mean(d, h.group_id.values, n_boot, rng)
        out.append(dict(model=model, task=task, setting=setting, depth=depth, **label_cols, n_prompts=len(h),
                        rate_a=h.rate_a.mean(), rate_b=h.rate_b.mean(), delta=d.mean(), ci_lo=lo, ci_hi=hi,
                        perm_p=signflip_p(d, h.group_id.values, rng)))
    return pd.DataFrame(out)


def per_prompt(df):
    """Mean compliance per prompt (averaged over seeds)."""
    return (df.groupby(["model", "precision", "uid", "task", "setting", "depth", "group_id"], as_index=False)
              .compliant.mean().rename(columns={"compliant": "rate"}))


# ---------------------------------------------------------------- tables
def cells(df, n_boot, rng):
    out = []
    for k, g in df.groupby(KEYS):
        lo, hi = cluster_boot_mean(g["compliant"].values, g["group_id"].values, n_boot, rng)
        out.append(dict(zip(KEYS, k), n=len(g), n_seeds=g.seed.nunique(), rate=g["compliant"].mean(),
                        ci_lo=lo, ci_hi=hi, loose_rate=g["loose"].mean(), trunc_rate=g["truncated"].mean(),
                        think_unfinished_rate=g["think_unfinished"].mean(), degenerate_rate=g["degenerate"].mean(),
                        empty_reasoning_rate=g["empty_reasoning"].mean(),
                        bits=g["bits"].iloc[0]))
    return pd.DataFrame(out)


def delta_vs_base(df, base, n_boot, rng):
    pp = per_prompt(df)
    b = pp[pp.precision == base]
    if b.empty:
        return pd.DataFrame()
    parts = [paired_table(b, pp[pp.precision == p], n_boot, rng, {"precision": p, "baseline": base})
             for p in prec_order(df) if p != base]
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out.rename(columns={"rate_a": "base_rate", "rate_b": "rate"})


def noise_floor(df, base, n_boot, rng):
    """Baseline vs. itself under a different sampling seed: the change expected from sampling alone."""
    b = df[df.precision == base]
    seeds = sorted(b.seed.unique())
    if len(seeds) < 2:
        return pd.DataFrame()
    pa = per_prompt(b[b.seed == seeds[0]])
    pb = per_prompt(b[b.seed == seeds[1]])
    return paired_table(pa, pb, n_boot, rng, {"precision": f"{base} s{seeds[1]} vs s{seeds[0]}"})


def headline(dvb, nf):
    """Per-depth conflict compliance change vs. baseline, beside the noise floor."""
    if dvb.empty:
        return pd.DataFrame()
    h = dvb[dvb.setting == "conflict"].copy()
    if not nf.empty:
        n = nf[nf.setting == "conflict"][["model", "task", "depth", "delta", "ci_lo", "ci_hi"]].rename(
            columns={"delta": "noise_delta", "ci_lo": "noise_ci_lo", "ci_hi": "noise_ci_hi"})
        h = h.merge(n, on=["model", "task", "depth"], how="left")
        # A change counts as beyond noise if its CI excludes 0 AND it is larger than the noise-floor CI bound.
        h["beyond_noise"] = (h.ci_hi < 0) & (h.delta < h.noise_ci_lo) | (h.ci_lo > 0) & (h.delta > h.noise_ci_hi)
    cols = [c for c in ["model", "task", "depth", "precision", "base_rate", "rate", "delta", "ci_lo", "ci_hi",
                        "perm_p", "noise_delta", "noise_ci_lo", "noise_ci_hi", "beyond_noise", "n_prompts"] if c in h]
    return h[cols].sort_values(["model", "task", "depth", "precision"])


def gaps(cell_df):
    piv = cell_df.pivot_table(index=["model", "task", "precision", "depth"], columns="setting", values="rate")
    piv = piv.reset_index()
    if "aligned" in piv and "conflict" in piv:
        piv["gap_aligned_minus_conflict"] = piv["aligned"] - piv["conflict"]
    if "reference" in piv and "conflict" in piv:
        piv["gap_reference_minus_conflict"] = piv["reference"] - piv["conflict"]
    return piv


def did(df, base, n_boot, rng):
    """H1: does quantization hurt conflict compliance more than non-conflict ability?

    DiD = (conflict_q - conflict_base) - (control_q - control_base), control in {aligned, reference}.
    The absolute DiD is biased by floor effects: a conflict rate of .09 can fall at most 9 points,
    an aligned rate of .83 can fall 83. So two scale-free versions are reported as well:
      did_log_ratio  log(conflict_q/conflict_base) - log(control_q/control_base)   (relative retention)
      did_log_odds   same on the logit scale
    (counts smoothed by +0.5 success / +1 trial). Negative = conflict drops more, on every scale.
    Bootstrap resamples example groups (shared across settings for rule-following)."""
    out = []
    for (model, task), g in df.groupby(["model", "task"]):
        for control in ["aligned", "reference"]:
            depths = sorted(set(g[g.setting == control].depth) & set(g[g.setting == "conflict"].depth))
            for depth in depths:
                h = g[(g.depth == depth) & g.setting.isin(["conflict", control])]
                for prec in [p for p in prec_order(h) if p != base]:
                    hh = h[h.precision.isin([base, prec])]
                    if hh.precision.nunique() < 2:
                        continue
                    groups = np.unique(hh.group_id)
                    gi = {gid: i for i, gid in enumerate(groups)}
                    cells_ = {}
                    for (s, p), c in hh.groupby(["setting", "precision"]):
                        sums = np.zeros(len(groups)); cnts = np.zeros(len(groups))
                        np.add.at(sums, c.group_id.map(gi).values, c.compliant.values)
                        np.add.at(cnts, c.group_id.map(gi).values, 1)
                        cells_[(s, p)] = (sums, cnts)
                    if len(cells_) < 4:
                        continue

                    def stat(idx):
                        r, rs = {}, {}
                        for k, v in cells_.items():
                            x, n = v[0][idx].sum(), v[1][idx].sum()
                            r[k] = x / max(n, 1)
                            rs[k] = (x + 0.5) / (n + 1)  # smoothed, never 0 or 1
                        lg = {k: np.log(v) for k, v in rs.items()}
                        lo_ = {k: np.log(v / (1 - v)) for k, v in rs.items()}
                        return np.array([
                            (r[("conflict", prec)] - r[("conflict", base)]) - (r[(control, prec)] - r[(control, base)]),
                            (lg[("conflict", prec)] - lg[("conflict", base)]) - (lg[(control, prec)] - lg[(control, base)]),
                            (lo_[("conflict", prec)] - lo_[("conflict", base)]) - (lo_[(control, prec)] - lo_[(control, base)])])

                    full = stat(np.arange(len(groups)))
                    boots = np.array([stat(rng.integers(0, len(groups), len(groups))) for _ in range(n_boot)])
                    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
                    out.append(dict(model=model, task=task, control=control, depth=depth, precision=prec,
                                    did=full[0], ci_lo=lo[0], ci_hi=hi[0],
                                    did_log_ratio=full[1], lr_ci_lo=lo[1], lr_ci_hi=hi[1],
                                    did_log_odds=full[2], lo_ci_lo=lo[2], lo_ci_hi=hi[2],
                                    note="negative = conflict compliance drops more than control"))
    return pd.DataFrame(out)


def regression(df) -> str:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    c = df[df.setting == "conflict"].copy()
    if c.precision.nunique() < 2 or c.depth.nunique() < 2:
        return "regression skipped: need >=2 precisions and >=2 depths in conflict rows\n"
    order = prec_order(c)
    bit_levels = sorted(c.bits.dropna().unique(), reverse=True)
    c["quant_level"] = c.bits.map({b: i for i, b in enumerate(bit_levels)})  # 0 = most bits
    c["depth_turns"] = c.depth.astype(float)
    c["grp"] = pd.factorize(c.task + "|" + c.group_id)[0]
    txt = [f"Conflict prompts only, pooled over seeds. quant_level by bits: "
           f"{ {int(b): i for i, b in enumerate(bit_levels)} } (precisions: {order}).",
           "H3 predicts a NEGATIVE quant_level:depth_turns coefficient (compliance falls faster with depth at",
           "lower precision). filler_violations = number of filler assistant turns' violations of the system",
           "rule in that prompt, so depth_turns is read net of history content. Cluster-robust SEs by example.\n"]

    def covars(d):
        # filler_violations is constant (0) on the clean subset and for extraction; drop it there.
        return " + filler_violations" if d.filler_violations.nunique() > 1 else ""

    def fit(f, d, tables_only=False):
        try:
            res = smf.glm(f, data=d, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": d["grp"]})
            return str(res.summary().tables[1] if tables_only else res.summary())
        except Exception as e:  # perfect separation etc.
            return f"failed: {e}"

    for model, m in c.groupby("model"):
        if m.depth.nunique() < 2 or m.precision.nunique() < 2:
            txt += [f"=== model={model}: skipped (needs >=2 depths and >=2 precisions; has depths "
                    f"{sorted(m.depth.unique())})", ""]
            continue
        task_term = " + C(task)" if m.task.nunique() > 1 else ""
        f1 = "compliant ~ quant_level * depth_turns" + covars(m) + task_term
        f2 = f"compliant ~ C(precision, Treatment('{order[0]}')) * depth_turns" + covars(m) + task_term
        for f in (f1, f2):
            txt += [f"=== model={model}\n{f}", fit(f, m), ""]
        for task, t in m.groupby("task"):
            f = "compliant ~ quant_level * depth_turns" + covars(t)
            txt += [f"=== model={model} task={task}\n{f}", fit(f, t, tables_only=True), ""]
    return "\n".join(txt)


# ---------------------------------------------------------------- plots
def plots(cell_df, figdir: Path, title_suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir.mkdir(parents=True, exist_ok=True)
    order = sorted(cell_df.precision.unique(), key=lambda p: -PRECISION_BITS.get(p, 0))
    cmap = plt.get_cmap("viridis", max(len(order), 2))
    color = {p: cmap(i) for i, p in enumerate(order)}
    for (model, task), g in cell_df.groupby(["model", "task"]):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, setting in zip(axes, ["conflict", "aligned"]):
            for p in order:
                h = g[(g.setting == setting) & (g.precision == p)].sort_values("depth")
                if h.empty:
                    continue
                ax.errorbar(h.depth, h.rate, yerr=[h.rate - h.ci_lo, h.ci_hi - h.rate], marker="o",
                            capsize=3, label=p, color=color[p])
            ax.set_title(setting + (" (compliance with system)" if setting == "conflict" else " (accuracy)"))
            ax.set_xlabel("filler turns between system prompt and final user turn")
            ax.set_xticks(sorted(g.depth.unique()))
            ax.grid(alpha=.3)
        axes[0].set_ylabel("pass rate")
        axes[0].set_ylim(0, 1.02)
        axes[1].legend(title="precision")
        n_ex = g[g.setting == "conflict"].groupby("depth").n.first().max() // max(g.n_seeds.max(), 1)
        fig.suptitle(f"{model} - {task}{title_suffix} (n={n_ex} examples/depth)")
        fig.tight_layout()
        fig.savefig(figdir / f"compliance_vs_depth_{model}_{task}.png", dpi=150)
        plt.close(fig)

        d0 = g[g.depth == g.depth.min()]
        fig, ax = plt.subplots(figsize=(5, 4))
        for setting, ls in [("conflict", "-"), ("aligned", "--"), ("reference", ":")]:
            h = d0[d0.setting == setting].sort_values("bits", ascending=False)
            if h.empty:
                continue
            ax.errorbar(h.bits, h.rate, yerr=[h.rate - h.ci_lo, h.ci_hi - h.rate], marker="o", ls=ls,
                        capsize=3, label=setting)
        ax.set_xscale("log", base=2)
        ticks = sorted(d0.bits.dropna().unique())
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(b)) for b in ticks])
        ax.invert_xaxis()
        ax.set_xlabel("weight bits")
        ax.set_ylabel("pass rate")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=.3)
        ax.legend()
        ax.set_title(f"{model} - {task}{title_suffix} (depth {int(d0.depth.min())})")
        fig.tight_layout()
        fig.savefig(figdir / f"accuracy_vs_bits_{model}_{task}.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------- driver
def run_all(df, out: Path, args, rng, title_suffix=""):
    out.mkdir(parents=True, exist_ok=True)
    c = cells(df, args.n_boot, rng)
    c.to_csv(out / "cells.csv", index=False)
    # Baseline per model: --baseline if that model has it, else its highest-bit precision
    # (gpt-oss has no bf16 run: its "native" checkpoint is the full-precision model).
    dvbs, nfs, dids = [], [], []
    for model, m in df.groupby("model"):
        base = args.baseline if args.baseline in set(m.precision) else prec_order(m)[0]
        dvbs.append(delta_vs_base(m, base, args.n_boot, rng))
        nfs.append(noise_floor(m, base, args.n_boot, rng))
        dids.append(did(m, base, max(args.n_boot // 4, 200), rng))
    cat = lambda xs: pd.concat([x for x in xs if x is not None and len(x)], ignore_index=True) \
        if any(x is not None and len(x) for x in xs) else pd.DataFrame()
    dvb, nf = cat(dvbs), cat(nfs)
    dvb.to_csv(out / "delta_vs_base.csv", index=False)
    nf.to_csv(out / "noise_floor.csv", index=False)
    hl = headline(dvb, nf)
    hl.to_csv(out / "headline.csv", index=False)
    gaps(c).to_csv(out / "gap.csv", index=False)
    cat(dids).to_csv(out / "did.csv", index=False)
    (out / "regression.txt").write_text(regression(df))
    plots(c, out / "figs", title_suffix)
    return c, hl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--baseline", default="bf16")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = load(args.files or sorted(SCORED.glob("*.jsonl")))
    pd.set_option("display.width", 200)

    c, hl = run_all(df, args.out_dir, args, rng)
    cdf = clean_subset(df)
    n_clean = cdf[cdf.setting == "conflict"].groupby("task").example_id.nunique().to_dict()
    run_all(cdf, args.out_dir / "clean_subset", args, rng, title_suffix=" [clean fillers]")

    print("\nPass rates, all prompts (rows: task/setting/depth, cols: precision; pooled over seeds):")
    print(c.pivot_table(index=["model", "task", "setting", "depth"], columns="precision", values="rate")
          [prec_order(df)].round(3).to_string())
    if not hl.empty:
        print(f"\nHEADLINE: conflict compliance change vs. {args.baseline}, per depth (all prompts):")
        show = [x for x in ["task", "depth", "precision", "delta", "ci_lo", "ci_hi", "perm_p", "noise_ci_lo",
                            "noise_ci_hi", "beyond_noise"] if x in hl]
        print(hl[show].round(3).to_string(index=False))
    print(f"\nclean subset: conflict examples per task = {n_clean}")
    for col, label in [("trunc_rate", "hit max_tokens"), ("think_unfinished_rate", "ran out of tokens while thinking"),
                       ("degenerate_rate", "are degenerate repetition (model collapse: inspect before trusting)")]:
        worst = c[col].max()
        if worst > 0.02:
            print(f"WARNING: up to {worst:.1%} of responses in a cell {label}; raise decoding.max_tokens.")
    print(f"\nwrote {args.out_dir}/ and {args.out_dir}/clean_subset/: headline.csv cells.csv delta_vs_base.csv "
          "noise_floor.csv gap.csv did.csv regression.txt figs/")


if __name__ == "__main__":
    main()
