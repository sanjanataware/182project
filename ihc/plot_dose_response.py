"""Figure 1: dose-response of instruction-hierarchy compliance vs. ability for gpt-oss-20b (RTN and GPTQ).

  python ihc/plot_dose_response.py      # -> results/figs/oss_dose_response.{pdf,png}

Regenerated from the saved scored responses (scored/gpt-oss-20b__*.jsonl), depth 0.
  * retention = pass rate at a configuration / bf16 pass rate, per task and kind
      compliance = conflict setting; ability = aligned + reference settings pooled
  * 95% CIs: the same cluster bootstrap as the tables (plot_oss_curve.boot_ratio): 1,000 resamples of
    IHEval example groups, the same resampled groups used for the configuration and for bf16. The
    random-number stream is consumed in exactly the order plot_oss_curve.main uses, so the intervals
    are identical to results/oss_curve.csv (checked below), not merely re-drawn.
  * GPTQ (C4 calibration) was not run at 4b/g32: its lines start at 3b/g32, nothing is interpolated.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common import RESULTS
from plot_oss_curve import boot_ratio, load

CONFIGS = ["BF16", "4b/g32", "3b/g32", "3b/g192", "2b/g16", "2b/g32"]
PREC = {  # method -> {config label: precision tag}
    "RTN": {"4b/g32": "rtn_w4", "3b/g32": "rtn_w3", "3b/g192": "rtn_w3_g192", "2b/g16": "rtn_w2_g16", "2b/g32": "rtn_w2"},
    "GPTQ": {"3b/g32": "gptq_w3", "3b/g192": "gptq_w3_g192", "2b/g16": "gptq_w2_g16", "2b/g32": "gptq_w2"},
}
KINDS = {"compliance (conflict)": "Compliance", "ability (aligned+reference)": "Ability"}
# Okabe-Ito blue / orange: distinguishable with colour-vision deficiency and dark vs light in greyscale;
# marker shape (circle / square) carries the same distinction redundantly.
COLOR = {"Compliance": "#0b4f9c", "Ability": "#e69f00"}
MARKER = {"Compliance": "o", "Ability": "s"}
STYLE = {"RTN": "-", "GPTQ": "--"}
DODGE = {("RTN", "Compliance"): -0.09, ("GPTQ", "Compliance"): -0.03,
         ("RTN", "Ability"): 0.03, ("GPTQ", "Ability"): 0.09}


def retention_table(n_boot: int = 1000) -> pd.DataFrame:
    """Same loop, data and RNG order as plot_oss_curve.main -> identical point estimates and CIs."""
    rng = np.random.default_rng(0)
    d = load()
    rows = []
    for (task, kind), t in d.groupby(["task", "kind"]):
        base = t[t.precision == "bf16"]
        for prec, g in t.groupby("precision"):
            r, lo, hi = boot_ratio(g, base, n_boot, rng)   # every precision, to keep the RNG stream aligned
            rows.append(dict(task=task, kind=kind, precision=prec, retention=r, ci_lo=lo, ci_hi=hi,
                             rate=g.compliant.mean(), n=len(g), n_seeds=g.seed.nunique()))
    return pd.DataFrame(rows)


def main():
    o = retention_table()
    # consistency with the tables' CSV (written by plot_oss_curve.py)
    ref = pd.read_csv(RESULTS / "oss_curve.csv")
    m = o.merge(ref, on=["task", "kind", "precision"], suffixes=("", "_ref"))
    dev = float(np.nanmax(np.abs(m[["retention", "ci_lo", "ci_hi"]].values - m[["retention_ref", "ci_lo_ref", "ci_hi_ref"]].values)))
    print(f"max |difference| vs results/oss_curve.csv over {len(m)} cells: {dev:.2e}")

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    x = {c: i for i, c in enumerate(CONFIGS)}
    printed = []
    for ax, (task, title) in zip(axes, [("extraction", "(a) Extraction"), ("rule_following", "(b) Rule-following")]):
        ax.axhline(1.0, color="0.75", lw=0.9, zorder=0)
        for method, cfgs in PREC.items():
            for kind, kl in KINDS.items():
                xs, ys, lo, hi = [x["BF16"]], [1.0], [0.0], [0.0]
                for c, prec in cfgs.items():
                    r = o[(o.task == task) & (o.kind == kind) & (o.precision == prec)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    xs.append(x[c]); ys.append(r.retention)
                    lo.append(r.retention - r.ci_lo); hi.append(r.ci_hi - r.retention)
                    printed.append(dict(task=task, measure=kl, method=method, config=c, retention=r.retention,
                                        ci_lo=r.ci_lo, ci_hi=r.ci_hi, pass_rate=r.rate, n=r.n, seeds=r.n_seeds))
                if method == "GPTQ":   # GPTQ has no 4b/g32 point: draw BF16 as its own marker, no connecting line
                    ax.plot([xs[0] + DODGE[(method, kl)]], [1.0], MARKER[kl], color=COLOR[kl], ms=5.5, mfc="white", mew=1.4)
                    xs, ys, lo, hi = xs[1:], ys[1:], lo[1:], hi[1:]
                xd = np.array(xs) + DODGE[(method, kl)]
                ax.errorbar(xd, ys, yerr=[lo, hi], ls=STYLE[method], marker=MARKER[kl], color=COLOR[kl], lw=1.8,
                            ms=5.5, mfc=COLOR[kl] if method == "RTN" else "white", mew=1.4,
                            elinewidth=1.0, capsize=2.5, capthick=1.0)
        ax.set(title=title, xticks=range(len(CONFIGS)), xlim=(-0.4, len(CONFIGS) - 0.6), ylim=(0, 1.05))
        ax.set_xticklabels(CONFIGS)
        ax.set_xlabel("expert-weight configuration (bits / group size)")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("retention relative to BF16")
    handles = [Line2D([], [], color=COLOR[k], marker=MARKER[k], lw=2.2, ms=6) for k in ("Compliance", "Ability")]
    handles += [Line2D([], [], color="0.25", ls=STYLE["RTN"], marker="o", mfc="0.25", lw=1.8),
                Line2D([], [], color="0.25", ls=STYLE["GPTQ"], marker="o", mfc="white", lw=1.8)]
    fig.legend(handles, ["Compliance (conflict)", "Ability (aligned + reference)", "RTN", "GPTQ (C4 calibration)"],
               loc="upper center", ncol=4, frameon=False, fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, .92))
    for ext in ("pdf", "png"):
        out = RESULTS / "figs" / f"oss_dose_response.{ext}"
        plt.savefig(out, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print("wrote", out)

    p = pd.DataFrame(printed)
    p["config"] = pd.Categorical(p.config, CONFIGS, ordered=True)
    p = p.sort_values(["task", "measure", "method", "config"])
    print("\nplotted retention (95% cluster-bootstrap CI), pass rate, n responses, seeds")
    for _, r in p.iterrows():
        print(f"  {r.task:15s} {r.measure:10s} {r.method:4s} {r.config:8s} {r.retention:5.2f} "
              f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}]  rate {r.pass_rate:.3f}  n={r.n}  seeds={r.seeds}")
    p.to_csv(RESULTS / "oss_dose_response_values.csv", index=False)


if __name__ == "__main__":
    main()
