"""Dose-response for gpt-oss-20b: conflict compliance vs. ability as expert-weight error grows.

  python ihc/plot_oss_curve.py      # -> results/oss_curve.csv, results/figs/oss_curve.png

x = median relative weight error of the expert tensors vs. the native weights (from each
checkpoint's ihc_quant_meta.json, recorded below), so levels with different bit widths and group
sizes sit on one axis. y = pass rate as a fraction of the bf16 rate ("retention"), so compliance and
ability are on the same scale even though they start at different levels. Ability = aligned and
reference settings pooled. 95% CIs: cluster bootstrap over IHEval example groups, pooled over seeds.
"""
from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RESULTS, SCORED

MODEL = "gpt-oss-20b"
# median expert weight_rel_err_vs_native, from /ckpt/gpt-oss-20b-*/ihc_quant_meta.json
LEVELS = {  # precision: (weight error, label, effective bits/weight incl. fp16 scale+zero per group)
    "bf16": (0.0, "bf16 (native values)", 4.25),
    "rtn_w4": (0.0795, "4b g32", 4 + 32 / 32),
    "rtn_w3": (0.1856, "3b g32", 3 + 32 / 32),
    "rtn_w3_g192": (0.2613, "3b g192", 3 + 32 / 192),
    "rtn_w2_g8": (0.2910, "2b g8", 2 + 32 / 8),
    "rtn_w2_g16": (0.3573, "2b g16", 2 + 32 / 16),
    "rtn_w2": (0.4218, "2b g32", 2 + 32 / 32),
}


# GPTQ on the same grids: precision -> (matching RTN level, label)
GPTQ = {"gptq_w3": ("rtn_w3", "3b g32"), "gptq_w3_g192": ("rtn_w3_g192", "3b g192"),
        "gptq_w2_g16": ("rtn_w2_g16", "2b g16"), "gptq_w2": ("rtn_w2", "2b g32")}


# Calibration-source experiment: GPTQ 2b g32, only the calibration data differs.
CALIB = {"gptq_w2": "C4 web text", "gptq_w2_cal_nosys": "chats, no system prompt",
         "gptq_w2_cal_syschat": "same chats + system prompt", "gptq_w2_cal_iheval": "IHEval-train"}


# Method x calibration at 2b g32: one-shot (RTN, GPTQ) vs gradient-trained (EfficientQAT).
METHODS = {"rtn_w2": "RTN", "gptq_w2": "GPTQ\nC4", "gptq_w2_cal_syschat": "GPTQ\nsys-chats",
           "blockap_w2": "Block-AP only\nC4", "blockap_w2_cal_syschat": "Block-AP only\nsys-chats",
           "eqat_w2": "EfficientQAT\nC4", "eqat_w2_cal_syschat": "EfficientQAT\nsys-chats"}


def load():
    rows = []
    for f in sorted(SCORED.glob(f"{MODEL}__*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    d = pd.DataFrame(rows)
    d = d[d.precision.isin(list(LEVELS) + list(GPTQ) + list(CALIB) + list(METHODS)) & (d.depth == 0)]
    d["kind"] = np.where(d.setting == "conflict", "compliance (conflict)", "ability (aligned+reference)")
    return d


def boot_ratio(g, base, n_boot, rng):
    """Retention = rate(level)/rate(bf16), resampling example groups jointly for both."""
    groups = np.unique(np.concatenate([g.group_id.values, base.group_id.values]))
    gi = {x: i for i, x in enumerate(groups)}

    def sums(df):
        s = np.zeros(len(groups)); n = np.zeros(len(groups))
        np.add.at(s, df.group_id.map(gi).values, df.compliant.values.astype(float))
        np.add.at(n, df.group_id.map(gi).values, 1)
        return s, n

    (sq, nq), (sb, nb) = sums(g), sums(base)
    stat = lambda i: (sq[i].sum() / nq[i].sum()) / max(sb[i].sum() / nb[i].sum(), 1e-9)
    full = stat(np.arange(len(groups)))
    bs = [stat(rng.integers(0, len(groups), len(groups))) for _ in range(n_boot)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return full, lo, hi


def main(n_boot=1000):
    rng = np.random.default_rng(0)
    d = load()
    out = []
    for (task, kind), t in d.groupby(["task", "kind"]):
        base = t[t.precision == "bf16"]
        for prec, g in t.groupby("precision"):
            r, lo, hi = boot_ratio(g, base, n_boot, rng)
            if prec.startswith(("eqat", "blockap")):
                method, label, err, eff = "eqat", METHODS[prec].replace("\n", " "), np.nan, 3.0
            elif prec in CALIB and prec != "gptq_w2":
                method, label, err, eff = "gptq_calib", CALIB[prec], np.nan, 3.0
            elif prec in GPTQ:
                method, (_, label) = "gptq", GPTQ[prec]
                err, eff = np.nan, LEVELS[GPTQ[prec][0]][2]
            else:
                method = "none" if prec == "bf16" else "rtn"
                err, label, eff = LEVELS[prec]
            out.append(dict(task=task, kind=kind, precision=prec, method=method, label=label,
                            weight_err=err, eff_bits=eff,
                            rate=g.compliant.mean(), n=len(g), n_seeds=g.seed.nunique(),
                            retention=r, ci_lo=lo, ci_hi=hi))
    o = pd.DataFrame(out).sort_values(["task", "kind", "weight_err"])
    o.to_csv(RESULTS / "oss_curve.csv", index=False)
    print(o.round(3).to_string(index=False))

    if (o.method == "gptq").any():
        compare_methods(o[~o.method.isin(["gptq_calib", "eqat"])])
    if (o.method == "gptq_calib").any():
        compare_calibration(o)
    if (o.method == "eqat").any():
        compare_eqat(o)
    o = o[~o.method.isin(["gptq", "gptq_calib", "eqat"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (task, t) in zip(axes, o.groupby("task")):
        for kind, k in t.groupby("kind"):
            ax.errorbar(k.weight_err, k.retention, yerr=[k.retention - k.ci_lo, k.ci_hi - k.retention],
                        marker="o", capsize=3, label=kind)
        for _, r in t[t.kind.str.startswith("compliance")].iterrows():
            ax.annotate(r.label, (r.weight_err, r.retention), textcoords="offset points", xytext=(4, -12),
                        fontsize=7, color="gray")
        ax.axhline(1, color="gray", lw=0.6, ls=":")
        ax.set_title(f"{MODEL} - {task} (depth 0)")
        ax.set_xlabel("expert weight error vs. native (relative L2)")
        ax.set_ylim(0, 1.15)
    axes[0].set_ylabel("pass rate / bf16 pass rate")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    (RESULTS / "figs").mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "figs" / "oss_curve.png", dpi=150)
    print("wrote", RESULTS / "oss_curve.csv", "and", RESULTS / "figs" / "oss_curve.png")


def compare_methods(o):
    """RTN vs GPTQ at each shared grid: retention of compliance and ability."""
    labels = [v[1] for v in GPTQ.values()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    x = np.arange(len(labels))
    style = {("rtn", "compliance"): ("tab:orange", "o", "--"), ("gptq", "compliance"): ("tab:orange", "s", "-"),
             ("rtn", "ability"): ("tab:blue", "o", "--"), ("gptq", "ability"): ("tab:blue", "s", "-")}
    for ax, (task, t) in zip(axes, o.groupby("task")):
        for (method, kind), (c, m, ls) in style.items():
            k = t[(t.method == method) & t.kind.str.startswith(kind)].set_index("label").reindex(labels)
            off = -0.06 if method == "rtn" else 0.06
            ax.errorbar(x + off, k.retention, yerr=[k.retention - k.ci_lo, k.ci_hi - k.retention], color=c,
                        marker=m, ls=ls, capsize=3, label=f"{kind} - {method.upper()}")
        ax.axhline(1, color="gray", lw=0.6, ls=":")
        ax.set_xticks(x, labels)
        ax.set_title(f"{MODEL} - {task}: RTN vs GPTQ (depth 0)")
        ax.set_xlabel("expert quantization grid (bits, group size)")
        ax.set_ylim(0, 1.15)
    axes[0].set_ylabel("pass rate / bf16 pass rate")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(RESULTS / "figs" / "oss_rtn_vs_gptq.png", dpi=150)


def compare_calibration(o):
    """GPTQ 2b g32 under four calibration sets: retention of compliance vs ability."""
    keys = list(CALIB)
    labels = [CALIB[k] for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    x = np.arange(len(keys))
    for ax, (task, t) in zip(axes, o.groupby("task")):
        for kind, c, off in [("compliance", "tab:orange", 0.18), ("ability", "tab:blue", -0.18)]:
            k = t[t.kind.str.startswith(kind)].set_index("precision").reindex(keys)
            ax.bar(x + off, k.retention, width=0.34, color=c, alpha=0.85, label=kind,
                   yerr=[k.retention - k.ci_lo, k.ci_hi - k.retention], capsize=3)
        ax.axhline(1, color="gray", lw=0.6, ls=":")
        ax.set_xticks(x, labels, rotation=15, fontsize=8)
        ax.set_title(f"{task}: GPTQ 2-bit g32, by calibration data", fontsize=10)
        ax.set_ylim(0, 1.15)
    axes[0].set_ylabel("pass rate / bf16 pass rate")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "figs" / "oss_calibration.png", dpi=150)


def compare_eqat(o):
    """2b g32: retention of compliance vs ability by quantizer and calibration data."""
    keys = [k for k in METHODS if k in set(o.precision)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), sharey=True)
    x = np.arange(len(keys))
    for ax, (task, t) in zip(axes, o.groupby("task")):
        for kind, c, off in [("compliance", "tab:orange", 0.18), ("ability", "tab:blue", -0.18)]:
            k = t[t.kind.str.startswith(kind)].set_index("precision").reindex(keys)
            ax.bar(x + off, k.retention, width=0.34, color=c, alpha=0.85, label=kind,
                   yerr=[k.retention - k.ci_lo, k.ci_hi - k.retention], capsize=3)
        ax.axhline(1, color="gray", lw=0.6, ls=":")
        ax.set_xticks(x, [METHODS[k] for k in keys], fontsize=7)
        ax.set_title(f"{task}: 2-bit g32 experts, by quantizer", fontsize=10)
        ax.set_ylim(0, 1.15)
    axes[0].set_ylabel("pass rate / bf16 pass rate")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "figs" / "oss_eqat.png", dpi=150)


if __name__ == "__main__":
    main()
