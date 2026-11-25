"""Appendix figure: Block-AP per-layer reconstruction error, gpt-oss-20b 2b g32.

  python ihc/plot_blockap_layers.py     # -> results/figs/oss_blockap_layers.png, results/oss_blockap_layers.csv

Reads results/eqat/blockap_stats_{c4,syschat}.json (written by quantize_oss_modal.build_eqat during
Block-AP; one record per transformer layer).
  held-out: MSE of the layer's output on held-out calibration-source chunks after Block-AP,
            divided by the same MSE at the RTN initialisation (heldout_mse_ratio)
  training: mean training loss over the last 1/16 of that layer's steps divided by the mean over the
            first 1/16 (train_loss_last / train_loss_first). The first 1/16 is close to, but not
            exactly, RTN, since step sizes are already moving.
"""
from __future__ import annotations

import json

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RESULTS

ARMS = {"c4": "C4 calibration", "syschat": "Chats + system prompt calibration"}


def main():
    rows = []
    for k in ARMS:
        for r in json.load(open(RESULTS / "eqat" / f"blockap_stats_{k}.json"))["block_ap"]:
            rows.append(dict(calib=k, layer=r["layer"], heldout_ratio=r["heldout_mse_ratio"],
                             train_ratio=r["train_loss_last"] / r["train_loss_first"],
                             frac_codes_changed=r["frac_codes_changed_vs_rtn"]))
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "oss_blockap_layers.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (k, title) in zip(axes, ARMS.items()):
        t = d[d.calib == k]
        ax.plot(t.layer, t.heldout_ratio, "-o", ms=4, color="#2a6fdb", label="Held-out: after Block-AP / RTN")
        ax.plot(t.layer, t.train_ratio, "--s", ms=4, color="gray", label="Training: last / first 1/16 of steps")
        ax.axhline(1.0, color="k", lw=0.7)
        ax.set(title=title, xlabel="Layer", ylim=(0, 1.05))
        ax.grid(alpha=.3)
    axes[0].set_ylabel("Layer-output MSE ratio (lower is better)")
    axes[0].legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    out = RESULTS / "figs" / "oss_blockap_layers.png"
    plt.savefig(out, dpi=150)
    print("wrote", out)
    for k in ARMS:
        t = d[d.calib == k]
        print(f"  {k:8s} held-out ratio {t.heldout_ratio.min():.2f}-{t.heldout_ratio.max():.2f}; "
              f"held-out <= training in {(t.heldout_ratio <= t.train_ratio).sum()}/{len(t)} layers")


if __name__ == "__main__":
    main()
