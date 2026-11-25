"""E2E-QP training curves for gpt-oss-20b 2-bit EfficientQAT (main-text figure) + reproducibility table.

  modal volume get ihc-checkpoints /_curves results/eqat/   # then
  python ihc/plot_eqat_curves.py

Reads results/eqat/_curves/eqat-w2-g32-cal-{c4-curves,syschat-curves,syschat-asst}.json
(written by quantize_oss_modal.build_eqat(curves=True)).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CUR = ROOT / "results" / "eqat" / "_curves"
ARMS = {  # file stem -> (label, colour, saved endpoints from the evaluated checkpoints or None)
    "eqat-w2-g32-cal-c4-curves": ("C4, all-token loss", "#2a6fdb", dict(first=4.2058, last=2.9254, ce_c4=3.0783, ce_chat=1.3884)),
    "eqat-w2-g32-cal-syschat-curves": ("Chats, all-token loss", "#d9822b", dict(first=1.0687, last=0.9068, ce_c4=4.2432, ce_chat=0.9272)),
    "eqat-w2-g32-cal-syschat-asst": ("Chats, assistant-token loss", "#2e9e6a", None),
}


def smooth(y, k=8):
    out = []
    for i in range(len(y)):
        w = y[max(0, i - k + 1):i + 1]
        out.append(sum(w) / len(w))
    return out


def main():
    arms = {k: json.load(open(CUR / f"{k}.json")) for k in ARMS if (CUR / f"{k}.json").exists()}
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    off = {k: d for k, d in zip(ARMS, (-0.012, 0.0, 0.012))}   # small vertical offsets so 0/30 lines don't hide each other
    for k, c in arms.items():
        lab, col, _ = ARMS[k]
        s = [r["step"] for r in c["train"]]; y = [r["loss"] for r in c["train"]]
        ax[0].plot(s, y, color=col, alpha=.18, lw=.8)
        ax[0].plot(s, smooth(y), color=col, lw=1.8)
        h = c["heldout"]
        ax[1].plot([r["step"] for r in h], [r["syschat"] for r in h], "-o", ms=3, color=col)
        ax[1].plot([r["step"] for r in h], [r["c4nl"] for r in h], ":", color=col, lw=1.6)
        p = c["probe"]; n = p[0]["n"]
        ax[2].plot([r["step"] for r in p], [r["invents_turn"] / n + off[k] for r in p], "-o", ms=4, color=col)
        ax[2].plot([r["step"] for r in p], [r["skips_reasoning"] / n + off[k] for r in p], "--^", ms=4, color=col, alpha=.75)
    ax[0].set(xlabel="E2E-QP optimizer step", ylabel="training loss (nats)", title="(a) Training loss (line: 8-step mean)")
    ax[1].axhline(2.18, color="gray", lw=.8, ls="--"); ax[1].text(255, 2.08, "bf16, chats", color="gray", fontsize=8, ha="right")
    ax[1].axhline(4.21, color="gray", lw=.8, ls=":"); ax[1].text(255, 4.11, "bf16, C4", color="gray", fontsize=8, ha="right")
    ax[1].set(xlabel="E2E-QP optimizer step", ylabel="held-out loss (nats)", title="(b) Held-out loss, every 16 steps")
    ax[1].legend([Line2D([], [], color="k", marker="o", ms=3), Line2D([], [], color="k", ls=":")],
                 ["chats (all tokens)", "C4 (blank-line separated)"], fontsize=8, loc="center right")
    ax[2].set(xlabel="E2E-QP optimizer step", ylabel="fraction of 30 probe prompts", ylim=(-.05, 1.05),
              title="(c) Format probe, every 32 steps")
    ax[2].legend([Line2D([], [], color="k", marker="o", ms=4), Line2D([], [], color="k", ls="--", marker="^", ms=4)],
                 ["invents a new turn", "skips reasoning"], fontsize=8, loc="center right")
    for a in ax:
        a.grid(alpha=.25)
    fig.legend([Line2D([], [], color=ARMS[k][1], lw=2.5) for k in arms], [ARMS[k][0] for k in arms],
               loc="upper center", ncol=3, frameon=False, fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, .92))
    out = ROOT / "results" / "figs" / "oss_e2e_curves.png"
    plt.savefig(out, dpi=150); print("wrote", out)

    print("\nreproducibility (mean of first / last 32 steps; final held-out CE with <|endoftext|>-joined C4)")
    for k, c in arms.items():
        lab, _, ref = ARMS[k]
        e, ce = c.get("e2e", {}), c.get("ce_final", {})
        print(f"  {lab:30s} rerun {e.get('loss_first', float('nan')):.3f} -> {e.get('loss_last', float('nan')):.3f}"
              f"  ce c4 {ce.get('c4', float('nan')):.3f} chats {ce.get('syschat', float('nan')):.3f}", end="")
        if ref:
            print(f"   | saved {ref['first']:.3f} -> {ref['last']:.3f}  ce c4 {ref['ce_c4']:.3f} chats {ref['ce_chat']:.3f}")
        else:
            print()
        print("    probes:", [(r["step"], r["invents_turn"], r["skips_reasoning"], r["reaches_final"]) for r in c["probe"]])


if __name__ == "__main__":
    main()
